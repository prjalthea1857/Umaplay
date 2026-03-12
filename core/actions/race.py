# core/actions/race.py
from __future__ import annotations

import random
import time
from core.controllers.android import ScrcpyController
from core.perception.analyzers.matching.race_banner import get_race_banner_matcher
from core.perception.yolo.interface import IDetector
from core.utils.waiter import Waiter
from typing import Dict, List, Optional, Tuple
from core.utils.race_index import RaceIndex

from PIL import Image
import cv2
import json
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import imagehash
from core.utils.img import to_bgr

from core.controllers.base import IController
from core.perception.analyzers.badge import (
    BADGE_PRIORITY,
    BADGE_PRIORITY_REVERSE,
    _badge_label,
)
from core.perception.is_button_active import ActiveButtonClassifier
from core.settings import Settings
from core.types import DetectionDict
from core.utils.geometry import crop_pil
from core.utils.logger import logger_uma
from core.utils.text import _normalize_ocr, fuzzy_ratio
from core.utils.yolo_objects import collect, find, bottom_most, inside
from core.utils.pointer import smart_scroll_small
from core.utils.abort import abort_requested, request_abort


class ConsecutiveRaceRefused(Exception):
    """Raised when a consecutive-race penalty is detected and settings forbid accepting it."""

    pass


class RaceFailureReason(Enum):
    NONE = "none"
    NAV_CONSECUTIVE_REFUSED = "nav_consecutive_refused"
    NO_RACE_SQUARE = "no_race_square"
    RACE_BUTTON_LIST_MISSING = "race_button_list_missing"
    ABORT_DURING_POPUP = "abort_during_popup"
    ABORT_DURING_LOBBY_WAIT = "abort_during_lobby_wait"
    PRE_LOBBY_TIMEOUT = "pre_lobby_timeout"
    LOBBY_FLOW_FAILED = "lobby_flow_failed"


class RaceFlow:
    """
    Clean, modular Race flow:
      - Self-contained: can start from Lobby and drive to Raceday, run, and exit.
      - One Waiter instance (no ad-hoc loops for waiting).
      - YOLO helpers & pointer utilities are reused.
      - OCR is used only when texts=... is provided.
    """

    def __init__(
        self, ctrl: IController, ocr, yolo_engine: IDetector, waiter: Waiter, *, telemetry=None,
    ) -> None:
        self.ctrl = ctrl
        self.ocr = ocr
        self.yolo_engine = yolo_engine
        self.waiter = waiter
        self._telemetry = telemetry
        self._banner_matcher = get_race_banner_matcher()
        self._race_result_counters = {
            "loss_indicators": 0,
            "retry_clicks": 0,
            "retry_skipped": 0,
            "wins_or_no_loss": 0,
        }
        self._waiting_for_manual_retry_decision = False
        self._last_failure_reason: RaceFailureReason = RaceFailureReason.NONE
        self._last_race_name: Optional[str] = None

    def _ensure_in_raceday(
        self, *, reason: str | None = None, from_raceday=False
    ) -> bool:
        """
        Idempotent. If we're still in the Lobby, click the lobby 'RACES' tile (or the
        'race_race_day' entry) to enter Raceday; tolerate the consecutive OK popup.
        """
        # Quick probe: do we already see race squares?
        try:
            img, dets = self._collect("race_nav_probe")
            squares = find(dets, "race_square")
            if squares:
                return True
        except Exception:
            # If detection fails for any reason, try to navigate anyway.
            pass
        if reason:
            logger_uma.debug(f"Looking for race buttons: {reason}")
        if Settings.ACTIVE_SCENARIO == "unity_cup":
            # Little delay before pressing race
            time.sleep(1.5)
        # Try to enter race screen from lobby (idempotent)
        clicked = self.waiter.click_when(
            classes=("lobby_races", "race_race_day"),
            prefer_bottom=True,
            timeout_s=2.5,
            tag="race_nav_from_lobby",
        )
        if clicked:
            logger_uma.debug(
                "Clicked 'RACES'. Fast-probing for squares vs penalty popup…"
            )
            # Fast race: as soon as 'race_square' is seen, bail out; otherwise opportunistically click OK.
            t0 = time.time()
            MAX_WAIT = 2.2  # upper bound; typical exit << 1.0s
            while (time.time() - t0) < MAX_WAIT:
                if abort_requested():
                    logger_uma.info("[race] Abort requested during nav to Raceday.")
                    return False
                # 1) If squares are already visible → done
                if self.waiter.seen(
                    classes=("race_square",), tag="race_nav_seen_squares"
                ):
                    return True
                # 2) If consecutive-race penalty popup is present → honor settings
                if self.waiter.seen(
                    classes=("button_green",),
                    texts=("OK",),
                    tag="race_nav_penalty_seen",
                ):
                    # from_raceday forces to accept consecutive, there is no another option
                    if not Settings.ACCEPT_CONSECUTIVE_RACE and not from_raceday:
                        logger_uma.info(
                            "[race] Consecutive race detected and refused by settings."
                        )
                        raise ConsecutiveRaceRefused(
                            "Consecutive race not accepted by settings."
                        )
                    # Accept the penalty promptly (single shot, no wait)
                    self.waiter.click_when(
                        classes=("button_green",),
                        texts=("OK",),
                        prefer_bottom=False,
                        allow_greedy_click=False,
                        timeout_s=0.5,
                        tag="race_nav_penalty_ok_click",
                    )
                    logger_uma.debug(
                        "Consecutive race. Accepted penalization per settings."
                    )

                time.sleep(0.12)
            # If loop expires, do one last probe:
            if self.waiter.seen(classes=("race_square",), tag="race_nav_seen_final"):
                return True
            return False
        return False

    # --------------------------
    # Internal helpers
    # --------------------------
    def _collect(self, tag: str) -> Tuple[Image.Image, List[DetectionDict]]:
        return collect(
            self.yolo_engine,
            imgsz=self.waiter.cfg.imgsz,
            conf=self.waiter.cfg.conf,
            iou=self.waiter.cfg.iou,
            tag=tag,
            agent=self.waiter.cfg.agent,
        )

    def _attempt_try_again_retry(self) -> bool:
        """Click the 'TRY AGAIN' button once loss was confirmed."""
        t0 = time.time()
        timeout_s = 2.0
        while (time.time() - t0) < timeout_s:
            clicked, det = self.waiter.click_when(
                classes=("button_green",),
                texts=("TRY AGAIN",),
                prefer_bottom=False,
                allow_greedy_click=False,
                timeout_s=0.3,
                forbid_texts=("RACE", "NEXT"),
                tag="race_try_again_try",
                return_object=True,
            )
            if clicked:
                bbox = det.get("xyxy") if det else None
                y_center = None
                if bbox:
                    _, y1, _, y2 = bbox
                    y_center = 0.5 * (y1 + y2)
                self._race_result_counters["retry_clicks"] += 1
                logger_uma.info(
                    "[race] TRY AGAIN clicked (y_center=%s) | counters=%s",
                    f"{y_center:.1f}" if y_center is not None else "?",
                    self._race_result_counters,
                )
                return True
            time.sleep(0.12)

        logger_uma.info(
            "[race] TRY AGAIN not clicked before timeout | counters=%s",
            self._race_result_counters,
        )
        return False

    def _handle_retry_transition(self) -> None:
        """Clear alarm-clock confirmations and wait until lobby buttons reappear."""
        logger_uma.debug("[race] Handling retry transition interstitials.")
        confirm_texts = ("USE", "USE ITEM", "TRY AGAIN", "RACE", "YES")
        cleanup_texts = ("OK", "CONFIRM")
        deadline = time.time() + 10.0

        while time.time() < deadline:
            if self.waiter.try_click_once(
                classes=("button_green",),
                texts=confirm_texts + cleanup_texts,
                prefer_bottom=False,
                allow_greedy_click=False,
                forbid_texts=("NEXT",),
                tag="race_try_again_confirm",
            ):
                logger_uma.debug("[race] Clicked retry interstitial confirmation.")
                time.sleep(0.45)
                continue

            if self.waiter.seen(
                classes=("button_white",),
                texts=("VIEW RESULTS",),
                tag="race_retry_view_results_ready",
            ):
                logger_uma.debug("[race] View Results ready after retry.")
                return

            if self.waiter.seen(
                classes=("button_green",),
                texts=("RACE",),
                tag="race_retry_race_ready",
            ):
                logger_uma.debug("[race] Race button ready after retry.")
                return

            time.sleep(0.35)

        logger_uma.warning("[race] Retry transition timed out; continuing anyway.")

    def _deduplicate_stars(self, stars: List[DetectionDict]) -> List[DetectionDict]:
        """
        Remove duplicate star detections by filtering overlapping bboxes.
        YOLO sometimes detects the same star twice with different confidences.
        """
        if len(stars) <= 1:
            return stars
        
        # Sort by confidence (highest first) to keep better detections
        sorted_stars = sorted(stars, key=lambda d: d.get("conf", 0.0), reverse=True)
        keep = []
        
        for star in sorted_stars:
            # Check if this star overlaps significantly with any already kept
            sx1, sy1, sx2, sy2 = star["xyxy"]
            s_area = (sx2 - sx1) * (sy2 - sy1)
            
            is_duplicate = False
            for kept in keep:
                kx1, ky1, kx2, ky2 = kept["xyxy"]
                # Calculate intersection
                ix1, iy1 = max(sx1, kx1), max(sy1, ky1)
                ix2, iy2 = min(sx2, kx2), min(sy2, ky2)
                
                if ix2 > ix1 and iy2 > iy1:
                    intersection = (ix2 - ix1) * (iy2 - iy1)
                    # If intersection is > 50% of star area, consider it a duplicate
                    if intersection / s_area > 0.5:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                keep.append(star)
        
        return keep

    def _pick_view_results_button(self) -> Optional[DetectionDict]:
        """Among white buttons, choose the one that OCR-matches 'VIEW RESULTS' best."""
        img, dets = self._collect("race_view_btn")
        whites = find(dets, "button_white")
        if not whites:
            return None

        best_d, best_s = None, 0.0
        for d in whites:
            txt = (self.ocr.text(crop_pil(img, d["xyxy"])) or "").strip()
            score = max(
                fuzzy_ratio(txt, "VIEW RESULTS"), fuzzy_ratio(txt, "VIEW RESULT")
            )
            if score > best_s and score > 0.01:
                best_d, best_s = d, score
        return best_d

    def _pick_race_square(
        self,
        *,
        prioritize_g1: bool,
        is_g1_goal: bool,
        desired_race_name: Optional[str] = None,
        max_scrolls: int = 3,
        date_key: Optional[str] = None,
    ) -> Tuple[Optional[DetectionDict], bool]:
        """
        Try to find a clickable race card:
          - consider 'race_square'
          - valid if it has >= 2 'race_star'
          - badge via color→OCR fallback
          - prioritize first G1 if requested; else EX>G1>G2>G3>OP (tie: topmost)
          - if desired_race_name is provided → do a one-way forward search for that name;
            if not found after scrolling up to max_scrolls, return (None, True) without fallback.

        """
        MINIMUM_RACE_OCR_MATCH = 0.91
        MIN_STARS = 2
        # OCR-based gating: minimum OCR score to accept a candidate during tie-breaks.
        OCR_DISCARD_MIN = 0.3
        # OCR signal weight to gently separate close candidates in the tie-breaker.
        OCR_SCORE_WEIGHT = 0.2
        moved_cursor = False
        did_scroll = False
        first_top_xyxy = None

        def clean_race_name(st: str) -> str:
            n_txt = (
                st.upper()
                .replace("RIGHT", "")
                .replace("TURT", "TURF")
                .replace("DIRF", "DIRT")
                .replace("LEFT", "")
                .replace("INNER", "")
                .replace("1NNER", "")
                .replace("OUTER", "")
                .replace("/", "")
                .strip()
            )
            n_txt = _normalize_ocr(n_txt)
            # remove isolated characters for example 'turf f 1b00' -> 'turf 1b00' we removed the isolated 'f'
            words = n_txt.split()
            cleaned_words = [word.strip() for word in words if len(word) > 1]
            n_txt = ' '.join(cleaned_words).strip()
            return n_txt
        best_non_g1: Optional[DetectionDict] = None
        best_rank: int = -1
        best_y: float = 1e9
        best_named: Optional[Tuple[DetectionDict, float]] = None  # (det, score)

        # Pre-compute the expected card titles and handles for desired races
        expected_cards: List[Tuple[str, str]] = []
        desired_order: int = 1
        seek_title: Optional[str] = None
        seek_rank: Optional[str] = None
        if desired_race_name:
            logger_uma.debug(f"Racing with desired_race_name={desired_race_name}")
            if date_key:
                e = RaceIndex.entry_for_name_on_date(desired_race_name, date_key)
                if e:
                    seek_title = str(e.get("display_title") or "").strip()
                    seek_rank = str(e.get("rank") or "").strip().upper() or "UNK"
                    desired_order = (
                        int(e.get("order", 1))
                        if str(e.get("order", 1)).isdigit()
                        else 1
                    )
                    expected_cards = [(seek_title, seek_rank)]
                    logger_uma.info(
                        "[race] Seeking '%s' on %s → title='%s', rank=%s, order=%d",
                        desired_race_name,
                        date_key,
                        seek_title,
                        seek_rank,
                        desired_order,
                    )
            if not expected_cards:
                expected_cards = RaceIndex.expected_titles_for_race(desired_race_name)
                if expected_cards:
                    logger_uma.info(
                        "[race] Seeking '%s' (no date binding) with titles: %s",
                        desired_race_name,
                        [t for t, _ in expected_cards],
                    )
            if not expected_cards:
                expected_cards = [(desired_race_name.strip(), "UNK")]
                logger_uma.warning(
                    "[race] Dataset has no entries for '%s'; falling back to literal name.",
                    desired_race_name,
                )

        # Template index cache
        _template_index_cache: Dict[str, str] = {}

        def _load_template_index() -> Dict[str, str]:
            nonlocal _template_index_cache
            if _template_index_cache:
                return _template_index_cache
            try:
                idx_path = Settings.ROOT_DIR / "assets" / "races" / "templates" / "index.json"
                with open(idx_path, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}
                _template_index_cache = {str(k): str(v) for k, v in data.items()}
            except Exception:
                _template_index_cache = {}
            return _template_index_cache

        def _resolve_template_path(race_name: str, date_key: Optional[str]) -> Optional[Path]:
            idx = _load_template_index()
            p = idx.get(race_name)
            if p:
                path = Settings.ROOT_DIR / p.lstrip("/\\")
                if path.exists():
                    return path
            try:
                if date_key:
                    e = RaceIndex.entry_for_name_on_date(race_name, date_key)
                    if e and e.get("public_banner_path"):
                        pbp = str(e.get("public_banner_path")).lstrip("/\\")
                        path = Settings.ROOT_DIR / "web" / "public" / Path(pbp).relative_to("race").as_posix()
                        if not path.exists():
                            path = Settings.ROOT_DIR / "web" / "public" / pbp
                        if path.exists():
                            return path
            except Exception:
                pass
            return None

        seen_title_counts: Dict[str, int] = {}

        for scroll_j in range(max_scrolls + 1):
            # Wait screen to stabilize
            time.sleep(1)
            game_img, dets = self._collect("race_pick")
            squares = find(dets, "race_square")
            if squares:
                squares.sort(key=lambda d: ((d["xyxy"][1] + d["xyxy"][3]) / 2.0))
                if first_top_xyxy is None:
                    first_top_xyxy = tuple(squares[0]["xyxy"])

                stars = find(dets, "race_star")
                original_star_count = len(stars)
                stars = self._deduplicate_stars(stars)  # Remove YOLO duplicate detections
                if len(stars) < original_star_count:
                    logger_uma.debug(
                        f"[race] Removed {original_star_count - len(stars)} duplicate star detection(s)"
                    )
                badges = find(dets, "race_badge")

                if len(squares) == 1 and scroll_j == 0:
                    sq = squares[0]
                    need_click = True
                    if (
                        (not did_scroll)
                        and first_top_xyxy is not None
                        and tuple(sq["xyxy"]) == first_top_xyxy
                    ):
                        need_click = False
                    return sq, need_click

                if desired_race_name:
                    page_scores: List[Tuple[DetectionDict, float]] = []
                    page_best: Optional[Tuple[DetectionDict, float]] = None
                    for idx_on_page, sq in enumerate(squares):
                        s_cnt = sum(
                            1 for st in stars if inside(st["xyxy"], sq["xyxy"], pad=1)
                        )
                        badge_det = next(
                            (b for b in badges if inside(b["xyxy"], sq["xyxy"], pad=1)),
                            None,
                        )
                        badge_label = "UNK"
                        right_of_badge_xyxy = None
                        if badge_det is not None:
                            badge_label = _badge_label(
                                self.ocr, game_img, badge_det["xyxy"]
                            )
                            bx1, by1, bx2, by2 = badge_det["xyxy"]
                            sx1, sy1, sx2, sy2 = sq["xyxy"]
                            badge_height = abs(by2 - by1)
                            padding = int(badge_height * 0.6)
                            right_of_badge_xyxy = (
                                bx2 + 1,
                                sy1 + padding,
                                sx2 - 6,
                                by2 + padding,
                            )
                        else:
                            sx1, sy1, sx2, sy2 = sq["xyxy"]
                            w = sx2 - sx1
                            right_of_badge_xyxy = (
                                sx1 + int(0.30 * w),
                                sy1 + 2,
                                sx2 - 6,
                                sy2 - 2,
                            )

                        try:
                            crop = crop_pil(game_img, right_of_badge_xyxy, pad=0)
                            txt = (self.ocr.text(crop) or "").strip()
                        except Exception:
                            txt = ""

                        best_score_here = 0
                        varies_token = clean_race_name("varies")
                        txt = clean_race_name(txt)
                        for expected_title, expected_rank in expected_cards:
                            expected_title_n = clean_race_name(expected_title)
                            s = fuzzy_ratio(txt, expected_title_n.upper())

                            if varies_token in expected_title_n:
                                tokens_actual = txt.split()
                                tokens_expected = [
                                    token
                                    for token in expected_title_n.split()
                                    if token and token != varies_token
                                ]
                                if tokens_expected and tokens_actual:
                                    matched = sum(
                                        1 for token in tokens_expected if token in tokens_actual
                                    )
                                    if matched:
                                        token_ratio = matched / len(tokens_expected)
                                        if matched == len(tokens_expected):
                                            token_ratio += 0.15
                                        s = max(s, token_ratio)
                            if expected_rank in ("G1", "G2", "G3", "OP", "EX"):
                                if badge_label.upper() != expected_rank.upper() and badge_label != "UNK":
                                    s -= 0.20
                            best_score_here = max(best_score_here, s)

                        if best_named is None or best_score_here > best_named[1]:
                            best_named = (sq, best_score_here)
                        if (page_best is None) or (best_score_here > page_best[1]):
                            page_best = (sq, best_score_here)
                        page_scores.append((sq, best_score_here))

                    if page_scores:
                        page_scores.sort(key=lambda t: t[1], reverse=True)

                        try:
                            card_candidates = [RaceIndex.banner_template(desired_race_name)]
                            card_candidates = [c for c in card_candidates if c]
                            

                            if card_candidates:
                                for idx, (cand_sq, base_score) in enumerate(page_scores[:4]):
                                    sx1, sy1, sx2, sy2 = cand_sq["xyxy"]
                                    badge = next((b for b in badges if inside(b["xyxy"], cand_sq["xyxy"], pad=3)), None)

                                    if badge:
                                        bx1, _, _, _ = badge["xyxy"]
                                        roi_xyxy = (sx1, sy1, max(sx1 + 4, bx1 - 2), sy2)
                                    else:
                                        roi_xyxy = (sx1, sy1, sx1 + int((sx2 - sx1) * 0.55), sy2)

                                    roi_img = crop_pil(game_img, roi_xyxy, pad=0)
                                    # OCR over the ROI to decide keep/discard and to add a small boost

                                    try:
                                        roi_txt_raw = (self.ocr.text(roi_img) or "").strip()
                                    except Exception:
                                        roi_txt_raw = ""
                                    roi_txt = clean_race_name(roi_txt_raw)
                                    # Compare ROI OCR against our expected titles
                                    best_ocr = 0.0
                                    varies_token = clean_race_name("varies")
                                    expected_title_desired = clean_race_name(desired_race_name)
                                    best_ocr = fuzzy_ratio(roi_txt, expected_title_desired.upper())
                                    match = self._banner_matcher.best_match(roi_img, candidates=[c["name"] for c in card_candidates], )
                                    match_score = match.score if match else 0.0
                                    # Keep/discard using OCR only
                                    if best_ocr < OCR_DISCARD_MIN and match_score < 0.5:
                                        adjusted_score = -1.0  # hard discard so it won't be picked
                                    else:
                                        adjusted_score = (base_score * 0.5) + (best_ocr * OCR_SCORE_WEIGHT) + (match_score * 0.5)  # extra 0.2
                                        if best_ocr > 0.99:
                                            pass
                                        logger_uma.debug(
                                            "[race] Candidate boosted by OCR: base=%.3f, ocr=%.3f, template_match=%.3f (w=%.2f) → total=%.3f | text='%s'",
                                            base_score,
                                            best_ocr,
                                            match_score,
                                            OCR_SCORE_WEIGHT,
                                            adjusted_score,
                                            roi_txt,
                                        )
                                    page_scores[idx] = (cand_sq, adjusted_score)

                                page_scores.sort(key=lambda t: t[1], reverse=True)
                            else:
                                # Not a special case, ignoring OCR and template matching for optimization, remove optimization in future if we will have much more of these cases
                                pass
                        except Exception as e:
                            logger_uma.debug("[race] banner tie-breaker failed: %s", e)

                        top_sq, top_score = page_scores[0]
                        if top_score >= MINIMUM_RACE_OCR_MATCH:
                            pick = top_sq
                            ymid = (pick["xyxy"][1] + pick["xyxy"][3]) / 2.0
                            logger_uma.info(
                                "[race] picked desired '%s' by card-title (score=%.2f) at y=%.1f",
                                desired_race_name,
                                top_score,
                                ymid,
                            )
                            need_click = True
                            if (
                                (not did_scroll)
                                and first_top_xyxy is not None
                                and tuple(pick["xyxy"]) == first_top_xyxy
                            ):
                                need_click = False
                            return pick, need_click
                else:
                    for sq in squares:
                        s_cnt = sum(
                            1 for st in stars if inside(st["xyxy"], sq["xyxy"], pad=1)
                        )
                        if s_cnt < MIN_STARS:
                            logger_uma.debug(f"Not enough stars, found: {s_cnt}")
                            continue
                        
                        logger_uma.debug(f"Found valid {s_cnt} stars, {len(badges)} badges. dets: {dets}")
                        badge_det = next(
                            (b for b in badges if inside(b["xyxy"], sq["xyxy"], pad=3)),
                            None,
                        )
                        label = "UNK"
                        if badge_det is not None:
                            label = _badge_label(self.ocr, game_img, badge_det["xyxy"])
                        rank = BADGE_PRIORITY.get(label, 0)
                        ymid = (sq["xyxy"][1] + sq["xyxy"][3]) / 2.0

                        if prioritize_g1 or is_g1_goal:
                            if label == "G1":
                                logger_uma.info(
                                    "[race] picked G1 with 2★ at y=%.1f", ymid
                                )
                                need_click = True
                                if (
                                    (not did_scroll)
                                    and first_top_xyxy is not None
                                    and tuple(sq["xyxy"]) == first_top_xyxy
                                ):
                                    need_click = False
                                return sq, need_click
                            # For prioritize_g1 (but not is_g1_goal), accumulate non-G1 races as fallback
                            if not is_g1_goal:
                                if best_non_g1 is None:
                                    best_non_g1, best_rank, best_y = sq, rank, ymid
                                elif (rank > best_rank) or (rank == best_rank and ymid < best_y):
                                    best_non_g1, best_rank, best_y = sq, rank, ymid
                        else:
                            if is_g1_goal:
                                continue
                            if best_non_g1 is None:
                                best_non_g1, best_rank, best_y = sq, rank, ymid
                            else:
                                if (rank > best_rank) or (
                                    rank == best_rank and ymid < best_y
                                ):
                                    best_non_g1, best_rank, best_y = sq, rank, ymid

            if best_non_g1 is not None:
                logger_uma.info(
                    f"[race] Picked best race found, rank={BADGE_PRIORITY_REVERSE[best_rank]}"
                )
                need_click = True
                if (
                    (not did_scroll)
                    and first_top_xyxy is not None
                    and tuple(best_non_g1["xyxy"]) == first_top_xyxy
                ):
                    need_click = False
                return best_non_g1, need_click

            # Use the first visible square as scroll anchor so drags start on the race list
            anchor_xy = None
            if squares:
                try:
                    anchor_xy = self.ctrl.center_from_xyxy(tuple(squares[0]["xyxy"]))
                    logger_uma.debug(
                        "[race] scroll anchor from first square: anchor_xy=%s bbox=%s",
                        anchor_xy,
                        squares[0]["xyxy"],
                    )
                except Exception as e:
                    logger_uma.debug("[race] failed to compute scroll anchor: %s", e)

            if squares and not moved_cursor:
                self.ctrl.move_xyxy_center(squares[0]["xyxy"])
                time.sleep(0.10)
                moved_cursor = True

            # probe next batch
            smart_scroll_small(self.ctrl, steps_pc=4, anchor_xy=anchor_xy)
            did_scroll = True
            time.sleep(0.35)

        if best_non_g1 is not None:
            # end-of-scroll fallback is always a click
            return best_non_g1, True
        return None, True

    # --------------------------
    # Public API
    # --------------------------
    def lobby(self) -> bool:
        """
        Handles the lobby where 'View Results' (white) and 'Race' (green) appear.
        Uses the unified Waiter API; no external polling loops.
        Returns False if the view button cannot be found after retries.
        """
        # Try resolving 'View Results' with progressive retries (up to 15s total)
        view_btn = self._pick_view_results_button()
        if view_btn is None:
            # Retry with progressive delays: 2s, 3s, 5s, 5s (total ~15s)
            retry_delays = [2, 3, 5, 5]
            for i, delay in enumerate(retry_delays, 1):
                logger_uma.warning(
                    "No view result button found, waiting %ds more (attempt %d/%d)...",
                    delay, i, len(retry_delays)
                )
                time.sleep(delay)
                view_btn = self._pick_view_results_button()
                if view_btn is not None:
                    logger_uma.info("View button found after %d retry attempt(s)", i)
                    break
        
        # If still not found after all retries, abort the operation
        if view_btn is None:
            logger_uma.error(
                "View Results button not found after ~15s of retries. "
                "Cannot determine lobby state. Aborting race operation."
            )
            return False

        is_view_active = False
        if view_btn is not None:
            clf = ActiveButtonClassifier.load(Settings.IS_BUTTON_ACTIVE_CLF_PATH)
            img, _ = self._collect("race_lobby_active")
            crop = crop_pil(img, view_btn["xyxy"])
            try:
                p = float(clf.predict_proba(crop))
                is_view_active = p >= 0.51
                logger_uma.debug("[race] View Results active probability: %.3f", p)
            except Exception:
                is_view_active = False
                logger_uma.debug("[race] View Results inactive")

        if is_view_active and view_btn is not None:
            # Tap 'View Results' a couple times to clear residual screens
            self.ctrl.click_xyxy_center(view_btn["xyxy"], clicks=random.randint(1, 2))
            time.sleep(random.uniform(3, 3.5))
            self.ctrl.click_xyxy_center(view_btn["xyxy"], clicks=random.randint(3, 3))
            time.sleep(random.uniform(0.3, 0.5))
        else:
            # Click green 'RACE' (prefer bottom-most; OCR disambiguation if needed)
            if not self.waiter.click_when(
                classes=("button_green",),
                texts=("RACE",),
                prefer_bottom=True,
                timeout_s=6,
                tag="race_lobby_race_click",
            ):
                logger_uma.error("[race] Race button not found after ~6s of retries. "
                "Cannot determine lobby state. Aborting race operation.")
                return False
            time.sleep(5)
            self.waiter.click_when(
                classes=("button_green",),
                texts=("RACE",),
                prefer_bottom=True,
                timeout_s=2,
                tag="race_lobby_race_click_just_in_case",
            )
            # Reactive second confirmation. Click as soon as popup appears,
            # or bail early if the pre-race lobby appears or skip buttons show up.
            t0 = time.time()
            seen_skip = False
            while (time.time() - t0) < 12.0:
                # If the confirmation 'RACE' appears, click it immediately.
                if self.waiter.click_when(
                    classes=("button_green",),
                    texts=("RACE", "NEXT"),
                    prefer_bottom=True,
                    timeout_s=0.3,
                    tag="race_lobby_race_confirm_try",
                ):
                    logger_uma.debug("[race] Clicked RACE confirmation")
                    time.sleep(0.5)
                # If we already transitioned into race (skip buttons), stop waiting.
                if self.waiter.seen(
                    classes=("button_skip",), tag="race_lobby_seen_skip"
                ):
                    seen_skip = True
                    logger_uma.debug("[race] Seen skip buttons, breaking to click them")
                    break
                time.sleep(0.5)
            logger_uma.debug(f"[race] Seen skip buttons: {seen_skip}")
            if not seen_skip:
                # search again for green 'Next' or 'RACE' button
                if not self.waiter.click_when(
                    classes=("button_green",),
                    texts=("RACE", "NEXT"),
                    prefer_bottom=True,
                    timeout_s=6,
                    tag="race_lobby_race_click_retry",
                ):
                    logger_uma.error("[race] Race button not found after ~6s of retries. "
                    "Cannot determine lobby state. Aborting race operation.")
                    return False
            time.sleep(4)
            logger_uma.debug("[race] Starting skip loop")
            # Greedy skip: keep pressing while present; stop as soon as 'CLOSE' or 'NEXT' shows.
            closed_early = False
            skip_clicks = 0
            t0 = time.time()
            total_time = 12.0
            while (time.time() - t0) < total_time:
                # Early-exit conditions:
                #  - close available → click once and stop
                if self.waiter.click_when(
                    classes=("button_white",),
                    texts=("CLOSE",),
                    prefer_bottom=False,
                    timeout_s=0.3,
                    tag="race_trophy_try_close",
                ):
                    closed_early = True
                    logger_uma.debug("[race] Clicked close Trophy button")
                    break
                #  - next visible → stop skipping; later logic will handle NEXT
                if self.waiter.seen(
                    classes=("button_green",), tag="race_skip_probe_next",
                    conf_min=0.65,
                ) and skip_clicks > 2:
                    logger_uma.debug("[race] Seen next button while looking for skip, breaking to click it")
                    break

                # Otherwise try to click a skip on this frame.
                if self.waiter.click_when(
                    classes=("button_skip",),
                    prefer_bottom=True,
                    timeout_s=1,
                    clicks=random.randint(3, 5),
                    tag="race_skip_try",
                ):
                    logger_uma.debug("[race] Clicked skip button")
                    skip_clicks += 1
                    total_time += 2
                    continue
                time.sleep(0.12)

            if not closed_early:
                logger_uma.debug("[race] Looking for CLOSE button.")
                self.waiter.click_when(
                    classes=("button_white",),
                    texts=("CLOSE",),
                    prefer_bottom=False,
                    allow_greedy_click=False,
                    timeout_s=3,
                    tag="race_trophy",
                )

        # Check if we loss
        time.sleep(1)
        clicked_try_again = False
        loss_indicator_seen = self.waiter.seen(
            classes=("button_green",),
            texts=("TRY AGAIN",),
            tag="race_try_again_probe",
            threshold=0.3,
        )
        if loss_indicator_seen:
            self._race_result_counters["loss_indicators"] += 1
            logger_uma.info(
                "[race] Loss indicator detected (toggle=%s) | counters=%s",
                Settings.TRY_AGAIN_ON_FAILED_GOAL,
                self._race_result_counters,
            )

        should_retry = bool(Settings.TRY_AGAIN_ON_FAILED_GOAL and loss_indicator_seen)

        if should_retry:
            clicked_try_again = self._attempt_try_again_retry()
        elif loss_indicator_seen:
            self._race_result_counters["retry_skipped"] += 1
            logger_uma.info(
                "[race] Retry disabled via settings despite loss indicator | counters=%s",
                self._race_result_counters,
            )
            logger_uma.warning(
                "[race] Stopping bot so user can choose Try Again or Cancel manually."
            )
            if self._telemetry:
                try:
                    self._telemetry.log_race_result(
                        turn=None,
                        race_name=self._last_race_name,
                        won=False,
                        retried=False,
                    )
                except Exception:
                    pass
            self._waiting_for_manual_retry_decision = True
            request_abort()
            return False

        if clicked_try_again:
            logger_uma.debug("[race] Lost the race, trying again.")
            if self._telemetry:
                try:
                    self._telemetry.log_race_result(
                        turn=None,
                        race_name=self._last_race_name,
                        won=False,
                        retried=True,
                    )
                except Exception:
                    pass
            self._handle_retry_transition()
            logger_uma.info(
                "[race] Loss metrics after retry: %s",
                self._race_result_counters,
            )
            return self.lobby()

        else:
            if not loss_indicator_seen:
                self._race_result_counters["wins_or_no_loss"] += 1
            elif should_retry:
                self._race_result_counters["retry_skipped"] += 1
                logger_uma.info(
                    "[race] Retry expected but TRY AGAIN not clicked; continuing. | counters=%s",
                    self._race_result_counters,
                )

            # Log race result
            if self._telemetry:
                try:
                    self._telemetry.log_race_result(
                        turn=None,
                        race_name=self._last_race_name,
                        won=not loss_indicator_seen,
                        retried=False,
                    )
                except Exception:
                    pass

            logger_uma.info(
                "[race] Continuing without retry (loss_indicator=%s) | counters=%s",
                loss_indicator_seen,
                self._race_result_counters,
            )
            # After the race/UI flow → 'NEXT' / 'OK' / 'PROCEED'
            logger_uma.debug(
                "[race] Looking for button_green 'Next' button. Shown after race."
            )
            self.waiter.click_when(
                classes=("button_green",),
                texts=("NEXT",),
                forbid_texts=("TRY AGAIN",),
                prefer_bottom=False,
                allow_greedy_click=False,
                timeout_s=4.6,
                clicks=3,
                tag="race_after_flow_next",
            )

            # 'Next' special
            logger_uma.debug(
                "[race] Looking for race_after_next special button. When Pyramid"
            )

            self.waiter.click_when(
                classes=("race_after_next",),
                texts=("NEXT",),
                prefer_bottom=True,
                timeout_s=6.0,
                clicks=random.randint(2, 4),
                tag="race_after",
            )

            # Optional: Confirm 'Next'. TODO understand when to use
            # self.waiter.click_when(
            #     classes=("button_green",),
            #     texts=("NEXT", ),
            #     prefer_bottom=False,
            #     allow_greedy_click=False,
            #     timeout_s=2.0,
            #     tag="race_after",
            # )

            logger_uma.info("[race] RaceDay flow finished.")
            return True

    # --------------------------
    # Strategy selector (End / Late / Pace / Front)
    # --------------------------
    def set_strategy(self, select_style: str, *, timeout_s: float = 2.0) -> bool:
        """
        Pick a running style inside the 'Change Strategy' modal, then press Confirm.
        `select_style` must be one of: 'end', 'late', 'pace', 'front' (case-insensitive).
        Returns True if both clicks (style + confirm) were performed.
        """
        game_img, dets = self._collect("change_style")
        elements = find(dets, "button_change")
        if elements and len(elements) == 1:
            button_change = elements[0]
            self.ctrl.click_xyxy_center(button_change["xyxy"], clicks=1)
            time.sleep(1.2)
        else:
            return False
        select_style = (select_style or "").strip().lower()
        STYLE_ORDER = ["end", "late", "pace", "front"]  # left → right in the modal
        if select_style not in STYLE_ORDER:
            logger_uma.warning(
                "[race] Unknown select_style=%r; defaulting to 'pace'", select_style
            )
            select_style = "front"

        # Read current modal
        img, dets = self._collect("change_style_modal")
        whites = find(dets, "button_white") or []
        greens = find(dets, "button_green") or []
        if not whites:
            logger_uma.error("[race] set_strategy: no white buttons detected.")
            return False

        # Confirm button: pick bottom-most green if present
        confirm_btn = bottom_most(greens)

        # Cancel button: bottom-most white (y center biggest)
        def y_center(d):
            x1, y1, x2, y2 = d["xyxy"]
            return 0.5 * (y1 + y2)

        cancel_btn = max(whites, key=y_center)

        # Candidate style buttons = white buttons above the confirm/cancel row
        style_btns = [
            d
            for d in whites
            if d is not cancel_btn and y_center(d) < (y_center(cancel_btn) - 10)
        ]
        if not style_btns:
            # fall back: all whites except the bottom-most
            style_btns = [d for d in whites if d is not cancel_btn]

        # Sort left → right by x center
        style_btns.sort(key=lambda d: (0.5 * (d["xyxy"][0] + d["xyxy"][2])))

        # We *expect* the order to be End, Late, Pace, Front (when all present).
        # If fewer are present, try OCR to map; otherwise rely on left-right order.
        idx_map = {name: i for i, name in enumerate(STYLE_ORDER)}

        chosen = None
        if len(style_btns) >= 4:
            chosen = style_btns[idx_map[select_style]]
        else:
            # OCR fallback for robustness on partial layouts
            def read_label(btn):
                x1, y1, x2, y2 = btn["xyxy"]
                # shrink a bit to avoid borders
                shrink = max(2, int(min(x2 - x1, y2 - y1) * 0.10))
                roi = (x1 + shrink, y1 + shrink, x2 - shrink, y2 - shrink)
                try:
                    t = (self.ocr.text(crop_pil(img, roi)) or "").strip().lower()
                except Exception:
                    t = ""
                return t

            best_btn, best_sc = None, 0.0
            for b in style_btns:
                t = read_label(b)
                # be permissive: compare against canonical label
                sc = fuzzy_ratio(t, select_style)
                if sc > best_sc:
                    best_sc, best_btn = sc, b
            # accept if somewhat confident; else fall back to order
            if best_btn is not None and best_sc >= 0.45:
                chosen = best_btn
            else:
                # fallback to the closest expected index available
                target_idx = idx_map[select_style]
                chosen = style_btns[min(target_idx, len(style_btns) - 1)]

        # Click selected style
        self.ctrl.click_xyxy_center(chosen["xyxy"], clicks=1)
        time.sleep(0.15)

        # Click Confirm
        if confirm_btn is None:
            # try waiter on text if green wasn't detected
            clicked = self.waiter.click_when(
                classes=("button_green",),
                texts=("CONFIRM",),
                prefer_bottom=True,
                timeout_s=timeout_s,
                tag="race_style_confirm_text",
            )
            return bool(clicked)
        else:
            self.ctrl.click_xyxy_center(confirm_btn["xyxy"], clicks=1)
            time.sleep(0.15)
            return True

    def run(
        self,
        *,
        prioritize_g1: bool = False,
        is_g1_goal: bool = False,
        desired_race_name: Optional[str] = None,
        date_key: Optional[str] = None,
        select_style=None,
        ensure_navigation: bool = True,
        from_raceday: bool = False,
        reason: str | None = None,
    ) -> bool:
        """
        End-to-end race-day routine. If called from Lobby, set ensure_navigation=True
        (default) and we will enter the Raceday list ourselves. This allows running
        RaceFlow without involving LobbyFlow/Agent orchestration.
        Behavior when consecutive-race penalty is detected and settings forbid it:
          - if from_raceday == True → raise ConsecutiveRaceRefused
          - else → return False (let caller continue with its skip logic)
        """
        # Reset manual retry decision flag and last failure reason at the start of a new race
        self._waiting_for_manual_retry_decision = False
        self._last_failure_reason = RaceFailureReason.NONE
        
        logger_uma.info(
            "[race] RaceDay begin (prioritize_g1=%s, is_g1_goal=%s)%s",
            prioritize_g1,
            is_g1_goal,
            f" | reason='{reason}'" if reason else "",
        )
        if ensure_navigation:
            try:
                _ = self._ensure_in_raceday(reason=reason, from_raceday=from_raceday)
            except ConsecutiveRaceRefused:
                logger_uma.info(
                    "[race] Returning False due to refused consecutive race (non-Raceday caller)."
                )
                self._last_failure_reason = RaceFailureReason.NAV_CONSECUTIVE_REFUSED
                return False

        time.sleep(2)
        # 1) Pick race card; scroll if needed
        square, need_click = self._pick_race_square(
            prioritize_g1=prioritize_g1,
            is_g1_goal=is_g1_goal,
            desired_race_name=desired_race_name,
            max_scrolls=3,
            date_key=date_key,
        )
        if square is None:
            logger_uma.debug("race square not found")
            self._last_failure_reason = RaceFailureReason.NO_RACE_SQUARE
            return False

        # 2) Click the race square
        if need_click:
            self.ctrl.click_xyxy_center(square["xyxy"], clicks=1)
            time.sleep(0.2)
            logger_uma.info("[race] Clicked race square")

        # Store race name for result logging
        self._last_race_name = desired_race_name or reason

        # Log race start
        if self._telemetry:
            try:
                self._telemetry.log_race_start(
                    turn=None,
                    race_name=self._last_race_name,
                    prioritize_g1=prioritize_g1,
                    is_goal=is_g1_goal,
                    select_style=select_style,
                )
            except Exception:
                pass

        # 3) Click green 'RACE' on the list (prefer bottom-most; OCR 'RACE' if needed)
        if not self.waiter.click_when(
            classes=("button_green",),
            texts=("RACE",),
            prefer_bottom=True,
            timeout_s=2,
            tag="race_list_race",
        ):
            logger_uma.warning("[race] couldn't find green 'Race' button (list).")
            self._last_failure_reason = RaceFailureReason.RACE_BUTTON_LIST_MISSING
            return False

        # Time to popup to grow, so we don't missclassify a mini button in the animation
        time.sleep(1.2)
        # Reactive confirm of the popup (if/when it appears). Bail out if pre-race lobby is already visible.
        t0 = time.time()
        while (time.time() - t0) < 5.0:
            if abort_requested():
                logger_uma.info("[race] Abort requested before popup confirm.")
                self._last_failure_reason = RaceFailureReason.ABORT_DURING_POPUP
                return False
            if self.waiter.seen(
                classes=("button_change",), tag="race_pre_lobby_seen_early"
            ):
                break
            if self.waiter.click_when(
                classes=("button_green",),
                texts=("RACE",),
                prefer_bottom=True,
                timeout_s=1,
                tag="race_popup_confirm_try",
            ):
                logger_uma.info("[race] Clicked green 'Race' button (popup) confirmation")
                # Give a short beat for the transition; continue probing.
                time.sleep(0.2)
                break
            else:
                logger_uma.warning("[race] couldn't find 'Race' button (popup) confirmation in this check.")
            time.sleep(0.1)

        # 4) Wait until the pre-race lobby is actually on screen (key: 'button_change')
        logger_uma.info("Waiting for race lobby to appear")
        time.sleep(7)
        t0 = time.time()
        max_wait = 14.0
        saw_pre_lobby = False
        while (time.time() - t0) < max_wait:
            if abort_requested():
                logger_uma.info(
                    "[race] Abort requested while waiting for pre-race lobby."
                )
                self._last_failure_reason = RaceFailureReason.ABORT_DURING_LOBBY_WAIT
                return False
            if self.waiter.seen(classes=("button_change",), tag="race_pre_lobby_gate"):
                saw_pre_lobby = True
                break
            time.sleep(0.15)

        if not saw_pre_lobby:
            logger_uma.warning("[race] Pre-race lobby not detected within timeout.")
            self._last_failure_reason = RaceFailureReason.PRE_LOBBY_TIMEOUT
            return False

        # 5) Optional: set strategy as soon as the Change button is available (no extra sleeps)
        if select_style and self.waiter.seen(
            classes=("button_change",), tag="race_pre_lobby_ready"
        ):
            logger_uma.debug(f"Setting style: {select_style}")
            self.set_strategy(select_style)
            time.sleep(3)  # wait for white buttons to dissapear

        # 6) Proceed with the result/lobby handling pipeline
        lobby_ok = self.lobby()
        if not lobby_ok:
            self._last_failure_reason = RaceFailureReason.LOBBY_FLOW_FAILED
            return False

        self._last_failure_reason = RaceFailureReason.NONE
        return True
