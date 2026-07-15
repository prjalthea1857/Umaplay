# core/actions/skills.py
from __future__ import annotations

import random
import time
from typing import List, Optional, Sequence, Tuple, Dict
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from PIL import Image

from core.controllers.android import ScrcpyController
from core.controllers.adb import ADBController
from core.controllers.bluestacks import BlueStacksController
from core.controllers.base import IController
from core.perception.ocr.interface import OCRInterface
from core.perception.yolo.interface import IDetector
from core.settings import Settings
from core.utils.logger import logger_uma
from core.utils.geometry import crop_pil
from core.utils.text import (
    fix_common_ocr_confusions,
    fuzzy_ratio,
    tokenize_ocr_text,
)
from core.utils.skill_matching import SkillMatcher
from core.utils.skill_memory import SkillMemoryManager
from core.perception.is_button_active import ActiveButtonClassifier
from core.types import DetectionDict
from core.utils.yolo_objects import inside, yolo_signature
from core.utils.waiter import Waiter
from core.utils.pointer import smart_scroll_small


class SkillsBuyStatus(Enum):
    SUCCESS = "success"
    NO_BUY = "no_buy"
    EXIT_FAILED = "exit_failed"


@dataclass(frozen=True)
class SkillsBuyResult:
    status: SkillsBuyStatus
    clicked_any: bool
    exit_recovered: bool

    def __bool__(self) -> bool:  # pragma: no cover - convenience helper
        return self.status is SkillsBuyStatus.SUCCESS

    def __str__(self) -> str:  # pragma: no cover - logging helper
        return self.status.value

    @property
    def exited_cleanly(self) -> bool:
        return self.exit_recovered


class SkillsFlow:
    """
    Skills screen automation (Learn view).
    - Uses Waiter for robust button clicking (OCR + position heuristics)
    - Speeds up by preloading the "active button" classifier
    - OCRs only the title band within each skills_square for accuracy
    - Mitigates scroll inertia by clicking the BUY button slightly above center
    """

    def __init__(
        self,
        ctrl: IController,
        ocr: OCRInterface,
        yolo_engine: IDetector,
        waiter: Waiter,
        skill_memory: Optional[SkillMemoryManager] = None,
    ) -> None:
        self.ctrl = ctrl
        self.ocr = ocr
        self.yolo_engine = yolo_engine
        self.waiter = waiter
        # Preload once for speed
        self._clf = ActiveButtonClassifier.load(Settings.IS_BUTTON_ACTIVE_CLF_PATH)
        self._skill_matcher = SkillMatcher.from_dataset()
        self._skill_memory = skill_memory or SkillMemoryManager(
            Settings.resolve_skill_memory_path(Settings.ACTIVE_SCENARIO),
            scenario=Settings.ACTIVE_SCENARIO,
        )

    # --------------------------
    # Public API
    # --------------------------

    def buy(
        self,
        skill_list: Sequence[str],
        *,
        max_scrolls: int = 15,
        ocr_threshold: float = 0.85,  # experimental, upgraded to 0.82 for the sake of avoiding false positives
        scroll_time_range: Tuple[int, int] = (6, 7),
        early_stop: bool = True,
        date_key: Optional[str] = None,
    ) -> SkillsBuyResult:
        """
        End-to-end skill buying.
        Returns a SkillsBuyResult representing success, a clean no-buy exit, or an exit failure.
        """
        if not skill_list:
            logger_uma.info("[skills] No targets configured.")
            return SkillsBuyResult(SkillsBuyStatus.NO_BUY, clicked_any=False, exit_recovered=True)

        logger_uma.info("[skills] Buying targets: %s", ", ".join(skill_list))

        any_clicked = False
        prev_sig: Optional[List[Tuple[str, int, int]]] = None
        prev_ocr_sig: Optional[List[Tuple[str, int, int]]] = None

        # Desired counts per target: "◎" requires at least 2 buys, otherwise 1.
        desired_counts: Dict[str, int] = {}
        purchases_made: Dict[str, int] = {}
        for t in skill_list:
            desired_counts[t] = 2 if "◎" in t else 1
            # Seed current-session counts from persisted memory so we don't rebuy when
            # reopening the skills screen mid-run (e.g., double-circle skills).
            if self._skill_memory:
                canonical = self._canonical_skill_name(t)
                grade_symbol = self._grade_from_name(t)
                if canonical:
                    bought = self._skill_memory.get_bought_count(
                        canonical, grade=grade_symbol
                    )
                    purchases_made[t] = min(bought, desired_counts[t])
                    continue
            purchases_made[t] = 0

        patience = 3
        for i in range(max_scrolls):
            clicked, game_img, dets, cur_ocr_sig = self._scan_and_click_buys(
                targets=skill_list,
                ocr_threshold=ocr_threshold,
                desired_counts=desired_counts,
                purchases_made=purchases_made,
                date_key=date_key,
            )
            any_clicked |= clicked

            # Early-stop if the scene didn't change between passes and nothing was clicked
            cur_sig = yolo_signature(dets)
            if (
                early_stop
                and (not clicked)
                and (prev_sig is not None)
                and self._nearly_same(prev_sig, cur_sig, prev_ocr_sig, cur_ocr_sig)
            ):
                logger_uma.info("[skills] Early stop (same view twice) patience -1.")
                patience -= 1
                if patience == 0:
                    logger_uma.info("[skills] Early stop buying.")
                    break
            else:
                patience = 3
            prev_sig = cur_sig
            prev_ocr_sig = cur_ocr_sig

            # Stop if we've satisfied all purchase requirements
            if all(purchases_made[t] >= desired_counts[t] for t in skill_list):
                logger_uma.info("[skills] All target purchase counts satisfied.")
                break

            # First pass focus nudge if nothing clicked
            if i == 0 and not any_clicked:
                self._focus_nudge(game_img, dets)

            self._scroll_once(scroll_time_range)

        if any_clicked:
            logger_uma.info("[skills] Confirming purchases...")
            confirmed = self._confirm_learn_close_back_flow()
            if confirmed:
                return SkillsBuyResult(SkillsBuyStatus.SUCCESS, clicked_any=True, exit_recovered=True)

            logger_uma.warning(
                "[skills] Confirmation flow failed; attempting recovery before returning control."
            )
            recovered = self._ensure_exit_to_lobby(tag_prefix="skills_flow_exit_recovery")
            if not recovered:
                logger_uma.error("[skills] Unable to confirm exit after confirmation failure.")
            return SkillsBuyResult(
                SkillsBuyStatus.EXIT_FAILED,
                clicked_any=True,
                exit_recovered=recovered,
            )

        logger_uma.info("[skills] No matching skills found to buy.")
        time.sleep(1.2)
        recovered = self._ensure_exit_to_lobby(
            tag_prefix="skills_flow_back_no_buys",
            prefer_back_only=True,
        )
        if not recovered:
            logger_uma.warning("[skills] Unable to confirm exit after no-buy flow.")
        status = SkillsBuyStatus.NO_BUY if recovered else SkillsBuyStatus.EXIT_FAILED
        return SkillsBuyResult(status, clicked_any=False, exit_recovered=recovered)

    # --------------------------
    # Internals
    # --------------------------

    def _collect(self, tag: str) -> Tuple[Image.Image, List[DetectionDict]]:
        img, _, dets = self.yolo_engine.recognize(
            imgsz=self.waiter.cfg.imgsz,
            conf=self.waiter.cfg.conf,
            iou=self.waiter.cfg.iou,
            tag=tag,
            agent=self.waiter.cfg.agent,
        )
        return img, dets

    @staticmethod
    def _nearly_same(
        a: List[Tuple[str, int, int]],
        b: List[Tuple[str, int, int]],
        a_ocr: Optional[List[Tuple[str, int, int]]] = None,
        b_ocr: Optional[List[Tuple[str, int, int]]] = None,
    ) -> bool:
        """
        Heuristic equivalence for scene signatures.
        Each signature item is (name, cx_bucket, cy_bucket). Two signatures are
        considered the same if:
          1) They have the same per-class counts (ignoring positions), and
          2) For every item in `a`, there exists an unmatched item in `b` with
             the same name and |dx|<=1 and |dy|<=1 bucket.
        Additionally (to avoid scroll false-positives), if OCR title signatures are provided,
        require a minimum overlap of normalized title texts across buckets.
        """
        if len(a) != len(b):
            return False
        ca = Counter(n for n, _, _ in a)
        cb = Counter(n for n, _, _ in b)
        if ca != cb:
            return False

        TOL = 1  # buckets (~8 px)
        pools = {}
        for name, x, y in b:
            pools.setdefault(name, []).append([x, y])

        for name, ax, ay in a:
            pool = pools.get(name, [])
            match_idx = -1
            best_metric = None
            for j, (bx, by) in enumerate(pool):
                dx, dy = abs(ax - bx), abs(ay - by)
                if dx <= TOL and dy <= TOL:
                    m = max(dx, dy)
                    if best_metric is None or m < best_metric:
                        best_metric = m
                        match_idx = j
                        if m == 0:
                            break
            if match_idx == -1:
                return False
            pool.pop(match_idx)
        # If we have OCR title fingerprints, enforce overlap to assert "same view".
        if a_ocr is not None and b_ocr is not None:
            if not a_ocr or not b_ocr:
                return True

            # Treat each entry as (norm_text, cx_bucket, cy_bucket).
            # Build multisets of texts, optionally requiring coarse spatial consistency.
            # We'll count matches by text first, then filter with a loose spatial check.
            def by_text_map(
                sig: List[Tuple[str, int, int]],
            ) -> Dict[str, List[Tuple[int, int]]]:
                d: Dict[str, List[Tuple[int, int]]] = {}
                for t, x, y in sig:
                    d.setdefault(t, []).append((x, y))
                return d

            A = by_text_map(a_ocr)
            B = by_text_map(b_ocr)
            matched = 0
            total = sum(len(v) for v in A.values())
            for t, apos in A.items():
                bpos = B.get(t, [])
                if not bpos:
                    continue
                # Greedy bipartite-ish match with TOL buckets.
                used = [False] * len(bpos)
                for ax, ay in apos:
                    best = -1
                    bestm = None
                    for j, (bx, by) in enumerate(bpos):
                        if used[j]:
                            continue
                        dx, dy = abs(ax - bx), abs(ay - by)
                        if dx <= TOL and dy <= TOL:
                            m = max(dx, dy)
                            if bestm is None or m < bestm:
                                bestm = m
                                best = j
                                if m == 0:
                                    break
                    if best != -1:
                        used[best] = True
                        matched += 1
            # Require a meaningful overlap: at least 2 titles AND ≥60% of the seen titles.
            return matched >= 2 and matched >= int(0.6 * max(1, total))

        return True

    @staticmethod
    def _canonical_skill_name(name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        cleaned = name
        for symbol in ("◎", "○", "×"):
            cleaned = cleaned.replace(symbol, "")
        cleaned = " ".join(cleaned.split()).strip()
        return cleaned or None

    @staticmethod
    def _grade_from_name(name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        for symbol in ("◎", "○", "×"):
            if symbol in name:
                return symbol
        return None

    @staticmethod
    def _grade_from_text(text: str) -> Optional[str]:
        for symbol in ("◎", "○", "×"):
            if symbol in (text or ""):
                return symbol
        return None

    def _already_bought(self, canonical_name: str, grade_symbol: Optional[str]) -> bool:
        if not self._skill_memory:
            return False
        # Only check exact grade match; allow upgrading from ○ to ◎
        return bool(
            grade_symbol
            and self._skill_memory.has_bought(canonical_name, grade=grade_symbol)
        )

    @staticmethod
    def _skill_title_roi(
        square_xyxy: Tuple[int, int, int, int],
    ) -> Tuple[int, int, int, int]:
        """
        Tight crop for the *title line* within a skills_square:
          - Skip left icon (~10%)
          - Crop a band near the top (~8%..38% height)
          - Leave some right margin (remove price/labels)
        """
        x1, y1, x2, y2 = square_xyxy
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        left = x1 + int(w * 0.10)
        right = x2 - int(w * 0.25)
        top = y1 + int(h * 0.08)
        bot = y1 + int(h * 0.38)
        if right <= left:
            right = left + 1
        if bot <= top:
            bot = top + 1
        return (left, top, right, bot)

    @staticmethod
    def _find_buy_inside(
        square: DetectionDict, candidates: List[DetectionDict]
    ) -> Optional[DetectionDict]:
        sq_xyxy = square.get("xyxy")
        if not sq_xyxy:
            return None
        for c in candidates:
            if inside(c["xyxy"], sq_xyxy, pad=4):
                return c
        return None

    def _scan_and_click_buys(
        self,
        *,
        targets: Sequence[str],
        ocr_threshold: float,
        desired_counts: Dict[str, int],
        purchases_made: Dict[str, int],
        date_key: Optional[str],
    ) -> Tuple[bool, Image.Image, List[DetectionDict], List[Tuple[str, int, int]]]:
        """
        Single pass: find all skills_square + their BUY button; OCR title-band and
        click BUY if matches a target. Returns (clicked_any, img, dets, ocr_title_signature).
        """
        game_img, dets = self._collect("skills_scan")

        squares = [d for d in dets if d["name"] == "skills_square"]
        buys = [d for d in dets if d["name"] == "skills_buy"]

        clicked_any = False
        ocr_titles_sig: List[Tuple[str, int, int]] = []
        seen_dirty = False

        for sq in squares:
            buy = self._find_buy_inside(sq, buys)
            if buy is None:
                continue

            # BUY must be active
            crop_buy = crop_pil(game_img, buy["xyxy"], pad=0)
            p = float(self._clf.predict_proba(crop_buy))
            if p < 0.55:
                continue

            # OCR only the title band for accuracy/speed
            title_crop = crop_pil(game_img, self._skill_title_roi(sq["xyxy"]), pad=2)
            raw_text = self.ocr.text(title_crop) or ""
            norm_text = self._norm_title(raw_text)
            tokens = tokenize_ocr_text(norm_text)
            # Record OCR title signature with coarse position buckets.
            x1, y1, x2, y2 = sq["xyxy"]
            cx = int((x1 + x2) / 2) // 8
            cy = int((y1 + y2) / 2) // 8
            if norm_text:
                ocr_titles_sig.append((norm_text, cx, cy))

            matches: List[Tuple[str, float, str]] = []
            diagnostics: List[Tuple[str, bool, float, str]] = []
            for target in targets:
                normalized_target = self._norm_title(target)
                ok, reason, score = self._skill_matcher.evaluate(
                    norm_text,
                    tokens,
                    target,
                    normalized_target,
                    threshold=ocr_threshold,
                )
                diagnostics.append((target, ok, score, reason))
                if ok:
                    matches.append((target, score, reason))

            diagnostics.sort(key=lambda x: x[2], reverse=True)

            contains_any = bool(matches)
            # Weighted best match: prioritize certain key terms
            KEY_UPWEIGHT = ("groundwork", "left-handed", "corner connoisseur")
            best_name, best_score = None, 0.0
            if matches:
                for tgt, score, reason in matches:
                    normalized_target = self._norm_title(tgt)
                    boost = 0.05 if any(k in normalized_target for k in KEY_UPWEIGHT) else 0.0
                    weighted = score + boost
                    if weighted > best_score:
                        best_score = weighted
                        best_name = tgt
                        match_reason = reason
            else:
                match_reason = "no_match"

            if diagnostics:
                logger_uma.debug(
                    "[skills] matcher diag title='%s' results=%s",
                    norm_text,
                    [
                        {
                            "target": t,
                            "ok": ok,
                            "score": round(s, 3),
                            "reason": r,
                        }
                        for t, ok, s, r in diagnostics[:3]
                    ],
                )

            grade_symbol = self._grade_from_name(best_name) or self._grade_from_text(raw_text)
            canonical_name = self._canonical_skill_name(best_name)

            if canonical_name and self._skill_memory:
                self._skill_memory.record_seen(
                    canonical_name,
                    grade=grade_symbol,
                    date_key=date_key,
                    commit=False,
                )
                seen_dirty = True

            # Respect purchase quotas before clicking
            if best_name is not None and (contains_any or best_score >= ocr_threshold):
                desired = desired_counts.get(best_name, 1)
                click_counts = desired
                if purchases_made.get(best_name, 0) >= desired:
                    continue
                if canonical_name and self._skill_memory:
                    bought_count = self._skill_memory.get_bought_count(
                        canonical_name, grade=grade_symbol
                    )
                    click_counts = abs(desired - bought_count)
                    if bought_count >= desired:
                        logger_uma.info(
                            "[skills] skipping '%s' grade='%s' (already purchased)",
                            best_name,
                            grade_symbol or SkillMemoryManager.ANY_GRADE,
                        )
                        continue
                # Click: center + slight upward offset to counter inertia
                bx1, by1, bx2, by2 = buy["xyxy"]
                bh = max(1, by2 - by1)

                upward_offset = 0.05

                if isinstance(self.ctrl, ScrcpyController):
                    upward_offset = 0.15
                dy = max(2, int(bh * upward_offset))  # ~X% upward
                shifted = (bx1, by1 - dy, bx2, by2 - dy)
                self.ctrl.click_xyxy_center(shifted, clicks=click_counts, jitter=0)
                purchases_made[best_name] = purchases_made.get(best_name, 0) + 1
                if canonical_name and self._skill_memory:
                    self._skill_memory.record_bought(
                        canonical_name,
                        grade=grade_symbol,
                        date_key=date_key,
                        commit=True,
                        boughts=click_counts
                    )
                logger_uma.info(
                    "Clicked BUY for '%s' (score=%.2f reason=%s) [%d/%d]",
                    best_name or "?",
                    best_score,
                    match_reason,
                    purchases_made.get(best_name, 0),
                    desired_counts.get(best_name, 1),
                )
                clicked_any = True

        if seen_dirty and self._skill_memory:
            # Persist sightings even when no purchases occur in this pass.
            self._skill_memory.save()

        return clicked_any, game_img, dets, ocr_titles_sig

    # --------------------------
    # Text normalization helpers
    # --------------------------
    @staticmethod
    def _norm_title(s: str) -> str:
        """
        Normalize OCR titles for robust signature matching:
          - strip/lowers
          - collapse spaces
          - remove trivial punctuation
        """
        s = fix_common_ocr_confusions(s or "")
        s = s.strip().lower()
        if not s:
            return ""
        # collapse whitespace
        s = " ".join(s.split())
        # drop superfluous punctuation commonly introduced by OCR
        # also strip skill-rank symbols which OCR may miss (○ ◎ ×)
        TABLE = str.maketrans("", "", "·•|[](){}:;,.!?\"'`’“”○◎×")
        s = s.translate(TABLE)
        return s.replace("-", " ").strip()

    def _confirm_learn_close_back_flow(self, waiting_popup: float = 1.0) -> bool:
        """
        Confirm → Learn → Close → Back using Waiter (OCR disambiguation under the hood).
        """
        # Confirm
        if not self.waiter.click_when(
            classes=("button_green",),
            texts=("CONFIRM",),
            prefer_bottom=True,
            timeout_s=3.0,
            tag="skills_flow_confirm",
        ):
            logger_uma.warning("Confirm button not found")
            return False
        time.sleep(waiting_popup)

        # Learn
        if not self.waiter.click_when(
            classes=("button_green",),
            texts=("LEARN",),
            prefer_bottom=True,
            timeout_s=1.2,
            tag="skills_flow_learn",
        ):
            logger_uma.warning("Confirm button not found")
            return False

        time.sleep(waiting_popup * 2)

        # Close
        if not self.waiter.click_when(
            classes=("button_white",),
            texts=("CLOSE",),
            prefer_bottom=False,
            allow_greedy_click=False,
            timeout_s=2,
            tag="skills_flow_close",
        ):
            logger_uma.warning("Close button not found")
            return False
        time.sleep(waiting_popup)

        # Back
        if not self.waiter.click_when(
            classes=("button_white",),
            texts=("BACK",),
            prefer_bottom=True,
            timeout_s=1.2,
            tag="skills_back",
        ):
            logger_uma.warning("Back button not found")
            return False
        time.sleep(0.15)
        return True

    def _ensure_exit_to_lobby(
        self,
        *,
        tag_prefix: str,
        prefer_back_only: bool = False,
        attempts: int = 3,
    ) -> bool:
        """Attempt to leave the Skills screen, optionally allowing CLOSE/OK fallbacks."""
        exit_targets = [("button_white", ("BACK",))]
        if not prefer_back_only:
            exit_targets.extend(
                [
                    ("button_white", ("CLOSE",)),
                    ("button_green", ("OK", "NEXT", "PROCEED")),
                ]
            )

        for attempt in range(attempts):
            for classes, texts in exit_targets:
                clicked = self.waiter.click_when(
                    classes=(classes,),
                    texts=texts,
                    prefer_bottom=True,
                    allow_greedy_click=False,
                    timeout_s=1.5,
                    tag=f"{tag_prefix}_{texts[0].lower()}_{attempt}",
                )
                if clicked:
                    time.sleep(0.6)
                    if self._is_lobby_or_raceday_visible():
                        return True

        return self._is_lobby_or_raceday_visible()

    def _is_lobby_or_raceday_visible(self) -> bool:
        if self.waiter.seen(
            classes=("lobby_races", "race_race_day"),
            tag="skills_exit_seen_lobby",
        ):
            return True
        return self.waiter.seen(
            classes=("button_green",),
            texts=("RACE", "NEXT"),
            tag="skills_exit_seen_green",
            threshold=0.5,
        )

    def _focus_nudge(self, game_img: Image.Image, dets: List[DetectionDict]) -> None:
        """
        If nothing clicked on first pass, gently move cursor onto the scrollable list to
        'wake up' the focus, then micro-scrolls will land properly.
        """
        try:
            squares = [d for d in dets if d.get("name") == "skills_square"]
            if squares:
                self.ctrl.move_xyxy_center(squares[0]["xyxy"])
                logger_uma.debug("[skills] Focus nudge: moved to first skills_square")
            else:
                W, H = game_img.size
                cx, cy = W // 2, H // 2
                sx, sy = self.ctrl.local_to_screen(cx, cy)
                j = 10
                self.ctrl.move_to(
                    sx + random.randint(-j, j),
                    sy + random.randint(-j, j),
                    duration=0.18,
                )
                logger_uma.debug("[skills] Focus nudge: moved to screen center")
            time.sleep(0.07)
        except Exception as e:
            logger_uma.debug("[skills] Focus nudge failed: %s", e)

    def _scroll_once(self, scroll_time_range: Tuple[int, int]) -> None:
        """
        One scroll step (PC: wheel nudges; Android: drag with end-hold to kill inertia).
        """
        if isinstance(self.ctrl, ScrcpyController):
            xywh = self.ctrl._client_bbox_screen_xywh()
            if xywh is None:
                return
            x, y, w, h = xywh
            cx, cy = (x + w // 2), int(y + h * 0.60)
            self.ctrl.move_to(cx, cy)
            time.sleep(0.25)
            # Larger, slower drag to cover more of the skills list per scroll
            self.ctrl.scroll(
                -int(h * 0.25),
                steps=2,
                duration_range=(0.22, 0.40),
                end_hold_range=(0.2, 0.40),
            )
            # Inertia wait
            time.sleep(0.15)
        elif isinstance(self.ctrl, ADBController):
            # Reuse smart_scroll_small for ADB so we get anchor-aware drags
            anchor_xy = None
            xywh = self.ctrl._client_bbox_screen_xywh()
            if xywh:
                x, y, w, h = xywh
                anchor_xy = (x + w // 2, y + h // 2)
            smart_scroll_small(
                self.ctrl,
                steps_android=2,
                fraction_android=0.18,
                settle_pre_s=0.02,
                settle_mid_s=0.05,
                settle_post_s=0.25,
                anchor_xy=anchor_xy,
                end_hold_range_android=(0.40, 0.6),
            )
            # Inertia wait
            time.sleep(0.15)
        elif isinstance(self.ctrl, BlueStacksController):
            xywh = self.ctrl._client_bbox_screen_xywh()
            if not xywh:
                return
            x, y, w, h = xywh
            cx, cy = (x + w // 2), int(y + h * 0.60)
            self.ctrl.move_to(cx, cy)
            time.sleep(0.25)
            self.ctrl.scroll(
                -int(h * 0.15),
                steps=2,
                duration_range=(0.22, 0.40),
                end_hold_range=(0.15, 0.30),
            )
            time.sleep(0.15)
        else:
            n = random.randint(scroll_time_range[0], scroll_time_range[1])
            for _ in range(n):
                self.ctrl.scroll(-1)
                time.sleep(0.01)
        time.sleep(0.12)
