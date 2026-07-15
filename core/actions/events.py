# core/actions/events.py
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple, Dict, Any, Set

import numpy as np
import cv2

from PIL import Image

from core.types import DetectionDict
from core.controllers.base import IController
from core.perception.ocr.interface import OCRInterface  # your interface type
from core.perception.yolo.interface import IDetector
from core.utils.logger import logger_uma
from core.utils.waiter import Waiter
from core.utils.text import fuzzy_contains

# Event retriever (local-only, CPU) you packaged
from core.utils.event_processor import (
    Catalog,
    UserPrefs,
    Query,
    retrieve_best,
    max_positive_energy,
    extract_reward_categories,
    select_candidate_by_priority,
)

# -----------------------------
# Helpers
# -----------------------------


def _clamp_box(
    box: Tuple[float, float, float, float], w: int, h: int
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def _crop(img: Image.Image, box: Tuple[float, float, float, float]) -> Image.Image:
    W, H = img.size
    x1, y1, x2, y2 = _clamp_box(box, W, H)
    return img.crop((x1, y1, x2, y2))


def _sort_top_to_bottom(dets: List[DetectionDict]) -> List[DetectionDict]:
    return sorted(dets, key=lambda d: float(d["xyxy"][1]))


_CHAIN_BLUE_CFG = {
    "h_center": 105,
    "h_tol": 12,
    "s_min": 80,
    "v_min": 100,
    "coverage_min": 0.20,
}


def _is_blue_chain(frame: Image.Image, det: DetectionDict) -> bool:
    box = det.get("xyxy")
    if not box:
        return False

    crop = _crop(frame, tuple(box))
    if crop.width <= 1 or crop.height <= 1:
        return False

    bgr = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    cfg = _CHAIN_BLUE_CFG
    h_center = cfg["h_center"]
    tol = cfg["h_tol"]
    lo = (h_center - tol) % 180
    hi = (h_center + tol) % 180
    if lo <= hi:
        band = (h >= lo) & (h <= hi)
    else:
        band = (h >= lo) | (h <= hi)
    mask = band & (s >= cfg["s_min"]) & (v >= cfg["v_min"])

    coverage = float(np.count_nonzero(mask)) / max(1.0, mask.size)
    return coverage >= cfg["coverage_min"]


def _count_chain_steps(parsed: List[DetectionDict], *, frame: Optional[Image.Image] = None) -> Optional[int]:
    steps = [d for d in parsed if d.get("name") == "event_chain"]
    if not steps:
        return None

    if frame is not None:
        steps = [d for d in steps if _is_blue_chain(frame, d)]

    return len(steps) if steps else None


def _pick_event_card(parsed: List[DetectionDict]) -> Optional[DetectionDict]:
    cards = [d for d in parsed if d.get("name") == "event_card"]
    if not cards:
        return None
    # highest confidence first
    cards.sort(key=lambda d: float(d.get("conf", 0.0)), reverse=True)
    return cards[0]


def _choices(parsed: List[DetectionDict], *, conf_min: float) -> List[DetectionDict]:
    return [
        d
        for d in parsed
        if d.get("name") == "event_choice" and float(d.get("conf", 0.0)) >= conf_min
    ]


def _extract_title_description_from_banner(
    ocr: OCRInterface,
    frame: Image.Image,
    card_box: Tuple[float, float, float, float],
) -> Tuple[str, str]:
    """
    The blue banner spans horizontally to the right of the portrait (event_card).
    The 'Support Card Event' header sits roughly in the top 30-40% of that banner,
    and the actual event title (we want) is below it (bigger white text).
    Strategy: crop the right-side band with slight vertical padding; OCR; pick the
    longest high-signal line from the lower half.
    """
    W, H = frame.size
    x1, y1, x2, y2 = card_box
    card_w = x2 - x1
    card_h = y2 - y1

    # Right-side blue banner crop
    pad_x = 0.05 * card_w
    vpad = 0.10 * card_h
    right = (
        x2 + pad_x,
        max(0.0, y1 - vpad),
        min(W - 1.0, x2 + 6.5 * card_w),
        min(H - 1.0, y2 + vpad),
    )
    banner = _crop(frame, right)

    # Split roughly: top zone (header), bottom zone (title)
    bw, bh = banner.size
    # If the portrait is nearly square, the header ribbon is typically shorter,
    # so use 30% for the header; if it's a taller vertical rectangle, use 40%.
    aspect = card_h / max(card_w, 1e-6)  # h/w
    squareish = 0.85 <= aspect <= 1.15

    if squareish:
        split_y = int(0.30 * bh)
    else:
        split_y = int(0.40 * bh)

    title_zone = banner.crop((0, 0, bw, split_y))
    description_zone = banner.crop((0, split_y, bw, bh))

    title_text = ocr.text(title_zone)
    description_text = ocr.text(description_zone)

    return title_text, description_text


# -----------------------------
# EventFlow
# -----------------------------


@dataclass
class EventDecision:
    """What we matched and what we clicked."""

    matched_key: Optional[str]
    matched_key_step: Optional[str]
    pick_option: int
    clicked_box: Optional[Tuple[float, float, float, float]]
    debug: Dict[str, Any]


class EventFlow:
    """
    Encapsulates *Event* screen behavior:
      - Reads OCR title from the blue banner (right of the portrait).
      - Counts chain arrows to infer chain_step_hint.
      - Uses portrait crop (PIL) to help retrieval.
      - Resolves user/default preferences and clicks the selected option.
      - Falls back to top option on any inconsistency.
    """

    def __init__(
        self,
        ctrl: IController,
        ocr: OCRInterface,
        yolo_engine: IDetector,
        waiter: Waiter,
        catalog: Catalog,
        prefs: UserPrefs,
        *,
        conf_min_choice: float = 0.60,
        debug_visual: bool = False,
    ) -> None:
        self.ctrl = ctrl
        self.ocr = ocr
        self.yolo_engine = yolo_engine
        self.waiter = waiter
        self.catalog = catalog
        self.prefs = prefs
        self.conf_min_choice = conf_min_choice
        self.debug_visual = debug_visual
        # Track last event clicked to detect confirmation phases (e.g., Acupuncturist)
        self._last_event_clicked: Optional[Tuple[str, int, int]] = None  # (key_step, pick, expected_n)

    # ----- public API -----

    def process_event_screen(
        self,
        frame: Image.Image,
        parsed_objects_screen: List[DetectionDict],
        *,
        current_energy: int | None = None,
        max_energy_cap: int = 100,
    ) -> EventDecision:
        """
        Main entry point when we're already on an Event screen.
        """
        debug: Dict[str, Any] = {}
        debug["current_energy"] = current_energy
        debug["max_energy_cap"] = max_energy_cap
        # 1) Collect detections
        card = _pick_event_card(parsed_objects_screen)
        chain_step_hint = _count_chain_steps(parsed_objects_screen, frame=frame)
        if chain_step_hint is None and card is not None:
            chain_step_hint = 1
        choices = _choices(parsed_objects_screen, conf_min=self.conf_min_choice)
        choices_sorted = _sort_top_to_bottom(choices)

        debug["chain_step_hint"] = chain_step_hint
        debug["num_choices"] = len(choices_sorted)
        debug["has_event_card"] = card is not None

        # 2) Extract OCR title from banner (right of portrait)
        ocr_title = ""
        ocr_description = ""
        if card is not None:
            ocr_title, ocr_description = _extract_title_description_from_banner(
                self.ocr, frame, tuple(card["xyxy"])
            )
        else:
            # fallback heuristic: try to OCR a central horizontal band (less reliable)
            W, H = frame.size
            band = _crop(
                frame, (int(0.10 * W), int(0.30 * H), int(0.90 * W), int(0.55 * H))
            )
            ocr_title = self.ocr.text(band)
            ocr_title = (
                ocr_title[0] if isinstance(ocr_title, list) and ocr_title else ""
            )

        debug["ocr_title"] = ocr_title
        debug["ocr_description"] = ocr_description

        # 3) Build query for retriever
        type_hint = None
        if "support" in ocr_title.lower():
            type_hint = "support"
        elif "trainee" in ocr_title.lower():
            type_hint = "trainee"
        portrait_img: Optional[Image.Image] = None
        if card is not None:
            portrait_img = _crop(frame, tuple(card["xyxy"]))

        # Description holds the most important part
        ocr_query = ocr_description or ocr_title or ""
        q = Query(
            ocr_title=ocr_query,
            type_hint=type_hint,
            name_hint=None,  # not available at runtime (deck-agnostic)
            rarity_hint=None,  # not available; portrait helps instead
            chain_step_hint=chain_step_hint,
            portrait_image=portrait_img,  # <- PIL accepted by retriever (see diff)
            preferred_trainee_name=self.prefs.preferred_trainee_name if type_hint == "trainee" else None,
        )

        q_used = q

        # 4) Retrieve & rank
        cands = retrieve_best(self.catalog, q, top_k=3, min_score=0.5)

        if not cands and q.chain_step_hint and q.chain_step_hint != 1:
            q_fallback = replace(q, chain_step_hint=1)
            cands = retrieve_best(self.catalog, q_fallback, top_k=3, min_score=0.6)
            debug["chain_step_hint_fallback"] = {
                "from": q.chain_step_hint,
                "to": 1,
                "candidates_found": len(cands),
            }
            if cands:
                logger_uma.info(
                    "[event] Chain hint fallback succeeded: %s -> 1.",
                    q.chain_step_hint,
                )
                q_used = q_fallback
            else:
                logger_uma.warning(
                    "[event] Chain hint fallback failed after forcing step 1."
                )

        if not cands:
            logger_uma.warning(
                "[event] No candidates from retriever; falling back to top option."
            )
            return self._fallback_click_top(choices_sorted, debug)

        best = cands[0]
        debug["chain_step_hint_used"] = q_used.chain_step_hint
        debug["top_match"] = {
            "key": best.rec.key,
            "key_step": best.rec.key_step,
            "score": best.score,
            "text_sim": best.text_sim,
            "img_sim": best.img_sim,
            "bonus": best.hint_bonus,
        }

        # 5) Resolve preference
        pick = self.prefs.pick_for(best.rec)
        debug["pick_resolved"] = pick

        # 5.5) Special case: Unity Cup "A Team at Last" - match by team name using OCR BEFORE validation
        unity_cup_override = False
        if best.rec.key_step == "scenario/Unity Cup/None/None/A Team at Last#s1":
            logger_uma.info("[event] Unity Cup 'A Team at Last' detected - using OCR-based team matching.")
            
            # Get the desired team text from the pick option
            desired_team_text = None
            option_data = best.rec.options.get(str(pick))
            if option_data and isinstance(option_data, list) and len(option_data) > 0:
                desired_team_text = option_data[0].get("team")
            
            if desired_team_text:
                debug["unity_cup_team_search"] = {
                    "desired_pick": pick,
                    "desired_team": desired_team_text,
                    "available_choices": len(choices_sorted)
                }
                
                # OCR all visible choices to find the matching team
                best_match_idx = None
                best_match_score = 0.0
                ocr_results = []
                
                for idx, choice_det in enumerate(choices_sorted):
                    choice_crop = _crop(frame, tuple(choice_det["xyxy"]))
                    choice_text = self.ocr.text(choice_crop)
                    if isinstance(choice_text, list):
                        choice_text = " ".join(choice_text)
                    choice_text = choice_text.strip()
                    
                    ocr_results.append(choice_text)
                    
                    # Use fuzzy matching to handle OCR errors (threshold 0.7)
                    is_match, match_score = fuzzy_contains(
                        choice_text, 
                        desired_team_text, 
                        threshold=0.55, 
                        return_ratio=True
                    )
                    
                    if is_match and match_score > best_match_score:
                        best_match_score = match_score
                        best_match_idx = idx
                
                debug["unity_cup_team_search"]["ocr_results"] = ocr_results
                
                if best_match_idx is not None:
                    original_pick = pick
                    pick = best_match_idx + 1  # Convert to 1-indexed
                    unity_cup_override = True
                    logger_uma.info(
                        "[event] Unity Cup team matched: '%s' found at visual index %d (was %d). Match score: %.2f",
                        desired_team_text,
                        pick,
                        original_pick,
                        best_match_score
                    )
                    debug["unity_cup_team_search"]["matched_index"] = pick
                    debug["unity_cup_team_search"]["original_pick"] = original_pick
                    debug["unity_cup_team_search"]["match_score"] = best_match_score
                else:
                    # Fallback: if user chose a team that doesn't exist, select bottom choice (Team Carrot)
                    original_pick = pick
                    pick = len(choices_sorted)  # Bottom choice
                    unity_cup_override = True
                    logger_uma.warning(
                        "[event] Unity Cup team '%s' not found in OCR results; falling back to bottom choice #%d (Team Carrot)",
                        desired_team_text,
                        pick
                    )
                    debug["unity_cup_team_search"]["no_match"] = True
                    debug["unity_cup_team_search"]["fallback_to_bottom"] = pick
                    debug["unity_cup_team_search"]["original_pick"] = original_pick
            else:
                logger_uma.warning("[event] Unity Cup event but no team text found in option data.")

        # 6) Validate number of options vs YOLO choices
        expected_n = len(best.rec.options or {})
        debug["expected_n_options"] = expected_n

        if expected_n <= 0:
            logger_uma.warning(
                "[event] Matched event has no options in DB; fallback to top."
            )
            return self._fallback_click_top(choices_sorted, debug)

        # Skip validation checks for Unity Cup override since pick is already matched to visible choices
        if not unity_cup_override:
            if len(choices_sorted) != expected_n:
                logger_uma.warning(
                    "[event] YOLO found %d choices but DB expects %d; waiting for options to render and retrying.",
                    len(choices_sorted),
                    expected_n,
                )
                # Retry: wait for UI to finish rendering and recapture
                time.sleep(0.8)  # Increased wait time for slow-rendering options
                retry_frame, _, retry_parsed = self.yolo_engine.recognize(
                    imgsz=832, conf=0.60, iou=0.45, tag="event_retry"
                )
                retry_choices = _choices(retry_parsed, conf_min=self.conf_min_choice)
                retry_choices_sorted = _sort_top_to_bottom(retry_choices)
                debug["retry_num_choices"] = len(retry_choices_sorted)

                if len(retry_choices_sorted) == expected_n:
                    logger_uma.info(
                        "[event] Retry successful: now found %d choices as expected.",
                        expected_n,
                    )
                    choices_sorted = retry_choices_sorted
                else:
                    logger_uma.warning(
                        "[event] Retry still found %d choices (expected %d).",
                        len(retry_choices_sorted),
                        expected_n,
                    )
                    # Use retry result if it has more detections
                    if len(retry_choices_sorted) > len(choices_sorted):
                        choices_sorted = retry_choices_sorted
                        debug["used_retry_choices"] = True

                    # Check if preferred option is within detected range
                    if pick <= len(choices_sorted):
                        logger_uma.info(
                            "[event] Preferred option %d is within detected %d choices; proceeding.",
                            pick,
                            len(choices_sorted),
                        )
                        debug["partial_match_fallback"] = True
                    else:
                        logger_uma.warning(
                            "[event] Preferred option %d exceeds detected %d choices; fallback to top.",
                            pick,
                            len(choices_sorted),
                        )
                        return self._fallback_click_top(choices_sorted, debug)

            if pick < 1 or pick > expected_n:
                logger_uma.warning(
                    "[event] Preference pick=%d out of range 1..%d; fallback to top.",
                    pick,
                    expected_n,
                )
                return self._fallback_click_top(choices_sorted, debug)

        # (7) If we know current energy, attempt to avoid overfilling it and honor reward priorities.
        # Skip for Unity Cup override since pick is already determined by OCR matching
        adjusted_pick = pick
        avoid_overflow = True
        if hasattr(self.prefs, "should_avoid_energy"):
            try:
                avoid_overflow = bool(self.prefs.should_avoid_energy(best.rec))
            except Exception:
                avoid_overflow = getattr(self.prefs, "avoid_energy_overflow", True)
        else:
            avoid_overflow = getattr(self.prefs, "avoid_energy_overflow", True)
        debug["avoid_energy_overflow"] = avoid_overflow

        if not unity_cup_override and avoid_overflow and current_energy is not None and expected_n >= 1:
            candidate_order = [((pick - 1 + shift) % expected_n) + 1 for shift in range(expected_n)]

            option_categories: Dict[int, Set[str]] = {}
            option_energy_gain: Dict[int, int] = {}
            safe_candidates: List[int] = []

            # Allow a small overcap window for PAL support dates (≤ +10)
            pal_overcap_extra = 0
            try:
                if (
                    str(getattr(best.rec, "type", "")).strip().lower() == "support"
                    and str(getattr(best.rec, "attribute", "")).strip().upper() == "PAL"
                ):
                    pal_overcap_extra = 10
            except Exception:
                pal_overcap_extra = 0

            for option_num in range(1, expected_n + 1):
                outcomes_raw = best.rec.options.get(str(option_num), []) or []
                if not isinstance(outcomes_raw, list):
                    outcomes = [outcomes_raw]
                else:
                    outcomes = outcomes_raw

                gain = max_positive_energy(outcomes)
                option_energy_gain[option_num] = gain

                if gain <= 0 or (current_energy + gain) <= (max_energy_cap + pal_overcap_extra):
                    safe_candidates.append(option_num)

                option_categories[option_num] = extract_reward_categories(outcomes)

            try:
                reward_priority = list(self.prefs.reward_priority_for(best.rec))
            except AttributeError:
                reward_priority = list(getattr(self.prefs, "reward_priority", []))
            selection = None
            if pick not in safe_candidates:
                selection = select_candidate_by_priority(
                    candidate_order,
                    safe_candidates,
                    option_categories,
                    reward_priority,
                )

            if selection:
                candidate, matched_category = selection
                adjusted_pick = candidate
                if adjusted_pick != pick:
                    debug["pick_adjusted_due_to_energy"] = {
                        "from": pick,
                        "to": adjusted_pick,
                        "reason": "reward_priority" if matched_category else "energy_safe",
                    }
                if matched_category:
                    debug["reward_priority_match"] = matched_category
            elif safe_candidates:
                for candidate in candidate_order:
                    if candidate in safe_candidates:
                        adjusted_pick = candidate
                        if adjusted_pick != pick:
                            debug["pick_adjusted_due_to_energy"] = {
                                "from": pick,
                                "to": adjusted_pick,
                                "reason": "energy_safe",
                            }
                        break

            debug.setdefault("energy_gain_by_option", option_energy_gain)
            debug.setdefault(
                "reward_categories_by_option",
                {str(k): sorted(v) for k, v in option_categories.items() if v},
            )
        pick = adjusted_pick

        # Detect confirmation phase: if same event as last time with fewer options, override to option 1
        if self._last_event_clicked is not None:
            last_key_step, last_pick, last_expected_n = self._last_event_clicked
            # Confirmation phase indicators:
            # - Same event (key_step matches)
            # - Fewer options detected than expected
            # - Original pick was > 1
            if (best.rec.key_step == last_key_step and 
                len(choices_sorted) < expected_n and 
                last_pick > 1):
                logger_uma.info(
                    "[event] Confirmation phase detected for '%s': overriding pick from %d to 1 (confirm choice).",
                    best.rec.event_name or best.rec.key_step,
                    pick
                )
                pick = 1
                debug["confirmation_phase_override"] = True

        available_n = len(choices_sorted)
        if pick > available_n:
            logger_uma.warning(
                "[event] Adjusted pick=%d exceeds detected %d choices; fallback to top.",
                pick,
                available_n,
            )
            debug["available_choices"] = available_n
            return self._fallback_click_top(choices_sorted, debug)

        target = choices_sorted[pick - 1]
        self.ctrl.click_xyxy_center(target["xyxy"], clicks=2)
        
        # Update last event state for confirmation phase detection
        self._last_event_clicked = (best.rec.key_step, pick, expected_n)
        
        logger_uma.info(
            "[event] Clicked option #%d for %s (score=%.3f, energy=%s/%s).",
            pick,
            best.rec.key_step,
            best.score,
            str(current_energy),
            str(max_energy_cap),
        )

        return EventDecision(
            matched_key=best.rec.key,
            matched_key_step=best.rec.key_step,
            pick_option=pick,
            clicked_box=tuple(target["xyxy"]),
            debug=debug,
        )

    # ----- internals -----

    def _fallback_click_top(
        self,
        choices_sorted: List[DetectionDict],
        debug: Dict[str, Any],
    ) -> EventDecision:
        # Reset state when falling back since we don't have a proper match
        self._last_event_clicked = None
        
        if not choices_sorted:
            logger_uma.info("[event] No event_choice to click.")
            return EventDecision(
                matched_key=None,
                matched_key_step=None,
                pick_option=1,
                clicked_box=None,
                debug=debug,
            )

        top_choice = choices_sorted[0]
        self.ctrl.click_xyxy_center(top_choice["xyxy"], clicks=1)
        logger_uma.info(
            "[event] Fallback: clicked top event_choice (conf=%.3f).",
            float(top_choice.get("conf", 0.0)),
        )
        return EventDecision(
            matched_key=None,
            matched_key_step=None,
            pick_option=1,
            clicked_box=tuple(top_choice["xyxy"]),
            debug=debug,
        )
