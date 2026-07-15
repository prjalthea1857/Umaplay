# core\utils\training_check_helpers.py
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import cv2
import time
from core.perception.extractors.training_metrics import extract_failure_pct_for_tile
from core.settings import Settings

from core.perception.analyzers.hint import (
    assign_hints_to_supports,
    build_support_geometries,
)

from core.utils.logger import logger_uma
from core.utils.analyzers import analyze_support_crop
from core.utils.support_matching import (
    get_card_priority,
    get_runtime_support_matcher,
    match_support_crop,
)

SUPPORT_NAMES = {
    "support_card",
    "support_card_rainbow",
    "support_etsuko",
    "support_director",
    "support_tazuna",
    "support_kashimoto",
}

# -------- helpers --------
def _center_x(xyxy):
    x1, _, x2, _ = xyxy
    return 0.5 * (x1 + x2)

def _center(xyxy: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

def get_buttons_ltr(parsed_objs: List[Dict]) -> List[Dict]:
    btns = [d for d in parsed_objs if d["name"] == "training_button"]
    btns.sort(key=lambda d: _center_x(d["xyxy"]))
    return btns

def raised_training_ltr_index(
    parsed: List[Dict], tol_px: int = 3, tol_frac_h: float = 0.06
) -> Optional[int]:
    btns = [d for d in parsed if d["name"] == "training_button"]
    if len(btns) < 2:
        return None
    tops = np.array([d["xyxy"][1] for d in btns], dtype=float)
    heights = np.array([d["xyxy"][3] - d["xyxy"][1] for d in btns], dtype=float)
    med_top = float(np.median(tops))
    min_top = float(np.min(tops))
    thr = max(float(tol_px), float(tol_frac_h) * float(np.median(heights)))
    if (med_top - min_top) > thr:
        xs = np.array(
            [(d["xyxy"][0] + d["xyxy"][2]) * 0.5 for d in btns], dtype=float
        )
        order = np.argsort(xs)
        btns_ltr = [btns[i] for i in order]
        tops_ltr = [b["xyxy"][1] for b in btns_ltr]
        return int(np.argmin(tops_ltr))
    return None

def reindex_left_to_right(rows: List[Dict]) -> List[Dict]:
    """
    Normalize logical tile indices by current on-screen geometry to avoid
    duplicating 'last raised' tiles (e.g., WIT) due to timing/animation.
    """
    # Sort by X and assign 0..N-1 as canonical indices
    rows_sorted = sorted(rows, key=lambda r: float(r.get("tile_center_x", 0.0)))
    for j, r in enumerate(rows_sorted):
        r["tile_idx"] = j
    return rows_sorted

def failure_pct(cur_img, cur_parsed, tile_xyxy, energy, ocr):
    ENERGY_TO_IGNORE_FAILURE = 45
    if energy >= ENERGY_TO_IGNORE_FAILURE:
        return 0

    failure_predict = extract_failure_pct_for_tile(
        cur_img, cur_parsed, tile_xyxy, ocr
    )
    if failure_predict == -1:
        # try again
        time.sleep(0.2)
        failure_predict = extract_failure_pct_for_tile(
            cur_img, cur_parsed, tile_xyxy, ocr
        )

        if failure_predict == -1:
            failure_predict = Settings.MAX_FAILURE + 1

    return failure_predict

def _classify_flame_pose(flx1, fly1, flx2, fly2, geom) -> str:
    """
    Decide 'filling_up' (left badge by portrait) vs 'exploded' (bottom-right bubble).

    We use the flame *center* in normalized support coordinates (rx, ry),
    with permissive windows that allow slight overhang outside [0,1].
    """
    sx1, sy1, sx2, sy2 = geom.bbox
    sw = max(1.0, float(sx2 - sx1))
    sh = max(1.0, float(sy2 - sy1))
    cx = 0.5 * (flx1 + flx2)
    cy = 0.5 * (fly1 + fly2)
    rx = (cx - sx1) / sw
    ry = (cy - sy1) / sh

    # Windows (normalized) for the two poses:
    # - left badge sits roughly around rx ~ [-0.20..0.35], ry ~ [0.15..0.95]
    # - exploded bubble sits around rx ~ [0.55..1.25],  ry ~ [0.45..1.20]
    in_left  = (-0.20 <= rx <= 0.35) and (0.15 <= ry <= 0.95)
    in_right = ( 0.55 <= rx <= 1.25) and (0.45 <= ry <= 1.20)

    if in_right and not in_left:
        return "exploded"
    if in_left and not in_right:
        return "filling_up"

    # Tie-breaker: pick the closer prototype (works if center lands on the border)
    d_left  = (rx - 0.10)**2 + (ry - 0.65)**2
    d_right = (rx - 1.00)**2 + (ry - 0.70)**2
    return "exploded" if d_right < d_left else "filling_up"


_SPIRIT_CLF = None
def _get_spirit_clf():
    """Lazy-load once; safe if the package is missing."""
    global _SPIRIT_CLF
    if _SPIRIT_CLF is not None:
        return _SPIRIT_CLF

    if Settings.USE_EXTERNAL_PROCESSOR:
        try:
            from core.perception.classifiers.spirit_remote import (
                RemoteUnityCupSpiritClassifier,
            )

            _SPIRIT_CLF = RemoteUnityCupSpiritClassifier(
                Settings.EXTERNAL_PROCESSOR_URL
            )
            logger_uma.info(
                "[spirit_clf] Using remote classifier at %s",
                Settings.EXTERNAL_PROCESSOR_URL,
            )
            return _SPIRIT_CLF
        except Exception as e:
            logger_uma.warning(
                "Remote spirit classifier unavailable (%s). Falling back to local load.",
                e,
            )
            _SPIRIT_CLF = None

    try:
        from core.perception.unity_cup_spirit_classifier import (
            UnityCupSpiritClassifier,
        )

        _SPIRIT_CLF = UnityCupSpiritClassifier.load_from_settings()
        logger_uma.info("[spirit_clf] Loaded local TinyCNN classifier")
    except Exception as e:
        logger_uma.warning("UnityCupSpiritClassifier load failed: %s", e)
        _SPIRIT_CLF = None
    return _SPIRIT_CLF

# Extreme Spirit Burst (July 2026) color heuristic. Calibrated qualitatively from
# in-game screenshots of the "Extreme Spirit Burst" trigger banner and its trainer-held
# flame effect: a bright, highly-saturated magenta/pink glow with a warm orange-yellow
# core, visually distinct from the existing blue (cool hue) and white (low saturation)
# spirit classes the trained CNN already knows. UNCALIBRATED against the actual small
# in-game icon crop (only the animated banner/effect was confirmed) -- retune
# PURPLE_HUE_MIN/MAX, PURPLE_SAT_MIN, PURPLE_VAL_MIN, PURPLE_FRAC_MIN below once real
# icon crops are available, or replace with a proper 3-class classifier retrain.
PURPLE_HUE_MIN = 140   # OpenCV H channel (0-179), magenta/pink band
PURPLE_HUE_MAX = 172
PURPLE_SAT_MIN = 90    # high saturation to avoid muted pink UI chrome / skin tones
PURPLE_VAL_MIN = 120   # bright/glowing, not a dark muted pink
PURPLE_FRAC_MIN = 0.15 # fraction of crop pixels that must match to call it purple


def _detect_extreme_spirit_purple(crop_rgb) -> bool:
    """Heuristic pre-check for Extreme Spirit Burst; see module-level constants above."""
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (
        (hue >= PURPLE_HUE_MIN) & (hue <= PURPLE_HUE_MAX)
        & (sat >= PURPLE_SAT_MIN) & (val >= PURPLE_VAL_MIN)
    )
    frac = float(np.count_nonzero(mask)) / max(1, mask.size)
    return frac >= PURPLE_FRAC_MIN


def _classify_spirit_icon(frame_bgr, xyxy, *, threshold: float = 0.51):
    """
    Returns dict with keys: spirit_label ('spirit_blue'|'spirit_white'|'spirit_purple_heuristic'|'unknown'),
    spirit_color ('blue'|'white'|'purple'|'unknown'), spirit_confidence (0..1).

    'purple' identifies Extreme Spirit Burst (added July 2026), detected via
    _detect_extreme_spirit_purple() below since the trained classifier bundle only
    knows 'spirit_blue'/'spirit_white' (see models/unity_spirit_classes.json) and has
    never seen the purple icon. The heuristic runs first and short-circuits before the
    CNN; if it doesn't fire, behavior is unchanged from before ESB support was added.
    """
    clf = _get_spirit_clf()
    if clf is None or not xyxy:
        return {"spirit_label": "unknown", "spirit_color": "unknown", "spirit_confidence": 0.0}

    x1, y1, x2, y2 = [int(v) for v in xyxy]
    h, w = frame_bgr.shape[:2]
    pad = 2
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return {"spirit_label": "unknown", "spirit_color": "unknown", "spirit_confidence": 0.0}

    crop_rgb = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)

    if _detect_extreme_spirit_purple(crop_rgb):
        return {
            "spirit_label": "spirit_purple_heuristic",
            "spirit_color": "purple",
            "spirit_confidence": 1.0,
        }

    pil_img = Image.fromarray(crop_rgb)

    try:
        pred = clf.predict(pil_img)  # {'pred_label':'spirit_blue', 'confidence':0.97, ...}
        label = str(pred.get("pred_label", "unknown"))
        conf = float(pred.get("confidence", 0.0))
        if conf < threshold:
            label = "unknown"
    except Exception as e:
        logger_uma.debug("Spirit color predict error: %s", e)
        label, conf = "unknown", 0.0

    if label == "spirit_blue":
        color = "blue"
    elif label == "spirit_white":
        color = "white"
    elif label == "spirit_purple":
        color = "purple"
    else:
        color = "unknown"
    return {"spirit_label": label, "spirit_color": color, "spirit_confidence": conf}


def collect_supports_enriched(
    cur_img: Image.Image, cur_parsed: List[Dict], conf_support: float = 0.60
) -> Tuple[List[Dict], bool]:
    """
    Take *all* supports visible in this capture — they correspond to the currently raised tile.
    Enrich each with bar/type pieces, hint, rainbow, etc.
    """
    frame_bgr = cv2.cvtColor(np.array(cur_img), cv2.COLOR_RGB2BGR)

    # --- helpers: IoU + NMS ---
    def _area(xyxy):
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _iou(a, b):
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        ua = _area(a) + _area(b) - inter
        return inter / ua if ua > 0 else 0.0

    def _nms_by_iou(dets, iou_thr=0.50):
        """
        Class-agnostic NMS: keep highest-conf per overlap cluster.
        Each det: {"xyxy":[...], "conf":float, "name":str, ...}
        """
        if not dets:
            return dets
        # sort by confidence DESC (missing conf -> 0.0)
        ordered = sorted(
            dets, key=lambda d: float(d.get("conf", 0.0)), reverse=True
        )
        kept = []
        for d in ordered:
            dx = d.get("xyxy")
            if not dx:
                continue
            drop = False
            for k in kept:
                if _iou(dx, k.get("xyxy")) >= iou_thr:
                    drop = True
                    break

                kc = k.get("xyxy")
                if kc is None:
                    continue

                cx_d, cy_d = _center(dx)
                cx_k, cy_k = _center(kc)

                kx1, ky1, kx2, ky2 = [float(v) for v in kc]
                dx1, dy1, dx2, dy2 = [float(v) for v in dx]

                center_overlap = (
                    kx1 <= cx_d <= kx2
                    and ky1 <= cy_d <= ky2
                ) or (
                    dx1 <= cx_k <= dx2
                    and dy1 <= cy_k <= dy2
                )

                if center_overlap:
                    drop = True
                    break
            if not drop:
                kept.append(d)
        return kept

    # Raw supports filtered by confidence
    supports_raw = [
        d
        for d in cur_parsed
        if d["name"] in SUPPORT_NAMES and d.get("conf", 0.0) >= conf_support
    ]
    # De-duplicate overlaps (e.g., double rainbow hits)
    supports = _nms_by_iou(supports_raw, iou_thr=0.50)
    parts_bar = [d for d in cur_parsed if d["name"] == "support_bar"]
    parts_type = [d for d in cur_parsed if d["name"] == "support_type"]

    # === SPIRIT detections (works like hint, but independent) ===
    conf_spirit = 0.55
    spirits_raw = [d for d in cur_parsed
                   if d["name"] == "unity_spirit" and d.get("conf", 0.0) >= conf_spirit]
    spirits = _nms_by_iou(spirits_raw, iou_thr=0.30)

    # Hint detections (primary source going forward)
    conf_support_hint = 0.55
    hints_raw = [
        d
        for d in cur_parsed
        if d["name"] == "support_hint" and d.get("conf", 0.0) >= conf_support_hint
    ]
    hints = _nms_by_iou(hints_raw, iou_thr=0.30)
    support_geoms = build_support_geometries(supports)

    # === FLAME detections (pose lives near the card; does NOT jump) ===
    conf_flame = 0.55
    flames_raw = [
        d for d in cur_parsed
        if d["name"] == "unity_flame" and d.get("conf", 0.0) >= conf_flame
    ]
    flames = _nms_by_iou(flames_raw, iou_thr=0.30)

    # Reuse the same geometry assignment for both markers
    support_assignments, tile_hints = assign_hints_to_supports(
        support_geoms, hints, canvas_height=frame_bgr.shape[0],
    )
    spirit_assignments, tile_spirits = assign_hints_to_supports(
        support_geoms, spirits, canvas_height=frame_bgr.shape[0],
    )
    # Flames tend to be either at left side of the circular portrait or at the bottom-right
    # of the portrait, sometimes slightly outside the bbox. Use a wider horizontal expansion
    # and extra bottom headroom to catch the 'exploded' bubble that hangs below.
    flame_assignments, tile_flames = assign_hints_to_supports(
        support_geoms,
        flames,
        canvas_height=frame_bgr.shape[0],
        expand_x_frac=0.50,     # allow overhang to left/right
        expand_top_frac=0.20,   # small margin above the portrait
        expand_bottom_frac=0.55 # larger margin below (exploded bubble)
    )

    if tile_spirits:
        logger_uma.debug("[unity_spirit] Tile spirits detected=%d", len(tile_spirits))
    if tile_hints:
        logger_uma.debug("[support_hint] Tile hints detected=%d", len(tile_hints))
    if tile_flames:
        logger_uma.debug("[unity_flame] Tile flames detected=%d", len(tile_flames))

    def _inside(inner, outer, pad=2):
        ix1, iy1, ix2, iy2 = inner
        ox1, oy1, ox2, oy2 = outer
        return (
            ix1 >= ox1 - pad
            and iy1 >= oy1 - pad
            and ix2 <= ox2 + pad
            and iy2 <= oy2 + pad
        )

    min_confidence = 0.25
    matcher: Optional[Any] = None
    has_priority_customization = bool(Settings.SUPPORT_PRIORITIES_HAVE_CUSTOMIZATION)
    custom_priority_keys = Settings.SUPPORT_CUSTOM_PRIORITY_KEYS

    enriched: List[Dict] = []
    any_rainbow = False

    for s, geom in zip(supports, support_geoms):
        x1, y1, x2, y2 = geom.bbox
        crop = frame_bgr[y1:y2, x1:x2].copy()

        # parts within this support
        bar_xyxy = None
        type_xyxy = None
        for pb in parts_bar:
            bx1, by1, bx2, by2 = [int(v) for v in pb["xyxy"]]
            if _inside((bx1, by1, bx2, by2), (x1, y1, x2, y2), pad=1):
                bar_xyxy = (bx1, by1, bx2, by2)
                break
        for pt in parts_type:
            tx1, ty1, tx2, ty2 = [int(v) for v in pt["xyxy"]]
            if _inside((tx1, ty1, tx2, ty2), (x1, y1, x2, y2), pad=1):
                type_xyxy = (tx1, ty1, tx2, ty2)
                break

        bar_crop = (
            None
            if bar_xyxy is None
            else frame_bgr[
                bar_xyxy[1] : bar_xyxy[3], bar_xyxy[0] : bar_xyxy[2]
            ].copy()
        )
        type_crop = (
            None
            if type_xyxy is None
            else frame_bgr[
                type_xyxy[1] : type_xyxy[3], type_xyxy[0] : type_xyxy[2]
            ].copy()
        )

        support_key = geom.key
        assigned_hints = support_assignments.get(support_key, [])
        hint_confidence_max = (
            max((h.get("conf", 0.0) for h in assigned_hints), default=0.0)
            if assigned_hints
            else 0.0
        )

        assigned_spirits = spirit_assignments.get(support_key, [])
        spirit_confidence_max = max((spt.get("conf", 0.0) for spt in assigned_spirits), default=0.0)

        # NEW: classify color of the best spirit box (if any)
        spirit_color = "unknown"
        spirit_label = "unknown"
        spirit_color_conf = 0.0
        if assigned_spirits:
            # pick the highest-conf spirit detection for color classification
            best_spt = max(assigned_spirits, key=lambda d: float(d.get("conf", 0.0)))
            spirit_cls = _classify_spirit_icon(frame_bgr, best_spt.get("xyxy"))
            spirit_label = spirit_cls["spirit_label"]
            spirit_color = spirit_cls["spirit_color"]
            spirit_color_conf = float(spirit_cls["spirit_confidence"])
            
        attrs = analyze_support_crop(
            s["name"],
            crop,
            piece_bar_bgr=bar_crop,
            piece_type_bgr=type_crop,
            hint_sources=assigned_hints,
            hint_confidence_max=hint_confidence_max,
        )

        has_rainbow = s["name"].endswith("_rainbow") or (
            s["name"] == "support_card_rainbow"
        )
        any_rainbow |= has_rainbow

        normalized_name = s["name"].replace("_rainbow", "")

        has_spirit = len(assigned_spirits) > 0
        has_hint_yolo = bool(attrs.get("has_hint_yolo", False))
        has_hint_hsv = bool(attrs.get("has_hint_hsv", False))
        hint_confidence_max = float(attrs.get("hint_confidence_max", 0.0))
        hint_sources = attrs.get("hint_sources", assigned_hints)

        assigned_flames = flame_assignments.get(support_key, [])
        flame_confidence_max = max((f.get("conf", 0.0) for f in assigned_flames), default=0.0)

        has_flame = len(assigned_flames) > 0
        flame_type: Optional[str] = None
        if has_flame:
            # If multiple flames got attached (rare), classify by the highest-conf one
            best = max(assigned_flames, key=lambda f: float(f.get("conf", 0.0)))
            fx1, fy1, fx2, fy2 = [float(v) for v in best.get("xyxy", (0, 0, 0, 0))]
            flame_type = _classify_flame_pose(fx1, fy1, fx2, fy2, geom)

        support_record: Dict[str, Any] = {
            **s,
            "name": normalized_name,
            "support_type": attrs["support_type"],
            "support_type_score": attrs["support_type_score"],
            "friendship_bar": attrs["friendship_bar"],
            "has_hint": bool(attrs.get("has_hint", False)),
            "has_rainbow": bool(has_rainbow),
            "has_hint_yolo": has_hint_yolo,
            "has_hint_hsv": has_hint_hsv,
            "hint_sources": hint_sources,
            "hint_confidence_max": hint_confidence_max,

            # Spirit fields
            "has_spirit": has_spirit,
            "spirit_sources": assigned_spirits,
            "spirit_confidence_max": float(spirit_confidence_max),
            "spirit_label": spirit_label,                 # 'spirit_blue' | 'spirit_white' | 'unknown'
            "spirit_color": spirit_color,                 # 'blue' | 'white' | 'unknown'
            "spirit_color_confidence": spirit_color_conf, # 0..1

            # flame fields
            "has_flame": bool(has_flame),
            "flame_type": flame_type,  # 'filling_up' | 'exploded' | None
            "flame_sources": assigned_flames,
            "flame_confidence_max": float(flame_confidence_max),
        }

        matched_priority_assignments: Dict[
            Tuple[str, str, str], Tuple[float, Dict[str, Any]]
        ] = {}
        def register_priority_match(
            card_key: Tuple[str, str, str],
            *,
            score: float,
            record: Dict[str, Any],
        ) -> bool:
            existing = matched_priority_assignments.get(card_key)
            if existing and existing[0] >= score:
                logger_uma.debug(
                    "[support_match] Duplicate match %s score=%.3f skipped (existing=%.3f)",
                    card_key,
                    score,
                    existing[0],
                )
                return False

            if existing:
                prev_record = existing[1]
                prev_record.pop("matched_card", None)
                prev_record.pop("priority_config", None)
                logger_uma.debug(
                    "[support_match] Reassigning priority match %s old_score=%.3f → new_score=%.3f",
                    card_key,
                    existing[0],
                    score,
                )

            matched_priority_assignments[card_key] = (score, record)
            return True

        if support_record["has_hint"] and has_priority_customization:
            if matcher is None:
                matcher = get_runtime_support_matcher(min_confidence=min_confidence)
            match = match_support_crop(crop, matcher=matcher)
            if match:
                name = match.get("name", "")
                rarity = match.get("rarity", "")
                attribute = match.get("attribute", "")
                support_record["matched_card"] = match
                key = (name, rarity, attribute)
                if key not in custom_priority_keys:
                    logger_uma.debug(
                        "[support_match] Match %s has no custom priority; skipping",
                        key,
                    )
                elif register_priority_match(key, score=float(match.get("score", 0.0)), record=support_record):
                    support_record["priority_config"] = get_card_priority(
                        name,
                        rarity,
                        attribute,
                    )

                    logger_uma.debug(
                        "[support_match] Hint support matched %s (%s/%s) score=%.3f",
                        name,
                        rarity,
                        attribute,
                        match.get("score", 0.0),
                    )
                else:
                    logger_uma.debug(
                        "[support_match] Match %s rejected due to superior assignment",
                        key,
                    )
            else:
                logger_uma.debug("[support_match] No confident match for hint support")

        enriched.append(support_record)

    return enriched, any_rainbow