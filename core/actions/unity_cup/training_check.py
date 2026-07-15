
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from core.settings import Settings
from core.utils.skill_memory import SkillMemoryManager
from typing import Any
from core.types import BLUE_GREEN, ORANGE_MAX, TileSV


# ---- knobs you may want to tweak later (kept here for clarity) ----
GREEDY_THRESHOLD_UNITY_CUP = 3.5
HIGH_SV_THRESHOLD = 3.5  # when SV >= this, allow risk up to ×RISK_RELAX_FACTOR
RISK_RELAX_FACTOR = 1.5  # e.g., 20% -> 30% when SV is high

# Director scoring by bar color (latest rule you wrote)
DIRECTOR_SCORE_BY_COLOR = {
    "blue": 0.25,  # "blue or less"
    "green": 0.15,
    "orange": 0.10,
    "yellow": 0.00,  # max (or treat is_max as yellow)
    "max": 0.00,  # alias
}

def compute_support_values(training_state: List[Dict]) -> List[Dict[str, Any]]:
    """
    Unity Cup variant. Same output shape as URA.
    Differences:
      • Hint defaults: 0.50 (blue/green), 0.25 (orange/max) — still overridable via priority_config
      • Rainbow combo: 0.25 * (#rainbows) if >= 2
      • Kashimoto: if support_type ∈ {SPD,STA,PWR,GUTS,WIT,PAL} → treat like PAL card (Tazuna rules);
                   otherwise treat like Director (color-based)
      • Spirits: per-spirit + combo bonus:
            per spirit: +0.50 if flame_type == 'filling_up', +0.12 if 'exploded'
            combo (only if ≥2 filling): 0.25*(2*n_fill-1) + 0.01*n_exploded
    """
    out: List[TileSV] = []
    skill_memory = SkillMemoryManager(
        Settings.resolve_skill_memory_path(Settings.ACTIVE_SCENARIO),
        scenario=Settings.ACTIVE_SCENARIO,
    )

    def _canon_skill(name: object) -> str:
        s = str(name or "")
        for sym in ("◎", "○", "×"):
            s = s.replace(sym, "")
        return " ".join(s.split()).strip()

    default_priority_cfg = Settings.default_support_priority()
    # Unity defaults (still overridable):
    adv_settings = Settings.UNITY_CUP_ADVANCED
    scores_cfg = adv_settings.get("scores", {}) if isinstance(adv_settings, dict) else {}

    def _score_value(key: str, fallback: float) -> float:
        try:
            return float(scores_cfg.get(key, fallback))
        except (TypeError, ValueError):
            return fallback

    SCORE_WHITE_FILL = _score_value("whiteSpiritFill", 0.40)
    SCORE_WHITE_EXPLODED = _score_value("whiteSpiritExploded", 0.13)
    SCORE_WHITE_COMBO_BASE = _score_value("whiteComboBase", 0.20)
    SCORE_WHITE_COMBO_PER_FILL = _score_value("whiteComboPerFill", 0.25)
    SCORE_WHITE_COMBO_EXPLODED = _score_value("whiteComboExplodedTiny", 0.01)
    SCORE_BLUE_EACH = _score_value("blueSpiritEach", 0.50)
    SCORE_BLUE_COMBO_PER_EXTRA = _score_value("blueComboPerExtraFill", 0.25)
    SCORE_RAINBOW_COMBO = _score_value("rainbowCombo", 0.50)
    # Extreme Spirit Burst (July 2026): purple variant, "significantly bigger stat boosts"
    # and 0% failure rate. Detection is an uncalibrated HSV heuristic (see
    # _detect_extreme_spirit_purple in training_check_helpers.py), not a trained
    # classifier -- verify against real gameplay before trusting it fully.
    SCORE_PURPLE_EACH = _score_value("purpleSpiritEach", 2.50)
    SCORE_PURPLE_COMBO_PER_EXTRA = _score_value("purpleComboPerExtraFill", 1.00)

    UNITY_BLUEGREEN_HINT_DEFAULT = 0.50
    UNITY_ORANGE_HINT_DEFAULT = 0.25

    def _support_label(support: Dict[str, Any]) -> str:
        matched = support.get("matched_card") or {}
        if isinstance(matched, dict) and matched.get("name"):
            name = str(matched.get("name", "")).strip()
            attr = str(matched.get("attribute", "")).strip()
            rarity = str(matched.get("rarity", "")).strip()
            suffix = " / ".join([p for p in (attr, rarity) if p])
            return f"{name} ({suffix})" if suffix else (name or "support")
        return str(support.get("name", "support")).strip() or "support"

    def _hint_candidate_for_support(
        support: Dict[str, Any],
        *,
        color_key: str,
        default_value: float,
        color_desc: str,
    ) -> Tuple[float, Dict[str, Any]]:
        priority_cfg = support.get("priority_config")
        matched_card = support.get("matched_card")
        matched = isinstance(matched_card, dict) and bool(matched_card)
        if not isinstance(priority_cfg, dict):
            priority_cfg = default_priority_cfg
            matched = False

        enabled = bool(priority_cfg.get("enabled", True))
        # Gate by required skills (if all already bought → disable)
        gated = False
        try:
            req = priority_cfg.get("skillsRequiredForPriority")
            req_list = []
            if isinstance(req, list):
                req_list = [n for n in (_canon_skill(x) for x in req) if n]
            elif isinstance(req, str):
                req_list = [n for n in (_canon_skill(x) for x in str(req).split(",")) if n]
            if req_list:
                gated = all(skill_memory.has_bought(n) for n in req_list)
        except Exception:
            gated = False
        if gated:
            enabled = False

        label = _support_label(support)
        config_value = float(priority_cfg.get(color_key, default_value))  # allow override
        base_value = config_value if matched else default_value
        important_mult = 3.0 if Settings.HINT_IS_IMPORTANT else 1.0
        effective_value = base_value * important_mult if enabled else 0.0
        meta = {
            "label": label,
            "color_desc": color_desc,
            "enabled": enabled,
            "matched": matched,
            "base_value": base_value,
            "important_mult": important_mult,
            "gated": gated,
        }
        return effective_value, meta

    def _format_hint_note(meta: Dict[str, Any], bonus: float) -> str:
        label = meta["label"]; color_desc = meta["color_desc"]; base_value = meta["base_value"]
        source = "priority" if meta["matched"] else "default"
        note = f"Hint on {label} ({color_desc}): +{bonus:.2f} (base={base_value:.2f} {source}"
        if meta.get("important_mult", 1.0) != 1.0:
            note += f", important×{meta['important_mult']:.1f}"
        note += ")"
        return note

    KNOWN_TYPES = {"SPD","STA","PWR","GUTS","WIT","PAL"}

    for tile in training_state:
        idx = int(tile.get("tile_idx", -1))
        failure_pct = int(tile.get("failure_pct", 0) or 0)
        supports = tile.get("supports", []) or []

        sv_total = 0.0
        sv_by_type: Dict[str, float] = {}
        notes: List[str] = []

        blue_hint_candidates: List[Tuple[float, Dict[str, Any]]] = []
        orange_hint_candidates: List[Tuple[float, Dict[str, Any]]] = []
        hint_disabled_notes: List[str] = []

        rainbow_count = 0

        # ---- 1) per-support contributions -----------------------------------
        for s in supports:
            sname = s.get("name", "")
            bar = s.get("friendship_bar", {}) or {}
            color = str(bar.get("color", "unknown")).lower()
            is_max = bool(bar.get("is_max", False))
            has_hint = bool(s.get("has_hint", False))
            has_rainbow = bool(s.get("has_rainbow", False))
            stype = (s.get("support_type") or "").strip().upper()
            label = _support_label(s)

            if is_max and color not in ("yellow", "max"):
                color = "yellow"

            # Special cameos
            if sname == "support_etsuko":
                sv_total += 0.10
                sv_by_type["special_reporter"] = sv_by_type.get("special_reporter", 0.0) + 0.10
                notes.append(f"Reporter ({label}): +0.10")
                continue

            if sname == "support_director":
                score = DIRECTOR_SCORE_BY_COLOR.get(color, DIRECTOR_SCORE_BY_COLOR["yellow"])
                if score > 0:
                    sv_total += score
                    sv_by_type["special_director"] = sv_by_type.get("special_director", 0.0) + score
                notes.append(f"Director ({label}, {color}): +{score:.2f}")
                continue

            if sname == "support_tazuna":
                # PAL rules
                if color in ("blue",):       score = 1.5
                else:                                 score = 0.5
                sv_total += score
                sv_by_type["special_tazuna"] = sv_by_type.get("special_tazuna", 0.0) + score
                notes.append(f"Tazuna ({label}, {color}): +{score:.2f}")
                continue

            if sname == "support_kashimoto":
                # If she shows any support_type → treat as PAL; else as Director
                if stype in KNOWN_TYPES and stype != "":
                    if color in ("blue",):       score = 1.5
                    else:                                 score = 0.5
                    sv_total += score
                    sv_by_type["special_kashimoto_pal"] = sv_by_type.get("special_kashimoto_pal", 0.0) + score
                    notes.append(f"Kashimoto as PAL ({label}, {color}): +{score:.2f}")
                else:
                    score = DIRECTOR_SCORE_BY_COLOR.get(color, DIRECTOR_SCORE_BY_COLOR["yellow"])
                    if score > 0:
                        sv_total += score
                        sv_by_type["special_kashimoto_director"] = sv_by_type.get("special_kashimoto_director", 0.0) + score
                    notes.append(f"Kashimoto as Director ({label}, {color}): +{score:.2f}")
                continue

            # Standard supports
            if has_rainbow:
                sv_total += 1.0
                rainbow_count += 1
                notes.append(f"rainbow ({label}): +1.00")

            if color in BLUE_GREEN:
                sv_total += 1.0
                sv_by_type["cards"] = sv_by_type.get("cards", 0.0) + 1.0
                notes.append(f"{label} {color}: +1.00")
                if has_hint:
                    bonus, meta = _hint_candidate_for_support(
                        s,
                        color_key="scoreBlueGreen",
                        default_value=UNITY_BLUEGREEN_HINT_DEFAULT,
                        color_desc="blue/green",
                    )
                    if not meta["enabled"]:
                        hint_disabled_notes.append(
                            f"Hint on {meta['label']} ({meta['color_desc']}): skipped (priority disabled)"
                        )
                    else:
                        blue_hint_candidates.append((bonus, meta))
            elif color in ORANGE_MAX or is_max:
                if has_hint:
                    bonus, meta = _hint_candidate_for_support(
                        s,
                        color_key="scoreOrangeMax",
                        default_value=UNITY_ORANGE_HINT_DEFAULT,
                        color_desc="orange/max",
                    )
                    if not meta["enabled"]:
                        hint_disabled_notes.append(
                            f"Hint on {meta['label']} ({meta['color_desc']}): skipped (priority disabled)"
                        )
                    else:
                        orange_hint_candidates.append((bonus, meta))
                notes.append(f"{label} {color}: +0.00")
            else:
                notes.append(f"{label} {color}: +0.00 (unknown color category)")

        # ---- tile-capped hint bonus (best only) ------------------------------
        for dn in hint_disabled_notes:
            notes.append(dn)

        best_hint_value = 0.0
        best_hint_meta: Optional[Dict[str, Any]] = None
        if blue_hint_candidates:
            v, m = max(blue_hint_candidates, key=lambda it: it[0])
            if v > best_hint_value:
                best_hint_value, best_hint_meta = v, {**m, "bucket": "hint_bluegreen"}
        if orange_hint_candidates:
            v, m = max(orange_hint_candidates, key=lambda it: it[0])
            if v > best_hint_value:
                best_hint_value, best_hint_meta = v, {**m, "bucket": "hint_orange_max"}

        if best_hint_meta and best_hint_value > 0:
            bucket = str(best_hint_meta.get("bucket", "hint_bluegreen"))
            sv_total += best_hint_value
            sv_by_type[bucket] = sv_by_type.get(bucket, 0.0) + best_hint_value
            notes.append(_format_hint_note(best_hint_meta, best_hint_value))
        elif best_hint_meta:
            notes.append(_format_hint_note(best_hint_meta, best_hint_value))

        # ---- rainbow combo (Unity) ------------------------------------------
        if rainbow_count >= 2:
            combo_bonus = SCORE_RAINBOW_COMBO * float(rainbow_count - 1)
            sv_total += combo_bonus
            sv_by_type["rainbow_combo"] = sv_by_type.get("rainbow_combo", 0.0) + combo_bonus
            notes.append(f"Rainbow combo ({rainbow_count}): +{combo_bonus:.2f}")

        # ---- spirits (colored) ----------------------------------------------
        spirits = [s for s in supports if s.get("has_spirit", False)]

        # Split by color
        whites = [s for s in spirits if (s.get("spirit_color") == "white" or s.get("spirit_color") == "unknown")]
        blues  = [s for s in spirits if s.get("spirit_color") == "blue"]
        purples = [s for s in spirits if s.get("spirit_color") == "purple"]

        # Per-spirit base value
        n_white_fill     = sum(1 for s in whites if s.get("has_flame") and s.get("flame_type") == "filling_up")
        n_white_exploded = sum(1 for s in whites if s.get("has_flame") and s.get("flame_type") == "exploded")
        n_blue_total     = len(blues)
        n_blue_fill      = sum(1 for s in blues  if s.get("has_flame") and s.get("flame_type") == "filling_up")
        n_purple_total   = len(purples)
        n_purple_fill    = sum(1 for s in purples if s.get("has_flame") and s.get("flame_type") == "filling_up")

        # White spirits: same rule as before (0.50 filling, 0.12 exploded)
        white_value = SCORE_WHITE_FILL * n_white_fill + SCORE_WHITE_EXPLODED * n_white_exploded
        if white_value > 0:
            sv_total += white_value
            sv_by_type["spirits_white"] = sv_by_type.get("spirits_white", 0.0) + white_value
            notes.append(f"White spirits value sum: +{white_value:.2f} (fill={n_white_fill}, exploded={n_white_exploded})")

        # White combo (only for not-exploded/flame filling) + tiny weight for exploded inside combo
        white_combo = 0.0
        if n_white_fill >= 2:
            white_combo += SCORE_WHITE_COMBO_BASE + SCORE_WHITE_COMBO_PER_FILL * n_white_fill
        if (n_white_fill + n_white_exploded) >= 2:
            white_combo += SCORE_WHITE_COMBO_EXPLODED * n_white_exploded
        if white_combo > 0:
            sv_total += white_combo
            sv_by_type["spirit_combo_white"] = sv_by_type.get("spirit_combo_white", 0.0) + white_combo
            notes.append(f"White spirit combo sum: +{white_combo:.2f} => SCORE_WHITE_COMBO_BASE={SCORE_WHITE_COMBO_BASE} + (SCORE_WHITE_COMBO_PER_FILL * n_white_fill)={SCORE_WHITE_COMBO_PER_FILL * n_white_fill} + (SCORE_WHITE_COMBO_EXPLODED * n_white_exploded)={SCORE_WHITE_COMBO_EXPLODED * n_white_exploded}")

        # Blue spirits: regardless of flame, 0.5 each
        blue_value = SCORE_BLUE_EACH * n_blue_total
        if blue_value > 0:
            sv_total += blue_value
            sv_by_type["spirits_blue"] = sv_by_type.get("spirits_blue", 0.0) + blue_value
            notes.append(f"Blue spirits: +{blue_value:.2f} (count={n_blue_total})")

        # Blue combo: if ≥2 blue 'to explode' (i.e., filling), +1 for each beyond the first
        blue_combo = 0.0
        if n_blue_fill > 1:
            # Blue is ADDITIVE, so combo is not as strong as white
            blue_combo = SCORE_BLUE_COMBO_PER_EXTRA * (n_blue_fill - 1)
            sv_total += blue_combo
            sv_by_type["spirit_combo_blue"] = sv_by_type.get("spirit_combo_blue", 0.0) + blue_combo
            notes.append(f"Blue spirit combo: +{blue_combo:.2f} (filling={n_blue_fill})")

        # Extreme Spirit Burst (purple): regardless of flame, scored far above blue/white
        # since it grants "significantly bigger stat boosts" and a 0% failure rate (see
        # risk gating below). Detection is dormant until the classifier learns 'purple'.
        purple_value = SCORE_PURPLE_EACH * n_purple_total
        if purple_value > 0:
            sv_total += purple_value
            sv_by_type["spirits_purple"] = sv_by_type.get("spirits_purple", 0.0) + purple_value
            notes.append(f"Extreme Spirit Burst (purple): +{purple_value:.2f} (count={n_purple_total})")

        purple_combo = 0.0
        if n_purple_fill > 1:
            purple_combo = SCORE_PURPLE_COMBO_PER_EXTRA * (n_purple_fill - 1)
            sv_total += purple_combo
            sv_by_type["spirit_combo_purple"] = sv_by_type.get("spirit_combo_purple", 0.0) + purple_combo
            notes.append(f"Extreme Spirit Burst combo: +{purple_combo:.2f} (filling={n_purple_fill})")

        sv_by_type["meta_white_fill_units"] = float(n_white_fill)
        sv_by_type["meta_white_exploded_units"] = float(n_white_exploded)
        sv_by_type["meta_blue_total_units"] = float(n_blue_total)
        sv_by_type["meta_blue_fill_units"] = float(n_blue_fill)
        sv_by_type["meta_blue_has_spirit"] = 1.0 if n_blue_total > 0 else 0.0
        sv_by_type["meta_purple_total_units"] = float(n_purple_total)
        sv_by_type["meta_has_extreme_spirit_burst"] = 1.0 if n_purple_total > 0 else 0.0
        # Preserve the base SV before any seasonal multipliers in policy.
        sv_by_type["meta_sv_base_unity"] = float(sv_total)

        # ---- risk gating (higher than URA) --------------------------------------
        base_limit = Settings.MAX_FAILURE
        has_any_hint = bool(blue_hint_candidates or orange_hint_candidates)
        if sv_total >= 7:
            risk_mult = 2.0
        elif sv_total > 5.5 and not (has_any_hint and Settings.HINT_IS_IMPORTANT):
            risk_mult = 1.65
        elif sv_total > 5 and not (has_any_hint and Settings.HINT_IS_IMPORTANT):
            risk_mult = 1.5
        elif sv_total >= 4.5 and not (has_any_hint and Settings.HINT_IS_IMPORTANT):
            risk_mult = 1.35
        elif sv_total >= 3.5:
            risk_mult = 1.25
        elif sv_total >= 2.5:
            risk_mult = 1.1
        else:
            risk_mult = 1.0

        risk_limit = int(min(100, base_limit * risk_mult))
        allowed = failure_pct <= risk_limit
        notes.append(
            f"Dynamic risk (base SV before seasonal multipliers): SV={sv_total:.2f} -> base {base_limit}% x {risk_mult:.2f} = {risk_limit}%"
        )

        # Extreme Spirit Burst: 0% failure rate on the facility while active, regardless
        # of the displayed failure% or computed risk limit.
        if n_purple_total > 0:
            allowed = True
            notes.append("Extreme Spirit Burst active: 0% failure rate override, tile always allowed")

        greedy_hit = (sv_total >= GREEDY_THRESHOLD_UNITY_CUP) and allowed
        if greedy_hit:
            notes.append(
                f"Greedy hit: SV {sv_total:.2f} ≥ {GREEDY_THRESHOLD_UNITY_CUP} and failure {failure_pct}% ≤ {risk_limit}%"
            )

        out.append(
            TileSV(
                tile_idx=idx,
                failure_pct=failure_pct,
                risk_limit_pct=risk_limit,
                allowed_by_risk=bool(allowed),
                sv_total=float(sv_total),
                sv_by_type=sv_by_type,
                greedy_hit=greedy_hit,
                notes=notes,
            )
        )

    return [t.as_dict() for t in out]
