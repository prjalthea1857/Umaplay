# SOP: Scraping Uma Musume “Event Viewer” HTML into Structured JSON (Multi-Support)

This document is a **self-contained handoff** so another engineer/LLM (without prior context) can maintain and improve the scraper that parses **support card** and **trainee** event blocks from the Gametora “Event Viewer” HTML into a normalized JSON format.

---

## 1) Goal & Scope

**Goal:** Convert the Event Viewer block (the UI that shows support cards side-by-side with “Chain/Random Events” tables) into a **machine-readable JSON** that our tooling can consume for planning and automation.

**In-scope:**
- Multiple support cards **and trainee profiles** in one HTML (e.g., *Matikanefukukitaru*, *Seeking the Pearl*, *Nice Nature*).
- Extract **events** (chain, random, special/after-race), **choice options** (Top/Mid/Bot or numeric labels), and **outcomes** (energy/stat changes, skill points, hints, bond, mood, statuses).
- Compute an event’s `default_preference` using a general heuristic that works “well enough” across cards **and trainees**.

**Out-of-scope (today):**
- Crawling the web: the script expects HTML already saved to disk or fetched by URL.
- Skill “Support hints” list (found on each card’s “Details page”)—that’s a different section/page and can be added later.

---

## 2) Input HTML & Relevant Structure

We parse the container **per participant** (support or trainee):

- Each support card lives in a block like:
  - `div.eventhelper_listgrid_item__*` (outer card container)
  - Inside: a centered header with image and **name**:
    - `div[style*="text-align: center"]`
    - Name appears in the **second** direct child `<div>` of that center block.
- Event sections:
  - `div.eventhelper_elist__*` → contains a section title
  - Section title element: `.sc-fc6527df-0` with text like:
    - `[ Chain Events ]` → **default_type = "chain"**
    - `[ Random Events ]` → **default_type = "random"**
    - `[ After a Race ]` → **default_type = "special"** (used by trainees)
- Within a section:
  - Wrapper for each event: `div.eventhelper_ewrapper__*`
  - Event heading: `.tooltips_ttable_heading__DK4_X`
    - Chain events sometimes include a **leading arrow indicator**:
      - e.g., `(❯❯) Guidance and Friends`
      - Number of arrows equals **chain step** (e.g., `❯❯` → `chain_step = 2`)
  - Event grid: `div.eventhelper_egrid__*` with two columns repeated:
    - Left label cell: `div.eventhelper_leftcell__Xzdy1` with text `Top / Mid / Bot`
    - Right cell (sibling `<div>`): contains a vertical list of `<div>` rows (effects)

Trainee blocks use the same inner wrappers but may skip `eventhelper_elist__*`. In that case the parser pairs each `.sc-fc6527df-0` heading with its immediate `div.eventhelper_listgrid__*` sibling.

**Special separators inside right cells:**
- `div.eventhelper_random_text__*` → “Randomly either”
- `div.eventhelper_divider_or__*` → “or”
- These split the right column into **multiple alternative outcomes** for the same option.

**Skill hint anchors inside right cells:**
- `a.utils_linkcolor__rvv3k` (or anchors to `/umamusume/skills/...`) → extract link text as hint name (e.g., “Right-Handed ○”, “Lucky Seven”).

---

## 3) Output JSON Schema

We produce an **array** of entries (support or trainee):

```json
[
  {
    "type": "support" | "trainee",
    "name": "Matikanefukukitaru",
    "rarity": "SR" | "None",
    "attribute": "WIT" | "None",
    "id": "Matikanefukukitaru_WIT_SR",          // supports only
    "choice_events": [
      {
        "type": "chain" | "random" | "special",
        "chain_step": 1 | 2 | 3,             // number of leading arrows for chain; 1 otherwise
        "name": "Guidance and Friends",
        "options": {
          "1": [ { ...outcome }, { ... } ],  // "Top"
          "2": [ ... ],                       // "Mid" if present
          "3": [ ... ]                        // "Bot"
        },
        "default_preference": 1 | 2 | 3       // computed heuristically (see §6)
      }
    ]
  }
]
```

**Outcome object fields (normalized):**

* `energy` (int; positive/negative)
* `skill_pts` (int)
* `bond` (int)
* `mood` (int; +1/-1; “Motivation up/down” mapped to ±1)
* Stats: `speed`, `stamina`, `power`, `guts`, `wit` (ints)
* `hints`: `["Right-Handed ○", "Lucky Seven", ...]`
* `status` (string) or `statuses` (array) when the line explicitly grants status effects
* Optional: `random_stats: {"count": X, "amount": N}` when more than one random stat is mentioned

**Expansion rules:**

* “All stats +N” → set all five stats to `+N`.
* “X random stats +N” → store `stats: N` for scoring. If `X > 1`, we also add `random_stats` for downstream consumers that need counts.
* Lines labelled `(random)` create additional optional outcomes; deterministic lines remain in the base outcome.

---

## 4) Script & CLI

**Filename:** `scrape_events.py`

**Dependencies:** `beautifulsoup4`, `lxml`, `requests` (optional; only for `--url`)

**One-liner run (as requested):**

```
cls && python scrape_events.py --html-file events_full_html.txt --support-defaults "Matikanefukukitaru-SR-SPD|Seeking the Pearl-SR-GUTS" --out supports_events.json --debug
```

**Flags:**

* `--html-file` *or* `--url` — input source.
* `--support-defaults "Name-RAR-ATTR|Name2-RAR-ATTR|..."`

  * Split from the **end** so names may contain hyphens.
  * Examples:

    * `Matikanefukukitaru-SR-SPD`
    * `Seeking the Pearl-SR-GUTS`
* `--out` — output JSON file path.
* `--debug` — verbose logs for troubleshooting.

**What the script does (high level):**

1. Find all `div.eventhelper_listgrid_item__*` blocks (each = **one support card**).
2. Extract **name** from the second direct child `<div>` of the centered header.
3. Resolve **rarity**/**attribute** from `--support-defaults`.
4. For each card, parse **events** section by section.
5. For each event, map `Top/Mid/Bot → "1"/"2"/"3"` (Top/Bot-only → "1"/"2").
   ✅ We **never** infer “Bot = 3” unless a **Mid** exists.
6. For trainees, also locate sections adjacent to headings when `eventhelper_elist__*` is absent.
7. Split right cell into **outcomes** at “Randomly either” and “or”, duplicating the base line whenever `(random)` branches appear.
8. Parse effects line-by-line (regex based), collect hints and statuses, build outcome dicts.
9. Compute `default_preference` by scoring options (see §6).
10. Emit the consolidated JSON (supports + trainees in the same array).

---

## 5) Parsing Details & Regexes

**Effects detection (case-insensitive):**

* `energy ±N` → `energy`
* `skill points +N` or `skill pts +N` → `skill_pts`
* `bond +N` → `bond`
* `mood +N` (or “Motivation up/down”) → `mood` (+1/−1)
* Stats:

  * `speed +N`, `stamina +N`, `power +N`, `guts +N`, `wit +N`
  * Also accepts “wis/wisdom/int/intelligence” for `wit`.
* *Special patterns*:

  * `All stats +N` / `All parameters +N` → expand all five stats to `+N`.
  * `X random stats +N` / `X random parameters +N` → store as `random_stats`.

**Chain step extraction:**

* Event title may begin with `(...)` where the **count of arrows** (`❯`, `»`, `>`) indicates the step:

  * `(❯)` → `chain_step = 1`
  * `(❯❯)` → `chain_step = 2`
  * `(❯❯❯)` → `chain_step = 3`
* Regex: `^\(\s*([»❯>]+)\s*\)\s*(.*)$` — we strip the arrows and keep the remainder as the **event name**.

**Skill hints inside lines:**

* Anchors with `.utils_linkcolor__rvv3k` (and anchors to `/umamusume/skills/...`) are captured as plain text and appended to `hints`.

---

## 6) Default Preference Heuristic

We score **each option** by averaging the scores of its outcomes:

```
score(outcome) =
  W_ENERGY   * energy
+ W_STAT     * (sum of stats + random_stats.count * random_stats.amount)
+ W_SKILLPTS * skill_pts
+ W_HINT     * (# of hints)
+ W_BOND     * bond
+ W_MOOD     * mood
```

**Current weights:**

* `W_ENERGY = 100.0` (dominant)
* `W_STAT = 10.0`
* `W_SKILLPTS = 2.0`
* `W_HINT = 1.0`
* `W_BOND = 0.3`
* `W_MOOD = 2.0`

**Stat priority:** when multiple stats increase together we weight them **Speed > Stamina > Power > Wit > Guts** (5→1). Random-stat bonuses use the average of those weights.

**Rationale:** prioritize **Energy** > **Stats** (with the preference above) > **Skill Pts** > **Hint**, with bond/mood as small tie-breakers.
If you want a “best-case” rather than “average” for multi-outcome options, swap the `average` with `max` in the selector (single line change).

---

## 7) Example: Expected Output (abridged)

### Matikanefukukitaru (SR/SPD)

> Note: earlier revisions of this doc listed a fabricated "Guidance and Friends"
> chain event with attribute WIT for this card — that example was wrong and, at
> one point, got copy-pasted directly into `datasets/in_game/events.json` instead
> of running the scraper against real data. Verified against the live GameTora
> page (`__NEXT_DATA__` JSON blob) as of 2026-07: she's SPD, and her three chain
> events are below. Always cross-check `--support-defaults` attribute/rarity
> against the live page before trusting a worked example.

* **Chain step 1**: *Mystery Fortune Ritual!*

  * `"1"`: two possible outcomes (Randomly either):

    * `{ "speed": 7, "stamina": 7, "power": 7, "guts": 7, "wit": 7, "bond": 7 }`
    * `{ "wit": 4 }`
  * `"2"`: `{ "energy": 5, "bond": 5 }`
* **Chain step 2**: *Fortune Favors the Friends!* — same shape as step 1.
* **Chain step 3**: *Finding Miraculous Joy!* — single option `"1"` with three
  Randomly-either outcomes (no Top/Bot split), each granting all stats +7 plus a
  `Super Lucky Seven`/`Lucky Seven` hint (skill ids 201561/201562) at varying
  levels, or just skill_pts + a `Lucky Seven (+3)` hint.
* **Random**: *Maximum Spirituality*

  * `"1"`: `{ "wit": 5, "skill_pts": 15, "bond": 5 }`
  * `"2"`: `{ "energy": -10, "speed": 5, "stamina": 5, "power": 5, "bond": 5 }`
* **Random**: *When Piety and Kindness Intersect*

  * `"1"`: `{ "skill_pts": 30, "bond": 5 }`
  * `"2"`: `{ "energy": 20, "bond": 5 }`

### Seeking the Pearl (SR/GUTS) — example patterns

* Chain (step 1) with **Top/Mid/Bot** and hint “Lucky Seven”
* Random event with **All stats +5** → all five stats set to `+5`
* Random event with **Uma Stan hint +3** recorded as `hints: ["Uma Stan"]`

### Nice Nature (trainee) — example patterns

* `[ Costume Events ]` and `[ After a Race ]` are parsed as `type: "random"` and `type: "special"` respectively.
* `(random) Get Charming ○ status` yields two outcomes: one without the status, one with `status: "Charming ○"`.
* Numeric left labels (“1.” / “2)”) are normalised to keys `"1"`, `"2"`, etc.

---

## 8) Quickstart Verification

1. Put the HTML into `events_full_html.txt`. The file must contain blocks like:

   * `div.eventhelper_listgrid_item__...` for each support.
   * Inside, `[ Chain Events ]` and/or `[ Random Events ]` sections.
2. Run:

```
cls && python scrape_events.py --html-file events_full_html.txt --support-defaults "Matikanefukukitaru-SR-SPD|Seeking the Pearl-SR-GUTS" --out supports_events.json --debug
```

3. Inspect logs for:

   * `[DEBUG] Found support items: N`
   * For each:

     * `[DEBUG] Support name: '…'`
     * Section headings, wrapper titles, left labels mapping (ensure `Top/Bot → 1/2`, `Top/Mid/Bot → 1/2/3`)
     * “right-cell lines: …” and effect parses
4. Validate `supports_events.json` structure and the chosen `default_preference`.

---

## 9) Troubleshooting & Edge Cases

* **Empty events:** If an event wrapper has no `eventhelper_egrid__*`, it’s skipped.
* **Bot incorrectly mapped to "3":** We only map `"3"` when **Mid** exists. Otherwise Bot → `"2"`.
* **No rarity/attribute in output:** Ensure the support name matches exactly (case-insensitive) in `--support-defaults`.
* **Hints not captured:** Confirm the anchor uses `.utils_linkcolor__rvv3k` or points to `/umamusume/skills/...`. If not, extend the selector.
* **Stats aliasing:** Some pages use “INT”/“Wisdom” for WIT—these are mapped via regex.
* **Random stats wording variants:** Extend `RE_RANDOM_STATS` if new text patterns appear (e.g., “random parameters” is already supported).
* **Trainee headings without wrappers:** If Gametora changes layout, ensure the fallback header→listgrid pairing still finds events.

---

## 10) Design Choices & Rationale

* **Average vs Max for multi-outcome options:** We use **average** to reduce bias towards rare spikes; change to **max** if you prefer greedy choice.
* **Random stats representation:** Kept as `random_stats` to avoid fabricating concrete stat names; still influences scoring via `count*amount`.
* **Strict DOM targeting:** Limited to stable class name patterns seen in the current HTML; selectors are resilient to hashed suffixes (e.g., `eventhelper_ewrapper__A_RGO`).

---

## 11) Backlog / Future Work

* **Support hints (Details page):** Build a companion scraper that collects the “Support hints” list for each card (e.g., “Triple 7s”, “A Small Breather”, etc.), including icon and description.
* **Full pipeline:** Add a small driver that fetches each card’s “Details page” and the Event Viewer snippet, then merges both into a single card object.
* **Internationalization:** Some pages may have other languages; add synonyms for effects (JP/EN).
* **Unit tests:** Create HTML fixtures for:

  * Top/Bot and Top/Mid/Bot variants
  * Randomly either with multiple separators
  * “All stats +N” and “X random stats +N”
  * Multiple hints on a single option
* **CLI option:** `--prefer-max-outcome` to switch scoring from average to maximum outcome.

---

## 12) Minimal Code Map (for quick edits)

* **Main file:** `scrape_events.py`
* **Key functions:**

  * `parse_support_defaults(raw)`: `"Name-RAR-ATTR|..."` → `{name: (rarity, attr)}`
  * `find_support_items(soup)`: locate each card block
  * `extract_support_name(item)`: name from centered header
  * `parse_events_in_card(item)`: full event parsing for a card
  * `parse_right_cell(right)`: split outcomes and parse lines
  * `parse_effects_from_text(txt)`: effect extraction (regex engine)
  * `choose_default_preference(options)`: scoring & selection
* **Regexes to maintain:** energy, bond, skill_pts, mood, stat names, “All stats”, “X random stats”, chain header arrows.

---

## 13) Sanity Tests (copy/paste)

* **Matikanefukukitaru** “Mystery Fortune Ritual!” (chain step 1, attribute SPD)

  * Expect chain_step = 1; options keys `"1"` and `"2"` (not `"3"`).
  * `"1"` should contain **two outcomes** split by “Randomly either”/“or”
    (`all stats +7 / bond +7` vs. `wit +4`).
  * `"2"` is a single outcome: `energy +5`, `bond +5`.
* **Seeking the Pearl** “Full-Power Thinking!”

  * `"1"`: `wit +20`
  * `"2"`: `energy -10`, hints include `"Uma Stan"`.

If the above holds, the parser is correctly aligned with the HTML.

---

## 14) License & Attribution

* This SOP documents a scraper for the **Event Viewer HTML** as rendered on Gametora. Class names include hashed suffixes but stable prefixes are used in selectors.

---

