# Product evolution — facts aligned with `main` (post–QA handoff)

Use this file so **narrative summaries and internal notes** stay consistent with the code. For line-by-line QA sign-off, see **QA_VERIFICATION.md**.

## Market Health “rev” / ID (not user-visible)

- **Config:** `config.DASHBOARD_MARKET_HEALTH_ID` (not `…_REV`). Value: **`"regime-macd-panel"`**.
- **UI:** The explainer block uses **`data-market-health-id="…"`** for QA / automation only. The string is **not** shown in the visible title line.
- **Do not** reintroduce user-visible internal tags such as `V3-FULL-ROW-BALANCE`. CI enforces this:
  - `tests/test_b5_banned_prose.py` — banned substrings in `views/tqqq_dashboard.py` (includes `V3-FULL-ROW-BALANCE` and a regex that rejects `V#-LIKE-THIS-TOKENS` in `views/*.py`).

## Market Health three-tile row

- **Tile height** is controlled by **`_mh_tile_h = "min-height: 120px"`** in `views/tqqq_dashboard.py` (QA row D4 / tall-card mitigation). If a future change intentionally uses 220px again, update **this file** and **QA_VERIFICATION.md** so notes stay in sync.

## Changelog-style release note (condensed)

The TQQQ Streamlit app’s model chart was aligned to the V6 backtest: QQQ + TQQQ context with entry/exit markers, real volume + average line, rangebreaks and no fake rangeslider strip, and Yahoo last-bar date handling consistent with the engine. Historical now reflects open lots with engine metadata; Market Health is a full-width three-tile row with regime/MACD explainer; docs and in-app copy match the real layout. A follow-up QA pass removed stale performance prose, reconciled trade counts and max drawdowns with the equity series, added wide layout, log/linear equity scaling, a unified data/engine footer, regression tests, and removed internal debug strings from the UI.

## Unified system state (single source of truth)

- **`core/system_state.py`** — `get_system_state()` derives every dashboard label from the V6 engine's own
  data pipeline and rule functions (`_fetch` / `_indicators` / `_make_weekly` / `_find_ftd_signal` + `LiveSnapshot`).
- The old parallel classifiers are gone from the UI: IBD pulse verdicts (top pills) → **Engine Regime** tile
  (Strong Bull / Cautious Bull / Bear = QQQ vs 200-day + weekly MACD sign) + **Selling Pressure** context chip
  (distribution days, explicitly non-driving). The golden-cross "Strong Bull" definition and the fabricated
  "2 closes below 50-day / 21-EMA" exit copy were removed.
- Docs/tiles now state the engine truth: the 200-day exit fires on the **first** close below (the old
  "2 consecutive closes" copy contradicted `_run_continuous`).
- `LiveSnapshot` gained reporting-only fields (`cooldown_until`, `ftd_cooldown_until`, `peak_portfolio`,
  `exited`); `get_dashboard_state` cache bust bumped to 4.

## SHAs (how to list)

```bash
cd tqqq-trading-system && git log -20 --oneline
```

Paste the range you care about into a PR or paste here for a hand-edited list.
