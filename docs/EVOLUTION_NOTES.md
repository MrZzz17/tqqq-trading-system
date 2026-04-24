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

## SHAs (how to list)

```bash
cd tqqq-trading-system && git log -20 --oneline
```

Paste the range you care about into a PR or paste here for a hand-edited list.
