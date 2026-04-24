# TQQQ App — Part A Engineering Verification (QA handoff)

Record **Confirmed** / **Rejected** / **Needs investigation** for each item from the handoff.  
Live reference: [tqqq-trade.streamlit.app](https://tqqq-trade.streamlit.app/).

## A.1 Content / data correctness

| ID | Item | Status | Notes |
|----|------|--------|-------|
| B1 | $8.6M vs hero equity | **Confirmed (fixed in code)** | “How the System Works” backtest line now uses `ending_value` from the V6 backtest, not a literal. |
| B2 | 2022 -18% vs -11.8% | **Confirmed (fixed)** | 2022 return pulled from `year_result_for_year(..., 2022)` / engine. |
| B3 | 71 vs 72 trades | **Confirmed** | Treat as **closed round-trips** (`sum(num_trades)`) **+** optional **open** leg. Dashboard header uses `trade_count_breakdown`. |
| B4 | Max DD label (Aug'19) | **Confirmed (fixed)** | `compute_equity_max_drawdown` on full equity; tile shows **peak month → trough month**. |
| B5 | Data source string mismatch | **Confirmed (fixed)** | Footer + sidebar: Yahoo (delayed) as primary; optional TradingView when API keys are set. |
| D5 | V3-FULL-ROW-BALANCE | **Rejected as user-visible** | **Removed** from visible title; `DASHBOARD_MARKET_HEALTH_ID` in HTML data-attr for QA only. |

## A.2 Layout / rendering

| ID | Item | Status | Notes |
|----|------|--------|-------|
| D1 | Live position wrap | **Confirmed (mitigated)** | No-mid-token grid + `white-space: nowrap` on value cells. |
| D2 | 633px vs wide | **Confirmed** | `layout="wide"`; narrow prose optional follow-up. |
| D3 | 2026 YTD % clip | **Confirmed (mitigated)** | Extra `min-width` / padding on hero third column. |
| D4 | Tall regime/MACD cards | **Confirmed (reduced)** | `min-height: 120px` (was ~400 in report). |
| D6 | Equity linear | **Confirmed (enhanced)** | **Log/Linear** toggle; short windows force linear. |
| D7 | Overlapping annotations | **Confirmed (mitigated)** | `label_mode` = markers-only on 3Y+; TQQQ log y on `All`. |
| D8 | "All" chart parity | **Confirmed (fixed)** | `DASHBOARD_YF_PERIOD = max` + same `_filter_*` helpers. |
| D10 | `loaded 11 :12` | **Confirmed (fixed)** | Manual 12h time; no `%I` strftime for that caption. |
| D11 | Sidebar in embed | **Not a code bug** | Streamlit `?embed` hides sidebar; document for users. |

## A.3 Strategy disclosure

| ID | Item | Status | Notes |
|----|------|--------|-------|
| L1 | Look-ahead | **Documented** | See disclaimers: same-day close for rules vs mark. |
| L2 | TQQQ history / divs | **Documented** | `auto_adjust=True` noted in disclaimer; TQQQ starts post-inception. |
| L3 | B&H definition | **Fixed in UI** | Tiles say **TQQQ buy&hold**; tables label TQQQ/QQQ B&H. |
| L4 | Engine version | **Fixed** | `config.ENGINE_VERSION` in footer and disclaimers. |

## A.4 Non-bugs (agree)

- Standard share URL / embed width: expected Streamlit behavior.
- Browser feature-policy warnings: platform.
- Fivetran POST: host analytics, not the app.
- 1 trade / 100% WR years: small-sample.
- Disclaimers: keep.

---

*Engineering: complete the **Status/Notes** column after each deploy and keep regression tests in `tests/` green.*
