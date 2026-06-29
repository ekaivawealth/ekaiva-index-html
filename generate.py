"""
generate.py
===========
Fetches NSE index data and writes a fully self-contained index.html dashboard.

Data pipeline (no server required at runtime):
  1. Load seed CSVs from the public ekaiva-tracker repo (raw GitHub URLs).
     Seeds provide long history needed for SMA-200 / EMA-200.
  2. Fetch recent closes (last 30 days) from Yahoo Finance for supported tickers.
  3. If yfinance doesn't have today's close, fall back to NiftyIndices daily snapshot.
  4. Merge seed + live → compute weekly SMA & EMA (periods 5/10/20/50/100/200).
  5. Embed all data as JSON inside index.html — no browser-side API calls needed.

Run locally:  python generate.py
GitHub Actions runs this automatically every weekday at 20:00 IST.
"""

import io
import json
import os
import re
import sys
import time
import datetime as dt

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Seeds live in the existing ekaiva-tracker public repo
SEED_REPO = "https://raw.githubusercontent.com/ekaivawealth/ekaiva-tracker/main/data/seeds"

SMA_PERIODS = [5, 10, 20, 50, 100, 200]

# (display name, yfinance ticker, NiftyIndices snapshot label)
INDICES = [
    # Broad
    ("Nifty 50",                          "^NSEI",                       "Nifty 50"),
    ("Nifty Next 50",                     "^NSMIDCP",                    "Nifty Next 50"),
    ("Nifty 500",                         "^CRSLDX",                     "Nifty 500"),
    ("Nifty Midcap 150",                  "NIFTYMIDCAP150.NS",           "Nifty Midcap 150"),
    ("Nifty Smallcap 250",                "NIFTYSMLCAP250.NS",           "Nifty Smallcap 250"),
    ("Nifty Microcap 250",                "NIFTY_MICROCAP250.NS",        "Nifty Microcap 250"),
    ("Nifty LargeMidcap 250",             "NIFTY_LARGEMID250.NS",        "Nifty LargeMidcap 250"),
    ("Nifty Alpha 50",                    "^NIFTYALPHA50",               "Nifty Alpha 50"),
    # Sectoral
    ("Nifty Pharma",                      "^CNXPHARMA",                  "Nifty Pharma"),
    ("Nifty Healthcare",                  "NIFTY_HEALTHCARE.NS",         "Nifty Healthcare Index"),
    ("Nifty MidSmall Healthcare",         "NIFTYMIDSMLHLTHCRE.NS",       "Nifty MidSmall Healthcare"),
    ("Nifty IT",                          "^CNXIT",                      "Nifty IT"),
    ("Nifty Auto",                        "^CNXAUTO",                    "Nifty Auto"),
    ("Nifty Metal",                       "^CNXMETAL",                   "Nifty Metal"),
    ("Nifty FMCG",                        "^CNXFMCG",                    "Nifty FMCG"),
    ("Nifty Media",                       "^CNXMEDIA",                   "Nifty Media"),
    ("Nifty Energy",                      "^CNXENERGY",                  "Nifty Energy"),
    ("Nifty Oil & Gas",                   "NIFTY_OIL_AND_GAS.NS",        "Nifty Oil & Gas"),
    ("Nifty Chemicals",                   "^CNXCHEM",                    "Nifty Chemicals"),
    ("Nifty Private Bank",                "NIFTYPVTBANK.NS",             "Nifty Private Bank"),
    ("Nifty PSU Bank",                    "^CNXPSUBNK",                  "Nifty PSU Bank"),
    ("Nifty Financial Services",          "NIFTY_FIN_SERVICE.NS",        "Nifty Financial Services"),
    ("Nifty MidSmall Financial Services", "^NIFTY_MIDSMALL_FINSRV",      "Nifty MidSmall Financial Services"),
    ("Nifty Capital Market",              "^NIFTY_CAP_MARKET",           "Nifty Capital Markets"),
    ("Nifty Consumer Durables",           "NIFTY_CONSR_DURBL.NS",        "Nifty Consumer Durables"),
    ("Nifty Realty",                      "^CNXREALTY",                  "Nifty Realty"),
    # Thematic
    ("Nifty CPSE",                        "^CNXCPSE",                    "Nifty CPSE"),
    ("Nifty India Tourism",               "NIFTY_IND_TOURISM.NS",        "Nifty India Tourism"),
    ("Nifty Commodities",                 "^CNXCMDT",                    "Nifty Commodities"),
    ("Nifty India Consumption",           "^CNXCONSUM",                  "Nifty India Consumption"),
    ("Nifty Rural",                       "^NIFTYRURAL",                 "Nifty Rural"),
    ("Nifty Housing",                     "^NIFTYHOUSING",               "Nifty Housing"),
    ("Nifty Infrastructure",              "^CNXINFRA",                   "Nifty Infrastructure"),
    ("Nifty Defence",                     "^NIFTYDEFENCE",               "Nifty India Defence"),
    ("Nifty India Manufacturing",         "NIFTY_INDIA_MFG.NS",          "Nifty India Manufacturing"),
    ("Nifty MNC",                         "^CNXMNC",                     "Nifty MNC"),
    # Volatility
    ("India VIX",                         "^INDIAVIX",                   "India VIX"),
]

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)

# ---------------------------------------------------------------------------
# Seed loading
# ---------------------------------------------------------------------------

def _safe_name(name: str) -> str:
    """Match the naming convention used in ekaiva-tracker/data_sources.py."""
    return name.replace(" ", "_").replace("&", "and").replace("/", "_")


def load_seed(name: str) -> pd.Series:
    url = f"{SEED_REPO}/{_safe_name(name)}.csv"
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": UA})
        if r.status_code != 200:
            return pd.Series(dtype="float64")
        df = pd.read_csv(io.BytesIO(r.content), index_col=0, parse_dates=True)
        s = df.iloc[:, 0].dropna()
        idx = pd.to_datetime(s.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(None)
        idx = idx.normalize()
        s = pd.Series(s.values, index=idx, dtype=float)
        return s[~s.index.duplicated(keep="last")].sort_index()
    except Exception as exc:
        print(f"  [seed] {name}: {exc}", file=sys.stderr)
        return pd.Series(dtype="float64")


# ---------------------------------------------------------------------------
# Yahoo Finance live prices
# ---------------------------------------------------------------------------

def fetch_yf(ticker: str) -> pd.Series:
    try:
        raw = yf.download(ticker, period="30d", interval="1d",
                          progress=False, auto_adjust=False)
        if raw is None or raw.empty:
            return pd.Series(dtype="float64")
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        s = close.dropna()
        idx = pd.to_datetime(s.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(None)
        idx = idx.normalize()
        s = pd.Series(s.values, index=idx, dtype=float)
        return s[~s.index.duplicated(keep="last")].sort_index()
    except Exception as exc:
        print(f"  [yf] {ticker}: {exc}", file=sys.stderr)
        return pd.Series(dtype="float64")


# ---------------------------------------------------------------------------
# NiftyIndices daily snapshot (fallback for latest close)
# ---------------------------------------------------------------------------

_nifty_session = None
_snapshot_cache: dict = {}


def _nifty_session_get():
    global _nifty_session
    if _nifty_session is None:
        _nifty_session = requests.Session()
        _nifty_session.headers.update({"User-Agent": UA,
                                        "Referer": "https://www.niftyindices.com/"})
        try:
            _nifty_session.get("https://www.niftyindices.com/", timeout=15)
        except Exception:
            pass
    return _nifty_session


def _load_snapshot(date: dt.date) -> dict:
    key = date.strftime("%d%m%Y")
    if key in _snapshot_cache:
        return _snapshot_cache[key]
    try:
        url = f"https://niftyindices.com/Daily_Snapshot/ind_close_all_{key}.csv"
        r = _nifty_session_get().get(url, timeout=15)
        if r.status_code != 200:
            _snapshot_cache[key] = {}
            return {}
        df = pd.read_csv(io.BytesIO(r.content))
        df.columns = [c.strip() for c in df.columns]
        name_col = next((c for c in df.columns if "Index Name" in c), None)
        close_col = next((c for c in df.columns if "Closing" in c), None)
        if name_col is None or close_col is None:
            _snapshot_cache[key] = {}
            return {}
        snap = {}
        for _, row in df.iterrows():
            raw_name = str(row[name_col])
            k = re.sub(r"[^a-z0-9 ]", " ",
                       raw_name.lower().replace("&", "and"))
            k = re.sub(r"\s+", " ", k).strip()
            try:
                snap[k] = float(row[close_col])
            except (ValueError, TypeError):
                pass
        _snapshot_cache[key] = snap
        return snap
    except Exception as exc:
        print(f"  [nifty snapshot] {date}: {exc}", file=sys.stderr)
        _snapshot_cache[key] = {}
        return {}


def nifty_latest(nse_label: str):
    """Return (date_str, close) from NiftyIndices snapshot, looking back up to 5 days."""
    def normalise(s: str) -> str:
        s = re.sub(r"\bindex\b", "", s.lower().replace("&", "and"))
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()

    nl = normalise(nse_label)
    today = dt.date.today()
    for back in range(5):
        d = today - dt.timedelta(days=back)
        if d.weekday() >= 5:
            continue
        snap = _load_snapshot(d)
        if nl in snap:
            return d.isoformat(), snap[nl]
        if nl.endswith("s") and nl[:-1] in snap:
            return d.isoformat(), snap[nl[:-1]]
    return None


# ---------------------------------------------------------------------------
# Weekly MA computation
# ---------------------------------------------------------------------------

def _to_weekly(daily: pd.Series) -> pd.Series:
    if daily.empty:
        return pd.Series(dtype="float64")
    return daily.resample("W-FRI").last().dropna()


def compute_score(weekly: pd.Series, periods: list, ma_type: str):
    if len(weekly) < 5:
        return None
    close = float(weekly.iloc[-1])
    crossed = available = 0
    mas = {}
    for p in periods:
        if ma_type == "sma":
            val = weekly.rolling(p, min_periods=p).mean().iloc[-1]
        else:
            mask = pd.Series(range(1, len(weekly) + 1),
                             index=weekly.index) >= p
            val = weekly.ewm(span=p, adjust=False).mean().where(mask).iloc[-1]
        if pd.notna(val):
            v = float(val)
            mas[str(p)] = round(v, 2)
            available += 1
            if close > v:
                crossed += 1
        else:
            mas[str(p)] = None
    return {
        "close": round(close, 2),
        "crossed": crossed,
        "available": available,
        "mas": mas,
    }


def build_history(daily: pd.Series, periods: list, ma_type: str, n_weeks: int = 52) -> list:
    weekly = _to_weekly(daily)
    if weekly.empty:
        return []
    rows = []
    start = max(0, len(weekly) - n_weeks)
    for i in range(start, len(weekly)):
        sub = weekly.iloc[: i + 1]
        s = compute_score(sub, periods, ma_type)
        if s:
            rows.append({
                "w": sub.index[-1].strftime("%Y-%m-%d"),
                "c": s["close"],
                "x": s["crossed"],
            })
    return rows


# ---------------------------------------------------------------------------
# Main data collection loop
# ---------------------------------------------------------------------------

def collect() -> list:
    print("Fetching live prices from Yahoo Finance...", flush=True)
    live_map = {}
    for name, ticker, _ in INDICES:
        time.sleep(0.3)
        s = fetch_yf(ticker)
        live_map[name] = s
        if not s.empty:
            print(f"  OK  {name}: {len(s)} rows, latest {s.index.max().date()}", flush=True)
        else:
            print(f"  --  {name}: empty", flush=True)

    print("\nLoading seeds + computing scores...", flush=True)
    results = []
    today = dt.date.today()

    for name, _, nse_label in INDICES:
        print(f"  {name}... ", end="", flush=True)

        seed = load_seed(name)
        live = live_map.get(name, pd.Series(dtype="float64"))

        if not seed.empty and not live.empty:
            daily = pd.concat([seed, live])
            daily = daily[~daily.index.duplicated(keep="last")].sort_index()
        elif len(seed) >= 50:
            daily = seed
        elif not live.empty:
            daily = live
        else:
            daily = pd.Series(dtype="float64")

        latest_date = None
        latest_close = None

        if not live.empty:
            latest_date = live.index.max().date().isoformat()
            latest_close = round(float(live.iloc[-1]), 2)

        if (latest_date != today.isoformat()) and (name != "India VIX"):
            ni = nifty_latest(nse_label)
            if ni:
                nd, nc = ni
                ts = pd.Timestamp(nd)
                if ts not in daily.index:
                    entry = pd.Series([nc], index=[ts], dtype=float)
                    daily = pd.concat([daily, entry]).sort_index()
                    daily = daily[~daily.index.duplicated(keep="last")]
                latest_date = nd
                latest_close = round(nc, 2)

        weekly = _to_weekly(daily)
        sma_score = compute_score(weekly, SMA_PERIODS, "sma")
        ema_score = compute_score(weekly, SMA_PERIODS, "ema")
        sma_hist = build_history(daily, SMA_PERIODS, "sma", 52)
        ema_hist = build_history(daily, SMA_PERIODS, "ema", 52)

        results.append({
            "name": name,
            "close": latest_close,
            "date": latest_date,
            "sma": sma_score,
            "ema": ema_score,
            "sma_hist": sma_hist,
            "ema_hist": ema_hist,
        })
        tag = f"{sma_score['crossed']}/6" if sma_score else "insuf."
        print(f"SMA {tag}, {len(weekly)}w", flush=True)

    return results


# ---------------------------------------------------------------------------
# HTML template
# NOTE: plain string (not f-string). Only __JSON_DATA__ is replaced.
# IMPORTANT: the _data script tag is placed BEFORE the main script so
# document.getElementById('_data') works when the JS runs.
# ---------------------------------------------------------------------------

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ekaiva &middot; NSE Index SMA/EMA Tracker</title>
<style>
:root {
  --bg:   #0d1117; --card: #161b22; --border: #30363d;
  --text: #e6edf3; --muted: #8b949e; --accent: #1f6feb;
  --c6: #1a7f37; --c6t: #3fb950;
  --c5: #2a6e18; --c5t: #85c450;
  --c4: #5a4000; --c4t: #d29922;
  --c3: #6b2800; --c3t: #f0883e;
  --c2: #6b1414; --c2t: #f85149;
  --c1: #4a0808; --c1t: #da3633;
  --c0: #1e1e1e; --c0t: #8b949e;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }

.site-header { border-bottom: 1px solid var(--border); padding: 20px 24px 14px; display: flex; flex-direction: column; align-items: center; gap: 4px; }
.site-header h1 { font-size: 1.25rem; font-weight: 800; letter-spacing: .06em; }
.site-header .tagline { color: var(--muted); font-size: .75rem; }
.meta-bar { color: var(--muted); font-size: .72rem; margin-top: 2px; }

.mode-toggle { display: flex; justify-content: center; gap: 10px; padding: 16px 0 8px; }
.mode-toggle button { padding: 7px 28px; border-radius: 6px; border: 1px solid var(--border); background: var(--card); color: var(--muted); font-weight: 700; font-size: .8rem; cursor: pointer; transition: all .18s; letter-spacing: .04em; }
.mode-toggle button.active { background: var(--accent); border-color: var(--accent); color: #fff; }

.container { max-width: 1080px; margin: 0 auto; padding: 0 16px 60px; }
.section-title { font-size: .7rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); padding: 16px 0 8px; }

.green-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; margin-bottom: 4px; }
.tile { background: var(--c6); border: 1px solid #2ea04380; border-radius: 10px; padding: 14px 16px; cursor: pointer; transition: filter .15s, transform .1s; }
.tile:hover { filter: brightness(1.15); transform: translateY(-1px); }
.tile-name { font-size: .82rem; font-weight: 700; line-height: 1.3; margin-bottom: 8px; color: var(--c6t); }
.tile-close { font-size: .75rem; color: rgba(255,255,255,.55); margin-bottom: 10px; font-variant-numeric: tabular-nums; }
.tile-pips { display: flex; gap: 4px; margin-bottom: 6px; }
.pip { width: 15px; height: 15px; border-radius: 3px; background: rgba(255,255,255,.15); }
.pip.on { background: var(--c6t); }
.tile-score { font-size: 1.2rem; font-weight: 800; color: var(--c6t); }

.index-list { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
.idx-row { display: flex; align-items: center; padding: 10px 16px; gap: 10px; border-top: 1px solid var(--border); cursor: pointer; transition: background .12s; }
.idx-row:first-child { border-top: none; }
.idx-row:hover { background: rgba(255,255,255,.04); }
.score-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.idx-name { flex: 1; font-size: .83rem; font-weight: 500; }
.idx-close { width: 86px; text-align: right; font-size: .8rem; color: var(--muted); font-variant-numeric: tabular-nums; }
.row-pips { display: flex; gap: 3px; flex-shrink: 0; }
.row-pip { width: 12px; height: 12px; border-radius: 2px; background: var(--border); }
.idx-score { width: 34px; text-align: right; font-size: .78rem; font-weight: 700; flex-shrink: 0; }
.ins-tag { font-size: .7rem; color: var(--muted); font-style: italic; }

#overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.72); z-index: 200; align-items: center; justify-content: center; }
#overlay.open { display: flex; }
#modal { background: var(--card); border: 1px solid var(--border); border-radius: 12px; width: min(620px, 96vw); max-height: 88vh; display: flex; flex-direction: column; }
#modal-hdr { padding: 16px 20px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; flex-shrink: 0; }
#modal-hdr h2 { font-size: .95rem; font-weight: 700; }
#modal-close { background: none; border: none; color: var(--muted); font-size: 1.2rem; cursor: pointer; }
#modal-body { overflow-y: auto; padding: 16px 20px; }
.modal-summary { display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 16px; }
.ms-item { display: flex; flex-direction: column; gap: 2px; min-width: 80px; }
.ms-label { font-size: .65rem; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
.ms-value { font-size: .95rem; font-weight: 700; }
.ms-ma { font-size: .82rem; font-weight: 600; }
.arrow-up { color: #3fb950; } .arrow-dn { color: #f85149; }
.hist-table { width: 100%; border-collapse: collapse; font-size: .76rem; }
.hist-table th { text-align: left; padding: 5px 8px; color: var(--muted); border-bottom: 1px solid var(--border); font-weight: 600; }
.hist-table td { padding: 4px 8px; border-bottom: 1px solid rgba(48,54,61,.4); font-variant-numeric: tabular-nums; }
.hist-table tr:hover td { background: rgba(255,255,255,.03); }

footer { text-align: center; padding: 20px 16px; color: var(--muted); font-size: .72rem; border-top: 1px solid var(--border); line-height: 1.8; }

@media (max-width: 520px) {
  .green-grid { grid-template-columns: 1fr 1fr; }
  .idx-close { display: none; }
}
</style>
</head>
<body>

<header class="site-header">
  <h1>EKAIVA &middot; NSE Index Tracker</h1>
  <div class="tagline">Weekly SMA &amp; EMA &middot; strict 6/6 scoring &middot; 37 indices</div>
  <div class="meta-bar" id="meta-bar"></div>
</header>

<div class="mode-toggle">
  <button id="btn-sma" class="active" onclick="setMode('sma')">SMA Model</button>
  <button id="btn-ema" onclick="setMode('ema')">EMA Model</button>
</div>

<div class="container" id="root"></div>

<div id="overlay" onclick="maybeClose(event)">
  <div id="modal">
    <div id="modal-hdr">
      <h2 id="modal-title"></h2>
      <button id="modal-close" onclick="closeModal()">&#x2715;</button>
    </div>
    <div id="modal-body">
      <div class="modal-summary" id="modal-summary"></div>
      <table class="hist-table">
        <thead><tr><th>Week</th><th>Close</th><th>Score</th><th>Signal</th></tr></thead>
        <tbody id="modal-tbody"></tbody>
      </table>
    </div>
  </div>
</div>

<footer>
  Ekaiva Wealth &nbsp;&bull;&nbsp; ekaivawealth.com &nbsp;&bull;&nbsp; +91 93766 98983 &nbsp;&bull;&nbsp; ARN 305896<br>
  Not investment advice. Score = weekly closes above each MA (periods: 5 / 10 / 20 / 50 / 100 / 200 weeks).
</footer>

<!-- DATA MUST come before the main script so getElementById works -->
<script id="_data" type="application/json">__JSON_DATA__</script>

<script>
const DATA = JSON.parse(document.getElementById('_data').textContent);
let MODE = 'sma';

const BG  = ['--c0','--c1','--c2','--c3','--c4','--c5','--c6'];
const TXT = ['--c0t','--c1t','--c2t','--c3t','--c4t','--c5t','--c6t'];

function css(v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }
function bgColor(x)  { return css(BG[Math.min(x,6)]); }
function txtColor(x) { return css(TXT[Math.min(x,6)]); }

function fmtNum(n) {
  if (n == null) return '—';
  return n.toLocaleString('en-IN', {minimumFractionDigits:2, maximumFractionDigits:2});
}

function signal(x) {
  if (x===6) return '&#x1F7E2; Fully above';
  if (x>=4)  return '&#x1F7E1; Mostly above';
  if (x>=2)  return '&#x1F7E0; Mixed';
  return '&#x1F534; Below most';
}

function setMode(m) {
  MODE = m;
  document.getElementById('btn-sma').className = m==='sma' ? 'active' : '';
  document.getElementById('btn-ema').className = m==='ema' ? 'active' : '';
  render();
}

function render() {
  document.getElementById('meta-bar').textContent =
    'Market data as of ' + DATA.as_of + '   ·   Generated ' + DATA.generated;

  const indices = DATA.indices;
  const green = indices.filter(d => d[MODE] && d[MODE].crossed===6 && d[MODE].available===6);
  const rest  = indices
    .filter(d => !(d[MODE] && d[MODE].crossed===6 && d[MODE].available===6))
    .sort((a,b) => {
      const xa = a[MODE] ? a[MODE].crossed : -1;
      const xb = b[MODE] ? b[MODE].crossed : -1;
      return xb - xa;
    });

  let html = '';

  if (green.length) {
    html += '<div class="section-title">&#x1F7E2; Green Alert — ' + MODE.toUpperCase() + ' 6/6 (' + green.length + ')</div>';
    html += '<div class="green-grid">';
    green.forEach(d => { html += tileHTML(d); });
    html += '</div>';
  } else {
    html += '<div class="section-title" style="color:#f85149">No index is 6/6 in ' + MODE.toUpperCase() + ' model right now</div>';
  }

  html += '<div class="section-title">Everything else</div>';
  html += '<div class="index-list">';
  rest.forEach(d => { html += rowHTML(d); });
  html += '</div>';

  document.getElementById('root').innerHTML = html;
}

function tileHTML(d) {
  const s = d[MODE];
  const x = s ? s.crossed : 0;
  const tc = txtColor(x);
  let pips = '';
  for (let i=0; i<6; i++) {
    pips += '<div class="pip' + (i<x?' on':'') + '" style="' + (i<x?'background:'+tc:'') + '"></div>';
  }
  return '<div class="tile" onclick="openModal(\'' + esc(d.name) + '\')">' +
    '<div class="tile-name">' + d.name + '</div>' +
    '<div class="tile-close">' + fmtNum(d.close) + '</div>' +
    '<div class="tile-pips">' + pips + '</div>' +
    '<div class="tile-score">' + x + '/6</div>' +
    '</div>';
}

function rowHTML(d) {
  const s = d[MODE];
  const x = s ? s.crossed : -1;
  const avail = s ? s.available : 0;
  const tc = x>=0 ? txtColor(x) : css('--muted');
  const dot = '<span class="score-dot" style="background:' + (x>=0 ? bgColor(x) : css('--c0')) + '"></span>';

  let pips = '<div class="row-pips">';
  for (let i=0; i<6; i++) {
    pips += '<span class="row-pip' + (i<x?' on':'') + '" style="' + (i<x?'color:'+tc:'') + '"></span>';
  }
  pips += '</div>';

  const scoreStr = avail>0
    ? '<span class="idx-score" style="color:'+tc+'">'+x+'/6</span>'
    : '<span class="ins-tag">insuf.</span>';

  return '<div class="idx-row" onclick="openModal(\'' + esc(d.name) + '\')">' +
    dot + '<span class="idx-name">' + d.name + '</span>' +
    '<span class="idx-close">' + fmtNum(d.close) + '</span>' +
    pips + scoreStr + '</div>';
}

function esc(s) { return s.replace(/\\/g,'\\\\').replace(/'/g,"\\'"); }

function openModal(name) {
  const d = DATA.indices.find(i => i.name===name);
  if (!d) return;
  const s = d[MODE];
  const hist = d[MODE+'_hist'] || [];
  const periods = [5,10,20,50,100,200];

  document.getElementById('modal-title').textContent = name;

  let sum = '';
  if (s) {
    const x = s.crossed;
    sum += '<div class="ms-item"><div class="ms-label">Close</div><div class="ms-value">' + fmtNum(d.close) + '</div></div>';
    sum += '<div class="ms-item"><div class="ms-label">Score</div><div class="ms-value" style="color:'+txtColor(x)+'">' + x + '/6</div></div>';
    periods.forEach(p => {
      const v = s.mas[String(p)];
      if (v!=null) {
        const above = d.close > v;
        sum += '<div class="ms-item"><div class="ms-label">' + MODE.toUpperCase()+p + '</div>' +
          '<div class="ms-ma">' + fmtNum(v) + ' <span class="'+(above?'arrow-up':'arrow-dn')+'">'+(above?'&#9650;':'&#9660;')+'</span></div></div>';
      }
    });
  } else {
    sum = '<span style="color:var(--muted);font-size:.8rem">Insufficient weekly history to compute scores.</span>';
  }
  document.getElementById('modal-summary').innerHTML = sum;

  let tbody = '';
  [...hist].reverse().forEach(row => {
    tbody += '<tr><td>' + row.w + '</td><td>' + fmtNum(row.c) + '</td>' +
      '<td style="color:'+txtColor(row.x)+';font-weight:700">' + row.x + '/6</td>' +
      '<td>' + signal(row.x) + '</td></tr>';
  });
  document.getElementById('modal-tbody').innerHTML = tbody ||
    '<tr><td colspan="4" style="color:var(--muted);text-align:center;padding:12px">No history available</td></tr>';

  document.getElementById('overlay').classList.add('open');
}

function closeModal() { document.getElementById('overlay').classList.remove('open'); }
function maybeClose(e) { if (e.target===document.getElementById('overlay')) closeModal(); }
document.addEventListener('keydown', e => { if (e.key==='Escape') closeModal(); });

render();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Build and write HTML
# ---------------------------------------------------------------------------

def build_html(results: list) -> str:
    today = dt.date.today()
    now = dt.datetime.now()

    dates = [r["date"] for r in results if r["date"]]
    as_of = max(dates) if dates else today.isoformat()

    payload = {
        "as_of": as_of,
        "generated": now.strftime("%d %b %Y, %H:%M"),
        "indices": [
            {
                "name":     r["name"],
                "close":    r["close"],
                "date":     r["date"],
                "sma":      r["sma"],
                "ema":      r["ema"],
                "sma_hist": r["sma_hist"],
                "ema_hist": r["ema_hist"],
            }
            for r in results
        ],
    }

    json_str = json.dumps(payload, separators=(",", ":"))
    json_str = json_str.replace("</", "<\\/")
    return HTML.replace("__JSON_DATA__", json_str)


if __name__ == "__main__":
    results = collect()
    html = build_html(results)
    out = "index.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nWritten -> {out}  ({len(html):,} bytes)")
