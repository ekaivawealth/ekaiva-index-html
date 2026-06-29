"""
generate.py
===========
Fetches NSE index data and writes a fully self-contained index.html dashboard.

Data pipeline (no server required at runtime):
  1. Load seed CSVs from the public ekaiva-tracker repo (raw GitHub URLs).
  2. Fetch recent closes (last 30 days) from Yahoo Finance.
  3. If yfinance misses today, fall back to NiftyIndices daily snapshot.
  4. Merge seed + live -> compute weekly SMA & EMA (5/10/20/50/100/200 weeks).
  5. Embed all data as JSON inside index.html — no browser-side API calls.

Run locally:  python generate.py
Auto-runs weekdays 20:00 IST via GitHub Actions.
"""

import io, json, re, sys, time, datetime as dt
import pandas as pd
import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED_REPO   = "https://raw.githubusercontent.com/ekaivawealth/ekaiva-tracker/main/data/seeds"
SMA_PERIODS = [5, 10, 20, 50, 100, 200]
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

INDICES = [
    ("Nifty 50",                          "^NSEI",                  "Nifty 50"),
    ("Nifty Next 50",                     "^NSMIDCP",               "Nifty Next 50"),
    ("Nifty 500",                         "^CRSLDX",                "Nifty 500"),
    ("Nifty Midcap 150",                  "NIFTYMIDCAP150.NS",      "Nifty Midcap 150"),
    ("Nifty Smallcap 250",                "NIFTYSMLCAP250.NS",      "Nifty Smallcap 250"),
    ("Nifty Microcap 250",                "NIFTY_MICROCAP250.NS",   "Nifty Microcap 250"),
    ("Nifty LargeMidcap 250",             "NIFTY_LARGEMID250.NS",   "Nifty LargeMidcap 250"),
    ("Nifty Alpha 50",                    "^NIFTYALPHA50",          "Nifty Alpha 50"),
    ("Nifty Pharma",                      "^CNXPHARMA",             "Nifty Pharma"),
    ("Nifty Healthcare",                  "NIFTY_HEALTHCARE.NS",    "Nifty Healthcare Index"),
    ("Nifty MidSmall Healthcare",         "NIFTYMIDSMLHLTHCRE.NS",  "Nifty MidSmall Healthcare"),
    ("Nifty IT",                          "^CNXIT",                 "Nifty IT"),
    ("Nifty Auto",                        "^CNXAUTO",               "Nifty Auto"),
    ("Nifty Metal",                       "^CNXMETAL",              "Nifty Metal"),
    ("Nifty FMCG",                        "^CNXFMCG",               "Nifty FMCG"),
    ("Nifty Media",                       "^CNXMEDIA",              "Nifty Media"),
    ("Nifty Energy",                      "^CNXENERGY",             "Nifty Energy"),
    ("Nifty Oil & Gas",                   "NIFTY_OIL_AND_GAS.NS",   "Nifty Oil & Gas"),
    ("Nifty Chemicals",                   "^CNXCHEM",               "Nifty Chemicals"),
    ("Nifty Private Bank",                "NIFTYPVTBANK.NS",        "Nifty Private Bank"),
    ("Nifty PSU Bank",                    "^CNXPSUBNK",             "Nifty PSU Bank"),
    ("Nifty Financial Services",          "NIFTY_FIN_SERVICE.NS",   "Nifty Financial Services"),
    ("Nifty MidSmall Financial Services", "^NIFTY_MIDSMALL_FINSRV", "Nifty MidSmall Financial Services"),
    ("Nifty Capital Market",              "^NIFTY_CAP_MARKET",      "Nifty Capital Markets"),
    ("Nifty Consumer Durables",           "NIFTY_CONSR_DURBL.NS",   "Nifty Consumer Durables"),
    ("Nifty Realty",                      "^CNXREALTY",             "Nifty Realty"),
    ("Nifty CPSE",                        "^CNXCPSE",               "Nifty CPSE"),
    ("Nifty India Tourism",               "NIFTY_IND_TOURISM.NS",   "Nifty India Tourism"),
    ("Nifty Commodities",                 "^CNXCMDT",               "Nifty Commodities"),
    ("Nifty India Consumption",           "^CNXCONSUM",             "Nifty India Consumption"),
    ("Nifty Rural",                       "^NIFTYRURAL",            "Nifty Rural"),
    ("Nifty Housing",                     "^NIFTYHOUSING",          "Nifty Housing"),
    ("Nifty Infrastructure",              "^CNXINFRA",              "Nifty Infrastructure"),
    ("Nifty Defence",                     "^NIFTYDEFENCE",          "Nifty India Defence"),
    ("Nifty India Manufacturing",         "NIFTY_INDIA_MFG.NS",     "Nifty India Manufacturing"),
    ("Nifty MNC",                         "^CNXMNC",                "Nifty MNC"),
    ("India VIX",                         "^INDIAVIX",              "India VIX"),
]

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _safe(name): return name.replace(" ","_").replace("&","and").replace("/","_")

def load_seed(name):
    try:
        r = requests.get(f"{SEED_REPO}/{_safe(name)}.csv", timeout=20,
                         headers={"User-Agent": UA})
        if r.status_code != 200: return pd.Series(dtype="float64")
        df = pd.read_csv(io.BytesIO(r.content), index_col=0, parse_dates=True)
        s  = df.iloc[:,0].dropna()
        idx = pd.to_datetime(s.index)
        if getattr(idx,"tz",None): idx = idx.tz_convert(None)
        s = pd.Series(s.values, index=idx.normalize(), dtype=float)
        return s[~s.index.duplicated(keep="last")].sort_index()
    except: return pd.Series(dtype="float64")

def fetch_yf(ticker):
    try:
        raw = yf.download(ticker, period="30d", interval="1d",
                          progress=False, auto_adjust=False)
        if raw is None or raw.empty: return pd.Series(dtype="float64")
        s = raw["Close"]
        if isinstance(s, pd.DataFrame): s = s.iloc[:,0]
        s = s.dropna()
        idx = pd.to_datetime(s.index)
        if getattr(idx,"tz",None): idx = idx.tz_convert(None)
        s = pd.Series(s.values, index=idx.normalize(), dtype=float)
        return s[~s.index.duplicated(keep="last")].sort_index()
    except: return pd.Series(dtype="float64")

_sess = None; _snaps = {}
def _nifty_sess():
    global _sess
    if not _sess:
        _sess = requests.Session()
        _sess.headers.update({"User-Agent":UA,"Referer":"https://www.niftyindices.com/"})
        try: _sess.get("https://www.niftyindices.com/",timeout=10)
        except: pass
    return _sess

def _snapshot(date):
    k = date.strftime("%d%m%Y")
    if k in _snaps: return _snaps[k]
    try:
        r = _nifty_sess().get(f"https://niftyindices.com/Daily_Snapshot/ind_close_all_{k}.csv",timeout=15)
        if r.status_code != 200: _snaps[k]={}; return {}
        df = pd.read_csv(io.BytesIO(r.content))
        df.columns = [c.strip() for c in df.columns]
        nc = next((c for c in df.columns if "Index Name" in c),None)
        vc = next((c for c in df.columns if "Closing" in c),None)
        if not nc or not vc: _snaps[k]={}; return {}
        snap={}
        for _,row in df.iterrows():
            key = re.sub(r"\s+"," ",re.sub(r"[^a-z0-9 ]"," ",
                  str(row[nc]).lower().replace("&","and"))).strip()
            try: snap[key]=float(row[vc])
            except: pass
        _snaps[k]=snap; return snap
    except: _snaps[k]={}; return {}

def nifty_latest(label):
    def n(s):
        s=re.sub(r"\bindex\b","",s.lower().replace("&","and"))
        return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9 ]"," ",s)).strip()
    nl = n(label); today = dt.date.today()
    for back in range(5):
        d = today - dt.timedelta(days=back)
        if d.weekday()>=5: continue
        snap = _snapshot(d)
        for key in [nl, nl.rstrip("s")]:
            if key in snap: return d.isoformat(), snap[key]
    return None

def weekly(daily):
    return daily.resample("W-FRI").last().dropna() if not daily.empty else pd.Series(dtype="float64")

def score(w, ma_type):
    if len(w)<5: return None
    close=float(w.iloc[-1]); crossed=avail=0; mas={}
    for p in SMA_PERIODS:
        if ma_type=="sma":
            v=w.rolling(p,min_periods=p).mean().iloc[-1]
        else:
            mask=pd.Series(range(1,len(w)+1),index=w.index)>=p
            v=w.ewm(span=p,adjust=False).mean().where(mask).iloc[-1]
        if pd.notna(v):
            v=float(v); mas[str(p)]=round(v,2); avail+=1
            if close>v: crossed+=1
        else: mas[str(p)]=None
    return {"close":round(close,2),"crossed":crossed,"available":avail,"mas":mas}

def history(daily, ma_type, n=52):
    w=weekly(daily)
    if w.empty: return []
    rows=[]
    for i in range(max(0,len(w)-n),len(w)):
        s=score(w.iloc[:i+1],ma_type)
        if s: rows.append({"w":w.index[i].strftime("%Y-%m-%d"),"c":s["close"],"x":s["crossed"]})
    return rows

# ---------------------------------------------------------------------------
# Collect
# ---------------------------------------------------------------------------
def collect():
    print("Fetching Yahoo Finance...", flush=True)
    live={}
    for name,ticker,_ in INDICES:
        time.sleep(0.3); s=fetch_yf(ticker); live[name]=s
        print(f"  {'OK' if not s.empty else '--'} {name}", flush=True)

    print("\nBuilding scores...", flush=True)
    out=[]; today=dt.date.today()
    for name,_,label in INDICES:
        seed=load_seed(name); lv=live.get(name,pd.Series(dtype="float64"))
        if not seed.empty and not lv.empty:
            daily=pd.concat([seed,lv]); daily=daily[~daily.index.duplicated(keep="last")].sort_index()
        elif len(seed)>=50: daily=seed
        elif not lv.empty: daily=lv
        else: daily=pd.Series(dtype="float64")

        ld=lc=None
        if not lv.empty:
            ld=lv.index.max().date().isoformat(); lc=round(float(lv.iloc[-1]),2)
        if ld!=today.isoformat() and name!="India VIX":
            ni=nifty_latest(label)
            if ni:
                nd,nc=ni; ts=pd.Timestamp(nd)
                if ts not in daily.index:
                    daily=pd.concat([daily,pd.Series([nc],index=[ts],dtype=float)]).sort_index()
                    daily=daily[~daily.index.duplicated(keep="last")]
                ld=nd; lc=round(nc,2)

        w=weekly(daily)
        out.append({"name":name,"close":lc,"date":ld,
                    "sma":score(w,"sma"),"ema":score(w,"ema"),
                    "sma_hist":history(daily,"sma"),"ema_hist":history(daily,"ema")})
        s=out[-1]["sma"]; print(f"  {name}: SMA {s['crossed']}/6" if s else f"  {name}: insuf",flush=True)
    return out

# ---------------------------------------------------------------------------
# HTML  — plain string, NOT f-string. Only __JSON_DATA__ is substituted.
# _data script tag is placed BEFORE the main script (critical for getElementById).
# ---------------------------------------------------------------------------
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ekaiva &middot; NSE Index Tracker</title>
<style>
:root{
  --bg:#0b0d11; --surface:#13161d; --border:#21262d;
  --text:#cdd9e5; --muted:#768390; --accent:#316dca;
  --green-bg:#0d1f14; --green-border:#238636; --green-text:#3fb950;
  --c6:#3fb950; --c5:#7cc866; --c4:#d4a017;
  --c3:#f0883e; --c2:#f85149; --c1:#cf3636; --c0:#768390;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;min-height:100vh}

/* header */
.hdr{border-bottom:1px solid var(--border);padding:18px 20px 12px;text-align:center}
.hdr h1{font-size:1.15rem;font-weight:800;letter-spacing:.07em;color:#e6edf3}
.hdr .sub{color:var(--muted);font-size:.73rem;margin-top:3px}
.hdr .meta{color:var(--muted);font-size:.68rem;margin-top:5px}

/* toggle */
.tog{display:flex;justify-content:center;gap:8px;padding:14px 0 6px}
.tog button{padding:6px 26px;border-radius:6px;border:1px solid var(--border);
  background:var(--surface);color:var(--muted);font-weight:700;font-size:.78rem;
  cursor:pointer;transition:all .15s;letter-spacing:.04em}
.tog button.on{background:var(--accent);border-color:var(--accent);color:#fff}

/* layout */
.wrap{max-width:980px;margin:0 auto;padding:0 14px 60px}
.sec-hdr{font-size:.68rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
  color:var(--muted);padding:14px 0 6px}

/* green alert panel */
.alert-panel{border:1px solid var(--green-border);border-radius:10px;overflow:hidden;margin-bottom:6px}
.alert-panel-hdr{background:var(--green-bg);padding:10px 16px;
  display:flex;justify-content:space-between;align-items:center}
.alert-panel-hdr .title{font-size:.78rem;font-weight:700;color:var(--green-text);letter-spacing:.04em}
.alert-panel-hdr .badge{background:#238636;color:#fff;font-size:.7rem;font-weight:700;
  padding:2px 10px;border-radius:12px}
.alert-panel-hdr .reset-note{font-size:.65rem;color:var(--muted)}
.alert-empty{padding:14px 16px;color:var(--muted);font-size:.8rem;font-style:italic;
  background:var(--green-bg)}

/* rows */
.idx-list{border:1px solid var(--border);border-radius:10px;overflow:hidden}
.row{display:flex;align-items:center;padding:9px 16px;gap:10px;
  border-top:1px solid var(--border);cursor:pointer;transition:background .12s}
.alert-panel .row{border-top:1px solid #238636;background:var(--green-bg)}
.alert-panel .row:first-child{border-top:none}
.row:hover,.alert-panel .row:hover{background:rgba(255,255,255,.05)}
.idx-list .row:first-child{border-top:none}

.dot{width:9px;height:9px;border-radius:50%;flex-shrink:0}
.rname{flex:1;font-size:.83rem;font-weight:500;color:var(--text)}
.alert-panel .rname{color:var(--green-text);font-weight:600}
.rclose{width:88px;text-align:right;font-size:.78rem;color:var(--muted);
  font-variant-numeric:tabular-nums}
.pips{display:flex;gap:3px;flex-shrink:0}
.pip{width:13px;height:13px;border-radius:2px;background:var(--border)}
.rscore{width:32px;text-align:right;font-size:.78rem;font-weight:700;flex-shrink:0}
.ins{font-size:.68rem;color:var(--muted);font-style:italic}

/* modal */
#ov{display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);
  z-index:200;align-items:center;justify-content:center}
#ov.open{display:flex}
#modal{background:var(--surface);border:1px solid var(--border);border-radius:12px;
  width:min(620px,96vw);max-height:88vh;display:flex;flex-direction:column}
#mhdr{padding:15px 20px;border-bottom:1px solid var(--border);
  display:flex;justify-content:space-between;align-items:center;flex-shrink:0}
#mhdr h2{font-size:.92rem;font-weight:700;color:var(--text)}
#mcls{background:none;border:none;color:var(--muted);font-size:1.15rem;cursor:pointer}
#mbody{overflow-y:auto;padding:16px 20px}
.msum{display:flex;flex-wrap:wrap;gap:16px;margin-bottom:16px}
.msi{display:flex;flex-direction:column;gap:3px}
.msl{font-size:.63rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.msv{font-size:.92rem;font-weight:700}
.msm{font-size:.8rem;font-weight:600}
.up{color:#3fb950}.dn{color:#f85149}
.htbl{width:100%;border-collapse:collapse;font-size:.75rem}
.htbl th{text-align:left;padding:5px 8px;color:var(--muted);
  border-bottom:1px solid var(--border);font-weight:600}
.htbl td{padding:4px 8px;border-bottom:1px solid rgba(33,38,45,.7);
  font-variant-numeric:tabular-nums}
.htbl tr:hover td{background:rgba(255,255,255,.03)}

footer{text-align:center;padding:18px 16px;color:var(--muted);
  font-size:.7rem;border-top:1px solid var(--border);line-height:1.9}
@media(max-width:500px){.rclose{display:none}}
</style>
</head>
<body>

<div class="hdr">
  <h1>EKAIVA &middot; NSE Index Tracker</h1>
  <div class="sub">Weekly SMA &amp; EMA &middot; strict 6/6 scoring &middot; 37 indices</div>
  <div class="meta" id="meta"></div>
</div>

<div class="tog">
  <button id="bs" class="on" onclick="setMode('sma')">SMA Model</button>
  <button id="be" onclick="setMode('ema')">EMA Model</button>
</div>

<div class="wrap">
  <!-- green alert -->
  <div class="sec-hdr">Today&rsquo;s Green Alert</div>
  <div class="alert-panel" id="alert-panel">
    <div class="alert-panel-hdr" id="alert-hdr">
      <span class="title" id="alert-title">&#x1F7E2; 6/6 indices</span>
      <span class="reset-note">Resets every trading day</span>
      <span class="badge" id="alert-count">0</span>
    </div>
    <div id="alert-rows"></div>
  </div>

  <!-- everything else -->
  <div class="sec-hdr">Everything else &mdash; sorted by score</div>
  <div class="idx-list" id="rest-list"></div>
</div>

<div id="ov" onclick="maybeClose(event)">
  <div id="modal">
    <div id="mhdr">
      <h2 id="mtitle"></h2>
      <button id="mcls" onclick="closeModal()">&#x2715;</button>
    </div>
    <div id="mbody">
      <div class="msum" id="msum"></div>
      <table class="htbl">
        <thead><tr><th>Week</th><th>Close</th><th>Score</th><th>Signal</th></tr></thead>
        <tbody id="mtbody"></tbody>
      </table>
    </div>
  </div>
</div>

<footer>
  Ekaiva Wealth &nbsp;&bull;&nbsp; ekaivawealth.com &nbsp;&bull;&nbsp;
  +91 93766 98983 &nbsp;&bull;&nbsp; ARN 305896<br>
  Not investment advice &nbsp;&bull;&nbsp;
  Score = weekly closes above each MA (5 / 10 / 20 / 50 / 100 / 200 weeks)
</footer>

<!-- DATA must be before the main script -->
<script id="_d" type="application/json">__JSON_DATA__</script>

<script>
const DATA = JSON.parse(document.getElementById('_d').textContent);
let MODE = 'sma';

const SCORE_CLR = ['#768390','#cf3636','#f85149','#f0883e','#d4a017','#7cc866','#3fb950'];

function clr(x){ return SCORE_CLR[Math.min(Math.max(x,0),6)]; }

function fmt(n){
  if(n==null) return '—';
  return n.toLocaleString('en-IN',{minimumFractionDigits:2,maximumFractionDigits:2});
}

function sig(x){
  if(x===6) return '&#x25CF; Fully above all';
  if(x>=4)  return '&#x25CF; Mostly above';
  if(x>=2)  return '&#x25CB; Mixed';
  return '&#x25CB; Below most';
}

function esc(s){ return s.replace(/\\/g,'\\\\').replace(/'/g,"\\'"); }

function setMode(m){
  MODE=m;
  document.getElementById('bs').className = m==='sma'?'on':'';
  document.getElementById('be').className = m==='ema'?'on':'';
  render();
}

function rowHTML(d, isGreen){
  const s=d[MODE], x=s?s.crossed:-1, avail=s?s.available:0;
  const c = x>=0 ? clr(x) : '#768390';
  const dot=`<span class="dot" style="background:${c}"></span>`;
  let pips='<div class="pips">';
  for(let i=0;i<6;i++){
    pips+=`<span class="pip" style="${i<x?'background:'+c:''}"></span>`;
  }
  pips+='</div>';
  const sc = avail>0
    ? `<span class="rscore" style="color:${c}">${x}/6</span>`
    : `<span class="ins">insuf.</span>`;
  return `<div class="row" onclick="openModal('${esc(d.name)}')">${dot}
    <span class="rname">${d.name}</span>
    <span class="rclose">${fmt(d.close)}</span>
    ${pips}${sc}</div>`;
}

function render(){
  document.getElementById('meta').textContent =
    'Data as of '+DATA.as_of+'   ·   Generated '+DATA.generated;

  const all   = DATA.indices;
  const green = all.filter(d=>d[MODE]&&d[MODE].crossed===6&&d[MODE].available===6);
  const rest  = all
    .filter(d=>!(d[MODE]&&d[MODE].crossed===6&&d[MODE].available===6))
    .sort((a,b)=>{
      const xa=a[MODE]?a[MODE].crossed:-1, xb=b[MODE]?b[MODE].crossed:-1;
      return xb-xa;
    });

  // alert panel
  document.getElementById('alert-count').textContent = green.length;
  document.getElementById('alert-title').innerHTML =
    '&#x1F7E2; '+MODE.toUpperCase()+' 6/6 today';
  if(green.length){
    document.getElementById('alert-rows').innerHTML = green.map(d=>rowHTML(d,true)).join('');
  } else {
    document.getElementById('alert-rows').innerHTML =
      '<div class="alert-empty">No index is at 6/6 in the '+MODE.toUpperCase()+' model today.</div>';
  }

  // rest list
  document.getElementById('rest-list').innerHTML = rest.map(d=>rowHTML(d,false)).join('');
}

function openModal(name){
  const d=DATA.indices.find(i=>i.name===name); if(!d) return;
  const s=d[MODE], hist=d[MODE+'_hist']||[];
  document.getElementById('mtitle').textContent=name;

  let sum='';
  if(s){
    const x=s.crossed, c=clr(x);
    sum+=`<div class="msi"><div class="msl">Close</div><div class="msv">${fmt(d.close)}</div></div>`;
    sum+=`<div class="msi"><div class="msl">Score</div><div class="msv" style="color:${c}">${x}/6</div></div>`;
    [5,10,20,50,100,200].forEach(p=>{
      const v=s.mas[String(p)];
      if(v!=null){
        const ab=d.close>v;
        sum+=`<div class="msi"><div class="msl">${MODE.toUpperCase()}${p}</div>
          <div class="msm">${fmt(v)} <span class="${ab?'up':'dn'}">${ab?'&#9650;':'&#9660;'}</span></div></div>`;
      }
    });
  } else {
    sum='<span style="color:var(--muted);font-size:.8rem">Insufficient weekly history.</span>';
  }
  document.getElementById('msum').innerHTML=sum;

  let tb='';
  [...hist].reverse().forEach(r=>{
    tb+=`<tr><td>${r.w}</td><td>${fmt(r.c)}</td>
      <td style="color:${clr(r.x)};font-weight:700">${r.x}/6</td>
      <td>${sig(r.x)}</td></tr>`;
  });
  document.getElementById('mtbody').innerHTML=tb||
    '<tr><td colspan="4" style="color:var(--muted);text-align:center;padding:12px">No history</td></tr>';

  document.getElementById('ov').classList.add('open');
}

function closeModal(){ document.getElementById('ov').classList.remove('open'); }
function maybeClose(e){ if(e.target===document.getElementById('ov')) closeModal(); }
document.addEventListener('keydown',e=>{ if(e.key==='Escape') closeModal(); });

render();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Build HTML
# ---------------------------------------------------------------------------
def build_html(results):
    today=dt.date.today(); now=dt.datetime.now()
    dates=[r["date"] for r in results if r["date"]]
    payload={
        "as_of": max(dates) if dates else today.isoformat(),
        "generated": now.strftime("%d %b %Y, %H:%M"),
        "indices":[{
            "name":r["name"],"close":r["close"],"date":r["date"],
            "sma":r["sma"],"ema":r["ema"],
            "sma_hist":r["sma_hist"],"ema_hist":r["ema_hist"]
        } for r in results]
    }
    js=json.dumps(payload,separators=(",",":")).replace("</","<\\/")
    return HTML.replace("__JSON_DATA__",js)

if __name__=="__main__":
    results=collect()
    html=build_html(results)
    with open("index.html","w",encoding="utf-8") as f: f.write(html)
    print(f"\nWritten -> index.html  ({len(html):,} bytes)")
