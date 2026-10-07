import sys, os, sqlite3, time, json, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from datetime import datetime, timezone

import streamlit as st
import pandas as pd
import altair as alt

from ml.assessment import assess_machine
from ml.inference import get_model_metadata
from backend.services.sustainability_service import compute_sustainability_estimate
from backend.services.whatif_service import run_whatif, MachineInputs
from backend.services.simulation_service import get_simulation_engine

# ══════════════════════════════════════════════════════════════════════════════
# SQLite persistence (built-in — no extra deps)
# ══════════════════════════════════════════════════════════════════════════════
DB = os.path.join(os.path.dirname(__file__), "industria_x.db")

def _conn():
    c = sqlite3.connect(DB, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c

def _init():
    c = _conn()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS machines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_identifier TEXT UNIQUE NOT NULL,
            type TEXT,
            created_at TEXT,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id INTEGER NOT NULL,
            failure_probability REAL,
            anomaly_score REAL,
            risk_score REAL,
            risk_level TEXT,
            model_version TEXT,
            air_temp_k REAL, process_temp_k REAL, rotational_speed_rpm REAL,
            torque_nm REAL, tool_wear_min REAL, machine_type TEXT,
            created_at TEXT
        );
    """)
    # Migration: add sensor columns to existing DBs that pre-date them
    existing = {row[1] for row in c.execute("PRAGMA table_info(assessments)").fetchall()}
    for col, typedef in [
        ("air_temp_k",          "REAL"),
        ("process_temp_k",      "REAL"),
        ("rotational_speed_rpm","REAL"),
        ("torque_nm",           "REAL"),
        ("tool_wear_min",       "REAL"),
        ("machine_type",        "TEXT"),
    ]:
        if col not in existing:
            c.execute(f"ALTER TABLE assessments ADD COLUMN {col} {typedef}")
    c.commit(); c.close()

_init()

def _now(): return datetime.now(timezone.utc).isoformat()

def db_machines():
    c = _conn()
    rows = c.execute("SELECT * FROM machines ORDER BY id").fetchall()
    c.close(); return [dict(r) for r in rows]

def db_machine(mid):
    c = _conn()
    r = c.execute("SELECT * FROM machines WHERE id=?", (mid,)).fetchone()
    c.close(); return dict(r) if r else None

def db_create_machine(ident, mtype):
    c = _conn()
    try:
        c.execute("INSERT INTO machines (machine_identifier,type,created_at,updated_at) VALUES (?,?,?,?)",
                  (ident, mtype or None, _now(), _now()))
        c.commit()
        r = c.execute("SELECT * FROM machines WHERE machine_identifier=?", (ident,)).fetchone()
        return dict(r)
    except sqlite3.IntegrityError:
        raise ValueError(f"Machine '{ident}' already exists.")
    finally: c.close()

def db_assessments(mid, limit=50):
    c = _conn()
    rows = c.execute(
        "SELECT * FROM assessments WHERE machine_id=? ORDER BY created_at DESC LIMIT ?",
        (mid, limit)).fetchall()
    c.close(); return [dict(r) for r in rows]

def db_all_assessments(limit=500):
    c = _conn()
    rows = c.execute("SELECT * FROM assessments ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    c.close(); return [dict(r) for r in rows]

def db_save_assessment(mid, r, inputs):
    c = _conn()
    c.execute("""INSERT INTO assessments
        (machine_id,failure_probability,anomaly_score,risk_score,risk_level,model_version,
         air_temp_k,process_temp_k,rotational_speed_rpm,torque_nm,tool_wear_min,machine_type,created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (mid, r.failure_probability, r.anomaly_score, r.risk_score, r.risk_level, r.model_version,
         inputs.get("air"), inputs.get("proc"), inputs.get("rpm"), inputs.get("torq"),
         inputs.get("wear"), inputs.get("mtype"), _now()))
    c.commit()
    row = c.execute("SELECT * FROM assessments WHERE machine_id=? ORDER BY created_at DESC LIMIT 1", (mid,)).fetchone()
    c.close(); return dict(row)

def run_ml(mid, air, proc, rpm, torq, wear, mtype):
    r = assess_machine(air_temp_k=air, process_temp_k=proc,
                       rotational_speed_rpm=rpm, torque_nm=torq,
                       tool_wear_min=wear, type_=mtype)
    saved = db_save_assessment(mid, r, {"air":air,"proc":proc,"rpm":rpm,"torq":torq,"wear":wear,"mtype":mtype})
    return saved, r

# ══════════════════════════════════════════════════════════════════════════════
# Artifact loaders
# ══════════════════════════════════════════════════════════════════════════════
ARTIFACTS = os.path.join(os.path.dirname(__file__), "..", "ml", "artifacts", "v1")

@st.cache_data
def load_metadata():
    try: return get_model_metadata()
    except: return {}

@st.cache_data
def load_shap():
    try:
        with open(os.path.join(ARTIFACTS, "global_shap_importance.json"), encoding="utf-8") as f:
            return json.load(f)
    except: return {}

@st.cache_data
def load_anomaly_meta():
    try:
        with open(os.path.join(ARTIFACTS, "anomaly_metadata.json"), encoding="utf-8") as f:
            return json.load(f)
    except: return {}

@st.cache_data
def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "ai4i2020.csv")
    try: return pd.read_csv(path)
    except: return None

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
RISK_COLOR = {"CRITICAL":"#dc2626","WARNING":"#d97706","NORMAL":"#16a34a","UNKNOWN":"#6b7280"}
RISK_BG    = {"CRITICAL":"#fee2e2","WARNING":"#fef9c3","NORMAL":"#dcfce7","UNKNOWN":"#f3f4f6"}
RISK_EMOJI = {"CRITICAL":"🔴","WARNING":"🟡","NORMAL":"🟢","UNKNOWN":"⚪"}
RISK_ORDER = {"CRITICAL":0,"WARNING":1,"NORMAL":2,"UNKNOWN":3}

def rl(l):  return f"{RISK_EMOJI.get((l or 'UNKNOWN').upper(),'⚪')} {(l or 'UNKNOWN').capitalize()}"
def ro(l):  return RISK_ORDER.get((l or 'UNKNOWN').upper(), 3)
def fp(v):  return f"{v*100:.1f}%" if v is not None else "—"
def fs(v):  return f"{v:.3f}"     if v is not None else "—"
def fts(ts):
    if not ts: return "Never"
    try: return datetime.fromisoformat(ts.replace("Z","+00:00")).strftime("%Y-%m-%d %H:%M")
    except: return ts

def risk_badge(level):
    c = RISK_COLOR.get((level or "UNKNOWN").upper(),"#6b7280")
    return f'<span style="background:{c};color:white;padding:2px 10px;border-radius:4px;font-size:12px;font-weight:700">{(level or "UNKNOWN").upper()}</span>'

# ══════════════════════════════════════════════════════════════════════════════
# Page config + sidebar
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="INDUSTRIA-X", page_icon="🏭", layout="wide")

with st.sidebar:
    st.markdown("## 🏭 INDUSTRIA-X")
    st.caption("Predictive Maintenance AI")
    st.divider()
    page = st.radio("Navigate", [
        "🏠 Home",
        "📊 Dashboard",
        "🖥️ Machines",
        "🔬 Machine Detail",
        "🔧 Maintenance",
        "🤖 ML Model Info",
        "📈 Dataset Explorer",
        "⚗️ What-If",
        "🌱 Sustainability",
        "📡 Simulation",
    ], label_visibility="collapsed")
    st.divider()
    st.caption("UN SDG 9 · Phase 12")
    st.caption("All logic runs in-process — no backend needed")

# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🏭 INDUSTRIA-X")
    st.subheader("Agentic AI for Predictive Maintenance — fully self-contained Streamlit app")
    st.info("All ML inference, data storage and UI run **inside this single Streamlit app**. No backend server, no Docker, no external database required.", icon="ℹ️")
    meta = load_metadata()
    if meta:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Active Model", meta.get("model_name","—"))
        c2.metric("Model Version", meta.get("model_version","—"))
        c3.metric("Threshold", f"{meta.get('threshold',0):.3f}")
        c4.metric("Training Rows", f"{meta.get('train_size',0):,}")
    st.divider()
    c1,c2,c3 = st.columns(3)
    c1.markdown("**📊 Monitoring**\n- Dashboard KPIs\n- Machine registry\n- Assessment history\n- Maintenance priority")
    c2.markdown("**🤖 AI / ML**\n- XGBoost failure model\n- IsolationForest anomaly\n- What-If analysis\n- Live simulation")
    c3.markdown("**🌱 Sustainability**\n- Energy estimates (P=τω)\n- CO₂e calculation\n- Dataset explorer\n- SHAP feature importance")
    st.warning("Decision-support only. UCI AI4I 2020 synthetic dataset. All decisions require qualified human review.", icon="⚠️")

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.title("📊 Executive Overview")
    if st.button("🔄 Refresh"): st.rerun()

    machines = db_machines()
    rows = [{"m":m, "a":(db_assessments(m["id"],1) or [None])[0]} for m in machines]
    total     = len(machines)
    assessed  = [r for r in rows if r["a"]]
    healthy   = sum(1 for r in assessed if r["a"]["risk_level"]=="NORMAL")
    warning   = sum(1 for r in assessed if r["a"]["risk_level"]=="WARNING")
    critical  = sum(1 for r in assessed if r["a"]["risk_level"]=="CRITICAL")
    anomalous = sum(1 for r in assessed if r["a"]["anomaly_score"]>0)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("🖥️ Total Machines", total)
    c2.metric("🟢 Healthy", healthy)
    c3.metric("🟡 Warning", warning)
    c4.metric("🔴 Critical", critical)
    c5.metric("⚡ Anomalies", anomalous)

    if total == 0:
        st.info("No machines yet — go to **Machines** to register one."); st.stop()
    st.divider()

    cl,cr = st.columns(2)
    with cl:
        st.subheader("Risk Distribution")
        if not assessed:
            st.caption("No assessments yet.")
        else:
            df_dist = pd.DataFrame([
                {"Level":"NORMAL",  "Count":healthy,  "Color":"#16a34a"},
                {"Level":"WARNING", "Count":warning,  "Color":"#d97706"},
                {"Level":"CRITICAL","Count":critical, "Color":"#dc2626"},
            ])
            for _,row in df_dist.iterrows():
                st.markdown(f'<div style="display:flex;justify-content:space-between;margin-bottom:2px"><span style="background:{row.Color};color:white;padding:1px 8px;border-radius:4px;font-size:12px">{row.Level}</span><span style="font-size:13px">{int(row.Count)}/{len(assessed)}</span></div>', unsafe_allow_html=True)
                st.progress(row.Count / max(len(assessed),1))
            ch = alt.Chart(df_dist).mark_bar(cornerRadiusTopLeft=4,cornerRadiusTopRight=4).encode(
                x=alt.X("Level:N",axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Count:Q"),
                color=alt.Color("Color:N",scale=None,legend=None),
                tooltip=["Level","Count"]
            ).properties(height=130)
            st.altair_chart(ch, use_container_width=True)

    with cr:
        st.subheader("⚠️ Attention Required")
        attn = sorted(
            [r for r in rows if r["a"] and r["a"]["risk_level"]!="NORMAL"],
            key=lambda r:(ro(r["a"]["risk_level"]),-(r["a"]["risk_score"] or 0))
        )[:5]
        if not attn:
            st.success("✅ No machines require immediate attention.")
        else:
            for r in attn:
                m,a = r["m"],r["a"]
                st.markdown(
                    f'<div style="background:{RISK_BG.get(a["risk_level"],"#f3f4f6")};border-radius:8px;padding:10px 14px;margin-bottom:8px;display:flex;justify-content:space-between">'
                    f'<div><strong>{m["machine_identifier"]}</strong><br><span style="font-size:12px;color:#777">{m["type"] or "—"}</span></div>'
                    f'<div style="text-align:right">{risk_badge(a["risk_level"])}<br><span style="font-size:12px">P(fail) {fp(a["failure_probability"])}</span></div></div>',
                    unsafe_allow_html=True)

    st.divider()
    st.subheader("Recent Assessments")
    recent = sorted([r for r in rows if r["a"]], key=lambda r:r["a"]["created_at"], reverse=True)[:10]
    if recent:
        st.dataframe(pd.DataFrame([{
            "Machine":  r["m"]["machine_identifier"],
            "Risk":     rl(r["a"]["risk_level"]),
            "Risk Score":  fs(r["a"]["risk_score"]),
            "P(Failure)":  fp(r["a"]["failure_probability"]),
            "Anomaly":  "⚠️ Anomalous" if r["a"]["anomaly_score"]>0 else "✅ Normal",
            "Timestamp":   fts(r["a"]["created_at"]),
        } for r in recent]), use_container_width=True, hide_index=True)
    st.warning("Decision-support only. UCI AI4I 2020 synthetic dataset.", icon="⚠️")

# ══════════════════════════════════════════════════════════════════════════════
# MACHINES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🖥️ Machines":
    st.title("🖥️ Machine Monitoring")
    with st.expander("➕ Register New Machine"):
        with st.form("reg"):
            ri = st.text_input("Machine Identifier *")
            rt = st.selectbox("Type (optional)", ["","L","M","H"])
            if st.form_submit_button("Register", type="primary"):
                if not ri.strip():
                    st.error("Identifier required.")
                else:
                    try:
                        m = db_create_machine(ri.strip(), rt or None)
                        st.success(f"Registered **{m['machine_identifier']}** (ID #{m['id']})"); st.rerun()
                    except ValueError as e: st.error(str(e))
                    except Exception as e: st.error(f"Failed: {e}")

    if st.button("🔄 Refresh"): st.rerun()
    machines = db_machines()
    if not machines: st.info("No machines registered."); st.stop()

    rows = [{"m":m,"a":(db_assessments(m["id"],1) or [None])[0]} for m in machines]
    sc1,sc2,sc3 = st.columns([3,2,2])
    search = sc1.text_input("🔍 Search", label_visibility="collapsed", placeholder="Name or type…")
    rf = sc2.selectbox("Risk", ["ALL","NORMAL","WARNING","CRITICAL","UNASSESSED"])
    sb = sc3.selectbox("Sort", ["Last Assessment","Risk","Risk Score","P(Failure)","Machine"])

    def _filt(rows,q,rf):
        out=[]
        for r in rows:
            m,a=r["m"],r["a"]
            if q and q.lower() not in m["machine_identifier"].lower() and q.lower() not in (m["type"] or "").lower(): continue
            if rf=="UNASSESSED" and a: continue
            if rf not in ("ALL","UNASSESSED") and (not a or a["risk_level"]!=rf): continue
            out.append(r)
        return out

    def _sort(rows,sb):
        if sb=="Machine":    return sorted(rows, key=lambda r: r["m"]["machine_identifier"])
        if sb=="Risk":       return sorted(rows, key=lambda r: ro(r["a"]["risk_level"] if r["a"] else None))
        if sb=="Risk Score": return sorted(rows, key=lambda r: -(r["a"]["risk_score"] if r["a"] else -1))
        if sb=="P(Failure)": return sorted(rows, key=lambda r: -(r["a"]["failure_probability"] if r["a"] else -1))
        return sorted(rows, key=lambda r: r["a"]["created_at"] if r["a"] else "", reverse=True)

    filtered = _sort(_filt(rows,search,rf),sb)
    st.caption(f"{len(filtered)} result(s)")
    if filtered:
        st.dataframe(pd.DataFrame([{
            "ID": r["m"]["id"],
            "Machine": r["m"]["machine_identifier"],
            "Type": r["m"]["type"] or "—",
            "Risk": rl(r["a"]["risk_level"]) if r["a"] else "⚪ Unassessed",
            "Risk Score": fs(r["a"]["risk_score"]) if r["a"] else "—",
            "P(Failure)": fp(r["a"]["failure_probability"]) if r["a"] else "—",
            "Anomaly": ("⚠️ Anomalous" if r["a"]["anomaly_score"]>0 else "✅ Normal") if r["a"] else "—",
            "Last Assessment": fts(r["a"]["created_at"]) if r["a"] else "Never",
        } for r in filtered]), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# MACHINE DETAIL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Machine Detail":
    st.title("🔬 Machine Detail")
    machines = db_machines()
    if not machines: st.info("No machines registered."); st.stop()

    opts = {f"#{m['id']} — {m['machine_identifier']}": m["id"] for m in machines}
    sel  = st.selectbox("Select Machine", list(opts.keys()))
    mid  = opts[sel]
    if st.button("🔄 Refresh"): st.rerun()

    machine = db_machine(mid)
    history = db_assessments(mid, limit=50)
    latest  = history[0] if history else None

    st.subheader(f"🖥️ {machine['machine_identifier']}")
    st.caption(f"Type: {machine['type'] or 'Unspecified'}")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("ID",         f"#{machine['id']}")
    c2.metric("Identifier", machine["machine_identifier"])
    c3.metric("Registered", fts(machine["created_at"]))
    c4.metric("Updated",    fts(machine["updated_at"]))

    st.divider()
    st.subheader("Current Risk")
    if latest:
        r1,r2,r3,r4,r5 = st.columns(5)
        rc = RISK_COLOR.get(latest["risk_level"],"#6b7280")
        r1.markdown(f'<div style="text-align:center"><p style="font-size:11px;color:#888;margin-bottom:4px">RISK LEVEL</p><span style="background:{rc};color:white;padding:4px 12px;border-radius:6px;font-weight:700">{latest["risk_level"]}</span></div>', unsafe_allow_html=True)
        r2.metric("Risk Score",    fs(latest["risk_score"]))
        r3.metric("P(Failure)",    fp(latest["failure_probability"]))
        r4.metric("Anomaly Score", fs(latest["anomaly_score"]))
        r5.metric("Anomaly",       "⚠️ Anomalous" if latest["anomaly_score"]>0 else "✅ Normal")
        st.caption(f"Latest: {fts(latest['created_at'])} · Model: {latest['model_version']}")
    else:
        st.info("No assessment yet. Use the form below.")

    st.divider()
    with st.expander("🔬 Run New Assessment", expanded=(latest is None)):
        with st.form("assess_form"):
            st.caption("Enter sensor readings — values are based on UCI AI4I 2020 synthetic dataset ranges.")
            ac1,ac2 = st.columns(2)
            with ac1:
                air  = st.number_input("Air Temp [K]",     value=300.0, min_value=200.0, max_value=400.0, step=0.5, help="Typical range: 295–305 K")
                proc = st.number_input("Process Temp [K]", value=310.0, min_value=200.0, max_value=400.0, step=0.5, help="Typical range: 305–315 K")
                rpm  = st.number_input("Speed [rpm]",      value=1500,  min_value=0,     max_value=5000,  step=10,  help="Typical range: 1100–2900 rpm")
            with ac2:
                torq = st.number_input("Torque [Nm]",      value=40.0,  min_value=0.0,   max_value=200.0, step=0.5, help="Typical range: 3–77 Nm")
                wear = st.number_input("Tool Wear [min]",  value=100,   min_value=0,     max_value=600,   step=1,   help="Typical range: 0–254 min")
                mtype= st.selectbox("Type", ["L","M","H"], index=1, help="L=Low, M=Medium, H=High quality variant")
            if st.form_submit_button("▶️ Assess Machine", type="primary"):
                try:
                    saved, r = run_ml(mid, air, proc, rpm, torq, wear, mtype)
                    col_a, col_b, col_c = st.columns(3)
                    col_a.success(f"Risk: **{r.risk_level}**")
                    col_b.metric("P(Failure)", fp(r.failure_probability))
                    col_c.metric("Risk Score", fs(r.risk_score))
                    st.rerun()
                except FileNotFoundError:
                    st.error("ML artifacts missing. Ensure `ml/artifacts/v1/` contains the model files.")
                except ValueError as e:
                    st.error(f"Invalid input: {e}")
                except Exception as e:
                    st.error(f"Assessment failed: {e}")

    st.divider()
    st.subheader("Assessment History")
    if not history:
        st.caption("No assessments yet.")
    else:
        t1,t2 = st.tabs(["📈 Chart","📋 Table"])
        with t1:
            cd = pd.DataFrame([{
                "Step": i+1,
                "Risk Score": round(a["risk_score"],3),
                "P(Failure) %": round(a["failure_probability"]*100,1),
                "Timestamp": fts(a["created_at"])
            } for i,a in enumerate(reversed(history[-20:]))])
            base = alt.Chart(cd).encode(x="Step:Q")
            l1 = base.mark_line(color="#3b82f6",strokeWidth=2).encode(y="Risk Score:Q",tooltip=["Step","Risk Score","Timestamp"])
            l2 = base.mark_line(color="#ef4444",strokeWidth=2,strokeDash=[4,2]).encode(y="P(Failure) %:Q",tooltip=["Step","P(Failure) %","Timestamp"])
            thr= alt.Chart(pd.DataFrame({"y":[50]})).mark_rule(color="#d97706",strokeDash=[3,3]).encode(y="y:Q")
            st.altair_chart((l1+l2+thr).properties(height=200), use_container_width=True)
            st.caption("🔵 Risk Score · 🔴 P(Failure)×100 · 🟡 0.5 threshold")
        with t2:
            rows_tbl = []
            for i,a in enumerate(history[:20]):
                rows_tbl.append({
                    "#": "★ Latest" if i==0 else f"#{a['id']}",
                    "Risk": rl(a["risk_level"]),
                    "Score": fs(a["risk_score"]),
                    "P(Failure)": fp(a["failure_probability"]),
                    "Anomaly": fs(a["anomaly_score"]),
                    "Air[K]": f"{a['air_temp_k']:.1f}" if a.get("air_temp_k") else "—",
                    "Torque[Nm]": f"{a['torque_nm']:.1f}" if a.get("torque_nm") else "—",
                    "Wear[min]": str(a.get("tool_wear_min","—")),
                    "Timestamp": fts(a["created_at"])
                })
            st.dataframe(pd.DataFrame(rows_tbl), use_container_width=True, hide_index=True)
    st.warning("Decision-support only. UCI AI4I 2020 synthetic dataset. Qualified human review required.", icon="⚠️")

# ══════════════════════════════════════════════════════════════════════════════
# MAINTENANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔧 Maintenance":
    st.title("🔧 Maintenance Center")
    if st.button("🔄 Refresh"): st.rerun()
    machines = db_machines()
    if not machines: st.info("No machines registered."); st.stop()

    rows = [{"m":m,"a":(db_assessments(m["id"],1) or [None])[0]} for m in machines]
    rows = sorted(rows, key=lambda r:(ro(r["a"]["risk_level"] if r["a"] else None),-(r["a"]["risk_score"] if r["a"] else 0)))
    crit  = [r for r in rows if r["a"] and r["a"]["risk_level"]=="CRITICAL"]
    warn  = [r for r in rows if r["a"] and r["a"]["risk_level"]=="WARNING"]
    norm  = [r for r in rows if r["a"] and r["a"]["risk_level"]=="NORMAL"]
    unass = [r for r in rows if not r["a"]]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🔴 Critical",  len(crit))
    c2.metric("🟡 Warning",   len(warn))
    c3.metric("🟢 Normal",    len(norm))
    c4.metric("⚪ Unassessed",len(unass))
    st.divider()

    def _rgroup(title, icon, rows, rec, empty):
        st.subheader(f"{icon} {title}")
        if not rows: st.caption(empty); return
        st.dataframe(pd.DataFrame([{
            "Machine":    r["m"]["machine_identifier"],
            "Type":       r["m"]["type"] or "—",
            "Risk":       rl(r["a"]["risk_level"]) if r["a"] else "⚪",
            "P(Failure)": fp(r["a"]["failure_probability"]) if r["a"] else "—",
            "Anomaly":    (f"⚠️ ({fs(r['a']['anomaly_score'])})" if r["a"]["anomaly_score"]>0 else "✅ Normal") if r["a"] else "—",
            "Last Assessment": fts(r["a"]["created_at"]) if r["a"] else "Never",
            "Recommendation": rec,
        } for r in rows]), use_container_width=True, hide_index=True)

    _rgroup("Critical — Immediate Attention","🔴",crit,"Recommended for maintenance review","✅ No critical machines.")
    _rgroup("Warning — Monitor Closely","🟡",warn,"Schedule maintenance review","✅ No warning machines.")
    _rgroup("Normal — Healthy","🟢",norm,"Continue routine monitoring","No normal-assessed machines yet.")
    if unass: _rgroup("Unassessed","⚪",unass,"Run initial assessment","")
    st.warning("Elevated risk indicators only — not guaranteed failure. Qualified human review required.", icon="⚠️")

# ══════════════════════════════════════════════════════════════════════════════
# ML MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Model Info":
    st.title("🤖 ML Model Information")
    st.caption("Live model artifacts — loaded directly from `ml/artifacts/v1/`")

    meta   = load_metadata()
    shap   = load_shap()
    a_meta = load_anomaly_meta()

    if not meta:
        st.error("Could not load model metadata. Ensure `ml/artifacts/v1/metadata.json` exists."); st.stop()

    # ── Model overview ────────────────────────────────────────────────────────
    st.subheader("Failure Prediction Model")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Model",     meta.get("model_name","—"))
    m2.metric("Version",   meta.get("model_version","—"))
    m3.metric("Threshold", f"{meta.get('threshold',0):.4f}")
    m4.metric("Dataset",   "UCI AI4I 2020")

    m5,m6,m7,m8 = st.columns(4)
    m5.metric("Train Rows",    f"{meta.get('train_size',0):,}")
    m6.metric("Test Rows",     f"{meta.get('test_size',0):,}")
    m7.metric("Train Failures",f"{meta.get('train_failures',0)}")
    m8.metric("Test Failures", f"{meta.get('test_failures',0)}")

    # ── Evaluation metrics table ──────────────────────────────────────────────
    st.divider()
    st.subheader("Model Evaluation Metrics (Test Set)")
    evals = meta.get("evaluation_metrics", {})
    if evals:
        rows_eval = []
        for model_name, m in evals.items():
            rows_eval.append({
                "Model": model_name,
                "Accuracy": f"{m.get('accuracy',0)*100:.2f}%",
                "Precision (Failure)": f"{m.get('precision_failure',0)*100:.2f}%",
                "Recall (Failure)": f"{m.get('recall_failure',0)*100:.2f}%",
                "F1 (Failure)": f"{m.get('f1_failure',0)*100:.2f}%",
                "ROC-AUC": f"{m.get('roc_auc',0):.4f}",
                "PR-AUC": f"{m.get('pr_auc',0):.4f}",
                "Threshold": f"{m.get('threshold',0):.4f}",
                "TP": m.get("tp",0), "FP": m.get("fp",0),
                "TN": m.get("tn",0), "FN": m.get("fn",0),
            })
        df_eval = pd.DataFrame(rows_eval)
        st.dataframe(df_eval, use_container_width=True, hide_index=True)

        # ROC-AUC comparison chart
        df_roc = pd.DataFrame([{"Model":r["Model"],"ROC-AUC":float(r["ROC-AUC"])} for r in rows_eval])
        ch = alt.Chart(df_roc).mark_bar(cornerRadiusTopLeft=4,cornerRadiusTopRight=4).encode(
            x=alt.X("Model:N",axis=alt.Axis(labelAngle=0)),
            y=alt.Y("ROC-AUC:Q",scale=alt.Scale(domain=[0.8,1.0])),
            color=alt.condition(alt.datum["Model"]=="XGBoost", alt.value("#3b82f6"), alt.value("#94a3b8")),
            tooltip=["Model","ROC-AUC"]
        ).properties(height=160, title="ROC-AUC Comparison")
        st.altair_chart(ch, use_container_width=True)

    # ── SHAP feature importance ───────────────────────────────────────────────
    st.divider()
    st.subheader("SHAP Feature Importance (Global)")
    if shap and shap.get("features"):
        df_shap = pd.DataFrame({
            "Feature":       shap["features"],
            "Mean |SHAP|":   shap["mean_abs_shap"],
        }).sort_values("Mean |SHAP|", ascending=False)
        ch2 = alt.Chart(df_shap).mark_bar(color="#3b82f6",cornerRadiusTopLeft=4,cornerRadiusTopRight=4).encode(
            x=alt.X("Mean |SHAP|:Q"),
            y=alt.Y("Feature:N", sort="-x"),
            tooltip=["Feature","Mean |SHAP|"]
        ).properties(height=220, title=f"Global SHAP (sample_size={shap.get('sample_size',0)})")
        st.altair_chart(ch2, use_container_width=True)
        st.dataframe(df_shap, use_container_width=True, hide_index=True)
    else:
        st.caption("SHAP data unavailable.")

    # ── Anomaly detector info ─────────────────────────────────────────────────
    st.divider()
    st.subheader("Anomaly Detector")
    if a_meta:
        a1,a2,a3,a4 = st.columns(4)
        a1.metric("Algorithm",    a_meta.get("algorithm","—"))
        a2.metric("Version",      a_meta.get("anomaly_version","—"))
        a3.metric("Contamination",f"{a_meta.get('contamination',0)*100:.1f}%")
        a4.metric("Train Rows",   f"{a_meta.get('train_rows',0):,}")
        st.caption(f"n_estimators={a_meta.get('n_estimators','—')} · random_state={a_meta.get('random_state','—')} · Labels used in training: {a_meta.get('labels_used_in_training','—')}")
    else:
        st.caption("Anomaly metadata unavailable.")

    # ── Features ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Feature Columns")
    features = meta.get("feature_columns", [])
    numerical = meta.get("numerical_features", [])
    categorical = meta.get("categorical_features", [])
    fc1,fc2,fc3 = st.columns(3)
    fc1.markdown("**All Features**\n" + "\n".join(f"- `{f}`" for f in features))
    fc2.markdown("**Numerical**\n"  + "\n".join(f"- `{f}`" for f in numerical))
    fc3.markdown("**Categorical**\n"+ "\n".join(f"- `{f}`" for f in categorical))

    # ── Risk formula ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Risk Score Formula")
    st.code("risk_score = 0.70 × failure_probability + 0.30 × anomaly_score\n\nNORMAL   : risk_score < 0.40\nWARNING  : 0.40 ≤ risk_score < 0.70\nCRITICAL : risk_score ≥ 0.70", language="text")
    st.caption("Weights and thresholds are configurable. These are the current defaults.")

# ══════════════════════════════════════════════════════════════════════════════
# DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Dataset Explorer":
    st.title("📈 Dataset Explorer")
    st.caption("UCI AI4I 2020 Predictive Maintenance Dataset — Synthetic")

    df = load_dataset()
    if df is None:
        st.error("Dataset not found at `data/raw/ai4i2020.csv`."); st.stop()

    # ── Overview KPIs ─────────────────────────────────────────────────────────
    total_rows   = len(df)
    total_fail   = int(df["Machine failure"].sum()) if "Machine failure" in df.columns else 0
    fail_rate    = total_fail / total_rows * 100 if total_rows else 0

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total Rows",    f"{total_rows:,}")
    k2.metric("Total Failures",f"{total_fail:,}")
    k3.metric("Failure Rate",  f"{fail_rate:.2f}%")
    k4.metric("Features",      str(len(df.columns)))

    st.divider()

    tab1,tab2,tab3,tab4,tab5 = st.tabs(["📊 Distributions","🔍 Filter & Browse","📉 Failure Analysis","🌡️ Correlations","📋 Raw Data"])

    with tab1:
        st.subheader("Feature Distributions")
        num_cols = ["Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]","Tool wear [min]"]
        num_cols = [c for c in num_cols if c in df.columns]
        sel_col = st.selectbox("Select feature", num_cols)
        if sel_col:
            fig_data = df[[sel_col]].rename(columns={sel_col:"value"})
            if "Machine failure" in df.columns:
                fig_data["Failure"] = df["Machine failure"].map({0:"No Failure",1:"Failure"})
                ch = alt.Chart(fig_data).mark_bar(opacity=0.7).encode(
                    x=alt.X("value:Q", bin=alt.Bin(maxbins=40), title=sel_col),
                    y=alt.Y("count():Q"),
                    color=alt.Color("Failure:N", scale=alt.Scale(
                        domain=["No Failure","Failure"],
                        range=["#3b82f6","#dc2626"]
                    )),
                    tooltip=["Failure","count()"]
                ).properties(height=280, title=f"Distribution of {sel_col}")
                st.altair_chart(ch, use_container_width=True)
            else:
                ch = alt.Chart(fig_data).mark_bar().encode(
                    x=alt.X("value:Q",bin=alt.Bin(maxbins=40),title=sel_col),
                    y="count():Q"
                ).properties(height=280)
                st.altair_chart(ch, use_container_width=True)

        if "Type" in df.columns:
            st.subheader("Machine Type Distribution")
            type_counts = df["Type"].value_counts().reset_index()
            type_counts.columns = ["Type","Count"]
            ch2 = alt.Chart(type_counts).mark_bar(cornerRadiusTopLeft=4,cornerRadiusTopRight=4).encode(
                x=alt.X("Type:N",axis=alt.Axis(labelAngle=0)),
                y="Count:Q",
                color=alt.Color("Type:N",scale=alt.Scale(scheme="tableau10")),
                tooltip=["Type","Count"]
            ).properties(height=200, title="Machine Type Counts")
            st.altair_chart(ch2, use_container_width=True)

    with tab2:
        st.subheader("Filter & Browse Dataset")
        fc1,fc2,fc3 = st.columns(3)
        type_filter = fc1.multiselect("Type", df["Type"].unique().tolist() if "Type" in df.columns else [], default=[])
        fail_filter = fc2.selectbox("Failure", ["All","Failures Only","No Failures"])
        n_rows = fc3.slider("Rows to show", 10, 500, 50, 10)

        dff = df.copy()
        if type_filter: dff = dff[dff["Type"].isin(type_filter)]
        if fail_filter == "Failures Only" and "Machine failure" in dff.columns:
            dff = dff[dff["Machine failure"]==1]
        if fail_filter == "No Failures" and "Machine failure" in dff.columns:
            dff = dff[dff["Machine failure"]==0]

        st.caption(f"{len(dff):,} rows after filter")
        st.dataframe(dff.head(n_rows), use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Failure Analysis")
        if "Machine failure" in df.columns:
            # Failure by type
            if "Type" in df.columns:
                grp = df.groupby("Type")["Machine failure"].agg(["sum","count"]).reset_index()
                grp.columns = ["Type","Failures","Total"]
                grp["Rate %"] = (grp["Failures"]/grp["Total"]*100).round(2)
                ch = alt.Chart(grp).mark_bar(cornerRadiusTopLeft=4,cornerRadiusTopRight=4).encode(
                    x=alt.X("Type:N",axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Rate %:Q"),
                    color=alt.Color("Type:N",scale=alt.Scale(scheme="tableau10")),
                    tooltip=["Type","Failures","Total","Rate %"]
                ).properties(height=200, title="Failure Rate by Machine Type")
                st.altair_chart(ch, use_container_width=True)
                st.dataframe(grp, use_container_width=True, hide_index=True)

            # Failure types breakdown
            fail_types = ["TWF","HDF","PWF","OSF","RNF"]
            available_ft = [c for c in fail_types if c in df.columns]
            if available_ft:
                st.subheader("Failure Type Breakdown")
                ft_counts = {c: int(df[c].sum()) for c in available_ft}
                df_ft = pd.DataFrame(list(ft_counts.items()), columns=["Failure Type","Count"])
                ch2 = alt.Chart(df_ft).mark_bar(cornerRadiusTopLeft=4,cornerRadiusTopRight=4,color="#dc2626").encode(
                    x=alt.X("Failure Type:N",axis=alt.Axis(labelAngle=0)),
                    y="Count:Q",
                    tooltip=["Failure Type","Count"]
                ).properties(height=200, title="Count of Each Failure Type")
                st.altair_chart(ch2, use_container_width=True)
        else:
            st.caption("No 'Machine failure' column found.")

    with tab4:
        st.subheader("Feature Correlations")
        num_cols2 = ["Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]","Tool wear [min]"]
        num_cols2 = [c for c in num_cols2 if c in df.columns]
        if "Machine failure" in df.columns:
            num_cols2.append("Machine failure")
        if len(num_cols2) >= 2:
            corr = df[num_cols2].corr().round(3)
            corr_long = corr.reset_index().melt(id_vars="index")
            corr_long.columns = ["Feature A","Feature B","Correlation"]
            ch3 = alt.Chart(corr_long).mark_rect().encode(
                x=alt.X("Feature A:N",axis=alt.Axis(labelAngle=-30)),
                y=alt.Y("Feature B:N"),
                color=alt.Color("Correlation:Q", scale=alt.Scale(scheme="redblue", domain=[-1,1])),
                tooltip=["Feature A","Feature B","Correlation"]
            ).properties(height=300, title="Correlation Heatmap")
            st.altair_chart(ch3, use_container_width=True)
            st.dataframe(corr, use_container_width=True)

    with tab5:
        st.subheader("Raw Dataset")
        st.caption(f"Showing first 1000 of {len(df):,} rows")
        st.dataframe(df.head(1000), use_container_width=True, hide_index=True)
        col_dl1, col_dl2 = st.columns(2)
        col_dl1.download_button("⬇️ Download CSV", df.to_csv(index=False).encode(), "ai4i2020.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# WHAT-IF
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚗️ What-If":
    st.title("⚗️ What-If Analysis")
    st.info("Model simulations based on AI4I 2020 synthetic training data — not guaranteed physical outcomes.", icon="ℹ️")

    col_b,col_s = st.columns(2)
    def _pinputs(pfx, defs):
        return {
            "Air temperature [K]":     st.number_input("Air Temp (K)",     value=float(defs.get("Air temperature [K]",300)),    min_value=250.0,max_value=400.0,step=0.5,key=f"{pfx}_air"),
            "Process temperature [K]": st.number_input("Process Temp (K)", value=float(defs.get("Process temperature [K]",310)),min_value=250.0,max_value=400.0,step=0.5,key=f"{pfx}_proc"),
            "Rotational speed [rpm]":  st.number_input("Speed (rpm)",      value=int(defs.get("Rotational speed [rpm]",1500)),  min_value=0,max_value=5000,step=10,key=f"{pfx}_rpm"),
            "Torque [Nm]":             st.number_input("Torque (Nm)",      value=float(defs.get("Torque [Nm]",45)),             min_value=0.0,max_value=200.0,step=0.5,key=f"{pfx}_torq"),
            "Tool wear [min]":         st.number_input("Tool Wear (min)",  value=int(defs.get("Tool wear [min]",100)),          min_value=0,max_value=600,step=1,key=f"{pfx}_wear"),
            "Type": st.selectbox("Type",["L","M","H"],index=["L","M","H"].index(defs.get("Type","M")),key=f"{pfx}_type"),
        }

    with col_b:
        st.markdown("**Baseline — Current Conditions**")
        baseline = _pinputs("b", st.session_state.get("wi_b",{}))
    with col_s:
        st.markdown("**Scenario — Modified Conditions**")
        scenario = _pinputs("s", st.session_state.get("wi_s",{}))
        if st.button("↩️ Reset scenario to baseline"):
            st.session_state["wi_s"] = dict(baseline); st.rerun()

    if st.button("▶️ Run Scenario", type="primary"):
        try:
            b = MachineInputs(air_temp_k=baseline["Air temperature [K]"],process_temp_k=baseline["Process temperature [K]"],rotational_speed_rpm=baseline["Rotational speed [rpm]"],torque_nm=baseline["Torque [Nm]"],tool_wear_min=baseline["Tool wear [min]"],type_=baseline["Type"])
            s = MachineInputs(air_temp_k=scenario["Air temperature [K]"],process_temp_k=scenario["Process temperature [K]"],rotational_speed_rpm=scenario["Rotational speed [rpm]"],torque_nm=scenario["Torque [Nm]"],tool_wear_min=scenario["Tool wear [min]"],type_=scenario["Type"])
            res = run_whatif(b, s)
            st.session_state["wi_result"] = res.to_dict()
            st.session_state["wi_b"] = dict(baseline)
            st.session_state["wi_s"] = dict(scenario)
        except FileNotFoundError: st.error("ML artifacts not found. Ensure `ml/artifacts/v1/` is populated.")
        except Exception as e: st.error(f"Analysis failed: {e}")

    if "wi_result" in st.session_state:
        res = st.session_state["wi_result"]
        b,s,c = res["baseline"],res["scenario"],res["change"]
        st.divider()
        if res.get("out_of_distribution_warning"):
            st.warning(res["out_of_distribution_warning"], icon="⚠️")

        def _dstr(d,fmt):
            if abs(d)<0.0005: return "— No change"
            return f"⬆️ +{fmt(abs(d))}" if d>0 else f"⬇️ -{fmt(abs(d))}"

        st.subheader("Model-Estimated Comparison")
        df_cmp = pd.DataFrame([
            {"Metric":"Failure Probability","Baseline":fp(b["failure_probability"]),"Scenario":fp(s["failure_probability"]),"Change":_dstr(c["failure_probability_delta"],fp)},
            {"Metric":"Anomaly Score","Baseline":fs(b["anomaly_score"]),"Scenario":fs(s["anomaly_score"]),"Change":_dstr(c["anomaly_score_delta"],fs)},
            {"Metric":"Risk Score","Baseline":fs(b["risk_score"]),"Scenario":fs(s["risk_score"]),"Change":_dstr(c["risk_score_delta"],fs)},
            {"Metric":"Risk Level","Baseline":rl(b["risk_level"]),"Scenario":rl(s["risk_level"]),"Change":"⚠️ Changed" if c["risk_level_changed"] else "— Unchanged"},
        ])
        st.dataframe(df_cmp, use_container_width=True, hide_index=True)

        # Delta bar chart
        df_bar = pd.DataFrame([
            {"Metric":"Failure Prob Δ","Delta":c["failure_probability_delta"]},
            {"Metric":"Anomaly Score Δ","Delta":c["anomaly_score_delta"]},
            {"Metric":"Risk Score Δ","Delta":c["risk_score_delta"]},
        ])
        df_bar["Color"] = df_bar["Delta"].apply(lambda x: "#dc2626" if x>0 else "#16a34a" if x<0 else "#94a3b8")
        ch = alt.Chart(df_bar).mark_bar(cornerRadiusTopLeft=3,cornerRadiusTopRight=3).encode(
            x=alt.X("Metric:N",axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Delta:Q",title="Change"),
            color=alt.Color("Color:N",scale=None,legend=None),
            tooltip=["Metric","Delta"]
        ).properties(height=160, title="Delta (Scenario − Baseline)")
        st.altair_chart(ch, use_container_width=True)

        d = c["failure_probability_delta"]
        chg = f"Failure probability **{'increased' if d>0 else 'decreased'}** by {fp(abs(d))}." if abs(d)>=0.0005 else "Failure probability unchanged."
        rchg = f"⚠️ Risk level changed **{b['risk_level']} → {s['risk_level']}**. " if c["risk_level_changed"] else f"Risk level unchanged (**{b['risk_level']}**). "
        st.markdown(f"**Summary:** {rchg}{chg}")
        if st.button("🗑️ Clear"): del st.session_state["wi_result"]; st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# SUSTAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌱 Sustainability":
    st.title("🌱 Sustainability & Energy Intelligence")
    st.caption("`ESTIMATED — NOT MEASURED`")
    st.info("SDG 9 & 13 — Engineering-based energy proxies from AI4I 2020 synthetic data. Not for regulatory use.", icon="ℹ️")

    machines = db_machines()
    col_l,col_r = st.columns([1,2])

    with col_l:
        st.subheader("Parameters")
        mopts = {"— None —": None}
        for m in machines: mopts[f"{m['machine_identifier']} ({m['type'] or '?'})"] = m
        sel_m     = st.selectbox("Associate Machine (optional)", list(mopts.keys()))
        sel_machine = mopts[sel_m]
        dt = 1
        if sel_machine and (sel_machine.get("type") or "").upper() in ["L","M","H"]:
            dt = ["L","M","H"].index(sel_machine["type"].upper())

        with st.form("sustain_form"):
            mtype  = st.selectbox("Machine Type", ["L","M","H"], index=dt)
            rpm    = st.number_input("Rotational Speed (RPM)", value=1500, min_value=1, max_value=10000, step=10, help="Typical AI4I range: 1100–2900 RPM")
            torq   = st.number_input("Torque (Nm)", value=45.0, min_value=0.1, max_value=1000.0, step=0.1, help="Typical AI4I range: 3–77 Nm")
            op_hrs = st.number_input("Operating Hours", value=8.0, min_value=0.1, max_value=8760.0, step=0.5)
            eff    = st.number_input("Efficiency (η)", value=0.88, min_value=0.01, max_value=1.0, step=0.01, help="Typical AC motor: 0.85–0.92")
            emf    = st.number_input("Emission Factor (kg CO₂e/kWh)", value=0.4, min_value=0.0, max_value=5.0, step=0.01, help="Illustrative value — substitute with your grid factor")
            if st.form_submit_button("📊 Calculate Estimates", type="primary"):
                try:
                    est = compute_sustainability_estimate(rpm, torq, op_hrs, mtype, eff, emf)
                    st.session_state["s_est"] = est
                except Exception as e:
                    st.error(f"Failed: {e}")

    with col_r:
        if "s_est" in st.session_state:
            est = st.session_state["s_est"]
            k1,k2,k3,k4 = st.columns(4)
            k1.metric("⚡ Est. Power",  f"{est.estimated_electrical_power_kw:.3f} kW")
            k2.metric("🔥 Est. Energy", f"{est.estimated_energy_kwh:.2f} kWh")
            k3.metric("💨 Est. CO₂e",  f"{est.estimated_co2e_kg:.2f} kg")
            k4.metric("🕐 Hours",       f"{est.operating_hours:.1f} h")
            st.caption("EST. = Engineering estimate — not measured")

            # Energy bar chart
            df_e = pd.DataFrame([
                {"Stage":"Mech. Power (kW)", "Value": est.estimated_mechanical_power_kw},
                {"Stage":"Elec. Power (kW)", "Value": est.estimated_electrical_power_kw},
            ])
            ch = alt.Chart(df_e).mark_bar(color="#3b82f6",cornerRadiusTopLeft=4,cornerRadiusTopRight=4).encode(
                x=alt.X("Stage:N",axis=alt.Axis(labelAngle=0)),
                y="Value:Q",
                tooltip=["Stage","Value"]
            ).properties(height=150, title="Power Estimates (kW)")
            st.altair_chart(ch, use_container_width=True)

            st.subheader("Energy Detail")
            st.table({
                "Parameter": ["Rotational Speed","Torque","Est. Mech. Power","Efficiency (η)","Est. Elec. Power"],
                "Value": [f"{est.rotational_speed_rpm:.1f} RPM",f"{est.torque_nm:.2f} Nm",
                          f"{est.estimated_mechanical_power_kw:.4f} kW",f"{est.assumptions.efficiency*100:.1f}%",
                          f"{est.estimated_electrical_power_kw:.4f} kW"],
            })
            st.subheader("Carbon Detail")
            st.table({
                "Parameter": ["Est. Energy","Emission Factor","Est. CO₂e"],
                "Value": [f"{est.estimated_energy_kwh:.4f} kWh",
                          f"{est.assumptions.emission_factor_kg_co2e_per_kwh} kg CO₂e/kWh",
                          f"{est.estimated_co2e_kg:.4f} kg CO₂e"],
            })
            st.warning(est.assumptions.disclaimer, icon="ℹ️")
            with st.expander("Formulas"):
                st.code("\n".join([
                    est.assumptions.formula_mechanical_power,
                    est.assumptions.formula_electrical_power,
                    est.assumptions.formula_energy,
                    est.assumptions.formula_co2e,
                ]))
            if st.button("🗑️ Clear"): del st.session_state["s_est"]; st.rerun()
        else:
            st.markdown('<div style="border:2px dashed #d1d5db;border-radius:8px;padding:48px;text-align:center;color:#9ca3af">Set parameters and click <strong>Calculate Estimates</strong></div>', unsafe_allow_html=True)

    st.warning("Engineering estimates only. Not for energy audits, carbon accounting, or regulatory submissions.", icon="⚠️")

# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📡 Simulation":
    st.title("📡 Live Simulation")
    st.caption("`SIMULATION · SYNTHETIC DATA`")

    sim = get_simulation_engine()
    SENSOR_FIELDS = [
        ("Air temperature [K]",    "Air Temp",    "K"),
        ("Process temperature [K]","Process Temp","K"),
        ("Rotational speed [rpm]", "Speed",       "rpm"),
        ("Torque [Nm]",            "Torque",      "Nm"),
        ("Tool wear [min]",        "Tool Wear",   "min"),
    ]
    state = sim.state.value
    STATUS = {"idle":"🔘 IDLE","running":"🟢 RUNNING","paused":"🟡 PAUSED","finished":"🔵 COMPLETE"}
    st.markdown(f"**{STATUS.get(state,'UNKNOWN')}** · *Synthetic AI4I 2020 Data*")

    sp_col,b1,b2,b3,_ = st.columns([2,1,1,1,3])
    speed = sp_col.selectbox("Speed",["slow","normal","fast"],index=1,label_visibility="collapsed")

    with b1:
        if state in ("idle","finished"):
            if st.button("▶️ Start" if state=="idle" else "🔄 Restart", use_container_width=True, type="primary"):
                if state=="finished": sim.reset()
                try: sim.start(speed)
                except FileNotFoundError: st.error("Place ai4i2020.csv in data/raw/")
                except Exception as e: st.error(str(e))
                st.rerun()
        elif state=="running":
            if st.button("⏸️ Pause", use_container_width=True): sim.pause(); st.rerun()
        else:
            if st.button("▶️ Resume", use_container_width=True, type="primary"): sim.resume(); st.rerun()
    with b2:
        if st.button("⏭️ Step", use_container_width=True):
            try:
                if state=="idle": sim.start(speed)
                sim.step()
            except IndexError: st.warning("End of dataset. Reset to restart.")
            except FileNotFoundError: st.error("Dataset not found.")
            except Exception as e: st.error(str(e))
            st.rerun()
    with b3:
        if st.button("↩️ Reset", use_container_width=True): sim.reset(); st.rerun()

    total   = sim.total_steps
    current = sim.current_step
    if total > 0:
        st.progress(min(1.0, current/total), text=f"Step {current} / {total} — {current/total*100:.1f}%")

    # Auto-advance
    if state == "running":
        DELAY = {"slow":2.0,"normal":0.8,"fast":0.2}
        try: sim.step()
        except (IndexError, RuntimeError): pass
        time.sleep(DELAY.get(speed,0.8))
        st.rerun()

    evt = sim.current_event
    if evt:
        sv   = evt.sensor_values
        rLvl = evt.risk_level
        st.markdown(
            f'<div style="background:{RISK_BG.get(rLvl,"#f3f4f6")};border-radius:8px;padding:12px 16px;margin:8px 0;display:flex;justify-content:space-between;align-items:center">'
            f'<div><p style="font-size:11px;color:#888;margin:0">MACHINE</p>'
            f'<p style="font-size:18px;font-weight:700;margin:2px 0">{evt.machine_identifier}</p>'
            f'<p style="font-size:12px;color:#666;margin:0">Type: {sv.get("Type","?")} · {evt.simulated_time_label}</p></div>'
            f'<span style="background:{RISK_COLOR.get(rLvl)};color:white;padding:6px 16px;border-radius:6px;font-weight:700">{rLvl}</span></div>',
            unsafe_allow_html=True)

        cl,cr = st.columns(2)
        with cl:
            st.markdown("**Sensor Values**")
            st.table(pd.DataFrame(
                [{"Sensor":label, "Value":f"{sv.get(key,0):.2f} {unit}"} for key,label,unit in SENSOR_FIELDS]
                + [{"Sensor":"Type","Value":str(sv.get("Type","—"))}]
            ))
        with cr:
            st.markdown("**AI Assessment**")
            st.table(pd.DataFrame([
                {"Metric":"Risk Level",          "Value": rl(rLvl)},
                {"Metric":"Failure Probability", "Value": fp(evt.failure_probability)},
                {"Metric":"Risk Score",          "Value": fs(evt.risk_score)},
                {"Metric":"Anomaly Score",       "Value": fs(evt.anomaly_score)},
                {"Metric":"Anomaly Detected",    "Value": "⚠️ Yes" if evt.is_anomaly else "✅ No"},
                {"Metric":"Predicted Failure",   "Value": "⚠️ Yes" if evt.predicted_failure else "No"},
                {"Metric":"Model",               "Value": f"{evt.model_version} · Thr: {evt.failure_threshold:.3f}"},
            ]))

        alerts = sim.alerts
        st.markdown("**🚨 Alert Panel**")
        if not alerts:
            st.caption("No alerts yet.")
        else:
            for a in list(reversed(alerts))[:8]:
                icon = "🔴" if a.alert_type=="RISK_CRITICAL" else "🟡" if a.alert_type=="RISK_WARNING" else "🟠"
                st.markdown(f'<div style="padding:8px 12px;border-radius:6px;background:#fef9c3;border-left:3px solid #d97706;margin-bottom:4px;font-size:13px">{icon} <strong>Step {a.step}:</strong> {a.message}</div>', unsafe_allow_html=True)
            st.caption(f"{len(alerts)} total alerts")

        hsim = sim.history
        if hsim:
            st.markdown("**Recent Event History**")

            # Mini trend chart from history
            df_hist = pd.DataFrame([{
                "Step": e.simulation_step,
                "Risk Score": e.risk_score,
                "P(Failure)": e.failure_probability,
            } for e in hsim[-50:]])
            base = alt.Chart(df_hist).encode(x="Step:Q")
            l1 = base.mark_line(color="#3b82f6",strokeWidth=1.5).encode(y="Risk Score:Q", tooltip=["Step","Risk Score"])
            l2 = base.mark_line(color="#ef4444",strokeWidth=1.5,strokeDash=[3,2]).encode(y="P(Failure):Q", tooltip=["Step","P(Failure)"])
            st.altair_chart((l1+l2).properties(height=120, title="Risk Score & P(Failure) over last 50 steps"), use_container_width=True)

            st.dataframe(pd.DataFrame([{
                "Step": e.simulation_step,
                "Risk": rl(e.risk_level),
                "P(Failure)": fp(e.failure_probability),
                "Risk Score": fs(e.risk_score),
                "Anomaly": "⚠️ Yes" if e.is_anomaly else "No",
                "Torque": f"{e.sensor_values.get('Torque [Nm]',0):.1f} Nm",
                "Wear": f"{e.sensor_values.get('Tool wear [min]',0):.0f} min",
            } for e in list(reversed(hsim))[:20]]), use_container_width=True, hide_index=True)

    elif state == "idle":
        st.markdown('<div style="border:2px dashed #d1d5db;border-radius:8px;padding:48px;text-align:center"><p style="font-size:32px">📡</p><p style="font-weight:600;color:#6b7280">Simulation not started</p><p style="color:#9ca3af">Press <strong>Start</strong> to step through the AI4I 2020 synthetic dataset.</p></div>', unsafe_allow_html=True)

    st.info("SIMULATION DISCLOSURE: Synthetic historical data — UCI AI4I 2020 (Matzka, 2021, CC BY 4.0). Not a live sensor connection.", icon="ℹ️")
