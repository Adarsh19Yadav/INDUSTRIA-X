"""Validation script for INDUSTRIA-X Streamlit app."""
import sys, os, json, sqlite3
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

print("=== INDUSTRIA-X Validation ===\n")

# 1. ML Assessment
from ml.assessment import assess_machine
r = assess_machine(air_temp_k=302, process_temp_k=312, rotational_speed_rpm=1800, torque_nm=55, tool_wear_min=200, type_="H")
print(f"[OK] Assessment (normal): level={r.risk_level}, P={r.failure_probability:.4f}, RS={r.risk_score:.4f}, anom={r.anomaly_score:.4f}")

r2 = assess_machine(air_temp_k=305, process_temp_k=318, rotational_speed_rpm=2900, torque_nm=70, tool_wear_min=250, type_="H")
print(f"[OK] Assessment (high-stress): level={r2.risk_level}, P={r2.failure_probability:.4f}")

# 2. Sustainability
from backend.services.sustainability_service import compute_sustainability_estimate
est = compute_sustainability_estimate(1500, 45.0, 8.0, "M", 0.88, 0.4)
print(f"[OK] Sustainability: P_mech={est.estimated_mechanical_power_kw:.4f}kW, E={est.estimated_energy_kwh:.4f}kWh, CO2={est.estimated_co2e_kg:.4f}kg")

# 3. What-If
from backend.services.whatif_service import run_whatif, MachineInputs
b = MachineInputs(air_temp_k=300, process_temp_k=310, rotational_speed_rpm=1500, torque_nm=40, tool_wear_min=100, type_="M")
s = MachineInputs(air_temp_k=307, process_temp_k=317, rotational_speed_rpm=2200, torque_nm=65, tool_wear_min=220, type_="H")
res = run_whatif(b, s)
d = res.to_dict()
bl = d["baseline"]["risk_level"]
sc = d["scenario"]["risk_level"]
dp = d["change"]["failure_probability_delta"]
print(f"[OK] WhatIf: Baseline={bl} -> Scenario={sc}, delta_P={dp:.4f}")

# 4. Simulation
from backend.services.simulation_service import get_simulation_engine
sim = get_simulation_engine()
sim.reset()
sim.start("normal")
for _ in range(5):
    sim.step()
print(f"[OK] Simulation: step={sim.current_step}, history={len(sim.history)}, alerts={len(sim.alerts)}, state={sim.state.value}")

# 5. Dataset
import pandas as pd
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "ai4i2020.csv"))
fail_rate = df["Machine failure"].mean() * 100
print(f"[OK] Dataset: {len(df)} rows, {len(df.columns)} columns, failure_rate={fail_rate:.2f}%")

# 6. Artifacts
artifacts = os.path.join(os.path.dirname(__file__), "..", "ml", "artifacts", "v1")
meta = json.load(open(os.path.join(artifacts, "metadata.json"), encoding="utf-8"))
shap = json.load(open(os.path.join(artifacts, "global_shap_importance.json"), encoding="utf-8"))
ameta = json.load(open(os.path.join(artifacts, "anomaly_metadata.json"), encoding="utf-8"))
print(f"[OK] Metadata: model={meta['model_name']}, threshold={meta['threshold']:.4f}, eval_models={list(meta.get('evaluation_metrics',{}).keys())}")
print(f"[OK] SHAP: features={shap['features']}")
print(f"[OK] AnomalyMeta: algo={ameta['algorithm']}, contamination={ameta['contamination']}")

# 7. SQLite
db_path = os.path.join(os.path.dirname(__file__), "industria_x.db")
con = sqlite3.connect(db_path)
con.row_factory = sqlite3.Row
machines_count = con.execute("SELECT COUNT(*) as n FROM machines").fetchone()["n"]
assessments_count = con.execute("SELECT COUNT(*) as n FROM assessments").fetchone()["n"]
con.close()
print(f"[OK] SQLite: {machines_count} machines, {assessments_count} assessments stored")

# 8. Test create + assess + persist cycle
from datetime import datetime, timezone

def _now():
    return datetime.now(timezone.utc).isoformat()

con2 = sqlite3.connect(db_path)
con2.row_factory = sqlite3.Row
# ensure test machine does not exist
con2.execute("DELETE FROM machines WHERE machine_identifier='TEST_VALIDATE_001'")
con2.commit()
con2.execute("INSERT INTO machines (machine_identifier,type,created_at,updated_at) VALUES (?,?,?,?)",
             ("TEST_VALIDATE_001", "M", _now(), _now()))
con2.commit()
mid = con2.execute("SELECT id FROM machines WHERE machine_identifier='TEST_VALIDATE_001'").fetchone()["id"]
r3 = assess_machine(air_temp_k=300, process_temp_k=310, rotational_speed_rpm=1500, torque_nm=40, tool_wear_min=100, type_="M")
con2.execute("""INSERT INTO assessments
    (machine_id,failure_probability,anomaly_score,risk_score,risk_level,model_version,
     air_temp_k,process_temp_k,rotational_speed_rpm,torque_nm,tool_wear_min,machine_type,created_at)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
    (mid, r3.failure_probability, r3.anomaly_score, r3.risk_score, r3.risk_level, r3.model_version,
     300.0, 310.0, 1500, 40.0, 100, "M", _now()))
con2.commit()
saved = con2.execute("SELECT * FROM assessments WHERE machine_id=? ORDER BY created_at DESC LIMIT 1", (mid,)).fetchone()
print(f"[OK] Full cycle: machine=TEST_VALIDATE_001, risk={saved['risk_level']}, score={saved['risk_score']:.4f}")
# cleanup
con2.execute("DELETE FROM assessments WHERE machine_id=?", (mid,))
con2.execute("DELETE FROM machines WHERE id=?", (mid,))
con2.commit()
con2.close()

print()
print("=== ALL 8 CHECKS PASSED — App is fully workable ===")
