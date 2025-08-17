# app.py
# Deps: streamlit, pandas, pyyaml
# Free-host friendly: no heavy libs, no background workers, simple file I/O in /tmp
# Behavior:
#  - Multi-cloud simulation (AWS/GCP/Azure)
#  - 20 CI-CT-CD steps with real-time logs
#  - 8 demo projects with truthful, scenario-specific failure rooted in generated assets
#  - FMEA details + "Apply Fix & Retry" (max 2 debug attempts) that actually edits assets

import os
import io
import json
import time
import shutil
import random
import textwrap
import streamlit as st
import pandas as pd
import yaml

# -------------------------------
# App Config / Styling
# -------------------------------
st.set_page_config(page_title="Multi-Cloud CI/CT/CD Simulator", layout="wide", page_icon="🚀")
st.markdown("""
<style>
.small {font-size:0.9rem; color:#666;}
.codechip {display:inline-block;padding:2px 8px;border-radius:6px;background:#eef;border:1px solid #ccd;margin-right:6px}
.ok {color:#0a0}
.warn {color:#b8860b}
.err {color:#b00}
.stepcard {padding:10px;border:1px solid #eee;border-radius:12px;margin-bottom:8px;background:#fafafa}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Helper: session init
# -------------------------------
def init_state():
    ss = st.session_state
    ss.setdefault("cloud", "AWS")
    ss.setdefault("demo_key", None)
    ss.setdefault("assets_dir", None)
    ss.setdefault("debug_attempts", 0)          # <= limit 2
    ss.setdefault("last_run_logs", [])
    ss.setdefault("last_failure", None)
    ss.setdefault("pipeline_completed", False)
    ss.setdefault("running", False)

init_state()

# -------------------------------
# Cloud service glossaries (for authenticity hints)
# -------------------------------
CLOUD_MAP = {
    "AWS": {
        1:"GitHub/CodeCommit", 2:"CodeBuild (static)", 3:"CodeBuild (deps)", 4:"PyTest in CodeBuild",
        5:"CodeBuild (integration)", 6:"Docker build (CodeBuild)", 7:"ECR push", 8:"CloudFormation/Terraform validate",
        9:"Inspector/CodeGuru Security", 10:"Provision test env (CloudFormation)",
        11:"Deploy to test (ECS/EKS)", 12:"Smoke tests (Canaries)",
        13:"Deploy staging (ECS/EKS)", 14:"Load test (CloudWatch Synthetics)",
        15:"Pre-prod deploy", 16:"UAT (AppConfig feature flags)", 17:"Approval (CodePipeline Manual)",
        18:"Prod deploy (Blue/Green)", 19:"Monitoring (CloudWatch/Prometheus)", 20:"CT retrain (SageMaker Pipelines)"
    },
    "GCP": {
        1:"GitHub/Cloud Source Repos", 2:"Cloud Build (static)", 3:"Cloud Build (deps)", 4:"PyTest in Cloud Build",
        5:"Cloud Build (integration)", 6:"Docker build (Cloud Build)", 7:"Artifact Registry push", 8:"Terraform/DM validate",
        9:"Security Command Center", 10:"Provision test env (Terraform)",
        11:"Deploy to test (GKE)", 12:"Smoke tests (Cloud Run Jobs)",
        13:"Deploy staging (GKE)", 14:"Load test (Cloud Monitoring SLOs)",
        15:"Pre-prod deploy", 16:"UAT (Launch Config)", 17:"Approval (Cloud Deploy)",
        18:"Prod deploy (Blue/Green)", 19:"Monitoring (Cloud Monitoring)", 20:"CT retrain (Vertex AI Pipelines)"
    },
    "Azure": {
        1:"GitHub/Azure Repos", 2:"Azure Pipelines (static)", 3:"Pipelines (deps)", 4:"PyTest in Pipelines",
        5:"Pipelines (integration)", 6:"Docker build (Pipelines)", 7:"ACR push", 8:"ARM/Bicep/Terraform validate",
        9:"Defender for Cloud", 10:"Provision test env (Bicep/Terraform)",
        11:"Deploy to test (AKS)", 12:"Smoke tests (Playwright/Func Tests)",
        13:"Deploy staging (AKS)", 14:"Load test (Azure Load Testing)",
        15:"Pre-prod deploy", 16:"UAT (Feature Management)", 17:"Approval (Environments)",
        18:"Prod deploy (Blue/Green)", 19:"Monitoring (Azure Monitor)", 20:"CT retrain (ML Pipelines)"
    }
}

# -------------------------------
# The 20 pipeline steps
# -------------------------------
STEPS = [
    "Code Commit",
    "Static Code Analysis",
    "Dependency Check",
    "Unit Tests",
    "Integration Tests",
    "Build Docker Image",
    "Push to Container Registry",
    "IaC Validation",
    "Security Scan",
    "Provision Test Environment",
    "Deploy to Test",
    "Smoke Tests",
    "Deploy to Staging",
    "Load Tests",
    "Deploy to Pre-Prod",
    "User Acceptance Testing",
    "Approval Gate",
    "Deploy to Production",
    "Post-Deploy Monitoring",
    "Feedback Loop & Continuous Training",
]

# -------------------------------
# 8 Demo Scenarios with real root-cause & mitigation
# Each scenario will generate assets in /tmp and we verify truthfully at its fail step.
# -------------------------------
SCENARIOS = {
    "Demo 1 · Linear Regression (dep missing)": {
        "type": "missing_dependency",
        "failure_step": 3,
        "fmea": {
            "mode": "ImportError: numpy not installed",
            "effect": "Build cannot run unit tests; pipeline blocked",
            "cause": "requirements.txt lacks numpy though code imports it",
            "detection": "Dependency audit at step 3",
            "mitigation_hint": "Add `numpy` to requirements.txt",
            "mitigation_fix": "Appended `numpy` to requirements.txt and reinstalled deps"
        }
    },
    "Demo 2 · Image Classifier (Dockerfile syntax)": {
        "type": "dockerfile_syntax",
        "failure_step": 6,
        "fmea": {
            "mode": "Dockerfile invalid: `COPY.` missing space",
            "effect": "Image can't be built",
            "cause": "Syntax error in Dockerfile",
            "detection": "Docker build at step 6",
            "mitigation_hint": "Replace `COPY.` with `COPY . /app`",
            "mitigation_fix": "Fixed Dockerfile COPY directive"
        }
    },
    "Demo 3 · NLP Sentiment (env credentials missing)": {
        "type": "env_missing",
        "failure_step": 10,
        "fmea": {
            "mode": "Missing DB_USER/DB_PASS/DB_HOST in .env",
            "effect": "Test environment provision fails",
            "cause": "Incomplete secrets/config",
            "detection": "Provision step 10",
            "mitigation_hint": "Populate required keys in .env",
            "mitigation_fix": "Wrote DB_USER/DB_PASS/DB_HOST to .env (placeholder values)"
        }
    },
    "Demo 4 · Fraud Detection (nulls in data)": {
        "type": "null_values",
        "failure_step": 12,
        "fmea": {
            "mode": "Smoke test fails due to NaNs",
            "effect": "API /predict crashes with null input",
            "cause": "Dataset contains missing values",
            "detection": "Smoke tests step 12",
            "mitigation_hint": "Impute/drop NaNs in CSV",
            "mitigation_fix": "Filled NaNs with column means"
        }
    },
    "Demo 5 · Recommender (high latency in profile)": {
        "type": "high_latency",
        "failure_step": 15,
        "fmea": {
            "mode": "p90 latency > SLO",
            "effect": "Pre-prod gate blocks release",
            "cause": "Inefficient queries / insufficient resources",
            "detection": "Load tests step 15",
            "mitigation_hint": "Tune or scale; set p90 <= 1000ms",
            "mitigation_fix": "Reduced p90 to 400ms in profile (proxy for tuning)"
        }
    },
    "Demo 6 · Forecasting (public bucket policy)": {
        "type": "public_bucket",
        "failure_step": 17,
        "fmea": {
            "mode": "S3/Bucket allows public write",
            "effect": "Security approval rejects",
            "cause": "Over-permissive ACL",
            "detection": "Approval gate step 17",
            "mitigation_hint": "Block public access",
            "mitigation_fix": "Set BlockPublicAcls=True / removed public write"
        }
    },
    "Demo 7 · Speech-to-Text (memory leak)": {
        "type": "memory_leak",
        "failure_step": 19,
        "fmea": {
            "mode": "Growing list in inference loop",
            "effect": "Pod OOM post-deploy",
            "cause": "Objects retained across calls",
            "detection": "Monitoring step 19",
            "mitigation_hint": "Free buffers; avoid global accumulation",
            "mitigation_fix": "Disabled leak flag; cleared buffers"
        }
    },
    "Demo 8 · Anomaly Detection (schema drift)": {
        "type": "schema_drift",
        "failure_step": 20,
        "fmea": {
            "mode": "Column `target` missing (found `label`)",
            "effect": "CT retrain pipeline fails",
            "cause": "Upstream ETL changed schema",
            "detection": "CT step 20",
            "mitigation_hint": "Map `label` → `target` before training",
            "mitigation_fix": "Created `target` column from `label`"
        }
    },
}

# -------------------------------
# Asset generation per scenario (truthful files in /tmp)
# -------------------------------
def write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def generate_assets(base_dir, scenario_key):
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    sc = SCENARIOS[scenario_key]
    s_type = sc["type"]

    # Common minimal app.py (not executed, just for realism)
    write(os.path.join(base_dir, "app.py"), textwrap.dedent(f"""
    # minimal app entrypoint for {scenario_key}
    def predict(x): 
        return x
    """).strip())

    # Scenario-specific assets
    if s_type == "missing_dependency":
        write(os.path.join(base_dir, "model.py"), "import numpy as np\n\nDEF_FLAG=1\n")
        write(os.path.join(base_dir, "requirements.txt"), "pandas==2.2.2\n# numpy intentionally missing\n")

    elif s_type == "dockerfile_syntax":
        write(os.path.join(base_dir, "Dockerfile"), "FROM python:3.11-slim\nWORKDIR /app\nCOPY.\nRUN pip install -r requirements.txt\n")
        write(os.path.join(base_dir, "requirements.txt"), "pandas==2.2.2\n")

    elif s_type == "env_missing":
        write(os.path.join(base_dir, ".env"), "DB_USER=\n# DB_PASS missing\nDB_HOST=\n")
        write(os.path.join(base_dir, "provision.yaml"), "service: nlp-sentiment\nresources: test-db\n")

    elif s_type == "null_values":
        df = pd.DataFrame({"amount":[100.0, None, 45.0, None], "merchant":["a","b","c","d"]})
        df.to_csv(os.path.join(base_dir, "data.csv"), index=False)

    elif s_type == "high_latency":
        write(os.path.join(base_dir, "endpoint_profile.json"), json.dumps({"p90_ms": 1400, "notes":"needs tuning"}, indent=2))

    elif s_type == "public_bucket":
        policy = {"bucket": "my-forecast-bucket", "public_write": True}
        write(os.path.join(base_dir, "iac.yaml"), yaml.safe_dump(policy, sort_keys=False))

    elif s_type == "memory_leak":
        leak_code = """
buffer = []
LEAK_DEMO = True
def infer(x):
    global buffer
    if LEAK_DEMO:
        buffer.append(bytes(1024*256))  # grow 256KB per call
    return x
"""
        write(os.path.join(base_dir, "model.py"), leak_code)

    elif s_type == "schema_drift":
        df = pd.DataFrame({"feature1":[1,2,3], "feature2":[0.1,0.2,0.3], "label":[0,1,0]})  # no 'target'
        df.to_csv(os.path.join(base_dir, "training.csv"), index=False)

    return base_dir

# -------------------------------
# Checks (truthful verification at the failing step)
# -------------------------------
def check_step_truth(base_dir, scenario_key, step_idx):
    sc = SCENARIOS[scenario_key]
    s_type = sc["type"]
    fail_step = sc["failure_step"]

    # Only enforce truth check at/after the defined fail step.
    if step_idx < fail_step:
        return True, "OK"

    # Evaluate problem & whether it's fixed already.
    try:
        if s_type == "missing_dependency" and step_idx == fail_step:
            req = open(os.path.join(base_dir, "requirements.txt"), encoding="utf-8").read()
            code = open(os.path.join(base_dir, "model.py"), encoding="utf-8").read()
            imports_numpy = "numpy" in code
            declared_numpy = "numpy" in req
            return (declared_numpy or not imports_numpy), ("numpy present" if declared_numpy else "numpy missing")

        if s_type == "dockerfile_syntax" and step_idx == fail_step:
            content = open(os.path.join(base_dir, "Dockerfile"), encoding="utf-8").read()
            invalid = "COPY." in content
            return (not invalid), ("Dockerfile COPY valid" if not invalid else "Invalid COPY directive")

        if s_type == "env_missing" and step_idx == fail_step:
            env = open(os.path.join(base_dir, ".env"), encoding="utf-8").read()
            has_user = "DB_USER=" in env and len(env.split("DB_USER=")[1].strip())>0
            has_pass = "DB_PASS=" in env and len(env.split("DB_PASS=")[1].strip())>0
            has_host = "DB_HOST=" in env and len(env.split("DB_HOST=")[1].strip())>0
            ok = has_user and has_pass and has_host
            return ok, ("env ok" if ok else "env keys missing")

        if s_type == "null_values" and step_idx == fail_step:
            df = pd.read_csv(os.path.join(base_dir, "data.csv"))
            ok = not df.isna().any().any()
            return ok, ("no NaNs" if ok else "NaNs present")

        if s_type == "high_latency" and step_idx == fail_step:
            prof = json.load(open(os.path.join(base_dir, "endpoint_profile.json")))
            return (prof.get("p90_ms", 9999) <= 1000), f"p90={prof.get('p90_ms')}ms"

        if s_type == "public_bucket" and step_idx == fail_step:
            policy = yaml.safe_load(open(os.path.join(base_dir, "iac.yaml")))
            return (not policy.get("public_write", False)), ("public_write=false" if not policy.get("public_write", False) else "public_write=true")

        if s_type == "memory_leak" and step_idx == fail_step:
            code = open(os.path.join(base_dir, "model.py"), encoding="utf-8").read()
            leaking = "LEAK_DEMO = True" in code
            return (not leaking), ("no leak" if not leaking else "leak flag true")

        if s_type == "schema_drift" and step_idx == fail_step:
            df = pd.read_csv(os.path.join(base_dir, "training.csv"))
            ok = "target" in df.columns
            return ok, ("target present" if ok else "target missing")

        # If other step beyond fail step, propagate previous result
        return True, "OK"
    except Exception as e:
        return False, f"check error: {e}"

# -------------------------------
# Apply mitigation (edits asset to fix real issue)
# -------------------------------
def apply_mitigation(base_dir, scenario_key):
    sc = SCENARIOS[scenario_key]
    s_type = sc["type"]

    try:
        if s_type == "missing_dependency":
            path = os.path.join(base_dir, "requirements.txt")
            with open(path, "a", encoding="utf-8") as f:
                f.write("\nnumpy==1.26.4\n")

        elif s_type == "dockerfile_syntax":
            p = os.path.join(base_dir, "Dockerfile")
            txt = open(p, encoding="utf-8").read().replace("COPY.", "COPY . /app\n")
            write(p, txt)

        elif s_type == "env_missing":
            p = os.path.join(base_dir, ".env")
            write(p, "DB_USER=demo\nDB_PASS=demo123\nDB_HOST=localhost\n")

        elif s_type == "null_values":
            p = os.path.join(base_dir, "data.csv")
            df = pd.read_csv(p)
            for col in df.columns:
                if df[col].dtype.kind in "biufc":
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = df[col].fillna(method="ffill").fillna(method="bfill")
            df.to_csv(p, index=False)

        elif s_type == "high_latency":
            p = os.path.join(base_dir, "endpoint_profile.json")
            prof = json.load(open(p))
            prof["p90_ms"] = 400
            write(p, json.dumps(prof, indent=2))

        elif s_type == "public_bucket":
            p = os.path.join(base_dir, "iac.yaml")
            policy = yaml.safe_load(open(p))
            policy["public_write"] = False
            write(p, yaml.safe_dump(policy, sort_keys=False))

        elif s_type == "memory_leak":
            p = os.path.join(base_dir, "model.py")
            txt = open(p, encoding="utf-8").read().replace("LEAK_DEMO = True", "LEAK_DEMO = False\n# fixed")
            write(p, txt)

        elif s_type == "schema_drift":
            p = os.path.join(base_dir, "training.csv")
            df = pd.read_csv(p)
            if "label" in df.columns and "target" not in df.columns:
                df["target"] = df["label"]
            elif "target" not in df.columns:
                df["target"] = 0
            df.to_csv(p, index=False)

        return True, sc["fmea"]["mitigation_fix"]
    except Exception as e:
        return False, f"mitigation error: {e}"

# -------------------------------
# UI: Header / Why / USP / UVP
# -------------------------------
st.title("🚀 Multi-Cloud CI/CT/CD Flight Simulator")
st.markdown("""
**Why use this app?** Train engineers, de-risk releases, and explain pipelines to stakeholders **without touching real cloud**.  
**USP:** Cross-cloud visual simulator with **authentic failures** and **FMEA + guided fixes**.  
**UVP:** Learn and practice the exact steps you’ll run on **AWS / GCP / Azure**, safely and cheaply.
""")

# -------------------------------
# Controls
# -------------------------------
colA, colB, colC = st.columns([1,2,1])
with colA:
    st.subheader("1) Cloud")
    st.session_state.cloud = st.selectbox("Choose Provider", ["AWS","GCP","Azure"], index=["AWS","GCP","Azure"].index(st.session_state.cloud))
with colB:
    st.subheader("2) Pick a Demo Project")
    demo_key = st.radio(
        "Each demo fails truthfully at a different step. Pick one:",
        list(SCENARIOS.keys()),
        index=0 if st.session_state.demo_key is None else list(SCENARIOS.keys()).index(st.session_state.demo_key)
    )
    st.session_state.demo_key = demo_key
with colC:
    st.subheader("3) Run Controls")
    run_btn = st.button("▶️ Run 20-Step Pipeline", use_container_width=True)
    fix_btn = st.button("🛠 Apply Fix & Retry (max 2)", use_container_width=True)

# Prepare assets dir for the selected scenario
assets_dir = os.path.join("/tmp", "ci_ct_cd_sim", demo_key.replace(" ","_").replace("·","_"))
st.session_state.assets_dir = assets_dir

# Handle Apply Fix
if fix_btn:
    if st.session_state.last_failure is None:
        st.warning("No failure detected yet. Run the pipeline first.")
    elif st.session_state.debug_attempts >= 2:
        st.error("Debug attempts exhausted (max 2). Reset by choosing another demo.")
    else:
        ok, msg = apply_mitigation(assets_dir, demo_key)
        if ok:
            st.session_state.debug_attempts += 1
            st.success(f"Mitigation applied: {msg}  • Attempts used: {st.session_state.debug_attempts}/2")
            # Immediately re-run pipeline after fix
            run_btn = True
        else:
            st.error(msg)

# -------------------------------
# Generate assets on first run or when demo changes
# -------------------------------
if (not os.path.exists(assets_dir)) or (st.session_state.last_run_logs == []) or run_btn:
    generate_assets(assets_dir, demo_key)

# -------------------------------
# Run pipeline when requested
# -------------------------------
if run_btn:
    st.session_state.running = True
    st.session_state.last_run_logs = []
    st.session_state.pipeline_completed = False
    st.session_state.last_failure = None

    # Guidance panel
    with st.expander("ℹ️ How to use this screen", expanded=True):
        st.markdown("""
- **Pick a cloud** (left) → the step badges will show the matching managed service.  
- **Pick a demo** (middle) → each is wired to fail at a different step for realism.  
- Click **Run** → watch steps execute; if it fails, you'll see **FMEA + mitigation**.  
- Click **Apply Fix & Retry** → the app **edits the problematic file** and runs again. (Max **2** attempts)  
- Use this to **teach, demo, and practice** CI-CT-CD without real cloud spend.
""")

    # Streaming log area
    log_placeholder = st.empty()
    progress = st.progress(0.0)

    # Step-by-step execution
    for idx, step in enumerate(STEPS, start=1):
        cloud_service = CLOUD_MAP[st.session_state.cloud].get(idx, "")
        header = f"Step {idx:02d} · {step}  —  <span class='small'>{cloud_service}</span>"
        st.markdown(f"<div class='stepcard'><b>{header}</b>", unsafe_allow_html=True)

        # Educational micro copy
        st.markdown(f"<span class='small'>Why:</span> Ensures quality before moving forward. "
                    f"<span class='small'>What to do here:</span> Follow logs; if this step fails, read FMEA and apply mitigation.", unsafe_allow_html=True)

        # Truth check at failure step
        ok, detail = check_step_truth(assets_dir, demo_key, idx)

        if ok:
            msg = f"✅ {step}: OK ({detail})"
            st.markdown(msg)
            st.markdown("</div>", unsafe_allow_html=True)
            st.session_state.last_run_logs.append(msg)
            progress.progress(idx/20)
            log_placeholder.code("\n".join(st.session_state.last_run_logs))
            time.sleep(0.15)  # gentle pacing for UX
        else:
            # Failure: show FMEA + mitigation path
            fmea = SCENARIOS[demo_key]["fmea"]
            st.markdown(f"**<span class='err'>❌ FAILED</span> — {detail}**", unsafe_allow_html=True)
            st.error(f"FMEA • Failure Mode: {fmea['mode']}\n\nEffect: {fmea['effect']}\n\nLikely Cause: {fmea['cause']}\n\nDetection Point: {fmea['detection']}")
            if st.session_state.debug_attempts < 2:
                st.warning(f"Mitigation Suggestion: {fmea['mitigation_hint']}  → Click **Apply Fix & Retry** (Attempts used: {st.session_state.debug_attempts}/2)")
            else:
                st.info(f"Max debug attempts used. Final Suggested Fix: {fmea['mitigation_fix']}  • Switch demo to reset attempts.")

            st.markdown("</div>", unsafe_allow_html=True)
            fail_line = f"❌ {step}: {detail}"
            st.session_state.last_run_logs.append(fail_line)
            log_placeholder.code("\n".join(st.session_state.last_run_logs))
            st.session_state.last_failure = {"step": idx, "detail": detail}
            st.session_state.running = False
            break

    # Completed?
    if st.session_state.last_failure is None:
        st.success("🎉 Pipeline completed successfully end-to-end!")
        st.session_state.pipeline_completed = True
        st.session_state.running = False

# -------------------------------
# Sidebar: quick reference & selling points
# -------------------------------
with st.sidebar:
    st.header("🎯 Quick Start")
    st.markdown("""
1) Choose **Cloud**  
2) Choose a **Demo**  
3) Click **Run**  
4) If it fails → click **Apply Fix & Retry** (up to 2x)
""")
    st.header("🌟 USP / UVP")
    st.markdown("""
- **USP:** Cross-cloud simulator + **authentic** failures rooted in files  
- **FMEA + Guided Mitigation**  
- **Hands-on training** without cloud spend
""")
    st.header("🏢 Enterprise Fit")
    st.markdown("""
- Team training & onboarding  
- Risk rehearsal (chaos/failure drills)  
- Multi-cloud literacy (AWS/GCP/Azure)
""")

# Footer helper
st.markdown("<div class='small'>Tip: Use different demos to experience failures at various stages (deps, Docker, env, data, latency, security, memory, schema).</div>", unsafe_allow_html=True)
