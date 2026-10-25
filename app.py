# app.py
import streamlit as st
import pandas as pd
from database_manager import MongoRegistry, MySQLVault

st.set_page_config(page_title="Smart MLOps - German Credit", layout="wide")
st.title("Smart MLOps: Dual-Database Model Gatekeeper")
st.caption("Domain: German Credit Risk | SIT DCDS Lab Project")

try:
    mongo = MongoRegistry()
    sql = MySQLVault()
except Exception as e:
    st.error(f"Database Connection Error: {e}. Make sure MySQL and MongoDB are running!")
    st.stop()

# KPI Header
active_prod = sql.get_active_production_model()
k1, k2, k3 = st.columns(3)
k1.metric("Active Prod Model", active_prod["model_name"] if active_prod else "None")
k2.metric("Production Version", active_prod["version_tag"] if active_prod else "N/A")
k3.metric("Prod Accuracy Baseline", f"{active_prod['accuracy']:.4f}" if active_prod else "N/A")

st.divider()

col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("1. Experiment Telemetry (MongoDB Aggregations)")
    tab_lead, tab_devs = st.tabs(["Leaderboard Pipeline", "Developer Productivity Pipeline"])

    with tab_lead:
        top_runs = mongo.get_leaderboard(limit=8)
        if top_runs:
            st.dataframe(pd.DataFrame(top_runs), use_container_width=True)
        else:
            st.warning("No runs found in MongoDB. Run your training script first to populate experiments.")

    with tab_devs:
        dev_stats = mongo.get_developer_stats()
        if dev_stats:
            st.dataframe(pd.DataFrame(dev_stats), use_container_width=True)

with col_right:
    st.subheader("2. Release Gatekeeper (MySQL)")
    if 'top_runs' in locals() and top_runs:
        run_options = [f"{r['run_id']} - {r['model']} (Acc: {r['accuracy']})" for r in top_runs]
        selected = st.selectbox("Select Candidate Run", run_options)
        cand = top_runs[run_options.index(selected)]

        version_input = st.text_input("New Release Version Tag", value="v2.0")

        if st.button("Evaluate & Promote Model", type="primary"):
            success, msg = sql.execute_promotion(
                "German_Credit_Classifier", version_input, float(cand["accuracy"]), float(cand["f1_score"]), cand["developer"]
            )
            if success:
                st.success(msg)
                st.rerun()
            else:
                st.error(f"Deployment Blocked by MySQL Gatekeeper: {msg}")
    else:
        st.info("Populate MongoDB experiments to enable model promotion.")

    st.divider()
    st.subheader("Production Audit History (MySQL INNER JOIN)")
    audit_data = sql.get_release_audit_trail()
    if audit_data:
        st.dataframe(pd.DataFrame(audit_data), use_container_width=True)
