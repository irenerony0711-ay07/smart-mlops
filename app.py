import mysql.connector
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

import db_config
from nosql.mongo_queries import (
    get_average_metrics_by_model,
    get_developer_productivity,
    get_leaderboard,
    get_mongo_collection,
)

st.set_page_config(page_title="Smart MLOps - German Credit", layout="wide")


def get_mysql_conn():
    return mysql.connector.connect(
        host=db_config.MYSQL_HOST,
        user=db_config.MYSQL_USER,
        password=db_config.MYSQL_PASS,
        database=db_config.MYSQL_DB,
        autocommit=False,
    )


def _safe_close(cursor=None, conn=None):
    try:
        if cursor is not None:
            cursor.close()
    except Exception:
        pass
    try:
        if conn is not None:
            conn.close()
    except Exception:
        pass


def render_executive_overview():
    st.title("Executive Overview & System Telemetry")

    total_experiments = 0
    try:
        total_experiments = get_mongo_collection().count_documents({})
    except Exception as exc:
        st.error(f"MongoDB unavailable: {exc}")

    prod_row = None
    conn = None
    cursor = None
    try:
        conn = get_mysql_conn()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT version_tag, accuracy, f1_score, created_by, registered_at
            FROM model_versions
            WHERE is_active_prod = TRUE
            LIMIT 1;
            """
        )
        prod_row = cursor.fetchone()
    except Exception as exc:
        st.error(f"MySQL unavailable: {exc}")
    finally:
        _safe_close(cursor, conn)

    if prod_row:
        prod_version, prod_accuracy, prod_f1 = prod_row[0], prod_row[1], prod_row[2]
    else:
        prod_version, prod_accuracy, prod_f1 = "None", "N/A", "N/A"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Experiments Logged", total_experiments)
    col2.metric("Active Prod Version", prod_version)
    col3.metric(
        "Prod Accuracy",
        f"{float(prod_accuracy):.4f}" if prod_accuracy != "N/A" else "N/A",
    )
    col4.metric(
        "Prod F1-Score",
        f"{float(prod_f1):.4f}" if prod_f1 != "N/A" else "N/A",
    )

    st.divider()

    left, right = st.columns(2)
    with left:
        st.subheader("Average Accuracy by Algorithm")
        try:
            avg_df = pd.DataFrame(get_average_metrics_by_model())
            if not avg_df.empty:
                fig = px.bar(
                    avg_df,
                    x="_id",
                    y="avg_accuracy",
                    text_auto=".3f",
                    title="Average Accuracy by Algorithm",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No experiment data available yet.")
        except Exception as exc:
            st.error(f"Could not load model metrics: {exc}")

    with right:
        st.subheader("Developer Productivity")
        try:
            dev_df = pd.DataFrame(get_developer_productivity())
            if not dev_df.empty:
                st.dataframe(dev_df, use_container_width=True)
            else:
                st.warning("No developer activity recorded yet.")
        except Exception as exc:
            st.error(f"Could not load developer productivity: {exc}")


def render_experiment_leaderboard():
    st.title("Experiment Runs Leaderboard")
    st.caption("Live queries executed via MongoDB aggregation pipeline.")

    try:
        leaderboard = get_leaderboard(limit=15)
    except Exception as exc:
        st.error(f"MongoDB unavailable: {exc}")
        return

    if not leaderboard:
        st.warning("No experiment runs found. Run the training pipeline first.")
        return

    df = pd.DataFrame(leaderboard)
    columns = [
        "run_id",
        "model",
        "developer",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "timestamp",
    ]
    df = df[[c for c in columns if c in df.columns]]
    st.dataframe(df, use_container_width=True)


def render_model_evaluation():
    st.title("In-Depth Model Evaluation")

    try:
        top_runs = get_leaderboard(limit=20)
    except Exception as exc:
        st.error(f"MongoDB unavailable: {exc}")
        return

    if not top_runs:
        st.warning("No experiment runs found. Run the training pipeline first.")
        return

    run_ids = [run["run_id"] for run in top_runs]
    selected_run_id = st.selectbox("Select an experiment run", run_ids)

    selected_doc = next(
        (run for run in top_runs if run["run_id"] == selected_run_id), None
    )
    if selected_doc is None:
        st.warning("Selected run could not be found.")
        return

    model_name = selected_doc.get("model", "Unknown")
    hyperparameters = selected_doc.get("hyperparameters", {})
    accuracy = selected_doc.get("accuracy")
    f1 = selected_doc.get("f1")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader(f"Hyperparameters: {model_name}")
        st.json(hyperparameters)
        st.metric("F1-Score", f1)
        st.metric("Accuracy", accuracy)

    with col_right:
        st.subheader("Confusion Matrix")
        confusion_matrix = None
        try:
            full_doc = get_mongo_collection().find_one({"run_id": selected_run_id})
            if full_doc:
                confusion_matrix = full_doc.get("confusion_matrix")
        except Exception as exc:
            st.error(f"Could not fetch full run document: {exc}")

        if confusion_matrix:
            fig = ff.create_annotated_heatmap(
                z=confusion_matrix,
                x=["Pred: Bad", "Pred: Good"],
                y=["Actual: Bad", "Actual: Good"],
                colorscale="Blues",
                showscale=True,
            )
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No confusion matrix available for this run.")


def render_production_gatekeeper():
    st.title("Production Release Gatekeeper")
    st.caption(
        "Enforcing ACID transactions and row locking via MySQL Stored Procedure."
    )

    st.subheader("Current Production Baseline")
    baseline = None
    conn = None
    cursor = None
    try:
        conn = get_mysql_conn()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT m.model_name, v.version_tag, v.accuracy, v.f1_score, v.created_by
            FROM model_versions v
            JOIN models m ON v.model_id = m.model_id
            WHERE v.is_active_prod = TRUE
            LIMIT 1;
            """
        )
        baseline = cursor.fetchone()
    except Exception as exc:
        st.error(f"MySQL unavailable: {exc}")
    finally:
        _safe_close(cursor, conn)

    if baseline:
        st.success(
            f"Active model: {baseline[0]} | version {baseline[1]} | "
            f"accuracy {float(baseline[2]):.4f} | f1 {float(baseline[3]):.4f} | "
            f"registered by {baseline[4]}"
        )
    else:
        st.warning("No production model is currently active.")

    st.divider()

    st.subheader("Evaluate Candidate Model for Deployment")

    try:
        candidate_runs = get_leaderboard(limit=10)
    except Exception as exc:
        st.error(f"MongoDB unavailable: {exc}")
        return

    if not candidate_runs:
        st.warning("No candidate runs available. Run the training pipeline first.")
        return

    candidate_ids = [run["run_id"] for run in candidate_runs]
    selected_candidate_id = st.selectbox(
        "Select candidate run", candidate_ids, key="candidate_select"
    )
    cand_doc = next(
        (run for run in candidate_runs if run["run_id"] == selected_candidate_id), None
    )
    if cand_doc is None:
        st.warning("Selected candidate could not be found.")
        return

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Candidate Model", cand_doc.get("model", "Unknown"))
    col_b.metric("Candidate Accuracy", f"{float(cand_doc['accuracy']):.4f}")
    col_c.metric("Developer", cand_doc.get("developer", "Unknown"))

    new_version_tag = st.text_input("Enter New Version Tag", value="v2.0")

    if st.button("Promote Candidate to Production", type="primary"):
        promote_conn = None
        promote_cursor = None
        try:
            promote_conn = get_mysql_conn()
            promote_cursor = promote_conn.cursor()
            promote_cursor.callproc(
                "PromoteModel",
                [
                    "German_Credit_Classifier",
                    new_version_tag,
                    float(cand_doc["accuracy"]),
                    float(cand_doc["f1"]),
                    cand_doc["developer"],
                ],
            )
            promote_conn.commit()
            st.success(
                f"Model {cand_doc.get('model')} promoted to production as "
                f"{new_version_tag}."
            )
            st.rerun()
        except mysql.connector.Error as err:
            st.error(f"Deployment Blocked by Gatekeeper: {err.msg}")
        except Exception as exc:
            st.error(f"Deployment failed: {exc}")
        finally:
            _safe_close(promote_cursor, promote_conn)

    st.divider()

    st.subheader("Release Audit History (MySQL)")
    audit_conn = None
    audit_cursor = None
    try:
        audit_conn = get_mysql_conn()
        audit_cursor = audit_conn.cursor()
        audit_cursor.execute(
            """
            SELECT h.release_id, v.version_tag, h.promoted_by,
                   h.promotion_status, h.reason, h.promoted_at
            FROM release_history h
            JOIN model_versions v ON h.version_id = v.version_id
            ORDER BY h.promoted_at DESC
            LIMIT 5;
            """
        )
        rows = audit_cursor.fetchall()
        audit_df = pd.DataFrame(
            rows,
            columns=[
                "release_id",
                "version_tag",
                "promoted_by",
                "promotion_status",
                "reason",
                "promoted_at",
            ],
        )
        st.table(audit_df)
    except Exception as exc:
        st.error(f"Could not load audit history: {exc}")
    finally:
        _safe_close(audit_cursor, audit_conn)


st.sidebar.title("Smart MLOps Hub")
st.sidebar.caption("Domain: German Credit Risk")

page = st.sidebar.radio(
    "Navigation",
    [
        "1. Executive Overview",
        "2. Experiment Leaderboard",
        "3. Model Evaluation",
        "4. Production Gatekeeper",
    ],
)

if page == "1. Executive Overview":
    render_executive_overview()
elif page == "2. Experiment Leaderboard":
    render_experiment_leaderboard()
elif page == "3. Model Evaluation":
    render_model_evaluation()
elif page == "4. Production Gatekeeper":
    render_production_gatekeeper()
