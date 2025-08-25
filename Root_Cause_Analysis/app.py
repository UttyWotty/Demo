from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pipeline.generate_demo import generate_rca_demo
from analyzer import RCADemoAnalyzer


@st.cache_data(ttl=1800)
def load_demo_data(
    start_date: str,
    days: int,
    suppliers: int,
    equipment_per_supplier: int,
    parts: int,
    seed: int,
) -> pd.DataFrame:
    """Generate and cache synthetic RCA dataset for responsiveness."""
    return generate_rca_demo(
        start_date=start_date,
        days=days,
        suppliers=suppliers,
        equipment_per_supplier=equipment_per_supplier,
        parts=parts,
        seed=seed,
    )


@st.cache_data(ttl=900)
def run_analysis(df: pd.DataFrame, top_n: int) -> Dict[str, Dict]:
    """Run RCADemoAnalyzer and cache the results."""
    analyzer = RCADemoAnalyzer(df)
    return analyzer.run(top_n=top_n)


def format_kpi(value: float) -> str:
    """Format numeric KPI values for compact display."""
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return f"{value:,.2f}"


def create_issue_rate_by_day(df: pd.DataFrame):
    """Bar chart for issue rate by day-of-week."""
    if df.empty:
        return None
    grp = df.groupby("DAY_OF_WEEK")["CT_ISSUE_FLAG"].mean().mul(100).reset_index()
    grp = grp.sort_values("CT_ISSUE_FLAG", ascending=False)
    fig = px.bar(
        grp,
        x="DAY_OF_WEEK",
        y="CT_ISSUE_FLAG",
        title="Issue Rate by Day of Week",
        labels={"CT_ISSUE_FLAG": "Issue Rate (%)"},
    )
    fig.update_layout(template="plotly_white", height=400)
    return fig


def create_issue_rate_by_equipment(df: pd.DataFrame, top: int = 10):
    """Bar chart for top-N equipment by issue rate."""
    if df.empty:
        return None
    grp = df.groupby("EQUIPMENT_CODE")["CT_ISSUE_FLAG"].mean().mul(100).reset_index()
    grp = grp.sort_values("CT_ISSUE_FLAG", ascending=False).head(top)
    fig = px.bar(
        grp,
        x="EQUIPMENT_CODE",
        y="CT_ISSUE_FLAG",
        title=f"Top {top} Equipment by Issue Rate",
        labels={"CT_ISSUE_FLAG": "Issue Rate (%)"},
    )
    fig.update_layout(template="plotly_white", height=400)
    return fig


def main():
    """Streamlit entrypoint for RCA Demo UI."""
    st.set_page_config(page_title="RCA Demo", layout="wide")
    st.title("RCA Demo Dashboard")
    st.caption("Portfolio-safe demo using synthetic data (no external services).")

    with st.sidebar:
        st.subheader("Demo Parameters")
        start_date = st.date_input("Start date", datetime(2025, 1, 1)).strftime(
            "%Y-%m-%d"
        )
        days = st.slider("Days", 7, 90, 30, 1)
        suppliers = st.slider("Suppliers", 1, 8, 3, 1)
        equipment_per_supplier = st.slider("Equipments per supplier", 1, 8, 4, 1)
        parts = st.slider("Parts", 1, 12, 6, 1)
        seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42)
        top_n = st.slider("Top targets", 1, 5, 3, 1)

    df = load_demo_data(
        start_date,
        days,
        suppliers,
        equipment_per_supplier,
        parts,
        seed,
    )

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Records", format_kpi(len(df)))
    with col2:
        st.metric("Suppliers", format_kpi(df["SUPPLIER_NAME"].nunique()))
    with col3:
        st.metric("Equipment", format_kpi(df["EQUIPMENT_CODE"].nunique()))
    with col4:
        issue_rate = (
            df["CT_ISSUE_FLAG"].mean() * 100 if "CT_ISSUE_FLAG" in df.columns else 0
        )
        st.metric("Overall Issue Rate", f"{issue_rate:.1f}%")

    # Charts
    st.subheader("Exploratory Charts")
    c1, c2 = st.columns(2)
    with c1:
        day_fig = create_issue_rate_by_day(df)
        if day_fig is not None:
            st.plotly_chart(day_fig, use_container_width=True)
        else:
            st.info("No data for day-of-week chart.")
    with c2:
        eq_fig = create_issue_rate_by_equipment(df)
        if eq_fig is not None:
            st.plotly_chart(eq_fig, use_container_width=True)
        else:
            st.info("No data for equipment chart.")

    # Analysis
    st.subheader("Targets and 5 Whys")
    results = run_analysis(df, top_n)
    if not results:
        st.info("No targets identified.")
        return

    # Target selector
    target_names = list(results.keys())
    selected = st.selectbox("Select Target", target_names, index=0)
    res = results[selected]

    # Supporting data KPIs
    sd = res.get("supporting_data", {})
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Avg CT (s)", f"{sd.get('avg_ct', 0):.2f}")
    with k2:
        st.metric("Efficiency (%)", f"{sd.get('efficiency_pct', 0):.1f}")
    with k3:
        st.metric("Issue Rate (%)", f"{sd.get('issue_rate_pct', 0):.1f}")
    with k4:
        st.metric("Samples", format_kpi(sd.get("samples", 0)))

    # Whys and recommendations
    st.markdown("#### 5 Whys")
    for i, why in enumerate(res.get("whys", []), 1):
        st.write(f"{i}. {why}")

    st.markdown("#### Recommendations")
    for rec in res.get("recommendations", []):
        st.write(f"- {rec}")

    st.caption("Demo data is synthetic. Figures are for demonstration only.")


if __name__ == "__main__":
    main()
