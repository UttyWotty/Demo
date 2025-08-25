import os
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pipeline.generate_demo import get_demo_summary
from utils.recommended_actions import ActionRecommender


@st.cache_data(ttl=3600)
def load_summary(
    schema_name: str, start_date: str, days: int, tools: int, suppliers: int, seed: int
) -> pd.DataFrame:
    """
    Load synthetic summary data for the demo app, cached for responsiveness.
    """
    return get_demo_summary(
        schema_name=schema_name or None,
        start_date=start_date,
        days=days,
        equipment_count=tools,
        supplier_count=suppliers,
        seed=seed,
    )


def safe_weighted_avg(
    df: pd.DataFrame, col: str, weight_col: str = "TOTAL_SHOTS"
) -> float:
    """
    Compute a weighted average safely, returning 0 on any failure.
    """
    try:
        return round(float(np.average(df[col], weights=df[weight_col])), 2)
    except Exception:
        return 0.0


def create_ct_trend_chart(filtered_data: pd.DataFrame, equipment_code: str):
    """
    Create a simple CT trend chart for a selected equipment.
    """
    if filtered_data.empty:
        return None

    trend_data = filtered_data.copy()
    trend_data["DATE"] = pd.to_datetime(trend_data["DAY"].astype(str), format="%Y%m%d")
    trend_data = trend_data.sort_values("DATE")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trend_data["DATE"],
            y=trend_data["APPROVED_CT"],
            mode="lines",
            name="Approved CT",
            line=dict(color="green", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trend_data["DATE"],
            y=trend_data["AVERAGE_CT"],
            mode="lines+markers",
            name="Average CT",
            line=dict(color="blue"),
        )
    )

    fig.update_layout(
        title=f"Cycle Time Trend - {equipment_code}",
        xaxis_title="Date",
        yaxis_title="Cycle Time (s)",
        template="plotly_white",
        height=400,
        margin=dict(l=60, r=30, t=50, b=50),
    )
    return fig


def create_efficiency_and_roi_charts(filtered_data: pd.DataFrame):
    """
    Create a 2x2 panel: Efficiency histogram, CT category pie, Net ROI bar, Stability vs Efficiency scatter.
    """
    if filtered_data.empty:
        return None

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Efficiency Distribution",
            "CT Categories",
            "Daily Net ROI",
            "Stability vs Efficiency",
        ),
        specs=[
            [{"type": "histogram"}, {"type": "pie"}],
            [{"type": "bar"}, {"type": "scatter"}],
        ],
    )

    realistic_eff = filtered_data[
        (filtered_data["TOOLING_EFFICIENCY(%)"] >= 10)
        & (filtered_data["TOOLING_EFFICIENCY(%)"] <= 200)
    ]["TOOLING_EFFICIENCY(%)"]
    fig.add_trace(
        go.Histogram(
            x=realistic_eff, nbinsx=20, name="Efficiency", marker_color="lightblue"
        ),
        row=1,
        col=1,
    )

    within_count = filtered_data["WITHIN_SHOT_COUNT"].sum()
    slower_count = filtered_data["SLOWER_SHOT_COUNT"].sum()
    faster_count = filtered_data["FASTER_SHOT_COUNT"].sum()
    fig.add_trace(
        go.Pie(
            labels=["Within", "Slower", "Faster"],
            values=[within_count, slower_count, faster_count],
            marker_colors=["green", "red", "orange"],
        ),
        row=1,
        col=2,
    )

    roi_by_day = filtered_data.groupby("DAY").agg({"NET_ROI": "sum"}).reset_index()
    roi_by_day["DATE"] = pd.to_datetime(roi_by_day["DAY"].astype(str), format="%Y%m%d")
    fig.add_trace(
        go.Bar(
            x=roi_by_day["DATE"],
            y=roi_by_day["NET_ROI"],
            name="Net ROI",
            marker_color="mediumpurple",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=filtered_data["TOOLING_EFFICIENCY(%)"],
            y=filtered_data["PROCESS_STABILITY"],
            mode="markers",
            name="Stability vs Efficiency",
            marker=dict(size=6, color="teal", opacity=0.7),
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        template="plotly_white",
        height=700,
        showlegend=False,
        margin=dict(l=60, r=30, t=50, b=50),
    )
    return fig


def main():
    """Streamlit entrypoint."""
    st.set_page_config(page_title="ROI Demo Dashboard", layout="wide")

    st.title("ROI Demo Dashboard")
    st.caption("Portfolio-safe demo using synthetic data (no external services).")

    with st.sidebar:
        st.subheader("Demo Parameters")
        schema = st.text_input("Schema name (label only)", value="DEMO")
        start_date = st.date_input("Start date", datetime(2025, 1, 1)).strftime(
            "%Y-%m-%d"
        )
        days = st.slider("Days", min_value=7, max_value=120, value=30, step=1)
        suppliers = st.slider(
            "Supplier count", min_value=1, max_value=5, value=2, step=1
        )
        tools = st.slider("Tool count", min_value=1, max_value=10, value=3, step=1)
        seed = st.number_input(
            "Random seed", min_value=0, max_value=999999, value=42, step=1
        )

        st.subheader("Cost Assumptions")
        machine_rate = st.number_input(
            "Machine rate ($/hour)",
            min_value=0.0,
            value=float(os.getenv("MACHINE_RATE_PER_HOUR", 25)),
        )
        labor_rate = st.number_input(
            "Labor rate ($/hour)",
            min_value=0.0,
            value=float(os.getenv("LABOR_RATE_PER_HOUR", 10)),
        )
        hour_value = machine_rate + labor_rate

    df = load_summary(schema, start_date, days, tools, suppliers, seed).copy()

    # Enrich with ROI for demo (based on hourly value)
    df["ROI_GAIN"] = (df["GAIN_HOURS"] * hour_value).round(2)
    df["ROI_LOSS"] = (df["LOSS_HOURS"] * hour_value).round(2)
    df["NET_ROI"] = df["ROI_GAIN"] - df["ROI_LOSS"]

    # Filters
    suppliers_list = sorted(df["SUPPLIER_NAME"].unique())
    supplier = st.selectbox(
        "Supplier", options=suppliers_list, index=0 if suppliers_list else None
    )

    eq_list = (
        sorted(df[df["SUPPLIER_NAME"] == supplier]["EQUIPMENT_CODE"].unique())
        if supplier
        else []
    )
    equipment = st.selectbox(
        "Equipment (Tool)", options=eq_list, index=0 if eq_list else None
    )

    # Filtered frame
    filtered = (
        df[(df["SUPPLIER_NAME"] == supplier) & (df["EQUIPMENT_CODE"] == equipment)]
        if supplier and equipment
        else df.head(0)
    )

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Avg CT (s)",
            value=f"{filtered['AVERAGE_CT'].mean():.2f}"
            if not filtered.empty
            else "0.00",
        )
    with col2:
        st.metric(
            "Efficiency (%)",
            value=f"{filtered['TOOLING_EFFICIENCY(%)'].mean():.1f}"
            if not filtered.empty
            else "0.0",
        )
    with col3:
        st.metric(
            "Process Stability",
            value=f"{filtered['PROCESS_STABILITY'].mean():.1f}"
            if not filtered.empty
            else "0.0",
        )
    with col4:
        st.metric(
            "Net ROI ($)",
            value=f"{filtered['NET_ROI'].sum():,.0f}" if not filtered.empty else "0",
        )

    # Charts
    st.subheader("Cycle Time Trend")
    trend_fig = create_ct_trend_chart(filtered, equipment or "")
    if trend_fig is not None:
        st.plotly_chart(trend_fig, use_container_width=True)
    else:
        st.info("No data for trend.")

    st.subheader("Efficiency, ROI, and Stability")
    panel = create_efficiency_and_roi_charts(filtered)
    if panel is not None:
        st.plotly_chart(panel, use_container_width=True)
    else:
        st.info("No data to display.")

    # Recommendations
    st.subheader("🎯 Recommended Actions")
    recommender = ActionRecommender()

    weighted_eff = safe_weighted_avg(filtered, "TOOLING_EFFICIENCY(%)")
    weighted_ct = filtered["AVERAGE_CT"].mean() if not filtered.empty else 0.0
    weighted_stability = (
        filtered["PROCESS_STABILITY"].mean() if not filtered.empty else 0.0
    )
    roi_loss = float(filtered["ROI_LOSS"].sum()) if not filtered.empty else 0.0
    roi_gain = float(filtered["ROI_GAIN"].sum()) if not filtered.empty else 0.0
    net_roi = float(filtered["NET_ROI"].sum()) if not filtered.empty else 0.0
    weighted_diff = filtered["DIFFERENCE"].mean() if not filtered.empty else 0.0
    approved_ct = filtered["APPROVED_CT"].mean() if not filtered.empty else 0.0
    total_shots = int(filtered["TOTAL_SHOTS"].sum()) if not filtered.empty else 0

    recs = recommender.generate_recommendations(
        equipment_code=equipment or "N/A",
        supplier_name=supplier or "N/A",
        weighted_eff=weighted_eff,
        weighted_ct=weighted_ct,
        weighted_stability=weighted_stability,
        roi_loss=roi_loss,
        roi_gain=roi_gain,
        net_roi=net_roi,
        weighted_diff=weighted_diff,
        approved_ct=approved_ct,
        total_shot_count=total_shots,
        filtered_data=filtered,
    )
    summary = recommender.get_action_summary(recs)

    colA, colB, colC = st.columns(3)
    colA.metric("Total Actions", summary["total_actions"])
    colB.metric("Critical Actions", summary["critical_actions"])
    colC.metric("High Priority", summary["high_priority_actions"])

    for i, rec in enumerate(recs, 1):
        st.markdown(
            f"**{i}. {rec.title}**  \n"
            f"- Priority: {rec.priority.value}  \n"
            f"- Category: {rec.category.value}  \n"
            f"- Impact: {rec.impact_score}/100  \n"
            f"- Timeline: {rec.timeline}  \n"
            f"- Responsible: {rec.responsible_party}  \n"
            f"- Estimated Cost: {('$' + format(rec.estimated_cost, ',.0f')) if rec.estimated_cost else 'N/A'}  \n"
            f"- Estimated Savings: {('$' + format(rec.estimated_savings, ',.0f')) if rec.estimated_savings else 'N/A'}  \n"
            f"{rec.description}"
        )

    st.caption("Demo data is synthetic. Figures are for demonstration only.")


if __name__ == "__main__":
    main()
