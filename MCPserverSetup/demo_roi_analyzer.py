#!/usr/bin/env python3
"""
ROI Analyzer - Portfolio Demo
==============================

Cycle Time Efficiency & ROI Analysis with synthetic data.
No external dependencies or credentials required.

Run with: streamlit run demo_roi_analyzer.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random

# ===========================
# Page Configuration
# ===========================

st.set_page_config(
    page_title="ROI Analyzer (Demo)",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""",
    unsafe_allow_html=True,
)

# ===========================
# Synthetic Data Generator
# ===========================


@st.cache_data
def generate_roi_data(supplier: str, equipment: str, days: int = 30):
    """Generate synthetic ROI analysis data"""

    data = []
    base_date = datetime.now() - timedelta(days=days)
    base_ct = 45.0  # Base approved cycle time
    delta_tolerance = 0.05  # ±5% tolerance

    for day in range(days):
        current_date = base_date + timedelta(days=day)
        shots_per_day = random.randint(800, 1200)

        for shot in range(shots_per_day):
            timestamp = current_date + timedelta(
                hours=random.randint(6, 22),
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59),
            )

            # Realistic cycle time variations
            variation = random.gauss(0, 0.1)  # ±10% standard deviation
            actual_ct = base_ct * (1 + variation)

            # Classify CT
            lower_bound = base_ct * (1 - delta_tolerance)
            upper_bound = base_ct * (1 + delta_tolerance)

            if actual_ct < lower_bound:
                classification = "FASTER"
            elif actual_ct > upper_bound:
                classification = "SLOWER"
            else:
                classification = "WITHIN"

            data.append(
                {
                    "SUPPLIER_NAME": supplier,
                    "EQUIPMENT_CODE": equipment,
                    "LOCAL_SHOT_TIME": timestamp,
                    "ACTUAL_CT": round(actual_ct, 2),
                    "APPROVED_CT": base_ct,
                    "CLASSIFICATION": classification,
                    "TIME_DIFF": round(actual_ct - base_ct, 2),
                }
            )

    df = pd.DataFrame(data)
    df = df.sort_values("LOCAL_SHOT_TIME").reset_index(drop=True)

    # Calculate additional metrics
    df["EFFICIENCY"] = (df["APPROVED_CT"] / df["ACTUAL_CT"] * 100).round(2)
    df["TIME_SAVED"] = np.where(
        df["TIME_DIFF"] < 0, abs(df["TIME_DIFF"]), 0
    )  # Saved when faster
    df["TIME_LOST"] = np.where(
        df["TIME_DIFF"] > 0, df["TIME_DIFF"], 0
    )  # Lost when slower

    return df


# ===========================
# Analysis Functions
# ===========================


def calculate_roi_metrics(df):
    """Calculate comprehensive ROI metrics"""

    total_shots = len(df)

    # Classification counts
    faster_shots = len(df[df["CLASSIFICATION"] == "FASTER"])
    within_shots = len(df[df["CLASSIFICATION"] == "WITHIN"])
    slower_shots = len(df[df["CLASSIFICATION"] == "SLOWER"])

    # Time analysis
    total_time_saved = df["TIME_SAVED"].sum()
    total_time_lost = df["TIME_LOST"].sum()
    net_time_impact = total_time_saved - total_time_lost

    # Efficiency metrics
    avg_efficiency = df["EFFICIENCY"].mean()
    avg_ct = df["ACTUAL_CT"].mean()
    approved_ct = df["APPROVED_CT"].iloc[0]

    # Financial impact (assuming $50/hour labor cost)
    hourly_cost = 50.0
    time_saved_hours = total_time_saved / 3600
    time_lost_hours = total_time_lost / 3600

    cost_savings = time_saved_hours * hourly_cost
    cost_losses = time_lost_hours * hourly_cost
    net_roi = cost_savings - cost_losses

    return {
        "total_shots": total_shots,
        "faster_shots": faster_shots,
        "within_shots": within_shots,
        "slower_shots": slower_shots,
        "faster_pct": (faster_shots / total_shots * 100) if total_shots > 0 else 0,
        "within_pct": (within_shots / total_shots * 100) if total_shots > 0 else 0,
        "slower_pct": (slower_shots / total_shots * 100) if total_shots > 0 else 0,
        "total_time_saved": total_time_saved,
        "total_time_lost": total_time_lost,
        "net_time_impact": net_time_impact,
        "avg_efficiency": avg_efficiency,
        "avg_ct": avg_ct,
        "approved_ct": approved_ct,
        "cost_savings": cost_savings,
        "cost_losses": cost_losses,
        "net_roi": net_roi,
    }


# ===========================
# Visualization Functions
# ===========================


def create_classification_pie(metrics):
    """Create cycle time classification pie chart"""

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["FASTER", "WITHIN", "SLOWER"],
                values=[
                    metrics["faster_shots"],
                    metrics["within_shots"],
                    metrics["slower_shots"],
                ],
                marker_colors=["#28a745", "#ffc107", "#dc3545"],
                hole=0.4,
            )
        ]
    )

    fig.update_layout(
        title="Cycle Time Classification Distribution", height=400, showlegend=True
    )

    return fig


def create_efficiency_timeline(df):
    """Create efficiency over time line chart"""

    # Aggregate by day
    daily_df = (
        df.groupby(df["LOCAL_SHOT_TIME"].dt.date)
        .agg({"EFFICIENCY": "mean"})
        .reset_index()
    )

    daily_df.columns = ["Date", "Avg_Efficiency"]

    fig = px.line(
        daily_df,
        x="Date",
        y="Avg_Efficiency",
        title="Daily Average Efficiency Trend",
        labels={"Avg_Efficiency": "Efficiency (%)"},
    )

    fig.add_hline(
        y=100, line_dash="dash", line_color="red", annotation_text="Target (100%)"
    )

    fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Efficiency (%)")

    return fig


def create_time_impact_chart(metrics):
    """Create time saved vs lost bar chart"""

    fig = go.Figure(
        data=[
            go.Bar(
                name="Time Saved",
                x=["Impact"],
                y=[metrics["total_time_saved"] / 3600],
                marker_color="#28a745",
            ),
            go.Bar(
                name="Time Lost",
                x=["Impact"],
                y=[metrics["total_time_lost"] / 3600],
                marker_color="#dc3545",
            ),
        ]
    )

    fig.update_layout(
        title="Time Impact Analysis (Hours)",
        yaxis_title="Hours",
        barmode="group",
        height=400,
    )

    return fig


def create_roi_waterfall(metrics):
    """Create ROI waterfall chart"""

    fig = go.Figure(
        go.Waterfall(
            name="ROI Analysis",
            orientation="v",
            measure=["relative", "relative", "total"],
            x=["Cost Savings", "Cost Losses", "Net ROI"],
            textposition="outside",
            y=[metrics["cost_savings"], -metrics["cost_losses"], metrics["net_roi"]],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#dc3545"}},
            increasing={"marker": {"color": "#28a745"}},
            totals={"marker": {"color": "#007bff"}},
        )
    )

    fig.update_layout(
        title="Financial ROI Waterfall ($)",
        yaxis_title="Amount ($)",
        height=400,
        showlegend=False,
    )

    return fig


def create_ct_distribution(df):
    """Create cycle time distribution histogram"""

    fig = px.histogram(
        df,
        x="ACTUAL_CT",
        nbins=50,
        title="Cycle Time Distribution",
        labels={"ACTUAL_CT": "Cycle Time (seconds)"},
        color_discrete_sequence=["#007bff"],
    )

    # Add approved CT line
    approved_ct = df["APPROVED_CT"].iloc[0]
    fig.add_vline(
        x=approved_ct, line_dash="dash", line_color="red", annotation_text="Approved CT"
    )

    fig.update_layout(height=400)

    return fig


# ===========================
# Main App
# ===========================

# Header
st.markdown(
    '<h1 class="main-header">💰 ROI Analyzer - Cycle Time Efficiency Analysis</h1>',
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.header("🔍 Analysis Configuration")
st.sidebar.markdown("*Using synthetic data - safe for portfolio*")

supplier = st.sidebar.selectbox(
    "Select Supplier", ["General Motors", "Tesla", "Ford", "BMW", "Toyota"]
)

equipment_options = {
    "General Motors": "GM-2822-01",
    "Tesla": "TSLA-3001-05",
    "Ford": "FORD-1501-03",
    "BMW": "BMW-4500-02",
    "Toyota": "TOYO-2100-04",
}

equipment = equipment_options[supplier]
st.sidebar.info(f"Equipment: **{equipment}**")

days = st.sidebar.slider("Days of Data", min_value=7, max_value=60, value=30)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Demo Mode**: This analyzer uses synthetic data. "
    "No real credentials or external services required."
)

# Generate data
with st.spinner("🔄 Generating synthetic ROI data..."):
    df = generate_roi_data(supplier, equipment, days)
    metrics = calculate_roi_metrics(df)

# Data Overview
st.subheader("📊 Analysis Overview")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Supplier", supplier)
with col2:
    st.metric("Equipment", equipment)
with col3:
    st.metric("Date Range", f"{days} days")
with col4:
    st.metric("Total Shots", f"{metrics['total_shots']:,}")

# Key Performance Indicators
st.subheader("🎯 Efficiency Metrics")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Avg Efficiency", f"{metrics['avg_efficiency']:.2f}%")
with col2:
    delta = metrics["avg_ct"] - metrics["approved_ct"]
    st.metric(
        "Avg Cycle Time",
        f"{metrics['avg_ct']:.2f}s",
        f"{delta:+.2f}s",
        delta_color="inverse",
    )
with col3:
    st.metric("Approved CT", f"{metrics['approved_ct']:.2f}s")
with col4:
    within_pct = metrics["within_pct"]
    st.metric("Within Tolerance", f"{within_pct:.1f}%")

# Classification Metrics
st.subheader("📈 Cycle Time Classification")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        "FASTER than Approved",
        f"{metrics['faster_shots']:,}",
        f"{metrics['faster_pct']:.1f}%",
    )
with col2:
    st.metric(
        "WITHIN Tolerance",
        f"{metrics['within_shots']:,}",
        f"{metrics['within_pct']:.1f}%",
    )
with col3:
    st.metric(
        "SLOWER than Approved",
        f"{metrics['slower_shots']:,}",
        f"{metrics['slower_pct']:.1f}%",
        delta_color="inverse",
    )

# Time Impact Analysis
st.subheader("⏱️ Time Impact Analysis")

col1, col2, col3 = st.columns(3)
with col1:
    time_saved_hours = metrics["total_time_saved"] / 3600
    st.metric("Time Saved", f"{time_saved_hours:.2f} hours")
with col2:
    time_lost_hours = metrics["total_time_lost"] / 3600
    st.metric("Time Lost", f"{time_lost_hours:.2f} hours", delta_color="inverse")
with col3:
    net_hours = metrics["net_time_impact"] / 3600
    delta_color = "normal" if net_hours >= 0 else "inverse"
    st.metric("Net Impact", f"{net_hours:+.2f} hours", delta_color=delta_color)

# Financial ROI
st.subheader("💰 Financial ROI Analysis")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Hourly Cost", "$50.00")
with col2:
    st.metric("Cost Savings", f"${metrics['cost_savings']:,.2f}")
with col3:
    st.metric("Cost Losses", f"${metrics['cost_losses']:,.2f}", delta_color="inverse")
with col4:
    delta_color = "normal" if metrics["net_roi"] >= 0 else "inverse"
    st.metric("Net ROI", f"${metrics['net_roi']:+,.2f}", delta_color=delta_color)

# Visualizations
st.subheader("📊 ROI Visualizations")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Classification", "📈 Trends", "💰 Financial", "📉 Distribution"]
)

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        classification_pie = create_classification_pie(metrics)
        st.plotly_chart(classification_pie, use_container_width=True)

    with col2:
        time_impact = create_time_impact_chart(metrics)
        st.plotly_chart(time_impact, use_container_width=True)

with tab2:
    efficiency_timeline = create_efficiency_timeline(df)
    st.plotly_chart(efficiency_timeline, use_container_width=True)

    # Additional stats
    col1, col2, col3 = st.columns(3)
    with col1:
        best_day_efficiency = (
            df.groupby(df["LOCAL_SHOT_TIME"].dt.date)["EFFICIENCY"].mean().max()
        )
        st.metric("Best Day Efficiency", f"{best_day_efficiency:.2f}%")
    with col2:
        worst_day_efficiency = (
            df.groupby(df["LOCAL_SHOT_TIME"].dt.date)["EFFICIENCY"].mean().min()
        )
        st.metric("Worst Day Efficiency", f"{worst_day_efficiency:.2f}%")
    with col3:
        efficiency_std = (
            df.groupby(df["LOCAL_SHOT_TIME"].dt.date)["EFFICIENCY"].mean().std()
        )
        st.metric("Efficiency Std Dev", f"{efficiency_std:.2f}%")

with tab3:
    roi_waterfall = create_roi_waterfall(metrics)
    st.plotly_chart(roi_waterfall, use_container_width=True)

    # ROI Summary
    st.markdown("### ROI Summary")

    summary_df = pd.DataFrame(
        {
            "Metric": [
                "Total Time Saved",
                "Total Time Lost",
                "Net Time Impact",
                "Cost Savings",
                "Cost Losses",
                "Net ROI",
            ],
            "Value": [
                f"{metrics['total_time_saved'] / 3600:.2f} hours",
                f"{metrics['total_time_lost'] / 3600:.2f} hours",
                f"{metrics['net_time_impact'] / 3600:+.2f} hours",
                f"${metrics['cost_savings']:,.2f}",
                f"${metrics['cost_losses']:,.2f}",
                f"${metrics['net_roi']:+,.2f}",
            ],
        }
    )

    st.dataframe(summary_df, use_container_width=True, hide_index=True)

with tab4:
    ct_distribution = create_ct_distribution(df)
    st.plotly_chart(ct_distribution, use_container_width=True)

    # Statistical summary
    st.markdown("### Statistical Summary")

    stats_df = pd.DataFrame(
        {
            "Statistic": ["Mean", "Median", "Std Dev", "Min", "Max"],
            "Actual CT (s)": [
                f"{df['ACTUAL_CT'].mean():.2f}",
                f"{df['ACTUAL_CT'].median():.2f}",
                f"{df['ACTUAL_CT'].std():.2f}",
                f"{df['ACTUAL_CT'].min():.2f}",
                f"{df['ACTUAL_CT'].max():.2f}",
            ],
            "Efficiency (%)": [
                f"{df['EFFICIENCY'].mean():.2f}",
                f"{df['EFFICIENCY'].median():.2f}",
                f"{df['EFFICIENCY'].std():.2f}",
                f"{df['EFFICIENCY'].min():.2f}",
                f"{df['EFFICIENCY'].max():.2f}",
            ],
        }
    )

    st.dataframe(stats_df, use_container_width=True, hide_index=True)

# Recommendations
st.subheader("💡 AI-Powered Recommendations")

if metrics["slower_pct"] > 30:
    st.error(
        f"⚠️ **High Inefficiency Alert**: {metrics['slower_pct']:.1f}% of shots are SLOWER than approved CT. "
        "Recommended actions: 1) Review equipment maintenance, 2) Check material quality, 3) Operator training."
    )
elif metrics["slower_pct"] > 20:
    st.warning(
        f"⚠️ **Moderate Inefficiency**: {metrics['slower_pct']:.1f}% of shots are SLOWER. "
        "Consider preventive maintenance and process optimization."
    )
else:
    st.success(
        f"✅ **Good Performance**: Only {metrics['slower_pct']:.1f}% of shots are SLOWER. "
        "Continue monitoring and maintain current best practices."
    )

if metrics["net_roi"] > 0:
    st.success(
        f"💰 **Positive ROI**: Net savings of ${metrics['net_roi']:,.2f}. "
        "Equipment is operating efficiently and generating cost savings."
    )
else:
    st.error(
        f"💸 **Negative ROI**: Net loss of ${abs(metrics['net_roi']):,.2f}. "
        "Immediate attention needed to improve cycle time performance."
    )

# Download Options
st.subheader("📥 Download Options")

col1, col2, col3 = st.columns(3)

with col1:
    # Raw data CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📊 Download Raw Data",
        csv,
        "roi_analysis_data.csv",
        "text/csv",
        help="Download raw analysis data as CSV",
    )

with col2:
    # Summary metrics CSV
    summary_data = pd.DataFrame([metrics])
    summary_csv = summary_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📋 Download Summary",
        summary_csv,
        "roi_summary_metrics.csv",
        "text/csv",
        help="Download summary metrics as CSV",
    )

with col3:
    st.info(
        "💡 In production, you can export to Excel with formatted reports and formulas"
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        <p>🎨 <strong>Portfolio Demo</strong> - Using synthetic data for demonstration purposes</p>
        <p>Production version connects to Snowflake and processes millions of shots daily</p>
        <p>Features: Cycle Time Analysis | ROI Calculation | Time/Cost Impact | Excel Report Generation</p>
    </div>
    """,
    unsafe_allow_html=True,
)
