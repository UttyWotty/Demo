#!/usr/bin/env python3
"""
Production Analytics Dashboard - Portfolio Demo
==============================================

Self-contained Streamlit dashboard with synthetic data.
No external dependencies or credentials required.

Run with: streamlit run demo_dashboard.py
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
    page_title="Production Analytics Dashboard (Demo)",
    page_icon="",
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
        border-left: 5px solid #1f4e79;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #28a745;
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
def generate_synthetic_data(supplier: str, days: int = 30, equipment_count: int = 3):
    """Generate realistic synthetic production data"""

    data = []
    base_date = datetime.now() - timedelta(days=days)

    for equipment_id in range(1, equipment_count + 1):
        equipment_code = f"{supplier[:3].upper()}-{equipment_id:03d}"
        base_ct = random.uniform(25, 45)  # Base cycle time
        efficiency = random.uniform(0.85, 0.95)  # Equipment efficiency

        for day in range(days):
            current_date = base_date + timedelta(days=day)
            shots_per_day = random.randint(200, 400)

            for shot in range(shots_per_day):
                timestamp = current_date + timedelta(
                    hours=random.randint(6, 22),
                    minutes=random.randint(0, 59),
                    seconds=random.randint(0, 59),
                )

                # Realistic cycle time with variations
                actual_ct = base_ct + random.gauss(0, 2)

                # Occasionally introduce stops (anomalies)
                is_stop = random.random() > efficiency

                data.append(
                    {
                        "SUPPLIER_NAME": supplier,
                        "EQUIPMENT_CODE": equipment_code,
                        "LOCAL_SHOT_TIME": timestamp,
                        "ACTUAL_CT": 999.9 if is_stop else max(10, actual_ct),
                        "APPROVED_CT": base_ct,
                        "STOP": 1 if is_stop else 0,
                    }
                )

    df = pd.DataFrame(data)
    df = df.sort_values("LOCAL_SHOT_TIME").reset_index(drop=True)

    # Calculate additional metrics
    df["EFFICIENCY"] = (1 - df["STOP"]) * 100
    df["SESSION_ID"] = (
        df["LOCAL_SHOT_TIME"].dt.date.astype(str) + "_" + df["EQUIPMENT_CODE"]
    )

    return df


# ===========================
# Visualization Functions
# ===========================


def create_production_timeline(df):
    """Create production timeline scatter plot"""

    if df.empty:
        return None

    # Filter valid data
    plot_df = df[df["ACTUAL_CT"] < 999].copy()

    if plot_df.empty:
        return None

    # Sample for performance
    if len(plot_df) > 2000:
        plot_df = plot_df.sample(n=2000, random_state=42).sort_values("LOCAL_SHOT_TIME")

    plot_df["Status"] = plot_df["STOP"].map({0: "Normal", 1: "Stop"})

    fig = px.scatter(
        plot_df,
        x="LOCAL_SHOT_TIME",
        y="ACTUAL_CT",
        color="Status",
        hover_data=["EQUIPMENT_CODE", "ACTUAL_CT"],
        title="Production Timeline - Cycle Time vs Time",
        color_discrete_map={"Normal": "#2E8B57", "Stop": "#DC143C"},
    )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Cycle Time (seconds)",
        height=450,
        showlegend=True,
    )

    return fig


def create_efficiency_gauge(efficiency_pct):
    """Create efficiency gauge chart"""

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=efficiency_pct,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Overall Efficiency (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 60], "color": "lightgray"},
                    {"range": [60, 80], "color": "yellow"},
                    {"range": [80, 100], "color": "green"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 80,
                },
            },
        )
    )

    fig.update_layout(height=300)
    return fig


def create_uptime_pie(uptime_pct):
    """Create uptime vs downtime pie chart"""

    downtime_pct = 100 - uptime_pct

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Uptime", "Downtime"],
                values=[uptime_pct, downtime_pct],
                hole=0.4,
                marker_colors=["#2E8B57", "#DC143C"],
            )
        ]
    )

    fig.update_layout(
        title="Uptime vs Downtime Distribution",
        height=400,
        showlegend=True,
    )

    return fig


def create_equipment_performance(df):
    """Create equipment performance comparison"""

    if df.empty:
        return None

    # Calculate per-equipment metrics
    equipment_metrics = (
        df.groupby("EQUIPMENT_CODE")
        .agg(
            {
                "STOP": lambda x: (1 - x.mean()) * 100,  # Efficiency
                "ACTUAL_CT": "count",  # Total shots
            }
        )
        .reset_index()
    )

    equipment_metrics.columns = ["Equipment", "Efficiency (%)", "Total Shots"]

    # Create subplot
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Efficiency by Equipment", "Total Shots by Equipment"),
        specs=[[{"type": "bar"}, {"type": "bar"}]],
    )

    # Efficiency
    fig.add_trace(
        go.Bar(
            x=equipment_metrics["Equipment"],
            y=equipment_metrics["Efficiency (%)"],
            name="Efficiency",
            marker_color="#4169E1",
        ),
        row=1,
        col=1,
    )

    # Total Shots
    fig.add_trace(
        go.Bar(
            x=equipment_metrics["Equipment"],
            y=equipment_metrics["Total Shots"],
            name="Total Shots",
            marker_color="#DC143C",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(height=400, showlegend=False, title_text="Equipment Performance")
    return fig


def create_cycle_time_distribution(df):
    """Create cycle time distribution histogram"""

    if df.empty:
        return None

    valid_df = df[df["ACTUAL_CT"] < 999].copy()

    if valid_df.empty:
        return None

    fig = px.histogram(
        valid_df,
        x="ACTUAL_CT",
        nbins=50,
        title="Cycle Time Distribution",
        labels={"ACTUAL_CT": "Cycle Time (seconds)", "count": "Frequency"},
        color_discrete_sequence=["#4169E1"],
    )

    fig.update_layout(height=400)
    return fig


# ===========================
# Main Dashboard
# ===========================

# Header
st.markdown(
    '<h1 class="main-header">🏭 Production Analytics Dashboard (Demo)</h1>',
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.header("🔍 Demo Configuration")
st.sidebar.markdown("*Using synthetic data - safe for portfolio*")

supplier_options = ["General Motors", "Tesla", "Ford", "BMW", "Toyota"]
supplier = st.sidebar.selectbox("Select Supplier", supplier_options)

days = st.sidebar.slider("Days of Data", min_value=7, max_value=60, value=30)
equipment_count = st.sidebar.slider(
    "Number of Equipment", min_value=1, max_value=5, value=3
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Demo Mode**: This dashboard uses synthetic data. "
    "No real credentials or external services required."
)

# Generate data
with st.spinner("🔄 Generating synthetic production data..."):
    df = generate_synthetic_data(supplier, days, equipment_count)

# Data Overview
st.subheader("📊 Data Overview")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Supplier", supplier)
with col2:
    first_shot = df["LOCAL_SHOT_TIME"].min()
    last_shot = df["LOCAL_SHOT_TIME"].max()
    date_range = f"{first_shot.strftime('%m/%d/%Y')} - {last_shot.strftime('%m/%d/%Y')}"
    st.metric("Date Range", date_range)
with col3:
    st.metric("Total Records", f"{len(df):,}")

# Key Performance Indicators
st.subheader("🎯 Key Performance Indicators")

# Calculate metrics
valid_df = df[df["ACTUAL_CT"] < 999].copy()
total_shots = len(df)
normal_shots = len(valid_df)
stop_shots = total_shots - normal_shots
efficiency_pct = (normal_shots / total_shots * 100) if total_shots > 0 else 0

# Uptime calculation (simplified)
uptime_pct = efficiency_pct

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Shots", f"{total_shots:,}")
with col2:
    st.metric("Efficiency", f"{efficiency_pct:.1f}%")
with col3:
    st.metric("Normal Shots", f"{normal_shots:,}")
with col4:
    st.metric("Stop Events", f"{stop_shots:,}")

# Reliability Metrics
st.subheader("🔧 Reliability Metrics")

col1, col2, col3, col4 = st.columns(4)

# Calculate simplified metrics
avg_cycle_time = valid_df["ACTUAL_CT"].mean() if not valid_df.empty else 0
std_cycle_time = valid_df["ACTUAL_CT"].std() if not valid_df.empty else 0

with col1:
    st.metric("Avg Cycle Time", f"{avg_cycle_time:.2f}s")
with col2:
    st.metric("Std Deviation", f"{std_cycle_time:.2f}s")
with col3:
    st.metric("Uptime", f"{uptime_pct:.1f}%")
with col4:
    total_sessions = df["SESSION_ID"].nunique()
    st.metric("Total Sessions", f"{total_sessions:,}")

# Visualizations
st.subheader("📈 Production Analytics")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "⚡ Efficiency", "🔧 Equipment", "📈 Distribution"]
)

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        efficiency_gauge = create_efficiency_gauge(efficiency_pct)
        if efficiency_gauge:
            st.plotly_chart(efficiency_gauge, use_container_width=True)

    with col2:
        uptime_pie = create_uptime_pie(uptime_pct)
        if uptime_pie:
            st.plotly_chart(uptime_pie, use_container_width=True)

    # Production Timeline
    st.markdown("### Production Timeline")
    timeline = create_production_timeline(df)
    if timeline:
        st.plotly_chart(timeline, use_container_width=True)

with tab2:
    # Efficiency metrics
    st.markdown("### Efficiency Analysis")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Target Efficiency", "85%", help="Industry standard target")
    with col2:
        delta = efficiency_pct - 85
        st.metric("Current Efficiency", f"{efficiency_pct:.1f}%", f"{delta:+.1f}%")
    with col3:
        status = (
            "🟢 Good"
            if efficiency_pct >= 85
            else "🟡 Review"
            if efficiency_pct >= 75
            else "🔴 Action Needed"
        )
        st.metric("Status", status)

    # Cycle time distribution
    ct_dist = create_cycle_time_distribution(df)
    if ct_dist:
        st.plotly_chart(ct_dist, use_container_width=True)

with tab3:
    # Equipment performance
    equipment_perf = create_equipment_performance(df)
    if equipment_perf:
        st.plotly_chart(equipment_perf, use_container_width=True)

with tab4:
    # Data table
    st.markdown("### Raw Data Sample")
    st.dataframe(
        df.head(100)[["LOCAL_SHOT_TIME", "EQUIPMENT_CODE", "ACTUAL_CT", "STOP"]],
        use_container_width=True,
    )

# Recommendations
st.subheader("💡 AI-Powered Recommendations")

if efficiency_pct < 85:
    st.warning(
        f"⚠️ **Low Efficiency Alert**: Current efficiency is {efficiency_pct:.1f}%, "
        "which is below the target of 85%. Consider reviewing equipment maintenance schedules."
    )
elif efficiency_pct < 90:
    st.info(
        f"📊 **Good Performance**: Current efficiency is {efficiency_pct:.1f}%. "
        "Continue monitoring and maintain current processes."
    )
else:
    st.success(
        f"✅ **Excellent Performance**: Current efficiency is {efficiency_pct:.1f}%. "
        "Outstanding performance - consider this as a best practice benchmark."
    )

# Download Options
st.subheader("📥 Download Options")

col1, col2, col3 = st.columns(3)

with col1:
    # CSV download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📊 Download CSV",
        csv,
        "production_data.csv",
        "text/csv",
        help="Download raw data as CSV",
    )

with col2:
    # Summary stats
    summary = df.describe().to_csv().encode("utf-8")
    st.download_button(
        "📋 Download Summary",
        summary,
        "production_summary.csv",
        "text/csv",
        help="Download statistical summary",
    )

with col3:
    st.info("💡 In production, you can export to Excel with formatted reports")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        <p>🎨 <strong>Portfolio Demo</strong> - Using synthetic data for demonstration purposes</p>
        <p>Production version connects to Snowflake and processes 50M+ records daily</p>
        <p>Built with: Python, Streamlit, Plotly, Pandas | FastAPI MCP Server Integration Ready</p>
    </div>
    """,
    unsafe_allow_html=True,
)
