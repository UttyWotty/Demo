#!/usr/bin/env python3
"""
Capacity Risk Analyzer - Portfolio Demo
========================================

OEE (Overall Equipment Effectiveness) & Capacity Analysis with synthetic data.
No external dependencies or credentials required.

Run with: streamlit run demo_capacity_analyzer.py
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
    page_title="Capacity Risk Analyzer (Demo)",
    page_icon="⚡",
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
        border-left: 5px solid #dc3545;
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
def generate_capacity_data(
    supplier: str, equipment: str, days: int = 30, oee_target: float = 0.85
):
    """Generate synthetic capacity and OEE data"""

    sessions = []
    base_date = datetime.now() - timedelta(days=days)
    approved_ct = 45.0  # seconds
    cavity_count = 4  # Multi-cavity mold

    session_id = 1

    for day in range(days):
        # 2-3 sessions per day
        sessions_per_day = random.randint(2, 3)

        for session_num in range(sessions_per_day):
            session_start = base_date + timedelta(days=day, hours=random.randint(6, 20))
            session_duration_hours = random.uniform(4, 8)
            session_end = session_start + timedelta(hours=session_duration_hours)

            # Simulate production
            shots_in_session = random.randint(300, 600)
            production_time_sec = (
                shots_in_session * approved_ct * random.uniform(0.95, 1.05)
            )

            # Stops and downtime
            stop_count = random.randint(2, 10)
            total_stop_time_sec = sum(
                [random.uniform(60, 600) for _ in range(stop_count)]
            )

            # Actual output (accounting for stops)
            actual_output = int(shots_in_session * random.uniform(0.85, 0.98))
            parts_output = actual_output * cavity_count

            # Availability calculation
            total_session_time_sec = session_duration_hours * 3600
            runtime_sec = total_session_time_sec - total_stop_time_sec
            availability = (
                runtime_sec / total_session_time_sec
                if total_session_time_sec > 0
                else 0
            )

            # Performance calculation
            optimal_output = int((runtime_sec / approved_ct) * oee_target)
            performance = actual_output / optimal_output if optimal_output > 0 else 0

            # Quality (assume 98-100% for demo)
            quality = random.uniform(0.98, 1.0)

            # OEE calculation
            oee = availability * performance * quality

            # Losses
            availability_loss = optimal_output - int(
                total_session_time_sec / approved_ct * oee_target
            )
            performance_loss = optimal_output - actual_output
            gap = optimal_output - actual_output

            sessions.append(
                {
                    "SESSION_ID": f"S{session_id:04d}",
                    "SUPPLIER_NAME": supplier,
                    "EQUIPMENT_CODE": equipment,
                    "SESSION_START": session_start,
                    "SESSION_END": session_end,
                    "SESSION_DURATION_HOURS": round(session_duration_hours, 2),
                    "APPROVED_CT": approved_ct,
                    "CAVITY_COUNT": cavity_count,
                    "TOTAL_SHOTS": shots_in_session,
                    "ACTUAL_OUTPUT": actual_output,
                    "PARTS_OUTPUT": parts_output,
                    "OPTIMAL_OUTPUT": optimal_output,
                    "STOP_COUNT": stop_count,
                    "TOTAL_STOP_TIME_SEC": round(total_stop_time_sec, 2),
                    "PRODUCTION_TIME_SEC": round(production_time_sec, 2),
                    "AVAILABILITY": round(availability, 4),
                    "PERFORMANCE": round(performance, 4),
                    "QUALITY": round(quality, 4),
                    "OEE": round(oee, 4),
                    "OEE_PCT": round(oee * 100, 2),
                    "OEE_TARGET": oee_target,
                    "AVAILABILITY_LOSS": availability_loss,
                    "PERFORMANCE_LOSS": performance_loss,
                    "GAP": gap,
                }
            )

            session_id += 1

    df = pd.DataFrame(sessions)
    return df


# ===========================
# Analysis Functions
# ===========================


def calculate_capacity_metrics(df, oee_target):
    """Calculate comprehensive capacity metrics"""

    total_sessions = len(df)
    total_shots = df["TOTAL_SHOTS"].sum()
    actual_output = df["ACTUAL_OUTPUT"].sum()
    optimal_output = df["OPTIMAL_OUTPUT"].sum()
    parts_output = df["PARTS_OUTPUT"].sum()

    # OEE metrics
    avg_oee = df["OEE_PCT"].mean()
    avg_availability = df["AVAILABILITY"].mean() * 100
    avg_performance = df["PERFORMANCE"].mean() * 100
    avg_quality = df["QUALITY"].mean() * 100

    # Losses
    total_availability_loss = df["AVAILABILITY_LOSS"].sum()
    total_performance_loss = df["PERFORMANCE_LOSS"].sum()
    total_gap = df["GAP"].sum()

    # Capacity utilization
    capacity_utilization = (
        (actual_output / optimal_output * 100) if optimal_output > 0 else 0
    )

    # Stops
    total_stops = df["STOP_COUNT"].sum()
    avg_stops_per_session = df["STOP_COUNT"].mean()
    total_downtime_hours = df["TOTAL_STOP_TIME_SEC"].sum() / 3600

    # Session statistics
    avg_session_duration = df["SESSION_DURATION_HOURS"].mean()
    total_production_hours = df["SESSION_DURATION_HOURS"].sum()

    return {
        "total_sessions": total_sessions,
        "total_shots": total_shots,
        "actual_output": actual_output,
        "optimal_output": optimal_output,
        "parts_output": parts_output,
        "avg_oee": avg_oee,
        "avg_availability": avg_availability,
        "avg_performance": avg_performance,
        "avg_quality": avg_quality,
        "total_availability_loss": total_availability_loss,
        "total_performance_loss": total_performance_loss,
        "total_gap": total_gap,
        "capacity_utilization": capacity_utilization,
        "total_stops": total_stops,
        "avg_stops_per_session": avg_stops_per_session,
        "total_downtime_hours": total_downtime_hours,
        "avg_session_duration": avg_session_duration,
        "total_production_hours": total_production_hours,
        "oee_target": oee_target * 100,
    }


# ===========================
# Visualization Functions
# ===========================


def create_oee_components_chart(metrics):
    """Create OEE components stacked bar chart"""

    fig = go.Figure(
        data=[
            go.Bar(
                name="Availability",
                x=["OEE Components"],
                y=[metrics["avg_availability"]],
                marker_color="#28a745",
            ),
            go.Bar(
                name="Performance",
                x=["OEE Components"],
                y=[metrics["avg_performance"]],
                marker_color="#ffc107",
            ),
            go.Bar(
                name="Quality",
                x=["OEE Components"],
                y=[metrics["avg_quality"]],
                marker_color="#007bff",
            ),
        ]
    )

    fig.update_layout(
        title="OEE Components Breakdown",
        yaxis_title="Percentage (%)",
        barmode="group",
        height=400,
    )

    return fig


def create_oee_gauge(oee_pct, target_pct):
    """Create OEE gauge chart"""

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=oee_pct,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Overall Equipment Effectiveness (%)"},
            delta={"reference": target_pct, "increasing": {"color": "green"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 50], "color": "lightgray"},
                    {"range": [50, 75], "color": "yellow"},
                    {"range": [75, 90], "color": "lightgreen"},
                    {"range": [90, 100], "color": "green"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": target_pct,
                },
            },
        )
    )

    fig.update_layout(height=350)
    return fig


def create_oee_timeline(df):
    """Create OEE over sessions timeline"""

    fig = px.line(
        df,
        x="SESSION_START",
        y="OEE_PCT",
        title="OEE Trend Over Sessions",
        labels={"OEE_PCT": "OEE (%)", "SESSION_START": "Session Start Time"},
        markers=True,
    )

    # Add target line
    target_pct = df["OEE_TARGET"].iloc[0] * 100
    fig.add_hline(
        y=target_pct,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Target ({target_pct}%)",
    )

    fig.update_layout(height=400)
    return fig


def create_losses_chart(metrics):
    """Create losses breakdown chart"""

    fig = go.Figure(
        data=[
            go.Bar(
                name="Availability Loss",
                x=["Losses"],
                y=[metrics["total_availability_loss"]],
                marker_color="#dc3545",
            ),
            go.Bar(
                name="Performance Loss",
                x=["Losses"],
                y=[metrics["total_performance_loss"]],
                marker_color="#ff6384",
            ),
        ]
    )

    fig.update_layout(
        title="Production Losses Analysis (Parts)",
        yaxis_title="Parts Lost",
        barmode="group",
        height=400,
    )

    return fig


def create_capacity_comparison(metrics):
    """Create actual vs optimal output comparison"""

    fig = go.Figure(
        data=[
            go.Bar(
                name="Actual Output",
                x=["Output"],
                y=[metrics["actual_output"]],
                marker_color="#007bff",
            ),
            go.Bar(
                name="Optimal Output",
                x=["Output"],
                y=[metrics["optimal_output"]],
                marker_color="#28a745",
            ),
        ]
    )

    fig.update_layout(
        title="Actual vs Optimal Output (Parts)",
        yaxis_title="Parts",
        barmode="group",
        height=400,
    )

    return fig


def create_stops_analysis(df):
    """Create stops per session scatter plot"""

    fig = px.scatter(
        df,
        x="SESSION_START",
        y="STOP_COUNT",
        size="TOTAL_STOP_TIME_SEC",
        color="OEE_PCT",
        title="Stops Analysis by Session",
        labels={
            "STOP_COUNT": "Number of Stops",
            "SESSION_START": "Session Start",
            "OEE_PCT": "OEE (%)",
        },
        color_continuous_scale="RdYlGn",
    )

    fig.update_layout(height=400)
    return fig


# ===========================
# Main App
# ===========================

# Header
st.markdown(
    '<h1 class="main-header">⚡ Capacity Risk Analyzer - OEE & Performance Analysis</h1>',
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.header("🔍 Analysis Configuration")
st.sidebar.markdown("*Using synthetic data - safe for portfolio*")

supplier = st.sidebar.selectbox(
    "Select Supplier", ["General Motors", "Tesla", "Ford", "BMW", "Toyota"]
)

equipment_options = {
    "General Motors": "GM-2BM30-80382",
    "Tesla": "TSLA-1BM25-80416",
    "Ford": "FORD-3BD3008371",
    "BMW": "BMW-3BD3008451",
    "Toyota": "TOYO-2BM30-90500",
}

equipment = equipment_options[supplier]
st.sidebar.info(f"Equipment: **{equipment}**")

days = st.sidebar.slider("Days of Data", min_value=7, max_value=60, value=30)

oee_target = (
    st.sidebar.slider("OEE Target (%)", min_value=50, max_value=100, value=85, step=5)
    / 100
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Demo Mode**: This analyzer uses synthetic data. "
    "No real credentials or external services required."
)

# Generate data
with st.spinner("🔄 Generating synthetic capacity data..."):
    df = generate_capacity_data(supplier, equipment, days, oee_target)
    metrics = calculate_capacity_metrics(df, oee_target)

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
    st.metric("Total Sessions", f"{metrics['total_sessions']:,}")

# OEE Metrics
st.subheader("🎯 OEE Performance Indicators")

col1, col2, col3, col4 = st.columns(4)
with col1:
    delta = metrics["avg_oee"] - metrics["oee_target"]
    st.metric("Average OEE", f"{metrics['avg_oee']:.2f}%", f"{delta:+.2f}%")
with col2:
    st.metric("Availability", f"{metrics['avg_availability']:.2f}%")
with col3:
    st.metric("Performance", f"{metrics['avg_performance']:.2f}%")
with col4:
    st.metric("Quality", f"{metrics['avg_quality']:.2f}%")

# Production Metrics
st.subheader("🏭 Production Metrics")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Shots", f"{metrics['total_shots']:,}")
with col2:
    st.metric("Actual Output", f"{metrics['actual_output']:,}")
with col3:
    st.metric("Optimal Output", f"{metrics['optimal_output']:,}")
with col4:
    st.metric("Parts Produced", f"{metrics['parts_output']:,}")

# Capacity & Losses
st.subheader("📉 Capacity Utilization & Losses")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Capacity Utilization", f"{metrics['capacity_utilization']:.2f}%")
with col2:
    st.metric("Total Gap", f"{metrics['total_gap']:,} parts", delta_color="inverse")
with col3:
    st.metric(
        "Availability Loss",
        f"{metrics['total_availability_loss']:,} parts",
        delta_color="inverse",
    )
with col4:
    st.metric(
        "Performance Loss",
        f"{metrics['total_performance_loss']:,} parts",
        delta_color="inverse",
    )

# Downtime Metrics
st.subheader("⏸️ Downtime Analysis")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Stops", f"{metrics['total_stops']:,}")
with col2:
    st.metric("Avg Stops/Session", f"{metrics['avg_stops_per_session']:.1f}")
with col3:
    st.metric("Total Downtime", f"{metrics['total_downtime_hours']:.2f} hours")
with col4:
    st.metric("Production Hours", f"{metrics['total_production_hours']:.2f} hours")

# Visualizations
st.subheader("📊 Capacity & OEE Visualizations")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 OEE Overview", "📈 Trends", "📉 Losses", "⏸️ Stops"]
)

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        oee_gauge = create_oee_gauge(metrics["avg_oee"], metrics["oee_target"])
        st.plotly_chart(oee_gauge, use_container_width=True)

    with col2:
        oee_components = create_oee_components_chart(metrics)
        st.plotly_chart(oee_components, use_container_width=True)

    # OEE Breakdown
    st.markdown("### OEE Formula")
    st.latex(r"OEE = Availability \times Performance \times Quality")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(
            f"**Availability**: {metrics['avg_availability']:.2f}%\n\n"
            "Measures uptime vs total time"
        )
    with col2:
        st.info(
            f"**Performance**: {metrics['avg_performance']:.2f}%\n\n"
            "Measures speed vs optimal speed"
        )
    with col3:
        st.info(
            f"**Quality**: {metrics['avg_quality']:.2f}%\n\n"
            "Measures good parts vs total parts"
        )

with tab2:
    oee_timeline = create_oee_timeline(df)
    st.plotly_chart(oee_timeline, use_container_width=True)

    # Session statistics
    st.markdown("### Session Statistics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        best_session_oee = df["OEE_PCT"].max()
        st.metric("Best Session OEE", f"{best_session_oee:.2f}%")
    with col2:
        worst_session_oee = df["OEE_PCT"].min()
        st.metric("Worst Session OEE", f"{worst_session_oee:.2f}%")
    with col3:
        oee_std = df["OEE_PCT"].std()
        st.metric("OEE Std Dev", f"{oee_std:.2f}%")
    with col4:
        sessions_above_target = len(df[df["OEE_PCT"] >= metrics["oee_target"]])
        pct_above = (
            (sessions_above_target / metrics["total_sessions"] * 100)
            if metrics["total_sessions"] > 0
            else 0
        )
        st.metric("Sessions Above Target", f"{pct_above:.1f}%")

with tab3:
    col1, col2 = st.columns(2)

    with col1:
        losses_chart = create_losses_chart(metrics)
        st.plotly_chart(losses_chart, use_container_width=True)

    with col2:
        capacity_comparison = create_capacity_comparison(metrics)
        st.plotly_chart(capacity_comparison, use_container_width=True)

    # Losses summary
    st.markdown("### Losses Impact Analysis")

    losses_df = pd.DataFrame(
        {
            "Loss Type": ["Availability Loss", "Performance Loss", "Total Gap"],
            "Parts Lost": [
                f"{metrics['total_availability_loss']:,}",
                f"{metrics['total_performance_loss']:,}",
                f"{metrics['total_gap']:,}",
            ],
            "% of Optimal": [
                f"{(metrics['total_availability_loss'] / metrics['optimal_output'] * 100):.2f}%",
                f"{(metrics['total_performance_loss'] / metrics['optimal_output'] * 100):.2f}%",
                f"{(metrics['total_gap'] / metrics['optimal_output'] * 100):.2f}%",
            ],
        }
    )

    st.dataframe(losses_df, use_container_width=True, hide_index=True)

with tab4:
    stops_analysis = create_stops_analysis(df)
    st.plotly_chart(stops_analysis, use_container_width=True)

    # Stops statistics
    st.markdown("### Stops Statistics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        max_stops = df["STOP_COUNT"].max()
        st.metric("Max Stops/Session", f"{max_stops}")
    with col2:
        min_stops = df["STOP_COUNT"].min()
        st.metric("Min Stops/Session", f"{min_stops}")
    with col3:
        avg_stop_duration = (
            df["TOTAL_STOP_TIME_SEC"].mean() / df["STOP_COUNT"].mean() / 60
        )
        st.metric("Avg Stop Duration", f"{avg_stop_duration:.1f} min")
    with col4:
        total_stop_time_pct = (
            df["TOTAL_STOP_TIME_SEC"].sum()
            / (df["SESSION_DURATION_HOURS"].sum() * 3600)
            * 100
        )
        st.metric("Downtime %", f"{total_stop_time_pct:.2f}%")

# Recommendations
st.subheader("💡 AI-Powered Recommendations")

if metrics["avg_oee"] < metrics["oee_target"]:
    gap_pct = metrics["oee_target"] - metrics["avg_oee"]
    st.error(
        f"⚠️ **Below Target OEE**: Current OEE is {metrics['avg_oee']:.2f}%, "
        f"which is {gap_pct:.2f}% below target. Recommended actions:\n"
        f"1. Focus on {'Availability' if metrics['avg_availability'] < 90 else 'Performance' if metrics['avg_performance'] < 95 else 'Quality'}\n"
        f"2. Reduce stop frequency (currently {metrics['avg_stops_per_session']:.1f} per session)\n"
        f"3. Optimize cycle time to match approved CT"
    )
elif metrics["avg_oee"] < (metrics["oee_target"] + 5):
    st.info(
        f"📊 **Near Target OEE**: Current OEE is {metrics['avg_oee']:.2f}%, "
        "close to target. Continue monitoring and fine-tune processes."
    )
else:
    st.success(
        f"✅ **Excellent OEE Performance**: Current OEE is {metrics['avg_oee']:.2f}%, "
        "exceeding target. Consider this as a best practice benchmark!"
    )

# Capacity risk assessment
utilization_pct = metrics["capacity_utilization"]
if utilization_pct < 70:
    st.warning(
        f"⚠️ **High Capacity Risk**: Only {utilization_pct:.1f}% capacity utilization. "
        "Significant room for improvement to meet production targets."
    )
elif utilization_pct < 85:
    st.info(
        f"📊 **Moderate Capacity Risk**: {utilization_pct:.1f}% capacity utilization. "
        "Good performance with potential for optimization."
    )
else:
    st.success(
        f"✅ **Low Capacity Risk**: {utilization_pct:.1f}% capacity utilization. "
        "Excellent equipment performance and capacity management!"
    )

# Download Options
st.subheader("📥 Download Options")

col1, col2, col3 = st.columns(3)

with col1:
    # Session data CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📊 Download Session Data",
        csv,
        "capacity_session_data.csv",
        "text/csv",
        help="Download session-level data as CSV",
    )

with col2:
    # Summary metrics CSV
    summary_data = pd.DataFrame([metrics])
    summary_csv = summary_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📋 Download Summary",
        summary_csv,
        "capacity_summary_metrics.csv",
        "text/csv",
        help="Download summary metrics as CSV",
    )

with col3:
    st.info(
        "💡 In production, you can export to Excel with multi-OEE scenarios (50%-100%)"
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        <p>🎨 <strong>Portfolio Demo</strong> - Using synthetic data for demonstration purposes</p>
        <p>Production version connects to Snowflake with multi-OEE analysis (50%-100% targets)</p>
        <p>Features: OEE Calculation | Session Analysis | Capacity Planning | Multi-Cavity Support</p>
    </div>
    """,
    unsafe_allow_html=True,
)
