from __future__ import annotations

from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from pipeline.generate_demo import generate_product_demo
from analyzer import preprocess_data, process_session, summarize_session


@st.cache_data(ttl=1800)
def load_data(
    start_date: str, days: int, suppliers: int, equipment_per_supplier: int, seed: int
) -> pd.DataFrame:
    return generate_product_demo(
        start_date=start_date,
        days=days,
        suppliers=suppliers,
        equipment_per_supplier=equipment_per_supplier,
        seed=seed,
    )


def main():
    st.set_page_config(page_title="Product Analysis Demo", layout="wide")
    st.title("Product Analysis Demo")
    st.caption("Synthetic session analytics (no external services).")

    with st.sidebar:
        st.subheader("Demo Parameters")
        start_date = st.date_input("Start date", datetime(2025, 1, 1)).strftime(
            "%Y-%m-%d"
        )
        days = st.slider("Days", 3, 30, 10, 1)
        suppliers = st.slider("Suppliers", 1, 6, 3, 1)
        equipment = st.slider("Equipments per supplier", 1, 6, 4, 1)
        seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42)

    df = load_data(start_date, days, suppliers, equipment, seed)
    st.write(
        f"Records: {len(df):,} | Suppliers: {df['SUPPLIER_NAME'].nunique()} | Equipment: {df['EQUIPMENT_CODE'].nunique()}"
    )

    # Session analytics
    df_prep = preprocess_data(df)
    df_proc = df_prep.groupby(["EQUIPMENT_CODE", "SESSION_ID"], group_keys=False).apply(
        process_session
    )
    session_summary = (
        df_proc.groupby(["EQUIPMENT_CODE", "SESSION_ID"], group_keys=False)
        .apply(summarize_session)
        .reset_index(drop=True)
    )

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sessions", f"{session_summary['SESSION_ID'].nunique():,}")
    with col2:
        st.metric("Avg Uptime (%)", f"{session_summary['Uptime_Pct'].mean():.1f}")
    with col3:
        st.metric("Avg Downtime (h)", f"{session_summary['Total_Downtime'].mean():.2f}")
    with col4:
        st.metric("Avg Stops", f"{session_summary['Stops'].mean():.1f}")

    st.subheader("Top Sessions by Downtime")
    top = session_summary.sort_values("Total_Downtime", ascending=False).head(20)
    fig = px.bar(
        top,
        x="EQUIPMENT_CODE",
        y="Total_Downtime",
        color="SUPPLIER_NAME",
        hover_data=["SESSION_ID"],
        title="Top Sessions by Downtime (hours)",
    )
    fig.update_layout(template="plotly_white", height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Session Summary (sample)")
    st.dataframe(session_summary.head(200))

    st.caption("Demo data is synthetic. Figures are for demonstration only.")


if __name__ == "__main__":
    main()
