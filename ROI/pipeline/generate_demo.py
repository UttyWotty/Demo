from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional


def _generate_demo_raw(
    schema_name: Optional[str] = None,
    start_date: str = "2025-01-01",
    days: int = 30,
    equipment_count: int = 3,
    supplier_count: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic raw ROI-like data for demo mode.

    The output includes columns compatible with the summary pipeline:
    SUPPLIER_NAME, EQUIPMENT_CODE, DAY (int), APPROVED_CT, AVERAGE_CT (CT),
    TOTAL_SHOT_COUNT, VOLUME.

    TODO: Add schema-specific parameter patterns for richer demos (e.g., 'JAGUAR', 'EATON').
    """
    rng = np.random.default_rng(seed)
    base_date = pd.to_datetime(start_date)
    days_idx = pd.date_range(base_date, periods=days, freq="D")

    suppliers = [f"SUPPLIER_{i+1}" for i in range(supplier_count)]
    equipments = [f"TOOL_{i+1:02d}" for i in range(equipment_count)]

    rows = []
    for sup in suppliers:
        for eq in equipments:
            approved_ct = max(5.0, float(rng.normal(20, 5)))
            # small per-tool variance
            tool_bias = float(rng.normal(0, 0.8))
            for d in days_idx:
                shots = int(rng.integers(500, 3000))
                drift = float(rng.normal(0, 1.5))
                avg_ct = max(2.0, approved_ct + tool_bias + drift + float(rng.normal(0, 1.0)))
                rows.append(
                    {
                        "SUPPLIER_NAME": sup,
                        "EQUIPMENT_CODE": eq,
                        "DAY": int(d.strftime("%Y%m%d")),
                        "APPROVED_CT": round(approved_ct, 2),
                        "AVERAGE_CT": round(avg_ct, 2),
                        "CT": round(avg_ct, 2),
                        "TOTAL_SHOT_COUNT": shots,
                        "VOLUME": shots,
                    }
                )

    df = pd.DataFrame(rows)
    if schema_name:
        df["SCHEMA"] = schema_name
    return df


def process_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize raw shot-level-like data into daily per-equipment aggregates and metrics.
    Mirrors the production pipeline’s core transformations for demo compatibility.
    """
    delta = 0.05
    df = df.copy()

    df["CT_CATEGORY"] = np.select(
        [
            np.abs(df["CT"] - df["APPROVED_CT"]) <= df["APPROVED_CT"] * delta,
            df["CT"] > df["APPROVED_CT"] * (1 + delta),
            df["CT"] < df["APPROVED_CT"] * (1 - delta),
        ],
        ["WITHIN", "SLOWER", "FASTER"],
        default="OTHER",
    )
    df["CT_WITHIN"] = (df["CT_CATEGORY"] == "WITHIN").astype(int)
    df["CT_SLOWER"] = (df["CT_CATEGORY"] == "SLOWER").astype(int)
    df["CT_FASTER"] = (df["CT_CATEGORY"] == "FASTER").astype(int)

    grouped = (
        df.groupby(["SUPPLIER_NAME", "EQUIPMENT_CODE", "DAY"])
        .agg(
            TOTAL_SHOTS=("TOTAL_SHOT_COUNT", "sum"),
            PARTS_PRODUCED=("VOLUME", "sum"),
            APPROVED_CT=("APPROVED_CT", "first"),
            AVERAGE_CT=(
                "AVERAGE_CT",
                lambda x: np.average(x, weights=df.loc[x.index, "TOTAL_SHOT_COUNT"]).round(2),
            ),
            WITHIN_SHOT_COUNT=("CT_WITHIN", "sum"),
            SLOWER_SHOT_COUNT=("CT_SLOWER", "sum"),
            FASTER_SHOT_COUNT=("CT_FASTER", "sum"),
        )
        .reset_index()
    )

    slow_ct = (
        df[df["CT_CATEGORY"] == "SLOWER"]
        .groupby(["SUPPLIER_NAME", "EQUIPMENT_CODE", "DAY"])
        .apply(lambda g: np.average(g["AVERAGE_CT"], weights=g["TOTAL_SHOT_COUNT"]))
        .rename("AVG_CT_SLOW")
    )
    fast_ct = (
        df[df["CT_CATEGORY"] == "FASTER"]
        .groupby(["SUPPLIER_NAME", "EQUIPMENT_CODE", "DAY"])
        .apply(lambda g: np.average(g["AVERAGE_CT"], weights=g["TOTAL_SHOT_COUNT"]))
        .rename("AVG_CT_FAST")
    )

    grouped = grouped.merge(slow_ct, on=["SUPPLIER_NAME", "EQUIPMENT_CODE", "DAY"], how="left")
    grouped = grouped.merge(fast_ct, on=["SUPPLIER_NAME", "EQUIPMENT_CODE", "DAY"], how="left")
    grouped["AVG_CT_SLOW"] = grouped["AVG_CT_SLOW"].fillna(grouped["APPROVED_CT"])
    grouped["AVG_CT_FAST"] = grouped["AVG_CT_FAST"].fillna(grouped["APPROVED_CT"])

    # Coefficient of variation-based stability
    daily_stats = (
        df.groupby(["SUPPLIER_NAME", "EQUIPMENT_CODE", "DAY"])
        .agg(CT_STD=("AVERAGE_CT", "std"), CT_MEAN=("AVERAGE_CT", "mean"))
        .reset_index()
    )
    daily_stats["CT_CV"] = np.where(
        daily_stats["CT_MEAN"] > 0, (daily_stats["CT_STD"] / daily_stats["CT_MEAN"] * 100), 0
    )
    daily_stats["CT_CV"] = daily_stats["CT_CV"].fillna(0)

    # Molding-oriented grading to 0-100
    cv = daily_stats["CT_CV"]
    stability = np.where(
        cv <= 20,
        100 - (cv * 0.5),
        np.where(
            cv <= 35,
            90 - ((cv - 20) * 1.33),
            np.where(
                cv <= 50,
                70 - ((cv - 35) * 1.33),
                np.where(cv <= 70, 50 - ((cv - 50) * 1.25), np.maximum(0, 25 - ((cv - 70) * 0.5))),
            ),
        ),
    )
    daily_stats["STABILITY_FROM_CV"] = np.clip(np.round(stability, 2), 0, 100)

    grouped = grouped.merge(
        daily_stats[["SUPPLIER_NAME", "EQUIPMENT_CODE", "DAY", "STABILITY_FROM_CV", "CT_CV"]],
        on=["SUPPLIER_NAME", "EQUIPMENT_CODE", "DAY"],
        how="left",
    )

    grouped["DIFFERENCE"] = (grouped["APPROVED_CT"] - grouped["AVERAGE_CT"]).round(2)
    grouped["WITHIN_SHOT_PCT"] = (grouped["WITHIN_SHOT_COUNT"] / grouped["TOTAL_SHOTS"] * 100).round(2)
    grouped["SLOWER_SHOT_PCT"] = (grouped["SLOWER_SHOT_COUNT"] / grouped["TOTAL_SHOTS"] * 100).round(2)
    grouped["FASTER_SHOT_PCT"] = (grouped["FASTER_SHOT_COUNT"] / grouped["TOTAL_SHOTS"] * 100).round(2)

    grouped["EXPECTED_HOURS"] = ((grouped["APPROVED_CT"] * grouped["TOTAL_SHOTS"]) / 3600).round(2)
    grouped["USED_HOURS"] = ((grouped["AVERAGE_CT"] * grouped["TOTAL_SHOTS"]) / 3600).round(2).fillna(0)
    grouped["HOURS_DIFF"] = (grouped["EXPECTED_HOURS"] - grouped["USED_HOURS"]).round(2).fillna(0)

    grouped["EXPECTED_HOURS_FAST"] = ((grouped["APPROVED_CT"] * grouped["FASTER_SHOT_COUNT"]) / 3600).round(2)
    grouped["EXPECTED_HOURS_SLOW"] = ((grouped["APPROVED_CT"] * grouped["SLOWER_SHOT_COUNT"]) / 3600).round(2)
    grouped["USED_HOURS_SLOW"] = ((grouped["AVG_CT_SLOW"] * grouped["SLOWER_SHOT_COUNT"]) / 3600).round(2)
    grouped["USED_HOURS_FAST"] = ((grouped["AVG_CT_FAST"] * grouped["FASTER_SHOT_COUNT"]) / 3600).round(2)

    grouped["GAIN_HOURS"] = np.clip((grouped["EXPECTED_HOURS_FAST"] - grouped["USED_HOURS_FAST"]).round(2), 0, None).fillna(0)
    grouped["LOSS_HOURS"] = np.clip((grouped["USED_HOURS_SLOW"] - grouped["EXPECTED_HOURS_SLOW"]).round(2), 0, None).fillna(0)

    grouped["TOOLING_EFFICIENCY(%)"] = np.where(
        grouped["USED_HOURS"] > 0,
        (grouped["TOTAL_SHOTS"] * grouped["APPROVED_CT"]) / (grouped["USED_HOURS"] * 3600) * 100,
        0,
    ).round(2)

    grouped["PROCESS_STABILITY"] = grouped["STABILITY_FROM_CV"].fillna(0)
    grouped["CT_COEFFICIENT_OF_VARIATION"] = grouped["CT_CV"].fillna(0)

    # Tidy columns (place DIFFERENCE after AVERAGE_CT)
    cols = grouped.columns.tolist()
    if "DIFFERENCE" in cols and "AVERAGE_CT" in cols:
        cols.remove("DIFFERENCE")
        avg_idx = cols.index("AVERAGE_CT")
        cols.insert(avg_idx + 1, "DIFFERENCE")
        grouped = grouped[cols]

    return grouped


def get_demo_summary(
    schema_name: Optional[str] = None,
    start_date: str = "2025-01-01",
    days: int = 30,
    equipment_count: int = 3,
    supplier_count: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build demo summary by generating synthetic raw data and applying process_summary.
    """
    raw_df = _generate_demo_raw(
        schema_name=schema_name,
        start_date=start_date,
        days=days,
        equipment_count=equipment_count,
        supplier_count=supplier_count,
        seed=seed,
    )
    return process_summary(raw_df)