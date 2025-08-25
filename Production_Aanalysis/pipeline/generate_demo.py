from __future__ import annotations

import numpy as np
import pandas as pd


def generate_product_demo(
    start_date: str = "2025-01-01",
    days: int = 7,
    suppliers: int = 2,
    equipment_per_supplier: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic shot-level data for product session analytics.

    Columns: SUPPLIER_NAME, EQUIPMENT_CODE, LOCAL_SHOT_TIME, ACTUAL_CT, APPROVED_CT

    TODO: Enrich with part info and per-shift parameter patterns for richer demos.
    """
    rng = np.random.default_rng(seed)
    base = pd.to_datetime(start_date)
    times = pd.date_range(base, periods=days * 24, freq="H")

    rows = []
    for s in range(suppliers):
        supplier = f"SUPPLIER_{s + 1}"
        for e in range(equipment_per_supplier):
            eq = f"EQ_{s + 1:02d}_{e + 1:02d}"
            approved_ct = max(10.0, float(rng.normal(120, 20)))
            # vary CT by hour to create realistic gaps and stops
            for ts in times:
                hour = ts.hour
                # expected inter-shot time in seconds ~ approved_ct with variation by hour
                bias = 0
                if hour in (6, 7, 18, 19):
                    bias = rng.normal(20, 10)  # busier periods
                elif hour in (2, 3):
                    bias = rng.normal(-10, 8)  # faster periods
                actual_ct = max(30.0, approved_ct + bias + float(rng.normal(0, 10)))
                # expand to multiple shots per hour (sample small number for demo)
                for _ in range(int(rng.integers(3, 8))):
                    rows.append(
                        {
                            "SUPPLIER_NAME": supplier,
                            "EQUIPMENT_CODE": eq,
                            "LOCAL_SHOT_TIME": ts,
                            "ACTUAL_CT": round(actual_ct, 2),
                            "APPROVED_CT": round(approved_ct, 2),
                        }
                    )

    df = pd.DataFrame(rows)
    # slight jitter to times to avoid perfect alignment
    df["LOCAL_SHOT_TIME"] = pd.to_datetime(df["LOCAL_SHOT_TIME"]) + pd.to_timedelta(
        np.random.default_rng(seed + 1).integers(0, 3600, size=len(df)), unit="s"
    )
    return df.sort_values(["EQUIPMENT_CODE", "LOCAL_SHOT_TIME"]).reset_index(drop=True)
