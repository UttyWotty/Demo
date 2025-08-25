from __future__ import annotations

import numpy as np
import pandas as pd


def generate_rca_demo(
    start_date: str = "2025-01-01",
    days: int = 30,
    suppliers: int = 3,
    equipment_per_supplier: int = 4,
    parts: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create synthetic shot-level data for RCA demos with fields used by analyses:
    SUPPLIER_NAME, COMPANY_ID, EQUIPMENT_CODE, PART_NAME, CT, APPROVED_CT,
    LOCAL_SHOT_TIME, HOUR, DAY_OF_WEEK, TOOLING_FAMILY (optional default).

    TODO: Add knob to vary issue rates by day/equipment for richer demos.
    """
    rng = np.random.default_rng(seed)
    base = pd.to_datetime(start_date)
    times = pd.date_range(base, periods=days * 24, freq="H")  # hourly shots window

    supplier_names = [f"SUPPLIER_{i + 1}" for i in range(suppliers)]
    company_ids = [f"C{i + 1:03d}" for i in range(suppliers)]
    part_names = [f"PART_{i + 1:02d}" for i in range(parts)]

    rows = []
    for s_idx, supplier in enumerate(supplier_names):
        company = company_ids[s_idx]
        equipment_codes = [
            f"EQ_{s_idx + 1:02d}_{j + 1:02d}" for j in range(equipment_per_supplier)
        ]

        for eq in equipment_codes:
            approved_ct = max(5.0, float(rng.normal(20, 3)))
            eq_bias = float(rng.normal(0, 1.0))

            for ts in times:
                # per-hour shot counts (we'll expand to multiple rows via sampling)
                shot_count = int(rng.integers(50, 200))
                # pick a part
                part = part_names[int(rng.integers(0, len(part_names)))]

                # hour/day effects to create patterns
                hour = ts.hour
                day_name = ts.day_name()
                hour_penalty = 1.0 + (0.1 if hour in (6, 7, 18, 19) else 0)
                day_penalty = 1.0 + (0.15 if day_name in ("Monday", "Friday") else 0)

                avg_ct = max(
                    2.0,
                    (approved_ct + eq_bias + float(rng.normal(0, 0.8)))
                    * hour_penalty
                    * day_penalty,
                )

                # Issue flag with higher chance on tough hours/days
                issue_prob = 0.05 * hour_penalty * day_penalty

                for _ in range(max(5, shot_count // 20)):
                    ct = max(2.0, float(rng.normal(avg_ct, 1.0)))
                    issue = 1 if rng.random() < issue_prob else 0
                    rows.append(
                        {
                            "SUPPLIER_NAME": supplier,
                            "COMPANY_ID": company,
                            "EQUIPMENT_CODE": eq,
                            "PART_NAME": part,
                            "CT": round(ct, 2),
                            "APPROVED_CT": round(approved_ct, 2),
                            "LOCAL_SHOT_TIME": ts,
                            "HOUR": ts.hour,
                            "DAY_OF_WEEK": day_name,
                            "CT_ISSUE_FLAG": issue,
                        }
                    )

    df = pd.DataFrame(rows)
    return df
