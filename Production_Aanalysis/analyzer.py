from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import mode


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["LOCAL_SHOT_TIME"] = pd.to_datetime(df["LOCAL_SHOT_TIME"])
    df = df.sort_values(["EQUIPMENT_CODE", "LOCAL_SHOT_TIME"])
    df = df.drop_duplicates(
        subset=["EQUIPMENT_CODE", "LOCAL_SHOT_TIME", "SUPPLIER_NAME"], keep="first"
    )
    df["SHOT_DIFF"] = (
        df.groupby("EQUIPMENT_CODE")["LOCAL_SHOT_TIME"].diff().dt.total_seconds()
    )
    df["NEW_SESSION"] = (df["SHOT_DIFF"] >= 28800).astype(int)
    df["SESSION_ID"] = df.groupby("EQUIPMENT_CODE")["NEW_SESSION"].cumsum()
    return df


def process_session(session_df: pd.DataFrame) -> pd.DataFrame:
    valid_diffs = session_df["SHOT_DIFF"].dropna()
    valid_diffs = valid_diffs[valid_diffs > 1]
    m = (
        mode(valid_diffs.round(), keepdims=True).mode[0]
        if not valid_diffs.empty
        else np.nan
    )
    lower = m * 0.95 if not np.isnan(m) else np.nan
    upper = m * 1.05 if not np.isnan(m) else np.nan

    out = session_df.copy()
    out["STOP"] = 0
    mask = out["SHOT_DIFF"].notna()
    out.loc[mask, "STOP"] = (
        (out.loc[mask, "SHOT_DIFF"] < lower) | (out.loc[mask, "SHOT_DIFF"] > upper)
    ).astype(int)
    out["MODE_CT"] = m
    out["LOWER_LIMIT"] = lower
    out["UPPER_LIMIT"] = upper
    return out


def calculate_run_stats(diffs_df: pd.DataFrame) -> float:
    run_durations = []
    current_run = 0.0
    for _, row in diffs_df.iterrows():
        if row["STOP"] == 0:
            current_run += row["SHOT_DIFF"]
        else:
            if current_run > 0:
                run_durations.append(current_run)
                current_run = 0.0
    if current_run > 0:
        run_durations.append(current_run)
    return float(np.mean(run_durations) if run_durations else 0.0)


def summarize_session(session_df: pd.DataFrame) -> pd.Series:
    s = session_df.copy()
    mode_ct = s["MODE_CT"].iloc[0]
    act_cts = s["ACTUAL_CT"]
    act_cts = act_cts[act_cts < 999]
    act = float(act_cts.mean()) if not act_cts.empty else np.nan
    min_ct = act * 0.95 if not np.isnan(act) else np.nan
    max_ct = act * 1.05 if not np.isnan(act) else np.nan
    diffs_df = s.loc[s["SHOT_DIFF"].notna()].copy()
    prod_time = float(diffs_df.loc[diffs_df["STOP"] == 0, "SHOT_DIFF"].sum())
    start_time = s["LOCAL_SHOT_TIME"].min()
    end_time = s["LOCAL_SHOT_TIME"].max()
    total_duration = (end_time - start_time).total_seconds()
    if not np.isnan(mode_ct):
        total_duration += mode_ct
    downtime = max(0.0, total_duration - prod_time)
    num_stops = int(diffs_df["STOP"].sum())
    uptime_pct = (prod_time / total_duration) * 100 if total_duration > 0 else 0
    downtime_pct = max(0.0, 100 - uptime_pct)
    total_shots = int(len(s))
    valid_shots = total_shots - num_stops
    is_valid = int(
        total_shots > 0 and total_duration >= 600 and 50 <= (mode_ct or 0) <= 600
    )
    avg_run_duration = calculate_run_stats(diffs_df)

    return pd.Series(
        {
            "SUPPLIER_NAME": s["SUPPLIER_NAME"].iloc[0],
            "EQUIPMENT_CODE": s["EQUIPMENT_CODE"].iloc[0],
            "SESSION_ID": s["SESSION_ID"].iloc[0],
            "Session_Start": start_time,
            "Session_End": end_time,
            "APPROVED_CT": s["APPROVED_CT"].iloc[0],
            "ACT": round(act, 2) if not np.isnan(act) else np.nan,
            "Mode_CT": round(mode_ct, 2) if not np.isnan(mode_ct) else np.nan,
            "Min_CT": round(min_ct, 2) if not np.isnan(min_ct) else np.nan,
            "Max_CT": round(max_ct, 2) if not np.isnan(max_ct) else np.nan,
            "Production_Time": round(prod_time / 3600, 2),
            "Total_Downtime": round(downtime / 3600, 2),
            "Total_Duration": round(total_duration / 3600, 2),
            "Uptime_Pct": round(uptime_pct, 1),
            "Downtime_Pct": round(downtime_pct, 1),
            "Total_Shots": total_shots,
            "Valid_Shots": valid_shots,
            "Stops": num_stops,
            "Avg_Run_Duration(min)": round(avg_run_duration / 60, 2),
            "Is_Valid_Session": is_valid,
        }
    )
