from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


@dataclass
class SupplierBenchmark:
    supplier_id: str
    supplier_name: str
    mean_normalized_efficiency: float
    tool_consistency_score: float
    total_tools: int
    performance_rank: int
    tier_classification: str
    adjusted_score: float = 0.0


class CTEfficiencyDemoAnalyzer:
    """Minimal benchmarking flow on synthetic data (no external deps)."""

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["tool_id"] = df["EQUIPMENT_CODE"].astype(str)
        df = df[df["CT"] > 0]
        df = df[df["APPROVED_CT"] > 0]
        # Trim extremes
        q1, q3 = df["CT"].quantile(0.01), df["CT"].quantile(0.99)
        df = df[(df["CT"] >= q1) & (df["CT"] <= q3)]
        return df

    def compute_efficiency(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["efficiency"] = df["APPROVED_CT"] / df["CT"]
        df["efficiency_pct"] = (df["efficiency"] - 1) * 100
        df.loc[df["efficiency_pct"] > 100, "efficiency_pct"] = 100
        df.loc[df["efficiency_pct"] < -50, "efficiency_pct"] = -50
        return df

    def aggregate_per_tool(self, df: pd.DataFrame) -> pd.DataFrame:
        tool_metrics = (
            df.groupby(
                [
                    "tool_id",
                    "EQUIPMENT_CODE",
                    "MOLD_ID",
                    "TOOLING_FAMILY",
                    "SUPPLIER_NAME",
                    "COMPANY_ID",
                ]
            )
            .agg(
                {
                    "efficiency_pct": ["mean", "std", "count"],
                    "CT": ["mean", "std"],
                    "APPROVED_CT": "first",
                }
            )
            .round(3)
        )
        tool_metrics.columns = ["_".join(col).strip() for col in tool_metrics.columns]
        tool_metrics = tool_metrics.reset_index()
        tool_metrics["efficiency_cov"] = (
            tool_metrics["efficiency_pct_std"]
            / tool_metrics["efficiency_pct_mean"].abs()
        ) * 100
        tool_metrics["ct_cov"] = tool_metrics["CT_std"] / tool_metrics["CT_mean"] * 100
        tool_metrics.replace([np.inf, -np.inf], np.nan, inplace=True)
        return tool_metrics

    def normalize_within_family(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["efficiency_percentile"] = df.groupby("TOOLING_FAMILY")[
            "efficiency_pct_mean"
        ].transform(lambda x: x.rank(pct=True))
        return df

    def aggregate_supplier(self, df: pd.DataFrame) -> List[SupplierBenchmark]:
        benchmarks: List[SupplierBenchmark] = []
        for supplier_id in df["COMPANY_ID"].unique():
            sdata = df[df["COMPANY_ID"] == supplier_id]
            supplier_name = sdata["SUPPLIER_NAME"].iloc[0]
            mean_norm = sdata["efficiency_percentile"].mean()
            std_eff = sdata["efficiency_pct_std"].mean()
            tool_consistency = (
                1.0 if pd.isna(std_eff) or std_eff == 0 else 1 / (1 + std_eff)
            )
            b = SupplierBenchmark(
                supplier_id=str(supplier_id),
                supplier_name=supplier_name,
                mean_normalized_efficiency=float(mean_norm),
                tool_consistency_score=float(tool_consistency),
                total_tools=len(sdata),
                performance_rank=0,
                tier_classification="",
            )
            benchmarks.append(b)

        # Adjusted score and ranking
        for b in benchmarks:
            penalty = min(1.0, b.total_tools / 3.0)
            b.adjusted_score = b.mean_normalized_efficiency * penalty
        benchmarks.sort(key=lambda x: x.adjusted_score, reverse=True)
        for i, b in enumerate(benchmarks):
            b.performance_rank = i + 1
            b.tier_classification = (
                "Excellent"
                if b.adjusted_score >= 0.8
                else "Good"
                if b.adjusted_score >= 0.6
                else "Average"
                if b.adjusted_score >= 0.4
                else "Needs Improvement"
                if b.adjusted_score >= 0.2
                else "Poor"
            )
        return benchmarks

    def create_plots(
        self, tool_metrics: pd.DataFrame, benchmarks: List[SupplierBenchmark]
    ) -> list[str]:
        os.makedirs("data", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths: list[str] = []

        # Supplier ranking
        fig = go.Figure()
        names = [b.supplier_name for b in benchmarks]
        scores = [b.mean_normalized_efficiency for b in benchmarks]
        colors = [
            (
                "green"
                if b.tier_classification in ("Excellent", "Good")
                else "orange"
                if b.tier_classification == "Average"
                else "red"
            )
            for b in benchmarks
        ]
        fig.add_trace(
            go.Bar(
                x=names,
                y=scores,
                marker_color=colors,
                text=[
                    f"{b.tier_classification}<br>Rank: {b.performance_rank}"
                    for b in benchmarks
                ],
                textposition="auto",
            )
        )
        fig.update_layout(
            title="Supplier Performance Ranking",
            xaxis_title="Supplier",
            yaxis_title="Normalized Efficiency Score",
            height=600,
        )
        p = f"data/benchmarking_supplier_ranking_{ts}.html"
        fig.write_html(p)
        paths.append(p)

        # Family box
        fig = px.box(
            tool_metrics,
            x="TOOLING_FAMILY",
            y="efficiency_pct_mean",
            title="Efficiency Distribution by Tooling Family",
        )
        fig.update_layout(height=500)
        p = f"data/benchmarking_tooling_family_{ts}.html"
        fig.write_html(p)
        paths.append(p)

        # Efficiency vs consistency
        fig = px.scatter(
            tool_metrics,
            x="efficiency_pct_mean",
            y="efficiency_cov",
            color="TOOLING_FAMILY",
            size="efficiency_pct_count",
            title="Efficiency vs Consistency by Tooling Family",
            labels={
                "efficiency_pct_mean": "Mean Efficiency (%)",
                "efficiency_cov": "Coefficient of Variation (%)",
            },
        )
        p = f"data/benchmarking_efficiency_consistency_{ts}.html"
        fig.write_html(p)
        paths.append(p)

        # Tier pie
        tiers = pd.Series([b.tier_classification for b in benchmarks]).value_counts()
        fig = px.pie(
            values=tiers.values,
            names=tiers.index,
            title="Supplier Performance Tier Distribution",
        )
        p = f"data/benchmarking_tier_distribution_{ts}.html"
        fig.write_html(p)
        paths.append(p)

        return paths

    def create_report(
        self,
        tool_metrics: pd.DataFrame,
        benchmarks: List[SupplierBenchmark],
        plots: list[str],
    ) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = f"data/supplier_benchmarking_report_{ts}.html"

        html = [
            "<!DOCTYPE html>",
            "<html><head><meta charset='utf-8'><title>Supplier Benchmarking Report</title>",
            "<style>body{font-family:Arial;margin:20px;background:#f5f5f5}.header{background:#1f4e79;color:#fff;padding:20px;border-radius:10px;margin-bottom:20px}.section{background:#fff;border-radius:8px;padding:20px;margin:20px 0;box-shadow:0 2px 4px rgba(0,0,0,.1)}.metric-card{background:#e3f2fd;border-left:4px solid #2196F3;padding:15px;margin:10px 0;border-radius:5px}.supplier-table{width:100%;border-collapse:collapse;margin:20px 0}.supplier-table th,.supplier-table td{border:1px solid #ddd;padding:12px;text-align:left}.supplier-table th{background:#1f4e79;color:#fff}.excellent{background:#d4edda}.good{background:#d1ecf1}.average{background:#fff3cd}.needs-improvement{background:#f8d7da}.poor{background:#f5c6cb}</style>",
            "</head><body>",
            f"<div class='header'><h1>🏭 Supplier Benchmarking Report</h1><p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p></div>",
            "<div class='section'><h2>📊 Executive Summary</h2>",
            f"<div class='metric-card'><p><strong>Total Suppliers Analyzed:</strong> {len(benchmarks)}</p>",
            f"<p><strong>Total Tools Analyzed:</strong> {len(tool_metrics)}</p>",
            f"<p><strong>Tooling Families:</strong> {tool_metrics['TOOLING_FAMILY'].nunique()}</p></div></div>",
            "<div class='section'><h2>🏆 Top Performing Suppliers</h2><table class='supplier-table'><tr><th>Rank</th><th>Supplier</th><th>Efficiency Score</th><th>Tool Consistency</th><th>Total Tools</th><th>Adjusted Score</th><th>Tier</th></tr>",
        ]
        for b in benchmarks[:10]:
            cls = b.tier_classification.lower().replace(" ", "-")
            html.append(
                f"<tr class='{cls}'><td>{b.performance_rank}</td><td>{b.supplier_name}</td>"
                f"<td>{b.mean_normalized_efficiency:.1%}</td><td>{b.tool_consistency_score:.1%}</td>"
                f"<td>{b.total_tools}</td><td>{b.adjusted_score:.1%}</td><td>{b.tier_classification}</td></tr>"
            )
        html.append("</table></div>")
        html.append("<div class='section'><h2>📈 Performance Analysis</h2>")
        for p in plots:
            fname = os.path.basename(p)
            html.append(
                f"<div class='metric-card'><h3>{fname.split('_')[-2].replace('-', ' ').title()}</h3><iframe src='{fname}' width='100%' height='500px' frameborder='0'></iframe></div>"
            )
        html.append("</div></body></html>")

        os.makedirs("data", exist_ok=True)
        with open(report, "w") as f:
            f.write("\n".join(html))
        return report
