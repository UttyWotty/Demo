from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import json
import os

import numpy as np
import pandas as pd


@dataclass
class Target:
    name: str
    code: str
    type: str  # Day, Equipment, Part
    issues: int
    total: int
    rate: float


class RCADemoAnalyzer:
    """Simplified Pareto-like targeting and 5 Whys analysis for demo data."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def identify_top_targets(self, top_n: int = 3) -> List[Target]:
        targets: List[Target] = []

        # 1) Day-of-week
        day = (
            self.df.groupby("DAY_OF_WEEK")
            .agg({"CT_ISSUE_FLAG": ["sum", "count"]})
            .round(2)
        )
        day.columns = ["Issues", "Total_Shots"]
        day["Issue_Rate"] = (day["Issues"] / day["Total_Shots"] * 100).round(2)
        day = day.sort_values("Issue_Rate", ascending=False)
        for d, row in day.head(top_n).iterrows():
            targets.append(
                Target(
                    name=f"Day {d}",
                    code=str(d),
                    type="Day",
                    issues=int(row["Issues"]),
                    total=int(row["Total_Shots"]),
                    rate=float(row["Issue_Rate"]),
                )
            )

        # 2) Equipment if needed
        if len(targets) < top_n:
            eq = (
                self.df.groupby("EQUIPMENT_CODE")
                .agg({"CT_ISSUE_FLAG": ["sum", "count"]})
                .round(2)
            )
            eq.columns = ["Issues", "Total_Shots"]
            eq["Issue_Rate"] = (eq["Issues"] / eq["Total_Shots"] * 100).round(2)
            eq = eq.sort_values("Issue_Rate", ascending=False)
            for e, row in eq.head(top_n - len(targets)).iterrows():
                targets.append(
                    Target(
                        name=f"Equipment {e}",
                        code=str(e),
                        type="Equipment",
                        issues=int(row["Issues"]),
                        total=int(row["Total_Shots"]),
                        rate=float(row["Issue_Rate"]),
                    )
                )

        # 3) Part if still needed
        if len(targets) < top_n:
            parts = (
                self.df.groupby("PART_NAME")
                .agg({"CT_ISSUE_FLAG": ["sum", "count"]})
                .round(2)
            )
            parts.columns = ["Issues", "Total_Shots"]
            parts["Issue_Rate"] = (parts["Issues"] / parts["Total_Shots"] * 100).round(
                2
            )
            parts = parts.sort_values("Issue_Rate", ascending=False)
            for p, row in parts.head(top_n - len(targets)).iterrows():
                targets.append(
                    Target(
                        name=f"Part {p}",
                        code=str(p),
                        type="Part",
                        issues=int(row["Issues"]),
                        total=int(row["Total_Shots"]),
                        rate=float(row["Issue_Rate"]),
                    )
                )

        return targets[:top_n]

    def five_whys(self, t: Target) -> Dict:
        """Generate a simple 5 Whys based on basic patterns."""
        df = self._subset(t)
        analysis = {
            "target": t.name,
            "type": t.type,
            "whys": [],
            "root_cause": "",
            "supporting_data": {},
            "recommendations": [],
        }

        # Basic metrics
        ct_mean = df["CT"].mean() if not df.empty else 0
        eff = (
            (df["APPROVED_CT"] / df["CT"]).clip(upper=1).mean() * 100
            if not df.empty
            else 0
        )
        issue_rate = (
            (df["CT_ISSUE_FLAG"].mean() * 100)
            if "CT_ISSUE_FLAG" in df.columns and not df.empty
            else 0
        )

        analysis["supporting_data"] = {
            "avg_ct": float(ct_mean),
            "efficiency_pct": float(eff),
            "issue_rate_pct": float(issue_rate),
            "samples": int(len(df)),
        }

        # Simple why chain
        analysis["whys"].append(
            f"Performance shows {issue_rate:.1f}% CT issues and avg CT {ct_mean:.1f}s"
        )
        analysis["whys"].append("Specific hours/days/tools show higher variation")
        analysis["whys"].append("Operating parameters vary outside of optimal ranges")
        analysis["whys"].append("Procedures and training differ across shifts")
        analysis["whys"].append(
            "Systematic process and maintenance controls need improvement"
        )

        analysis["root_cause"] = (
            "Operational variability due to non-standardized procedures"
        )
        analysis["recommendations"] = [
            "Standardize operating procedures",
            "Implement shift-aligned training",
            "Tune process parameters and monitor",
            "Plan preventive maintenance on worst tools",
            "Establish daily performance review",
        ]

        return analysis

    def _subset(self, t: Target) -> pd.DataFrame:
        if t.type.lower() == "day":
            return self.df[self.df["DAY_OF_WEEK"] == t.code]
        if t.type.lower() == "equipment":
            return self.df[self.df["EQUIPMENT_CODE"] == t.code]
        if t.type.lower() == "part":
            return self.df[self.df["PART_NAME"] == t.code]
        return self.df

    def run(self, top_n: int = 3) -> Dict:
        targets = self.identify_top_targets(top_n)
        results: Dict[str, Dict] = {}
        for t in targets:
            results[t.name] = self.five_whys(t)

        os.makedirs("data", exist_ok=True)
        out = os.path.join("data", "five_whys_results.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        return results

    def create_report(self, results: Dict[str, Dict]) -> str:
        os.makedirs("data", exist_ok=True)
        path = os.path.join("data", "rca_report.html")
        rows = []
        for name, res in results.items():
            whys = "".join([f"<li>{w}</li>" for w in res.get("whys", [])])
            recs = "".join([f"<li>{r}</li>" for r in res.get("recommendations", [])])
            sd = res.get("supporting_data", {})
            rows.append(
                f"<div class='card'><h3>{name}</h3><p><b>Type:</b> {res.get('type')}</p>"
                f"<p><b>Avg CT:</b> {sd.get('avg_ct', 0):.2f}s, <b>Efficiency:</b> {sd.get('efficiency_pct', 0):.1f}%, "
                f"<b>Issue Rate:</b> {sd.get('issue_rate_pct', 0):.1f}%</p>"
                f"<p><b>Root Cause:</b> {res.get('root_cause')}</p>"
                f"<p><b>Whys:</b><ul>{whys}</ul></p>"
                f"<p><b>Recommendations:</b><ul>{recs}</ul></p></div>"
            )

        html = (
            "<html><head><meta charset='utf-8'><title>RCA Demo Report</title>"
            "<style>body{font-family:Arial;margin:20px;background:#f5f5f5}.card{background:#fff;border-radius:8px;padding:16px;margin:12px 0;box-shadow:0 2px 4px rgba(0,0,0,.08)}</style>"
            "</head><body><h1>RCA Demo Report</h1>" + "".join(rows) + "</body></html>"
        )
        with open(path, "w") as f:
            f.write(html)
        return path
