import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class ActionPriority(Enum):
    """Priority levels for recommended actions."""
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class ActionCategory(Enum):
    """Categories for recommended actions."""
    PROCESS_OPTIMIZATION = "Process Optimization"
    MAINTENANCE = "Maintenance"
    TRAINING = "Training"
    EQUIPMENT_UPGRADE = "Equipment Upgrade"
    MONITORING = "Monitoring"
    INVESTIGATION = "Investigation"


@dataclass
class RecommendedAction:
    """Data class for a single recommended action."""
    title: str
    description: str
    priority: ActionPriority
    category: ActionCategory
    impact_score: float  # 0-100
    estimated_cost: Optional[float] = None
    estimated_savings: Optional[float] = None
    timeline: str = "Immediate"
    responsible_party: str = "Production Team"


class ActionRecommender:
    """
    Generates recommended actions based on ROI analysis results.

    TODO: Expose thresholds via UI and environment for more flexible demos.
    """

    def __init__(self):
        self.thresholds = {
            "efficiency_critical": 70,
            "efficiency_warning": 85,
            "stability_critical": 60,
            "stability_warning": 80,
            "roi_loss_critical": 10000,
            "roi_loss_warning": 5000,
            "ct_deviation_critical": 20,
            "ct_deviation_warning": 10,
        }

    def generate_recommendations(
        self,
        equipment_code: str,
        supplier_name: str,
        weighted_eff: float,
        weighted_ct: float,
        weighted_stability: float,
        roi_loss: float,
        roi_gain: float,
        net_roi: float,
        weighted_diff: float,
        approved_ct: float,
        total_shot_count: int,
        filtered_data: pd.DataFrame,
    ) -> List[RecommendedAction]:
        """Generate a list of recommendations based on KPIs and trends."""
        recommendations: List[RecommendedAction] = []

        ct_deviation_pct = (abs(weighted_diff / approved_ct * 100) if approved_ct > 0 else 0)
        efficiency_gap = 100 - weighted_eff
        stability_gap = 100 - weighted_stability

        # Efficiency
        if weighted_eff < self.thresholds["efficiency_critical"]:
            recommendations.append(
                RecommendedAction(
                    title="Immediate Process Optimization Required",
                    description=f"Equipment {equipment_code} efficiency is critically low at {weighted_eff:.1f}%.",
                    priority=ActionPriority.CRITICAL,
                    category=ActionCategory.PROCESS_OPTIMIZATION,
                    impact_score=95,
                    estimated_cost=5000,
                    estimated_savings=efficiency_gap * 1000,
                    timeline="1-2 weeks",
                    responsible_party="Production Engineering",
                )
            )
        elif weighted_eff < self.thresholds["efficiency_warning"]:
            recommendations.append(
                RecommendedAction(
                    title="Process Efficiency Improvement",
                    description=f"Equipment {equipment_code} efficiency at {weighted_eff:.1f}% is below optimal levels.",
                    priority=ActionPriority.HIGH,
                    category=ActionCategory.PROCESS_OPTIMIZATION,
                    impact_score=75,
                    estimated_cost=2000,
                    estimated_savings=efficiency_gap * 500,
                    timeline="2-4 weeks",
                    responsible_party="Production Team",
                )
            )

        # Stability
        if weighted_stability < self.thresholds["stability_critical"]:

            recommendations.append(
                RecommendedAction(
                    title="Process Stability Investigation",
                    description=f"Critical process instability detected ({weighted_stability:.1f}%).",
                    priority=ActionPriority.CRITICAL,
                    category=ActionCategory.INVESTIGATION,
                    impact_score=90,
                    estimated_cost=8000,
                    timeline="1-3 weeks",
                    responsible_party="Quality Engineering",
                )
            )
        elif weighted_stability < self.thresholds["stability_warning"]:
            recommendations.append(
                RecommendedAction(
                    title="Process Monitoring Enhancement",
                    description=f"Process stability below optimal levels ({weighted_stability:.1f}%).",
                    priority=ActionPriority.MEDIUM,
                    category=ActionCategory.MONITORING,
                    impact_score=60,
                    estimated_cost=1500,
                    timeline="2-3 weeks",
                    responsible_party="Maintenance Team",
                )
            )

        # ROI
        if net_roi < 0:
            recommendations.append(
                RecommendedAction(
                    title="Profitability Recovery Plan",
                    description=f"Negative net ROI of ${abs(net_roi):,.0f}.",
                    priority=ActionPriority.HIGH,
                    category=ActionCategory.PROCESS_OPTIMIZATION,
                    impact_score=90,
                    estimated_cost=15000,
                    estimated_savings=abs(net_roi) * 0.8,
                    timeline="4-8 weeks",
                    responsible_party="Operations Management",
                )
            )

        # Cycle time deviation
        if ct_deviation_pct > self.thresholds["ct_deviation_warning"]:
            recommendations.append(
                RecommendedAction(
                    title="Cycle Time Consistency Improvement",
                    description=f"Cycle time deviation of {ct_deviation_pct:.1f}% indicates inconsistent performance.",
                    priority=ActionPriority.MEDIUM,
                    category=ActionCategory.PROCESS_OPTIMIZATION,
                    impact_score=65,
                    estimated_cost=2000,
                    timeline="2-4 weeks",
                    responsible_party="Production Team",
                )
            )

        # Trend-based example
        if not filtered_data.empty:
            daily_stability = filtered_data.groupby("DAY")["PROCESS_STABILITY"].mean()
            if daily_stability.std() > 15:
                recommendations.append(
                    RecommendedAction(
                        title="Daily Process Consistency Improvement",
                        description="High daily process variability detected. Standardize daily procedures.",
                        priority=ActionPriority.HIGH,
                        category=ActionCategory.TRAINING,
                        impact_score=70,
                        estimated_cost=2500,
                        timeline="4-6 weeks",
                        responsible_party="Training Department",
                    )
                )

        recommendations.sort(key=lambda x: (x.priority.value, -x.impact_score), reverse=True)
        return recommendations

    def get_action_summary(self, recommendations: List[RecommendedAction]) -> Dict:
        """Summarize recommended actions for quick KPIs."""
        if not recommendations:
            return {
                "total_actions": 0,
                "critical_actions": 0,
                "high_priority_actions": 0,
                "estimated_total_cost": 0,
                "estimated_total_savings": 0,
                "categories": {},
            }

        summary = {
            "total_actions": len(recommendations),
            "critical_actions": len([r for r in recommendations if r.priority == ActionPriority.CRITICAL]),
            "high_priority_actions": len([r for r in recommendations if r.priority in (ActionPriority.CRITICAL, ActionPriority.HIGH)]),
            "estimated_total_cost": sum([(r.estimated_cost or 0) for r in recommendations]),
            "estimated_total_savings": sum([(r.estimated_savings or 0) for r in recommendations]),
            "categories": {},
        }
        for rec in recommendations:
            summary["categories"][rec.category.value] = summary["categories"].get(rec.category.value, 0) + 1
        return summary