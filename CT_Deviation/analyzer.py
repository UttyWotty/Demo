"""
Cycle Time Deviation Analysis Demo Analyzer

This module demonstrates comprehensive cycle time deviation analysis
for injection molding manufacturing operations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


class CTDeviationAnalyzer:
    """
    Comprehensive cycle time deviation analyzer for manufacturing analytics
    """

    def __init__(self):
        """Initialize the CT deviation analyzer"""
        self.performance_categories = {
            "Excellent": {"max_deviation": 5, "min_stability": 90},
            "Good": {"max_deviation": 10, "min_stability": 80},
            "Acceptable": {"max_deviation": 20, "min_stability": 70},
            "Poor": {"max_deviation": float("inf"), "min_stability": 0},
        }

        self.analysis_config = {
            "min_shots_threshold": 100,
            "outlier_threshold": 3,  # Z-score threshold
            "confidence_level": 0.95,
            "trend_window": 30,  # days
        }

    def generate_demo_data(self, n_records=10000):
        """
        Generate synthetic cycle time data with realistic deviation patterns

        Returns:
            pd.DataFrame: Demo dataset with cycle time deviations
        """
        np.random.seed(42)

        # Generate equipment and part data
        equipment_codes = ["2822-01", "2822-02", "2829-05", "2830-01", "2831-02"]
        approved_ct_map = {
            "2822-01": 42.0,
            "2822-02": 48.0,
            "2829-05": 35.0,
            "2830-01": 45.0,
            "2831-02": 52.0,
        }

        # Generate realistic data distribution
        data_records = []

        for i in range(n_records):
            # Select equipment with weighted probability
            equipment = np.random.choice(equipment_codes, p=[0.3, 0.25, 0.2, 0.15, 0.1])
            approved_ct = approved_ct_map[equipment]

            # Generate realistic actual cycle time with equipment-specific patterns
            if equipment == "2822-01":  # Best performing equipment
                deviation_factor = np.random.normal(1.02, 0.08)  # 2% average deviation
            elif equipment == "2829-05":  # Moderate performer
                deviation_factor = np.random.normal(1.08, 0.12)  # 8% average deviation
            elif equipment == "2831-02":  # Poor performer
                deviation_factor = np.random.normal(1.18, 0.15)  # 18% average deviation
            else:
                deviation_factor = np.random.normal(1.12, 0.10)  # 12% average deviation

            actual_ct = approved_ct * deviation_factor

            # Add some extreme outliers (equipment issues)
            if np.random.random() < 0.02:  # 2% chance of extreme outlier
                actual_ct = approved_ct * np.random.uniform(1.5, 2.5)

            # Generate timestamp
            start_date = datetime(2024, 1, 1)
            random_days = np.random.randint(0, 365)
            random_hours = np.random.randint(0, 24)
            timestamp = start_date + timedelta(days=random_days, hours=random_hours)

            # Calculate derived metrics
            deviation_pct = ((actual_ct - approved_ct) / approved_ct) * 100
            efficiency = (approved_ct / actual_ct) * 100

            # Add supplier information
            supplier_map = {
                "2822-01": "Premium_Supplier_A",
                "2822-02": "Premium_Supplier_A",
                "2829-05": "Standard_Supplier_B",
                "2830-01": "Standard_Supplier_C",
                "2831-02": "Budget_Supplier_D",
            }

            data_records.append(
                {
                    "timestamp": timestamp,
                    "equipment_code": equipment,
                    "approved_ct": approved_ct,
                    "actual_ct": actual_ct,
                    "deviation_pct": deviation_pct,
                    "efficiency": efficiency,
                    "supplier": supplier_map[equipment],
                    "part_family": f"Part_Family_{equipment[-2:]}",
                    "shift": "Day" if 8 <= timestamp.hour <= 16 else "Night",
                }
            )

        df = pd.DataFrame(data_records)

        # Add rolling statistics
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["rolling_avg_ct"] = df.groupby("equipment_code")["actual_ct"].transform(
            lambda x: x.rolling(window=10, min_periods=1).mean()
        )
        df["rolling_std_ct"] = df.groupby("equipment_code")["actual_ct"].transform(
            lambda x: x.rolling(window=10, min_periods=1).std()
        )

        return df

    def calculate_performance_metrics(self, df):
        """
        Calculate comprehensive performance metrics for each equipment

        Args:
            df (pd.DataFrame): Input cycle time data

        Returns:
            pd.DataFrame: Performance metrics by equipment
        """
        # Group by equipment and calculate metrics
        metrics = (
            df.groupby("equipment_code")
            .agg(
                {
                    "actual_ct": ["mean", "std", "count"],
                    "approved_ct": "first",
                    "deviation_pct": ["mean", "std"],
                    "efficiency": ["mean", "std"],
                    "supplier": "first",
                }
            )
            .round(3)
        )

        # Flatten column names
        metrics.columns = [
            "avg_actual_ct",
            "std_actual_ct",
            "total_shots",
            "approved_ct",
            "avg_deviation_pct",
            "std_deviation_pct",
            "avg_efficiency",
            "std_efficiency",
            "supplier",
        ]

        # Calculate additional metrics
        metrics["coefficient_of_variation"] = (
            metrics["std_actual_ct"] / metrics["avg_actual_ct"]
        ) * 100
        metrics["stability_score"] = 100 / (1 + metrics["coefficient_of_variation"])

        # Calculate confidence intervals
        confidence_level = self.analysis_config["confidence_level"]
        alpha = 1 - confidence_level

        metrics["efficiency_ci_lower"] = metrics["avg_efficiency"] - stats.t.ppf(
            1 - alpha / 2, metrics["total_shots"] - 1
        ) * (metrics["std_efficiency"] / np.sqrt(metrics["total_shots"]))

        metrics["efficiency_ci_upper"] = metrics["avg_efficiency"] + stats.t.ppf(
            1 - alpha / 2, metrics["total_shots"] - 1
        ) * (metrics["std_efficiency"] / np.sqrt(metrics["total_shots"]))

        return metrics.reset_index()

    def categorize_performance(self, metrics_df):
        """
        Categorize equipment performance based on deviation and stability

        Args:
            metrics_df (pd.DataFrame): Performance metrics

        Returns:
            pd.DataFrame: Metrics with performance categories
        """

        def assign_category(row):
            abs_deviation = abs(row["avg_deviation_pct"])
            stability = row["stability_score"]

            for category, thresholds in self.performance_categories.items():
                if (
                    abs_deviation <= thresholds["max_deviation"]
                    and stability >= thresholds["min_stability"]
                ):
                    return category
            return "Poor"

        metrics_df["performance_category"] = metrics_df.apply(assign_category, axis=1)

        # Add ranking
        metrics_df["efficiency_rank"] = metrics_df["avg_efficiency"].rank(
            ascending=False, method="min"
        )
        metrics_df["stability_rank"] = metrics_df["stability_score"].rank(
            ascending=False, method="min"
        )

        # Calculate composite score
        metrics_df["composite_score"] = (metrics_df["avg_efficiency"] * 0.6) + (
            metrics_df["stability_score"] * 0.4
        )
        metrics_df["overall_rank"] = metrics_df["composite_score"].rank(
            ascending=False, method="min"
        )

        return metrics_df

    def analyze_trends(self, df):
        """
        Analyze temporal trends in cycle time performance

        Args:
            df (pd.DataFrame): Input data

        Returns:
            dict: Trend analysis results
        """
        # Daily trend analysis
        daily_stats = (
            df.groupby([df["timestamp"].dt.date, "equipment_code"])
            .agg({"efficiency": "mean", "actual_ct": "mean", "deviation_pct": "mean"})
            .reset_index()
        )

        # Monthly trend analysis
        monthly_stats = (
            df.groupby([df["timestamp"].dt.to_period("M"), "equipment_code"])
            .agg(
                {
                    "efficiency": "mean",
                    "actual_ct": "mean",
                    "deviation_pct": "mean",
                    "timestamp": "count",
                }
            )
            .reset_index()
        )
        monthly_stats.rename(columns={"timestamp": "shot_count"}, inplace=True)

        # Shift analysis
        shift_stats = (
            df.groupby(["equipment_code", "shift"])
            .agg({"efficiency": "mean", "actual_ct": "mean", "deviation_pct": "mean"})
            .reset_index()
        )

        return {
            "daily_trends": daily_stats,
            "monthly_trends": monthly_stats,
            "shift_analysis": shift_stats,
        }

    def create_comprehensive_visualizations(
        self, df, metrics_df, trends, save_path=None
    ):
        """
        Create comprehensive visualizations for CT deviation analysis

        Args:
            df (pd.DataFrame): Raw data
            metrics_df (pd.DataFrame): Performance metrics
            trends (dict): Trend analysis results
            save_path (str): Path to save plots
        """
        fig, axes = plt.subplots(3, 3, figsize=(24, 18))
        fig.suptitle(
            "Cycle Time Deviation Analysis Dashboard", fontsize=20, fontweight="bold"
        )

        # 1. Equipment Efficiency Comparison
        equipment_order = metrics_df.sort_values("avg_efficiency", ascending=False)[
            "equipment_code"
        ]
        colors = [
            "#2E8B57"
            if cat == "Excellent"
            else "#4169E1"
            if cat == "Good"
            else "#FF8C00"
            if cat == "Acceptable"
            else "#DC143C"
            for cat in metrics_df.set_index("equipment_code").loc[
                equipment_order, "performance_category"
            ]
        ]

        bars = axes[0, 0].bar(
            equipment_order,
            metrics_df.set_index("equipment_code").loc[
                equipment_order, "avg_efficiency"
            ],
            color=colors,
        )
        axes[0, 0].axhline(y=100, color="red", linestyle="--", label="Target (100%)")
        axes[0, 0].set_title(
            "Equipment Efficiency Comparison", fontsize=14, fontweight="bold"
        )
        axes[0, 0].set_ylabel("Efficiency (%)")
        axes[0, 0].set_xlabel("Equipment Code")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[0, 0].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.5,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 2. Deviation Distribution
        axes[0, 1].hist(
            df["deviation_pct"], bins=50, alpha=0.7, color="skyblue", edgecolor="black"
        )
        axes[0, 1].axvline(
            x=0, color="green", linestyle="--", linewidth=2, label="Target (0%)"
        )
        axes[0, 1].axvline(
            x=df["deviation_pct"].mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean ({df['deviation_pct'].mean():.1f}%)",
        )
        axes[0, 1].set_title(
            "Cycle Time Deviation Distribution", fontsize=14, fontweight="bold"
        )
        axes[0, 1].set_xlabel("Deviation (%)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Performance Category Distribution
        category_counts = metrics_df["performance_category"].value_counts()
        colors_pie = ["#2E8B57", "#4169E1", "#FF8C00", "#DC143C"]
        axes[0, 2].pie(
            category_counts.values,
            labels=category_counts.index,
            autopct="%1.1f%%",
            colors=colors_pie,
            startangle=90,
        )
        axes[0, 2].set_title(
            "Performance Category Distribution", fontsize=14, fontweight="bold"
        )

        # 4. Efficiency vs Stability Scatter
        for category in metrics_df["performance_category"].unique():
            subset = metrics_df[metrics_df["performance_category"] == category]
            axes[1, 0].scatter(
                subset["avg_efficiency"],
                subset["stability_score"],
                label=category,
                s=100,
                alpha=0.7,
            )

        axes[1, 0].set_title(
            "Efficiency vs Stability Analysis", fontsize=14, fontweight="bold"
        )
        axes[1, 0].set_xlabel("Average Efficiency (%)")
        axes[1, 0].set_ylabel("Stability Score")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Monthly Efficiency Trends
        for equipment in metrics_df["equipment_code"].head(3):  # Top 3 equipment
            equipment_data = trends["monthly_trends"][
                trends["monthly_trends"]["equipment_code"] == equipment
            ]
            if not equipment_data.empty:
                axes[1, 1].plot(
                    equipment_data.index,
                    equipment_data["efficiency"],
                    marker="o",
                    label=equipment,
                    linewidth=2,
                )

        axes[1, 1].set_title(
            "Monthly Efficiency Trends (Top 3)", fontsize=14, fontweight="bold"
        )
        axes[1, 1].set_xlabel("Month")
        axes[1, 1].set_ylabel("Efficiency (%)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis="x", rotation=45)

        # 6. Shift Performance Comparison
        shift_pivot = trends["shift_analysis"].pivot(
            index="equipment_code", columns="shift", values="efficiency"
        )
        shift_pivot.plot(kind="bar", ax=axes[1, 2], color=["lightcoral", "lightblue"])
        axes[1, 2].set_title("Efficiency by Shift", fontsize=14, fontweight="bold")
        axes[1, 2].set_xlabel("Equipment Code")
        axes[1, 2].set_ylabel("Efficiency (%)")
        axes[1, 2].legend()
        axes[1, 2].tick_params(axis="x", rotation=45)

        # 7. Supplier Performance
        supplier_perf = (
            df.groupby("supplier")["efficiency"]
            .agg(["mean", "std"])
            .sort_values("mean", ascending=False)
        )
        axes[2, 0].bar(
            supplier_perf.index,
            supplier_perf["mean"],
            yerr=supplier_perf["std"],
            capsize=5,
            color="lightgreen",
            alpha=0.8,
        )
        axes[2, 0].set_title(
            "Supplier Performance Comparison", fontsize=14, fontweight="bold"
        )
        axes[2, 0].set_xlabel("Supplier")
        axes[2, 0].set_ylabel("Average Efficiency (%)")
        axes[2, 0].tick_params(axis="x", rotation=45)
        axes[2, 0].grid(True, alpha=0.3)

        # 8. Control Chart (Equipment 2822-01)
        equipment_data = (
            df[df["equipment_code"] == "2822-01"].sort_values("timestamp").tail(100)
        )
        if not equipment_data.empty:
            control_mean = equipment_data["actual_ct"].mean()
            control_std = equipment_data["actual_ct"].std()
            ucl = control_mean + 3 * control_std
            lcl = control_mean - 3 * control_std

            axes[2, 1].plot(
                equipment_data.index,
                equipment_data["actual_ct"],
                "b-o",
                markersize=3,
                linewidth=1,
            )
            axes[2, 1].axhline(
                y=control_mean, color="green", linestyle="-", label="Mean"
            )
            axes[2, 1].axhline(y=ucl, color="red", linestyle="--", label="UCL")
            axes[2, 1].axhline(y=lcl, color="red", linestyle="--", label="LCL")
            axes[2, 1].set_title(
                "Statistical Process Control (2822-01)", fontsize=14, fontweight="bold"
            )
            axes[2, 1].set_xlabel("Sample Number")
            axes[2, 1].set_ylabel("Actual Cycle Time (s)")
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)

        # 9. Performance Metrics Summary Table
        axes[2, 2].axis("tight")
        axes[2, 2].axis("off")

        # Create summary table
        summary_data = [
            ["Total Equipment", len(metrics_df)],
            [
                "Excellent Performance",
                sum(metrics_df["performance_category"] == "Excellent"),
            ],
            ["Average Efficiency", f"{metrics_df['avg_efficiency'].mean():.1f}%"],
            [
                "Best Performer",
                metrics_df.loc[metrics_df["avg_efficiency"].idxmax(), "equipment_code"],
            ],
            [
                "Highest Stability",
                metrics_df.loc[
                    metrics_df["stability_score"].idxmax(), "equipment_code"
                ],
            ],
            ["Total Shots Analyzed", f"{metrics_df['total_shots'].sum():,}"],
        ]

        table = axes[2, 2].table(
            cellText=summary_data,
            colLabels=["Metric", "Value"],
            cellLoc="center",
            loc="center",
            colWidths=[0.6, 0.4],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        axes[2, 2].set_title(
            "Performance Summary", fontsize=14, fontweight="bold", pad=20
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"CT deviation visualizations saved to: {save_path}")

        plt.show()

    def run_demo_analysis(self, save_results=True):
        """
        Run complete CT deviation analysis demo

        Args:
            save_results (bool): Whether to save results to files

        Returns:
            dict: Complete analysis results
        """
        print("🚀 Starting Cycle Time Deviation Analysis Demo...")
        print("=" * 60)

        # Generate demo data
        print("📊 Generating synthetic cycle time data...")
        df = self.generate_demo_data(n_records=8000)

        # Calculate performance metrics
        print("📈 Calculating performance metrics...")
        metrics_df = self.calculate_performance_metrics(df)
        metrics_df = self.categorize_performance(metrics_df)

        # Analyze trends
        print("📊 Analyzing temporal trends...")
        trends = self.analyze_trends(df)

        # Create visualizations
        print("📈 Creating comprehensive visualizations...")
        viz_path = "data/ct_deviation_analysis.png" if save_results else None
        import os

        os.makedirs("data", exist_ok=True)
        self.create_comprehensive_visualizations(df, metrics_df, trends, viz_path)

        # Print summary
        print("\n📋 Analysis Summary:")
        print(f"• Total equipment analyzed: {len(metrics_df)}")
        print(f"• Total shots analyzed: {metrics_df['total_shots'].sum():,}")
        print(f"• Average efficiency: {metrics_df['avg_efficiency'].mean():.1f}%")
        print(
            f"• Best performer: {metrics_df.loc[metrics_df['avg_efficiency'].idxmax(), 'equipment_code']} ({metrics_df['avg_efficiency'].max():.1f}%)"
        )
        print(
            f"• Equipment in 'Excellent' category: {sum(metrics_df['performance_category'] == 'Excellent')}/{len(metrics_df)}"
        )

        # Save results
        if save_results:
            df.to_csv("data/demo_ct_data.csv", index=False)
            metrics_df.to_csv("data/equipment_performance_metrics.csv", index=False)

            # Save trend analysis
            trends["monthly_trends"].to_csv("data/monthly_trends.csv", index=False)
            trends["shift_analysis"].to_csv("data/shift_analysis.csv", index=False)

            print(f"\n💾 Results saved to data/ directory")

        return {
            "raw_data": df,
            "performance_metrics": metrics_df,
            "trend_analysis": trends,
        }


def main():
    """Main demo execution"""
    analyzer = CTDeviationAnalyzer()
    results = analyzer.run_demo_analysis()

    print("\n✅ Demo completed successfully!")
    print("🔗 This demo showcases:")
    print("  • Advanced statistical analysis")
    print("  • Performance categorization")
    print("  • Trend analysis and forecasting")
    print("  • Statistical process control")
    print("  • Data-driven decision support")


if __name__ == "__main__":
    main()

