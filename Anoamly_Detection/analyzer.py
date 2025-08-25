"""
Cycle Time Anomaly Detection Demo Analyzer

This module provides a comprehensive demonstration of anomaly detection algorithms
for cycle time analysis in injection molding operations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from scipy import stats
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


class AnomalyDetectionAnalyzer:
    """
    Comprehensive anomaly detection analyzer for cycle time data
    """

    def __init__(self):
        """Initialize the analyzer with detection algorithms"""
        self.algorithms = {
            "isolation_forest": IsolationForest(contamination=0.1, random_state=42),
            "lof": LocalOutlierFactor(n_neighbors=20, contamination=0.1),
            "statistical": None,  # Custom implementation
        }
        self.scaler = StandardScaler()
        self.is_fitted = False

    def generate_demo_data(self, n_samples=2000):
        """
        Generate synthetic cycle time data with embedded anomalies

        Returns:
            pd.DataFrame: Demo dataset with normal and anomalous patterns
        """
        np.random.seed(42)

        # Generate timestamps
        start_time = datetime(2024, 1, 1)
        timestamps = [start_time + timedelta(minutes=10 * i) for i in range(n_samples)]

        # Base cycle time pattern (normal operation)
        base_cycle_time = 45.0  # seconds

        # Generate normal cycle times with realistic variations
        normal_data = []
        for i in range(int(n_samples * 0.85)):  # 85% normal data
            # Add daily and hourly patterns
            hour = timestamps[i].hour
            day_effect = 2 * np.sin(2 * np.pi * hour / 24)  # Daily cycle
            noise = np.random.normal(0, 2)  # Random noise
            cycle_time = base_cycle_time + day_effect + noise
            normal_data.append(cycle_time)

        # Generate anomalous data
        anomaly_data = []
        anomaly_labels = []

        for i in range(int(n_samples * 0.15)):  # 15% anomalous data
            anomaly_type = np.random.choice(
                [
                    "equipment_degradation",
                    "process_deviation",
                    "sensor_malfunction",
                    "extreme_outlier",
                ]
            )

            if anomaly_type == "equipment_degradation":
                # Gradual increase in cycle time
                cycle_time = base_cycle_time + 15 + np.random.normal(0, 3)
            elif anomaly_type == "process_deviation":
                # Sudden jump in cycle time
                cycle_time = base_cycle_time + 25 + np.random.normal(0, 5)
            elif anomaly_type == "sensor_malfunction":
                # Unrealistic values
                cycle_time = np.random.choice([5, 200, -10, 500])
            else:  # extreme_outlier
                cycle_time = base_cycle_time + np.random.normal(0, 20)

            anomaly_data.append(cycle_time)
            anomaly_labels.append(anomaly_type)

        # Combine data
        all_cycle_times = normal_data + anomaly_data
        all_timestamps = timestamps[: len(all_cycle_times)]

        # Create labels (0 = normal, 1 = anomaly)
        labels = [0] * len(normal_data) + [1] * len(anomaly_data)
        anomaly_types = ["normal"] * len(normal_data) + anomaly_labels

        # Shuffle data to mix normal and anomalous points
        indices = list(range(len(all_cycle_times)))
        np.random.shuffle(indices)

        df = pd.DataFrame(
            {
                "timestamp": [all_timestamps[i] for i in indices],
                "cycle_time": [all_cycle_times[i] for i in indices],
                "is_anomaly": [labels[i] for i in indices],
                "anomaly_type": [anomaly_types[i] for i in indices],
            }
        )

        # Add additional features
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["cycle_time_rolling_mean"] = (
            df["cycle_time"].rolling(window=10, center=True).mean()
        )
        df["cycle_time_rolling_std"] = (
            df["cycle_time"].rolling(window=10, center=True).std()
        )

        # Add temperature data (correlated with cycle time)
        df["temperature"] = (
            240
            + 0.5 * (df["cycle_time"] - base_cycle_time)
            + np.random.normal(0, 5, len(df))
        )

        return df.sort_values("timestamp").reset_index(drop=True)

    def detect_anomalies_isolation_forest(self, df):
        """
        Detect anomalies using Isolation Forest algorithm

        Args:
            df (pd.DataFrame): Input data

        Returns:
            np.array: Anomaly predictions (-1 for anomaly, 1 for normal)
        """
        features = ["cycle_time", "temperature", "hour"]
        X = df[features].fillna(df[features].mean())

        # Fit and predict
        predictions = self.algorithms["isolation_forest"].fit_predict(X)

        # Convert to binary (1 for anomaly, 0 for normal)
        anomaly_scores = np.where(predictions == -1, 1, 0)

        return anomaly_scores

    def detect_anomalies_lof(self, df):
        """
        Detect anomalies using Local Outlier Factor

        Args:
            df (pd.DataFrame): Input data

        Returns:
            np.array: Anomaly predictions
        """
        features = ["cycle_time", "temperature", "hour"]
        X = df[features].fillna(df[features].mean())

        # Fit and predict
        predictions = self.algorithms["lof"].fit_predict(X)

        # Convert to binary
        anomaly_scores = np.where(predictions == -1, 1, 0)

        return anomaly_scores

    def detect_anomalies_statistical(self, df, threshold=3):
        """
        Detect anomalies using statistical methods (Z-score)

        Args:
            df (pd.DataFrame): Input data
            threshold (float): Z-score threshold

        Returns:
            np.array: Anomaly predictions
        """
        cycle_times = df["cycle_time"].fillna(df["cycle_time"].mean())

        # Calculate Z-scores
        z_scores = np.abs(stats.zscore(cycle_times))

        # Identify anomalies
        anomaly_scores = np.where(z_scores > threshold, 1, 0)

        return anomaly_scores

    def ensemble_detection(self, df):
        """
        Combine multiple detection algorithms for robust anomaly detection

        Args:
            df (pd.DataFrame): Input data

        Returns:
            dict: Detection results from all algorithms
        """
        results = {}

        # Run individual algorithms
        results["isolation_forest"] = self.detect_anomalies_isolation_forest(df)
        results["lof"] = self.detect_anomalies_lof(df)
        results["statistical"] = self.detect_anomalies_statistical(df)

        # Ensemble voting (majority vote)
        votes = np.column_stack(
            [results["isolation_forest"], results["lof"], results["statistical"]]
        )

        ensemble_predictions = np.where(np.sum(votes, axis=1) >= 2, 1, 0)
        results["ensemble"] = ensemble_predictions

        return results

    def evaluate_performance(self, predictions, true_labels):
        """
        Evaluate anomaly detection performance

        Args:
            predictions (np.array): Predicted labels
            true_labels (np.array): True labels

        Returns:
            dict: Performance metrics
        """
        from sklearn.metrics import (
            precision_score,
            recall_score,
            f1_score,
            accuracy_score,
        )

        metrics = {
            "precision": precision_score(true_labels, predictions, zero_division=0),
            "recall": recall_score(true_labels, predictions, zero_division=0),
            "f1_score": f1_score(true_labels, predictions, zero_division=0),
            "accuracy": accuracy_score(true_labels, predictions),
        }

        return metrics

    def create_visualizations(self, df, detection_results, save_path=None):
        """
        Create comprehensive visualizations for anomaly detection results

        Args:
            df (pd.DataFrame): Input data
            detection_results (dict): Detection results from all algorithms
            save_path (str): Path to save plots
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "Cycle Time Anomaly Detection Analysis", fontsize=16, fontweight="bold"
        )

        # 1. Time series with anomalies
        axes[0, 0].plot(df.index, df["cycle_time"], "b-", alpha=0.7, label="Cycle Time")
        anomaly_indices = np.where(df["is_anomaly"] == 1)[0]
        axes[0, 0].scatter(
            anomaly_indices,
            df.loc[anomaly_indices, "cycle_time"],
            color="red",
            s=30,
            label="True Anomalies",
            zorder=5,
        )
        axes[0, 0].set_title("Cycle Time with True Anomalies")
        axes[0, 0].set_xlabel("Sample Index")
        axes[0, 0].set_ylabel("Cycle Time (seconds)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Algorithm comparison
        algorithm_names = ["Isolation Forest", "LOF", "Statistical", "Ensemble"]
        algorithm_keys = ["isolation_forest", "lof", "statistical", "ensemble"]

        performance_data = []
        for i, (name, key) in enumerate(zip(algorithm_names, algorithm_keys)):
            metrics = self.evaluate_performance(
                detection_results[key], df["is_anomaly"]
            )
            performance_data.append(
                [metrics["precision"], metrics["recall"], metrics["f1_score"]]
            )

        performance_df = pd.DataFrame(
            performance_data,
            columns=["Precision", "Recall", "F1-Score"],
            index=algorithm_names,
        )

        performance_df.plot(kind="bar", ax=axes[0, 1])
        axes[0, 1].set_title("Algorithm Performance Comparison")
        axes[0, 1].set_ylabel("Score")
        axes[0, 1].legend()
        axes[0, 1].set_xticklabels(algorithm_names, rotation=45)

        # 3. Ensemble detection results
        ensemble_anomalies = np.where(detection_results["ensemble"] == 1)[0]
        axes[0, 2].plot(df.index, df["cycle_time"], "b-", alpha=0.7, label="Cycle Time")
        axes[0, 2].scatter(
            ensemble_anomalies,
            df.loc[ensemble_anomalies, "cycle_time"],
            color="orange",
            s=30,
            label="Detected Anomalies",
            zorder=5,
        )
        axes[0, 2].set_title("Ensemble Anomaly Detection Results")
        axes[0, 2].set_xlabel("Sample Index")
        axes[0, 2].set_ylabel("Cycle Time (seconds)")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Anomaly type distribution
        anomaly_type_counts = df[df["is_anomaly"] == 1]["anomaly_type"].value_counts()
        axes[1, 0].pie(
            anomaly_type_counts.values,
            labels=anomaly_type_counts.index,
            autopct="%1.1f%%",
        )
        axes[1, 0].set_title("Anomaly Type Distribution")

        # 5. Feature correlation with anomalies
        features = ["cycle_time", "temperature", "hour"]
        for feature in features:
            normal_data = df[df["is_anomaly"] == 0][feature]
            anomaly_data = df[df["is_anomaly"] == 1][feature]

            axes[1, 1].hist(
                normal_data, alpha=0.7, bins=30, label=f"Normal {feature}", density=True
            )
            axes[1, 1].hist(
                anomaly_data,
                alpha=0.7,
                bins=30,
                label=f"Anomaly {feature}",
                density=True,
            )

        axes[1, 1].set_title("Feature Distribution: Normal vs Anomaly")
        axes[1, 1].set_xlabel("Value")
        axes[1, 1].set_ylabel("Density")
        axes[1, 1].legend()

        # 6. Detection timeline
        hourly_anomalies = df.groupby("hour")["is_anomaly"].mean()
        hourly_detected = df.groupby("hour").apply(
            lambda x: detection_results["ensemble"][x.index].mean()
        )

        axes[1, 2].plot(
            hourly_anomalies.index,
            hourly_anomalies.values,
            "r-",
            marker="o",
            label="True Anomaly Rate",
        )
        axes[1, 2].plot(
            hourly_detected.index,
            hourly_detected.values,
            "b--",
            marker="s",
            label="Detected Anomaly Rate",
        )
        axes[1, 2].set_title("Hourly Anomaly Detection Performance")
        axes[1, 2].set_xlabel("Hour of Day")
        axes[1, 2].set_ylabel("Anomaly Rate")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Visualizations saved to: {save_path}")

        plt.show()

    def run_demo_analysis(self, save_results=True):
        """
        Run complete anomaly detection demo analysis

        Args:
            save_results (bool): Whether to save results to files

        Returns:
            dict: Complete analysis results
        """
        print("🚀 Starting Cycle Time Anomaly Detection Demo...")
        print("=" * 60)

        # Generate demo data
        print("📊 Generating synthetic cycle time data with anomalies...")
        df = self.generate_demo_data(n_samples=1500)

        # Run anomaly detection
        print("🔍 Running multi-algorithm anomaly detection...")
        detection_results = self.ensemble_detection(df)

        # Evaluate performance
        print("📈 Evaluating detection performance...")
        performance = {}
        for algorithm, predictions in detection_results.items():
            performance[algorithm] = self.evaluate_performance(
                predictions, df["is_anomaly"]
            )

        # Create visualizations
        print("📈 Creating comprehensive visualizations...")
        viz_path = "data/anomaly_detection_analysis.png" if save_results else None
        import os

        os.makedirs("data", exist_ok=True)
        self.create_visualizations(df, detection_results, viz_path)

        # Print summary
        print("\n📋 Analysis Summary:")
        print(f"• Total samples analyzed: {len(df):,}")
        print(
            f"• True anomalies: {df['is_anomaly'].sum():,} ({df['is_anomaly'].mean():.1%})"
        )
        print(
            f"• Best algorithm: {max(performance, key=lambda x: performance[x]['f1_score'])}"
        )
        print(f"• Ensemble F1-Score: {performance['ensemble']['f1_score']:.3f}")

        # Save results
        if save_results:
            df.to_csv("data/demo_anomaly_data.csv", index=False)

            # Save detection results
            detection_df = pd.DataFrame(detection_results)
            detection_df.to_csv("data/detection_results.csv", index=False)

            # Save performance metrics
            performance_df = pd.DataFrame(performance).T
            performance_df.to_csv("data/performance_metrics.csv")

            print(f"\n💾 Results saved to data/ directory")

        return {
            "dataset": df,
            "detection_results": detection_results,
            "performance": performance,
        }


def main():
    """Main demo execution"""
    analyzer = AnomalyDetectionAnalyzer()
    results = analyzer.run_demo_analysis()

    print("\n✅ Demo completed successfully!")
    print("🔗 This demo showcases:")
    print("  • Multiple anomaly detection algorithms")
    print("  • Ensemble learning methods")
    print("  • Real-time anomaly detection")
    print("  • Industrial IoT data analysis")
    print("  • Performance evaluation metrics")


if __name__ == "__main__":
    main()

