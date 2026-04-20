"""
Utility module for analyzing tracker results and generating reports.
"""
import json
from typing import Dict, List
from datetime import datetime


class TrackingAnalyzer:
    """Analyze tracking results and generate reports."""

    def __init__(self):
        self.analysis_results = []

    def analyze_detections(self, anomalies: Dict) -> Dict:
        """
        Analyze anomalies and generate a report.
        
        Args:
            anomalies: Dictionary from tracker.get_anomalies()
            
        Returns:
            Analysis report with insights
        """
        false_positives = anomalies.get('false_positives', [])
        missed_detections = anomalies.get('missed_detections', [])
        total_tracks = anomalies.get('total_tracks', 0)
        active_tracks = anomalies.get('active_tracks', 0)

        fp_rate = len(false_positives) / total_tracks if total_tracks > 0 else 0
        md_rate = len(missed_detections) / total_tracks if total_tracks > 0 else 0

        analysis = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_objects_tracked': total_tracks,
                'currently_active': active_tracks,
                'false_positives_found': len(false_positives),
                'missed_detections_found': len(missed_detections),
                'false_positive_rate': round(fp_rate * 100, 2),
                'missed_detection_rate': round(md_rate * 100, 2)
            },
            'false_positives': self._categorize_issues(false_positives),
            'missed_detections': self._categorize_issues(missed_detections),
            'recommendations': self._generate_recommendations(fp_rate, md_rate)
        }

        self.analysis_results.append(analysis)
        return analysis

    def _categorize_issues(self, issues: List[Dict]) -> Dict:
        """Categorize issues by class label."""
        categorized = {}
        for issue in issues:
            label = issue.get('label', 'unknown')
            if label not in categorized:
                categorized[label] = []
            categorized[label].append({
                'track_id': issue['track_id'],
                'avg_confidence': round(issue['avg_confidence'], 3),
                'confidence_std': round(issue['confidence_std'], 3),
                'anomaly_score': round(issue['anomaly_score'], 3),
                'detections_count': issue['detections_count'],
                'missed_frames': issue['missed_frames']
            })
        return categorized

    def _generate_recommendations(self, fp_rate: float, md_rate: float) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        if fp_rate > 0.3:
            recommendations.append(
                "🔴 HIGH false positive rate (>30%). Consider:"
                "\n   - Lowering confidence threshold (test_conf)"
                "\n   - Increasing NMS threshold to filter overlapping boxes"
                "\n   - Retraining model with harder examples"
            )
        elif fp_rate > 0.15:
            recommendations.append(
                "🟡 MODERATE false positive rate. Consider fine-tuning thresholds."
            )

        if md_rate > 0.2:
            recommendations.append(
                "🔴 HIGH missed detection rate (>20%). Consider:"
                "\n   - Lowering confidence threshold"
                "\n   - Adjusting model architecture for better recall"
                "\n   - Adding more training data for specific classes"
            )
        elif md_rate > 0.1:
            recommendations.append(
                "🟡 MODERATE missed detection rate. Monitor closely."
            )

        if fp_rate < 0.1 and md_rate < 0.1:
            recommendations.append(
                "✅ Model performance is good!"
            )

        return recommendations

    def export_report(self, filepath: str):
        """Export analysis report to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.analysis_results, f, indent=2)

    def get_last_analysis(self) -> Dict:
        """Get the most recent analysis."""
        return self.analysis_results[-1] if self.analysis_results else None

    def print_analysis(self, analysis: Dict = None):
        """Print analysis in human-readable format."""
        if analysis is None:
            analysis = self.get_last_analysis()
        
        if analysis is None:
            print("No analysis available")
            return

        print("\n" + "="*60)
        print("TRACKING ANALYSIS REPORT")
        print("="*60)
        
        summary = analysis['summary']
        print(f"\nSummary:")
        print(f"  Total Objects Tracked: {summary['total_objects_tracked']}")
        print(f"  Currently Active: {summary['currently_active']}")
        print(f"  False Positives: {summary['false_positives_found']} ({summary['false_positive_rate']}%)")
        print(f"  Missed Detections: {summary['missed_detections_found']} ({summary['missed_detection_rate']}%)")

        if analysis['false_positives']:
            print(f"\nFalse Positives by Class:")
            for label, issues in analysis['false_positives'].items():
                print(f"  {label}: {len(issues)} issues")
                for issue in issues[:3]:  # Show first 3
                    print(f"    - ID {issue['track_id']}: avg_conf={issue['avg_confidence']}, "
                          f"std={issue['confidence_std']}")

        if analysis['missed_detections']:
            print(f"\nMissed Detections by Class:")
            for label, issues in analysis['missed_detections'].items():
                print(f"  {label}: {len(issues)} issues")
                for issue in issues[:3]:  # Show first 3
                    print(f"    - ID {issue['track_id']}: {issue['missed_frames']} missed frames")

        if analysis['recommendations']:
            print(f"\nRecommendations:")
            for rec in analysis['recommendations']:
                print(f"  {rec}")

        print("\n" + "="*60)
