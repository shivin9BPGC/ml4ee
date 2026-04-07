"""
Interactive Scoring Dashboard for Salary Prediction Competition
Displays RMSPE scores with multiple representations and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

from rmspe_scoring import (
    evaluate_student_submissions, 
    create_leaderboard, 
    get_performance_summary,
    rmspe
)


class RMSPEScoringDashboard:
    """
    Dashboard for comprehensive RMSPE-based student evaluation
    """
    
    def __init__(self, submissions_df: pd.DataFrame, ground_truth_df: pd.DataFrame, 
                 baseline_rmspe: float = 25.0):
        """
        Initialize dashboard with student submissions and ground truth
        
        Args:
            submissions_df: DataFrame with student predictions
            ground_truth_df: DataFrame with true values
            baseline_rmspe: Baseline RMSPE for comparison
        """
        self.submissions_df = submissions_df
        self.ground_truth_df = ground_truth_df
        self.baseline_rmspe = baseline_rmspe
        self.results_df = None
        self.leaderboard_df = None
        self.summary_stats = None
        
        # Evaluate submissions
        self._evaluate_submissions()
    
    def _evaluate_submissions(self):
        """Evaluate all student submissions"""
        print("Evaluating student submissions...")
        
        self.results_df = evaluate_student_submissions(
            self.submissions_df, 
            self.ground_truth_df,
            baseline_rmspe=self.baseline_rmspe
        )
        
        self.leaderboard_df = create_leaderboard(self.results_df)
        self.summary_stats = get_performance_summary(self.results_df)
        
        print(f"Evaluated {len(self.results_df)} students successfully.")
    
    def display_leaderboard(self, top_n: Optional[int] = None, 
                          include_details: bool = False):
        """
        Display formatted leaderboard
        
        Args:
            top_n: Number of top students to show
            include_details: Whether to show additional metrics
        """
        print("=" * 60)
        print("🏆 SALARY PREDICTION COMPETITION LEADERBOARD")
        print("=" * 60)
        print(f"Metric: RMSPE (Root Mean Squared Percentage Error)")
        print(f"Scoring: Lower is Better | Baseline: {self.baseline_rmspe}%")
        print("=" * 60)
        
        display_board = self.leaderboard_df.copy()
        if top_n:
            display_board = display_board.head(top_n)
            print(f"Showing Top {top_n} Students:")
        else:
            print("Complete Rankings:")
        
        # Format the display
        display_board_formatted = display_board.copy()
        display_board_formatted['RMSPE (%)'] = display_board_formatted['RMSPE (%)'].apply(
            lambda x: f"{x:.2f}%"
        )
        display_board_formatted['vs Baseline'] = display_board_formatted['vs Baseline'].apply(
            lambda x: f"{x:.3f}x"
        )
        
        print(display_board_formatted.to_string(index=False))
        
        if include_details:
            print("\n" + "=" * 60)
            print("📊 DETAILED METRICS")
            print("=" * 60)
            
            detailed_cols = [
                'student_id', 'rmspe', 'rank', 'percentile', 'tier',
                'penalty_rmspe', 'num_predictions', 'avg_prediction', 'std_prediction'
            ]
            
            detailed_df = self.results_df[detailed_cols].head(top_n if top_n else len(self.results_df))
            print(detailed_df.to_string(index=False))
    
    def display_summary_statistics(self):
        """Display comprehensive summary statistics"""
        print("\n" + "=" * 60)
        print("📈 COMPETITION SUMMARY STATISTICS")
        print("=" * 60)
        
        stats = self.summary_stats
        
        print(f"Total Students: {stats['total_students']}")
        print(f"Best RMSPE: {stats['best_rmspe']:.2f}%")
        print(f"Worst RMSPE: {stats['worst_rmspe']:.2f}%") 
        print(f"Median RMSPE: {stats['median_rmspe']:.2f}%")
        print(f"Mean RMSPE: {stats['mean_rmspe']:.2f}%")
        print(f"Standard Deviation: {stats['std_rmspe']:.2f}%")
        print()
        print(f"Students Better than Baseline ({self.baseline_rmspe}%): {stats['students_below_baseline']}")
        print(f"Students Worse than Baseline: {stats['students_above_baseline']}")
        
        print("\n📚 Grade Distribution:")
        grade_labels = {1: 'A+', 2: 'A', 3: 'B+', 4: 'B', 5: 'C+', 6: 'C', 7: 'F'}
        for tier, count in sorted(stats['grade_distribution'].items()):
            grade = grade_labels.get(tier, f'Tier {tier}')
            percentage = (count / stats['total_students']) * 100
            print(f"  {grade}: {count} students ({percentage:.1f}%)")
    
    def plot_score_distribution(self, figsize: tuple = (12, 8)):
        """Plot RMSPE score distribution with multiple views"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('RMSPE Score Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Histogram of RMSPE scores
        axes[0, 0].hist(self.results_df['rmspe'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(self.baseline_rmspe, color='red', linestyle='--', 
                          label=f'Baseline ({self.baseline_rmspe}%)')
        axes[0, 0].axvline(self.results_df['rmspe'].median(), color='green', linestyle='--', 
                          label=f'Median ({self.results_df["rmspe"].median():.1f}%)')
        axes[0, 0].set_xlabel('RMSPE (%)')
        axes[0, 0].set_ylabel('Number of Students')
        axes[0, 0].set_title('RMSPE Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Box plot
        axes[0, 1].boxplot(self.results_df['rmspe'], vert=True)
        axes[0, 1].set_ylabel('RMSPE (%)')
        axes[0, 1].set_title('RMSPE Box Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Rank vs RMSPE scatter
        axes[1, 0].scatter(self.results_df['rank'], self.results_df['rmspe'], 
                          alpha=0.7, color='coral', s=50)
        axes[1, 0].set_xlabel('Rank')
        axes[1, 0].set_ylabel('RMSPE (%)')
        axes[1, 0].set_title('Rank vs RMSPE')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Grade distribution pie chart
        grade_labels = {1: 'A+', 2: 'A', 3: 'B+', 4: 'B', 5: 'C+', 6: 'C', 7: 'F'}
        grade_counts = self.results_df['tier'].value_counts().sort_index()
        grade_names = [grade_labels.get(tier, f'T{tier}') for tier in grade_counts.index]
        
        colors = ['#2E8B57', '#32CD32', '#9ACD32', '#FFD700', '#FFA500', '#FF6347', '#DC143C']
        axes[1, 1].pie(grade_counts.values, labels=grade_names, autopct='%1.1f%%', 
                      colors=colors[:len(grade_counts)])
        axes[1, 1].set_title('Grade Distribution')
        
        plt.tight_layout()
        plt.show()
    
    def plot_performance_comparison(self, figsize: tuple = (14, 6)):
        """Plot detailed performance comparison"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Performance Comparison Analysis', fontsize=16, fontweight='bold')
        
        # 1. RMSPE vs Competition Score
        colors = ['green' if x < 1 else 'red' if x > 1 else 'gray' 
                 for x in self.results_df['competition_score']]
        
        axes[0].scatter(self.results_df['competition_score'], self.results_df['rmspe'],
                       c=colors, alpha=0.7, s=60)
        axes[0].axvline(1.0, color='black', linestyle='--', alpha=0.5, 
                       label='Baseline Performance')
        axes[0].set_xlabel('Competition Score (vs Baseline)')
        axes[0].set_ylabel('RMSPE (%)')
        axes[0].set_title('Performance vs Baseline')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Add annotations for best and worst
        best_idx = self.results_df['rmspe'].idxmin()
        worst_idx = self.results_df['rmspe'].idxmax()
        
        axes[0].annotate(f'Best: {self.results_df.loc[best_idx, "student_id"]}',
                        xy=(self.results_df.loc[best_idx, 'competition_score'], 
                           self.results_df.loc[best_idx, 'rmspe']),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=9, ha='left',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
        
        axes[0].annotate(f'Worst: {self.results_df.loc[worst_idx, "student_id"]}',
                        xy=(self.results_df.loc[worst_idx, 'competition_score'], 
                           self.results_df.loc[worst_idx, 'rmspe']),
                        xytext=(10, -20), textcoords='offset points',
                        fontsize=9, ha='left',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
        
        # 2. Percentile vs Rank
        axes[1].scatter(self.results_df['rank'], self.results_df['percentile'],
                       alpha=0.7, color='purple', s=60)
        axes[1].set_xlabel('Rank')
        axes[1].set_ylabel('Percentile')
        axes[1].set_title('Rank vs Percentile')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_individual_report(self, student_id: str):
        """Generate detailed report for individual student"""
        student_data = self.results_df[self.results_df['student_id'] == student_id]
        
        if len(student_data) == 0:
            print(f"Student {student_id} not found in results.")
            return
        
        student = student_data.iloc[0]
        
        print("=" * 60)
        print(f"📋 INDIVIDUAL PERFORMANCE REPORT")
        print(f"Student ID: {student_id}")
        print("=" * 60)
        
        print(f"🎯 Primary Metrics:")
        print(f"  RMSPE Score: {student['rmspe']:.2f}%")
        print(f"  Rank: {student['rank']} of {len(self.results_df)}")
        print(f"  Grade: {['A+', 'A', 'B+', 'B', 'C+', 'C', 'F'][student['tier']-1]}")
        print()
        
        print(f"📊 Comparative Metrics:")
        print(f"  Percentile: {student['percentile']:.1f}%")
        print(f"  vs Baseline: {student['competition_score']:.3f}x")
        performance = "Better" if student['competition_score'] < 1 else "Worse"
        print(f"  Performance: {performance} than baseline")
        print()
        
        print(f"⚠️  Quality Metrics:")
        print(f"  Penalty RMSPE: {student['penalty_rmspe']:.2f}%")
        penalty_applied = student['penalty_rmspe'] > student['rmspe']
        print(f"  Penalties Applied: {'Yes' if penalty_applied else 'No'}")
        print()
        
        print(f"📈 Prediction Statistics:")
        print(f"  Number of Predictions: {student['num_predictions']}")
        print(f"  Average Prediction: ${student['avg_prediction']:,.2f}")
        print(f"  Standard Deviation: ${student['std_prediction']:,.2f}")
        
        # Comparison with class statistics
        print("\n" + "=" * 60)
        print(f"📚 CLASS COMPARISON")
        print("=" * 60)
        
        better_students = len(self.results_df[self.results_df['rmspe'] < student['rmspe']])
        worse_students = len(self.results_df[self.results_df['rmspe'] > student['rmspe']])
        
        print(f"Students performing better: {better_students}")
        print(f"Students performing worse: {worse_students}")
        print(f"Class median RMSPE: {self.summary_stats['median_rmspe']:.2f}%")
        print(f"Class mean RMSPE: {self.summary_stats['mean_rmspe']:.2f}%")
        
        vs_median = "above" if student['rmspe'] > self.summary_stats['median_rmspe'] else "below"
        print(f"Performance vs class median: {vs_median} average")
    
    def export_results(self, filename: str = "competition_results.csv"):
        """Export complete results to CSV"""
        self.results_df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")
    
    def quick_summary(self):
        """Display quick summary for immediate overview"""
        print("🚀 QUICK COMPETITION SUMMARY")
        print("=" * 40)
        
        print(f"🏆 Winner: {self.results_df.iloc[0]['student_id']}")
        print(f"   Score: {self.results_df.iloc[0]['rmspe']:.2f}% RMSPE")
        print()
        
        print(f"📊 Class Stats:")
        print(f"   Students: {self.summary_stats['total_students']}")
        print(f"   Best: {self.summary_stats['best_rmspe']:.2f}%")
        print(f"   Median: {self.summary_stats['median_rmspe']:.2f}%")
        print(f"   Worst: {self.summary_stats['worst_rmspe']:.2f}%")
        print()
        
        below_baseline = self.summary_stats['students_below_baseline']
        total = self.summary_stats['total_students']
        print(f"🎯 {below_baseline}/{total} students beat baseline ({self.baseline_rmspe}%)")
        
        return self.results_df.iloc[0]['student_id']


def demo_dashboard():
    """Demonstrate the dashboard with sample data"""
    # Generate sample data
    np.random.seed(42)
    
    # Ground truth
    ground_truth = pd.DataFrame({
        'ID': range(200),
        'salary_average': np.random.lognormal(10.8, 0.5, 200)  # More realistic salary distribution
    })
    
    # Student submissions with varying performance levels
    students = [f'student_{i:03d}' for i in range(1, 21)]  # 20 students
    submissions = []
    
    for i, student in enumerate(students):
        # Create different performance levels
        if i < 3:  # Top performers
            error_factor = 0.05
            bias = 0
        elif i < 8:  # Good performers  
            error_factor = 0.15
            bias = 0.05
        elif i < 15:  # Average performers
            error_factor = 0.25
            bias = 0.1
        else:  # Poor performers
            error_factor = 0.4
            bias = 0.2
        
        noise = np.random.normal(0, error_factor, 200)
        predictions = ground_truth['salary_average'] * (1 + bias + noise)
        predictions = np.maximum(predictions, 1000)  # Ensure positive predictions
        
        for j in range(200):
            submissions.append({
                'student_id': student,
                'ID': j,
                'salary_average': predictions[j]
            })
    
    submissions_df = pd.DataFrame(submissions)
    
    # Create dashboard
    dashboard = RMSPEScoringDashboard(submissions_df, ground_truth, baseline_rmspe=20.0)
    
    # Display various views
    dashboard.quick_summary()
    print("\n")
    
    dashboard.display_leaderboard(top_n=10)
    dashboard.display_summary_statistics()
    
    # Generate individual report for top student
    winner = dashboard.results_df.iloc[0]['student_id']
    print("\n")
    dashboard.generate_individual_report(winner)
    
    # Show visualizations
    dashboard.plot_score_distribution()
    dashboard.plot_performance_comparison()
    
    return dashboard


if __name__ == "__main__":
    demo_dashboard()