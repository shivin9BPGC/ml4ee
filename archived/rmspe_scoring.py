"""
Student Scoring System for Salary Prediction Competition
Based on RMSPE (Root Mean Squared Percentage Error) where lower scores = better performance
"""

import pandas as pd
import numpy as np
from scipy.stats import rankdata, percentileofscore
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Union
import warnings


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Percentage Error
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        
    Returns:
        RMSPE as percentage (lower = better)
    """
    # Handle edge cases
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    if len(y_true) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    # Create mask for non-zero true values to avoid division by zero
    mask = y_true != 0
    
    if np.sum(mask) == 0:
        warnings.warn("All true values are zero, returning 0 RMSPE")
        return 0.0
    
    # Calculate percentage errors only for non-zero true values
    percentage_errors = ((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2
    
    # Return RMSPE as percentage
    return np.sqrt(np.mean(percentage_errors)) * 100


def direct_rmspe_score(student_predictions: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Direct RMSPE scoring - simplest approach
    
    Args:
        student_predictions: Student's predicted salary values
        ground_truth: True salary values
        
    Returns:
        RMSPE score (lower = better)
    """
    return rmspe(ground_truth, student_predictions)


def rank_score(all_student_rmspe_scores: List[float]) -> List[int]:
    """
    Convert RMSPE scores to ranks
    
    Args:
        all_student_rmspe_scores: List of RMSPE scores for all students
        
    Returns:
        List of ranks where 1 = best (lowest RMSPE), N = worst
    """
    if not all_student_rmspe_scores:
        return []
    
    # Use 'min' method to assign same rank to ties
    return rankdata(all_student_rmspe_scores, method='min').tolist()


def percentile_score(student_rmspe: float, all_rmspe_scores: List[float]) -> float:
    """
    Convert RMSPE to percentile score
    
    Args:
        student_rmspe: Individual student's RMSPE
        all_rmspe_scores: All students' RMSPE scores
        
    Returns:
        Percentile score (0 = best, 100 = worst)
    """
    if not all_rmspe_scores:
        return 0.0
    
    return percentileofscore(all_rmspe_scores, student_rmspe, kind='strict')


def competition_score(student_rmspe: float, baseline_rmspe: float = 25.0) -> float:
    """
    Score relative to baseline performance
    
    Args:
        student_rmspe: Student's RMSPE
        baseline_rmspe: Baseline RMSPE for comparison
        
    Returns:
        Relative score (1.0 = baseline, < 1.0 = better, > 1.0 = worse)
    """
    if baseline_rmspe <= 0:
        raise ValueError("Baseline RMSPE must be positive")
    
    return student_rmspe / baseline_rmspe


def tiered_score(student_rmspe: float) -> int:
    """
    Convert RMSPE to academic grade tiers
    
    Args:
        student_rmspe: Student's RMSPE
        
    Returns:
        Tier score (1 = A+, 7 = F) where lower = better
    """
    if student_rmspe <= 10:
        return 1  # A+ tier
    elif student_rmspe <= 15:
        return 2  # A tier
    elif student_rmspe <= 20:
        return 3  # B+ tier
    elif student_rmspe <= 25:
        return 4  # B tier
    elif student_rmspe <= 30:
        return 5  # C+ tier
    elif student_rmspe <= 35:
        return 6  # C tier
    else:
        return 7  # F tier


def penalty_rmspe_score(
    base_rmspe: float, 
    predictions: np.ndarray,
    submission_time: Optional[datetime] = None,
    deadline: Optional[datetime] = None
) -> float:
    """
    Apply penalties to RMSPE for poor submission practices
    
    Args:
        base_rmspe: Original RMSPE score
        predictions: Predicted values for quality checks
        submission_time: When submission was made
        deadline: Submission deadline
        
    Returns:
        Penalized RMSPE (higher = worse due to penalties)
    """
    score = base_rmspe
    penalties = []
    
    # Check for negative predictions
    if np.any(predictions < 0):
        score *= 1.1  # 10% penalty
        penalties.append("Negative predictions")
    
    # Check for extreme outliers (beyond reasonable salary range)
    if np.any(predictions > 1000000) or np.any(predictions < 1000):
        score *= 1.05  # 5% penalty
        penalties.append("Extreme outlier predictions")
    
    # Check for late submission
    if submission_time and deadline and submission_time > deadline:
        score *= 1.15  # 15% penalty
        penalties.append("Late submission")
    
    # Check for constant predictions (no learning)
    if len(np.unique(predictions)) == 1:
        score *= 1.2  # 20% penalty
        penalties.append("Constant predictions")
    
    return score


def inverse_rmspe_score(student_rmspe: float, max_score: float = 100.0) -> float:
    """
    Convert RMSPE to "higher is better" scoring if needed
    
    Args:
        student_rmspe: Student's RMSPE
        max_score: Maximum possible score
        
    Returns:
        Inverse score where higher = better
    """
    if student_rmspe >= max_score:
        return 0.0
    
    return max_score - student_rmspe


def evaluate_student_submissions(
    submissions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    id_col: str = 'ID',
    prediction_col: str = 'salary_average',
    baseline_rmspe: float = 25.0
) -> pd.DataFrame:
    """
    Comprehensive evaluation of all student submissions
    
    Args:
        submissions_df: DataFrame with columns ['student_id', 'ID', 'salary_average']
        ground_truth_df: DataFrame with columns ['ID', 'salary_average'] 
        id_col: Column name for sample IDs
        prediction_col: Column name for predictions
        baseline_rmspe: Baseline for comparison scoring
        
    Returns:
        DataFrame with comprehensive scoring results
    """
    results = []
    
    # Get unique students
    students = submissions_df['student_id'].unique()
    all_rmspe_scores = []
    
    # First pass: calculate all RMSPE scores
    for student_id in students:
        student_data = submissions_df[submissions_df['student_id'] == student_id]
        
        # Merge with ground truth
        merged = pd.merge(student_data, ground_truth_df, on=id_col)
        
        if len(merged) == 0:
            warnings.warn(f"No matching samples for student {student_id}")
            continue
        
        # Calculate RMSPE
        student_rmspe = rmspe(
            merged[f'{prediction_col}_y'].values,  # ground truth
            merged[f'{prediction_col}_x'].values   # predictions
        )
        all_rmspe_scores.append(student_rmspe)
    
    # Second pass: calculate all scoring metrics
    ranks = rank_score(all_rmspe_scores)
    
    for i, student_id in enumerate(students):
        student_data = submissions_df[submissions_df['student_id'] == student_id]
        merged = pd.merge(student_data, ground_truth_df, on=id_col)
        
        if len(merged) == 0:
            continue
        
        predictions = merged[f'{prediction_col}_x'].values
        ground_truth = merged[f'{prediction_col}_y'].values
        
        # Calculate all scores
        student_rmspe = all_rmspe_scores[i]
        
        result = {
            'student_id': student_id,
            'rmspe': round(student_rmspe, 4),
            'rank': ranks[i],
            'percentile': round(percentile_score(student_rmspe, all_rmspe_scores), 2),
            'competition_score': round(competition_score(student_rmspe, baseline_rmspe), 4),
            'tier': tiered_score(student_rmspe),
            'penalty_rmspe': round(penalty_rmspe_score(student_rmspe, predictions), 4),
            'inverse_score': round(inverse_rmspe_score(student_rmspe), 2),
            'num_predictions': len(predictions),
            'avg_prediction': round(np.mean(predictions), 2),
            'std_prediction': round(np.std(predictions), 2)
        }
        
        results.append(result)
    
    # Create results DataFrame and sort by RMSPE (best first)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('rmspe').reset_index(drop=True)
    
    return results_df


def create_leaderboard(results_df: pd.DataFrame, top_n: Optional[int] = None) -> pd.DataFrame:
    """
    Create formatted leaderboard for display
    
    Args:
        results_df: Results from evaluate_student_submissions
        top_n: Number of top students to show (None for all)
        
    Returns:
        Formatted leaderboard DataFrame
    """
    leaderboard = results_df.copy()
    
    if top_n:
        leaderboard = leaderboard.head(top_n)
    
    # Add tier labels
    tier_labels = {1: 'A+', 2: 'A', 3: 'B+', 4: 'B', 5: 'C+', 6: 'C', 7: 'F'}
    leaderboard['grade'] = leaderboard['tier'].map(tier_labels)
    
    # Select and rename columns for display
    display_cols = {
        'rank': 'Rank',
        'student_id': 'Student ID',
        'rmspe': 'RMSPE (%)',
        'grade': 'Grade',
        'percentile': 'Percentile',
        'competition_score': 'vs Baseline'
    }
    
    leaderboard = leaderboard[list(display_cols.keys())].rename(columns=display_cols)
    
    return leaderboard


def get_performance_summary(results_df: pd.DataFrame) -> Dict:
    """
    Generate performance summary statistics
    
    Args:
        results_df: Results from evaluate_student_submissions
        
    Returns:
        Dictionary with summary statistics
    """
    return {
        'total_students': len(results_df),
        'best_rmspe': results_df['rmspe'].min(),
        'worst_rmspe': results_df['rmspe'].max(),
        'median_rmspe': results_df['rmspe'].median(),
        'mean_rmspe': results_df['rmspe'].mean(),
        'std_rmspe': results_df['rmspe'].std(),
        'students_above_baseline': len(results_df[results_df['competition_score'] > 1.0]),
        'students_below_baseline': len(results_df[results_df['competition_score'] < 1.0]),
        'grade_distribution': results_df['tier'].value_counts().to_dict()
    }


# Example usage function
def example_usage():
    """
    Demonstrate how to use the scoring system
    """
    # Create example data
    np.random.seed(42)
    
    # Sample ground truth
    ground_truth = pd.DataFrame({
        'ID': range(100),
        'salary_average': np.random.normal(50000, 15000, 100)
    })
    
    # Sample student submissions
    students = ['student_001', 'student_002', 'student_003']
    submissions = []
    
    for student in students:
        # Simulate different quality predictions
        if student == 'student_001':  # Good student
            noise = np.random.normal(0, 5000, 100)
        elif student == 'student_002':  # Average student  
            noise = np.random.normal(0, 10000, 100)
        else:  # Poor student
            noise = np.random.normal(5000, 15000, 100)
        
        student_preds = ground_truth['salary_average'] + noise
        
        for i in range(100):
            submissions.append({
                'student_id': student,
                'ID': i,
                'salary_average': max(1000, student_preds[i])  # Ensure positive
            })
    
    submissions_df = pd.DataFrame(submissions)
    
    # Evaluate submissions
    results = evaluate_student_submissions(submissions_df, ground_truth)
    
    # Create leaderboard
    leaderboard = create_leaderboard(results)
    
    # Get summary
    summary = get_performance_summary(results)
    
    print("=== Student Performance Leaderboard ===")
    print(leaderboard)
    print("\n=== Performance Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    return results, leaderboard, summary


if __name__ == "__main__":
    example_usage()