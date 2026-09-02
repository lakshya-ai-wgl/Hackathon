#!/usr/bin/env python3
"""
PS 1: Predictive Maintenance Evaluation
Metrics: ROC-AUC (target_failure classification) + RMSE (rul_hours regression)
Formula: final_score = AUC - (RMSE / 1000)
"""

import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error
import numpy as np
import sys
import json

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def validate_submission(submission, sample):
    """Validate submission format"""
    required_cols = ['unit_id', 'predicted_failure_prob', 'predicted_rul_hours']
    
    if list(submission.columns) != required_cols:
        return False, f"Column mismatch: expected {required_cols}, got {list(submission.columns)}"
    
    if len(submission) != len(sample):
        return False, f"Row count mismatch: expected {len(sample)}, got {len(submission)}"
    
    if submission.isnull().any().any():
        return False, "Submission contains NaN values"
    
    if not ((submission['predicted_failure_prob'] >= 0).all() and 
            (submission['predicted_failure_prob'] <= 1).all()):
        return False, "Probabilities must be between 0 and 1"
    
    if not (submission['predicted_rul_hours'] >= 0).all():
        return False, "RUL must be non-negative"
    
    return True, "Valid"

def evaluate(submission_path, solution_path, sample_path=None):
    """
    Evaluate PS 1 submissions
    
    Returns:
        dict: {AUC, RMSE, final_score, valid, message}
    """
    submission = pd.read_csv(submission_path)
    solution = pd.read_csv(solution_path)
    
    if sample_path:
        sample = pd.read_csv(sample_path)
        valid, message = validate_submission(submission, sample)
        if not valid:
            return {'valid': False, 'message': message}
            
    # Calculate metrics
    auc = roc_auc_score(solution['target_failure'], submission['predicted_failure_prob'])
    rmse = np.sqrt(mean_squared_error(solution['rul_hours'], submission['predicted_rul_hours']))
    
    # Final composite score
    final_score = auc - (rmse / 1000.0)
    
    return {
        'valid': True,
        'AUC': round(float(auc), 4),
        'RMSE': round(float(rmse), 4),
        'final_score': round(float(final_score), 4)
    }

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python evaluate_PS1.py <submission.csv> <solution.csv> [<sample.csv>]")
        sys.exit(1)
        
    sample_path = sys.argv[3] if len(sys.argv) >= 4 else None
    results = evaluate(sys.argv[1], sys.argv[2], sample_path)
    
    if not results['valid']:
        print(f"❌ INVALID: {results['message']}")
        sys.exit(1)
        
    print(json.dumps(results, indent=2))
