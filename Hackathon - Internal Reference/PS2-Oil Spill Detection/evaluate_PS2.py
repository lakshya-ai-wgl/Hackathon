#!/usr/bin/env python3
"""
PS 2: Oil Spill Detection Evaluation
Metrics: LogLoss (oil detection) + RMSE (thickness estimation)
"""

import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
import numpy as np
import sys
import json

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def validate_submission(submission, sample):
    """Validate submission format"""
    required_cols = ['sample_id', 'oil_present_prob', 'oil_thickness_mm']
    
    if list(submission.columns) != required_cols:
        return False, f"Column mismatch: expected {required_cols}, got {list(submission.columns)}"
    
    if len(submission) != len(sample):
        return False, f"Row count mismatch: expected {len(sample)}, got {len(submission)}"
    
    if submission.isnull().any().any():
        return False, "Submission contains NaN values"
    
    if not ((submission['oil_present_prob'] >= 0).all() and 
            (submission['oil_present_prob'] <= 1).all()):
        return False, "Probabilities must be between 0 and 1"
    
    if not (submission['oil_thickness_mm'] >= 0).all():
        return False, "Thickness must be non-negative"
    
    return True, "Valid"

def evaluate(submission_path, solution_path, sample_path):
    """
    Evaluate PS 2 submissions
    
    Returns:
        dict: {logloss, auc, rmse, final_score, valid, message}
    """
    # Load data
    submission = pd.read_csv(submission_path)
    solution = pd.read_csv(solution_path)
    sample = pd.read_csv(sample_path)
    
    # Validate
    valid, message = validate_submission(submission, sample)
    if not valid:
        return {'valid': False, 'message': message}
    
    # Merge to ensure correct order
    submission = submission.merge(solution[['sample_id']], on='sample_id')
    solution = solution.merge(submission[['sample_id']], on='sample_id')
    
    # Clip probabilities to avoid log(0)
    submission['oil_present_prob'] = np.clip(submission['oil_present_prob'], 1e-7, 1-1e-7)
    
    # Calculate metrics
    logloss = log_loss(solution['oil_present'], submission['oil_present_prob'])
    auc = roc_auc_score(solution['oil_present'], submission['oil_present_prob'])
    rmse = np.sqrt(mean_squared_error(solution['oil_thickness_mm'], submission['oil_thickness_mm']))
    
    # Final score (lower is better)
    final_score = logloss + (rmse / 10)
    
    return {
        'valid': True,
        'LogLoss': round(float(logloss), 4),
        'AUC': round(float(auc), 4),
        'RMSE': round(float(rmse), 2),
        'final_score': round(float(final_score), 4)
    }

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python evaluate_PS2.py <submission.csv> <solution.csv> <sample.csv>")
        sys.exit(1)
    
    results = evaluate(sys.argv[1], sys.argv[2], sys.argv[3])
    
    if not results['valid']:
        print(f"❌ INVALID: {results['message']}")
        sys.exit(1)
    
    print(json.dumps(results, indent=2))
