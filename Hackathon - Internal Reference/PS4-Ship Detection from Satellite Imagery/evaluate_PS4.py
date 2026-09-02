#!/usr/bin/env python3
"""
PS 4: Ship Detection Evaluation
Metrics: AUC-ROC (ship detection) + MAE (ship count)
Formula: final_score = AUC - (MAE / 10)
"""

import pandas as pd
from sklearn.metrics import roc_auc_score, mean_absolute_error
import sys
import json

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def evaluate(submission_path, solution_path, sample_path=None):
    """Evaluate PS 4 submissions"""
    submission = pd.read_csv(submission_path)
    solution = pd.read_csv(solution_path)
    
    # Validate
    if len(submission) != len(solution):
        return {'valid': False, 'message': 'Row count mismatch'}
    
    if list(submission.columns) != ['image_id', 'ship_present_prob', 'ship_count']:
        return {'valid': False, 'message': 'Column mismatch: expected [image_id, ship_present_prob, ship_count]'}
    
    # Merge to ensure correct order
    submission = submission.merge(solution[['image_id']], on='image_id')
    solution = solution.merge(submission[['image_id']], on='image_id')
    
    # Calculate metrics
    auc = roc_auc_score(solution['ship_present'], submission['ship_present_prob'])
    mae = mean_absolute_error(solution['ship_count'], submission['ship_count'])
    
    # Final score (higher is better)
    final_score = auc - (mae / 10.0)
    
    return {
        'valid': True,
        'AUC': round(float(auc), 4),
        'MAE': round(float(mae), 4),
        'final_score': round(float(final_score), 4)
    }

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python evaluate_PS4.py <submission.csv> <solution.csv>")
        sys.exit(1)
    
    results = evaluate(sys.argv[1], sys.argv[2])
    
    if not results['valid']:
        print(f"❌ INVALID: {results['message']}")
        sys.exit(1)
    
    print(json.dumps(results, indent=2))
