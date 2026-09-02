#!/usr/bin/env python3
"""
PS 5: Acoustic Species Classification Evaluation
Metrics: mAP (Mean Average Precision for multi-label classification)
Formula: final_score = mAP
"""

import pandas as pd
from sklearn.metrics import average_precision_score
import numpy as np
import sys
import json

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def evaluate(submission_path, solution_path, sample_path=None):
    """Evaluate PS 5 submissions (multi-label classification)"""
    submission = pd.read_csv(submission_path)
    solution = pd.read_csv(solution_path)
    
    # Validate
    if len(submission) != len(solution):
        return {'valid': False, 'message': 'Row count mismatch'}
    
    # Get species columns (all except audio_id)
    species_cols = [col for col in solution.columns if col != 'audio_id']
    
    # Merge to ensure correct order
    submission = submission.merge(solution[['audio_id']], on='audio_id')
    solution = solution.merge(submission[['audio_id']], on='audio_id')
    
    # Calculate mAP (mean average precision)
    aps = {}
    for col in species_cols:
        prob_col = f'{col}_prob'
        if prob_col in submission.columns:
            # If all 0s or all 1s in ground truth, handle edge cases
            if len(np.unique(solution[col])) > 1:
                ap = average_precision_score(solution[col], submission[prob_col])
            else:
                ap = 1.0 if (solution[col].values == (submission[prob_col].values >= 0.5)).all() else 0.5
            aps[col] = float(ap)
    
    mAP = np.mean(list(aps.values())) if aps else 0.0
    
    return {
        'valid': True,
        'mAP': round(float(mAP), 4),
        'per_class_AP': {k: round(v, 4) for k, v in aps.items()},
        'final_score': round(float(mAP), 4)
    }

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python evaluate_PS5.py <submission.csv> <solution.csv>")
        sys.exit(1)
    
    results = evaluate(sys.argv[1], sys.argv[2])
    
    if not results['valid']:
        print(f"❌ INVALID: {results['message']}")
        sys.exit(1)
    
    print(json.dumps(results, indent=2))
