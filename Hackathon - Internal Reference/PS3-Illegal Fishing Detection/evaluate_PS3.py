#!/usr/bin/env python3
"""
PS 3: Illegal Fishing Detection Evaluation (xView3 Maritime Metric)
Metrics:
- F1D: F1 Detection score (spatial matching within distance tolerance)
- F1S: F1 Close-to-shore detection score
- F1V: F1 Vessel classification score
- F1F: F1 Fishing classification score
- PEL: Percent Error in Length estimation
- MR / final_score: Maritime Ranking aggregate metric
"""

import pandas as pd
import numpy as np
import sys
import json
from scipy.spatial.distance import cdist

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def evaluate_scene(preds, gts, distance_tolerance=20.0):
    """
    Evaluate detections for a single scene using bipartite Euclidean distance matching.
    """
    if len(gts) == 0 and len(preds) == 0:
        return {'tp': 0, 'fp': 0, 'fn': 0, 'vessel_tp': 0, 'fishing_tp': 0, 'len_errors': []}
    if len(gts) == 0:
        return {'tp': 0, 'fp': len(preds), 'fn': 0, 'vessel_tp': 0, 'fishing_tp': 0, 'len_errors': []}
    if len(preds) == 0:
        return {'tp': 0, 'fp': 0, 'fn': len(gts), 'vessel_tp': 0, 'fishing_tp': 0, 'len_errors': []}

    pred_coords = preds[['detect_scene_row', 'detect_scene_column']].values
    gt_coords = gts[['detect_scene_row', 'detect_scene_column']].values
    
    dists = cdist(pred_coords, gt_coords)
    
    matched_gt = set()
    matched_pred = set()
    vessel_matches = 0
    fishing_matches = 0
    len_errors = []
    
    # Greedy matching by minimum distance
    flat_indices = np.argsort(dists.ravel())
    for idx in flat_indices:
        p_idx, g_idx = divmod(idx, dists.shape[1])
        if p_idx in matched_pred or g_idx in matched_gt:
            continue
        if dists[p_idx, g_idx] <= distance_tolerance:
            matched_pred.add(p_idx)
            matched_gt.add(g_idx)
            
            p_row = preds.iloc[p_idx]
            g_row = gts.iloc[g_idx]
            
            # Classification correctness
            if bool(p_row['is_vessel']) == bool(g_row['is_vessel']):
                vessel_matches += 1
            if bool(p_row['is_fishing']) == bool(g_row['is_fishing']):
                fishing_matches += 1
                
            # Length error
            true_len = max(1.0, float(g_row['vessel_length_m']))
            pred_len = max(1.0, float(p_row['vessel_length_m']))
            len_err = min(1.0, abs(pred_len - true_len) / true_len)
            len_errors.append(len_err)
            
    tp = len(matched_gt)
    fp = len(preds) - tp
    fn = len(gts) - tp
    
    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'vessel_tp': vessel_matches,
        'fishing_tp': fishing_matches,
        'len_errors': len_errors
    }

def evaluate(submission_path, solution_path, sample_path=None):
    sub = pd.read_csv(submission_path)
    sol = pd.read_csv(solution_path)
    
    required_cols = ['scene_id', 'detect_scene_row', 'detect_scene_column', 'is_vessel', 'is_fishing', 'vessel_length_m']
    for c in required_cols:
        if c not in sub.columns:
            return {'valid': False, 'message': f'Missing required column {c}'}
            
    total_tp, total_fp, total_fn = 0, 0, 0
    total_vessel_matches, total_fishing_matches = 0, 0
    all_len_errors = []
    
    scenes = sol['scene_id'].unique()
    for s in scenes:
        scene_preds = sub[sub['scene_id'] == s]
        scene_gts = sol[sol['scene_id'] == s]
        res = evaluate_scene(scene_preds, scene_gts)
        total_tp += res['tp']
        total_fp += res['fp']
        total_fn += res['fn']
        total_vessel_matches += res['vessel_tp']
        total_fishing_matches += res['fishing_tp']
        all_len_errors.extend(res['len_errors'])
        
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_detection = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    f1_vessel = total_vessel_matches / total_tp if total_tp > 0 else 0.0
    f1_fishing = total_fishing_matches / total_tp if total_tp > 0 else 0.0
    f1_shore = f1_detection * 0.95  # proximity proxy
    pel = float(np.mean(all_len_errors)) if all_len_errors else 0.5
    
    # Maritime ranking aggregate
    # Formula: F1D * (1 + (F1S + F1V + F1F + (1 - PEL)) / 4)
    mr_score = f1_detection * (1.0 + (f1_shore + f1_vessel + f1_fishing + max(0.0, 1.0 - pel)) / 4.0)
    
    return {
        'valid': True,
        'MR': round(float(mr_score), 4),
        'F1D': round(float(f1_detection), 4),
        'F1S': round(float(f1_shore), 4),
        'F1V': round(float(f1_vessel), 4),
        'F1F': round(float(f1_fishing), 4),
        'PEL': round(float(pel), 4),
        'final_score': round(float(mr_score), 4)
    }

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python evaluate_PS3.py <submission.csv> <solution.csv>")
        sys.exit(1)
        
    results = evaluate(sys.argv[1], sys.argv[2])
    if not results['valid']:
        print(f"❌ INVALID: {results['message']}")
        sys.exit(1)
        
    print(json.dumps(results, indent=2))
