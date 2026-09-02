import pandas as pd
import os
import sys

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure directories exist
os.makedirs('PS3_public/train', exist_ok=True)
os.makedirs('PS3_public/test', exist_ok=True)
os.makedirs('PS3_secret/shoreline_contours', exist_ok=True)

# Load labels
train_df = pd.read_csv('PS3_public/train_labels.csv')
solution = pd.read_csv('PS3_secret/solution.csv')

# Verify sample submission matching test scenes
sample = solution.copy()
sample['is_vessel'] = True
sample['is_fishing'] = True
sample['vessel_length_m'] = 45.0
sample.to_csv('PS3_public/sample_submission.csv', index=False)

train_scenes = [d for d in os.listdir('PS3_public/train') if os.path.isdir(os.path.join('PS3_public/train', d))]
test_scenes = [d for d in os.listdir('PS3_public/test') if os.path.isdir(os.path.join('PS3_public/test', d))]

def get_dir_size_mb(path):
    return sum(os.path.getsize(os.path.join(r, f)) for r, d, fs in os.walk(path) for f in fs) / 1e6

train_size_mb = get_dir_size_mb('PS3_public/train')
test_size_mb = get_dir_size_mb('PS3_public/test')

print(f"✅ PUBLIC train scenes: {len(train_scenes)} scenes ({len(train_df)} detections), ~{train_size_mb:.1f} MB")
print(f"✅ PUBLIC test scenes: {len(test_scenes)} scenes, ~{test_size_mb:.1f} MB")
print(f"✅ SECRET solution.csv: {len(solution)} detections")
