import pandas as pd
import os
import sys

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure directories exist
os.makedirs('PS5_public/train', exist_ok=True)
os.makedirs('PS5_public/test', exist_ok=True)
os.makedirs('PS5_secret', exist_ok=True)

# Load labels
train_df = pd.read_csv('PS5_public/train_labels.csv')
solution = pd.read_csv('PS5_secret/solution.csv')

# Verify sample submission matching test audio files
species_cols = [c for c in solution.columns if c != 'audio_id']
sample = solution[['audio_id']].copy()
for col in species_cols:
    sample[f'{col}_prob'] = 0.5
sample.to_csv('PS5_public/sample_submission.csv', index=False)

train_wavs = os.listdir('PS5_public/train')
test_wavs = os.listdir('PS5_public/test')

train_size_mb = sum(os.path.getsize(os.path.join('PS5_public/train', f)) for f in train_wavs) / 1e6
test_size_mb = sum(os.path.getsize(os.path.join('PS5_public/test', f)) for f in test_wavs) / 1e6
sol_size_mb = os.path.getsize('PS5_secret/solution.csv') / 1e6

print(f"✅ PUBLIC train: {len(train_wavs)} audio clips, ~{train_size_mb:.1f} MB")
print(f"✅ PUBLIC test: {len(test_wavs)} audio clips, ~{test_size_mb:.1f} MB")
print(f"✅ SECRET solution.csv: {len(solution)} rows, {sol_size_mb:.3f} MB")
