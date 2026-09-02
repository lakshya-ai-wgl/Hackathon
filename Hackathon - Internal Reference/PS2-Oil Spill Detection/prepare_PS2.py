import pandas as pd
import os
import sys

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure directories exist
os.makedirs('PS2_public/train/images', exist_ok=True)
os.makedirs('PS2_public/test/images', exist_ok=True)
os.makedirs('PS2_secret', exist_ok=True)

# Load labels
train_df = pd.read_csv('PS2_public/train_labels.csv')
solution = pd.read_csv('PS2_secret/solution.csv')

# Verify and create sample submission matching test set
sample = solution[['sample_id']].copy()
sample['oil_present_prob'] = 0.5
sample['oil_thickness_mm'] = 1.0
sample.to_csv('PS2_public/sample_submission.csv', index=False)

train_imgs = os.listdir('PS2_public/train/images')
test_imgs = os.listdir('PS2_public/test/images')

train_size_mb = sum(os.path.getsize(os.path.join('PS2_public/train/images', f)) for f in train_imgs) / 1e6
test_size_mb = sum(os.path.getsize(os.path.join('PS2_public/test/images', f)) for f in test_imgs) / 1e6
sol_size_mb = os.path.getsize('PS2_secret/solution.csv') / 1e6

print(f"✅ PUBLIC train/images: {len(train_imgs)} images, ~{train_size_mb:.1f} MB")
print(f"✅ PUBLIC test/images: {len(test_imgs)} images, ~{test_size_mb:.1f} MB")
print(f"✅ SECRET solution.csv: {len(solution)} rows, {sol_size_mb:.3f} MB")
