import pandas as pd
import os
import sys

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure directories exist
os.makedirs('PS4_public/train', exist_ok=True)
os.makedirs('PS4_public/test', exist_ok=True)
os.makedirs('PS4_secret', exist_ok=True)

# Load labels
train_df = pd.read_csv('PS4_public/train_labels.csv')
solution = pd.read_csv('PS4_secret/solution.csv')

# Verify sample submission matching test images
sample = solution[['image_id']].copy()
sample['ship_present_prob'] = 0.5
sample['ship_count'] = 1
sample.to_csv('PS4_public/sample_submission.csv', index=False)

train_imgs = os.listdir('PS4_public/train')
test_imgs = os.listdir('PS4_public/test')

train_size_mb = sum(os.path.getsize(os.path.join('PS4_public/train', f)) for f in train_imgs) / 1e6
test_size_mb = sum(os.path.getsize(os.path.join('PS4_public/test', f)) for f in test_imgs) / 1e6
sol_size_mb = os.path.getsize('PS4_secret/solution.csv') / 1e6

print(f"✅ PUBLIC train: {len(train_imgs)} satellite images, ~{train_size_mb:.1f} MB")
print(f"✅ PUBLIC test: {len(test_imgs)} satellite images, ~{test_size_mb:.1f} MB")
print(f"✅ SECRET solution.csv: {len(solution)} rows, {sol_size_mb:.3f} MB")
