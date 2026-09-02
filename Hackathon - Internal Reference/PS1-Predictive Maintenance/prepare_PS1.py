import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Load full dataset
df = pd.read_csv('original_train.csv')

# Split: 80% train, 20% test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create directories
os.makedirs('PS1_public', exist_ok=True)
os.makedirs('PS1_secret', exist_ok=True)

# Save PUBLIC train (with targets)
train_df.to_csv('PS1_public/train.csv', index=False)

# Save PUBLIC test (WITHOUT targets)
test_public = test_df.drop(columns=['target_failure', 'rul_hours'])
test_public.to_csv('PS1_public/test.csv', index=False)

# Save PRIVATE solution (ground truth for test)
solution = test_df[['unit_id', 'target_failure', 'rul_hours']]
solution.to_csv('PS1_secret/solution.csv', index=False)

# Create sample submission
sample = test_public[['unit_id']].copy()
sample['predicted_failure_prob'] = 0.5
sample['predicted_rul_hours'] = 200.0
sample.to_csv('PS1_public/sample_submission.csv', index=False)

print(f"✅ PUBLIC train.csv: {len(train_df)} rows, {train_df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
print(f"✅ PUBLIC test.csv: {len(test_public)} rows, {test_public.memory_usage(deep=True).sum() / 1e6:.2f} MB")
print(f"✅ SECRET solution.csv: {len(solution)} rows, {solution.memory_usage(deep=True).sum() / 1e6:.2f} MB")
