import pandas as pd

# Read the parquet file
df = pd.read_parquet("/gemini/platform/public/embodiedAI/huggingface_cache/rhodes_lerobot/RoboTwin2/v30/demo_clean_50tasks/meta/episodes/chunk-000/file-000.parquet")

# Print the first few rows
print(df.head())

import ipdb;ipdb.set_trace()
print()
