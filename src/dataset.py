import os
import pandas as pd
from utils import train_test_split

# Prepare dataset
path = os.path.join(os.path.dirname(__file__), '..', 'data')
df = pd.read_csv(os.path.join(path, 'heart.csv'))
train_df, test_df = train_test_split(df, 0.2)

train_df.to_csv(os.path.join(path, 'train.csv'), index = False)
test_df.to_csv(os.path.join(path, 'test.csv'), index = False)