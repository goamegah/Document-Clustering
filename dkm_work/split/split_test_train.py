import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
file_path="/home/khaldi/Downloads/classic4.csv"

# Step 1: Load your DataFrame
df = pd.read_csv(file_path)

# Step 2: Split indices into training and test sets
train_indices, test_indices = train_test_split(df.index, test_size=0.9, random_state=42)

# Step 3: Save indices into separate files
np.savetxt("validation", train_indices, fmt='%d')
np.savetxt('test', test_indices, fmt='%d')


