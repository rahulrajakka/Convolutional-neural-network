
import pandas as pd

# Load the sales data
print("Loading sales data...")
df = pd.read_csv('Download file and add file path ')
print("Sales data loaded successfully.")
print("DataFrame columns:", df.columns)


# Remove rows where any column contains a zero value
df = df[(df != 0).all(axis=1)]
