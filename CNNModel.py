
import pandas as pd

df = pd.read_csv('Download file and add file path ')
print("Sales data loaded successfully.")
print("DataFrame columns:", df.columns)


df = df[(df != 0).all(axis=1)]
