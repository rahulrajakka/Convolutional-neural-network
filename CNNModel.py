
import pandas as pd

# Load the sales data
print("Loading sales data...")
df = pd.read_csv('Download file and add file path ')
print("Sales data loaded successfully.")
print("DataFrame columns:", df.columns)


# Remove rows where any column contains a zero value
df = df[(df != 0).all(axis=1)]

# Calculate average sales
print("Calculating average sales...")
avg_sales = df.groupby(['product_id', 'sales_week of year'])['sales_product_quantity'].mean().reset_index()
avg_sales['sales_product_quantity'] = avg_sales['sales_product_quantity'].astype(float)
avg_sales.columns = ['product_id', 'sales_week of year', 'avg_sales_product_quantity']
print("Average sales calculated.")
print(avg_sales.head())


# Normalize the data
print("Normalizing the data...")
avg_sales['product_id'] = avg_sales['product_id'] / avg_sales['product_id'].max()
avg_sales['sales_week of year'] = avg_sales['sales_week of year'] / avg_sales['sales_week of year'].max()
print("Data normalized.")
print(avg_sales.head())


# Prepare input and output data for the model
print("Preparing input and output data for the model...")
X = avg_sales[['product_id', 'sales_week of year']].values
y = avg_sales['avg_sales_product_quantity'].values
print("Input and output data prepared.")