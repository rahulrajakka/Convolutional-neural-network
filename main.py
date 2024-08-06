import base64
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dash import Dash, html
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

# Load the sales data
print("Loading sales data...")
df = pd.read_csv('sales_data.csv')
print("Sales data loaded successfully.")
print("DataFrame columns:", df.columns)

# Data cleaning
print("Cleaning sales data...")
df.dropna(inplace=True)
df = df[(df != 0).all(axis=1)]
print("Sales data cleaned successfully.")
print("DataFrame shape after cleaning:", df.shape)
print("DataFrame columns after cleaning:", df.columns)

# Calculate average sales
print("Calculating average sales...")
avg_sales = df.groupby(['product_id', 'sales_week of year'])['sales_product_quantity'].mean().reset_index()
avg_sales['sales_product_quantity'] = avg_sales['sales_product_quantity'].astype(float)
avg_sales['avg_sales_product_quantity'] = avg_sales['sales_product_quantity'].astype(int)
avg_sales = avg_sales.drop(columns=['sales_product_quantity'])
print(f"Average sales calculated {avg_sales.head()}")

# Prepare input and output data for the model
print("Preparing input and output data for the model...")
X = avg_sales[['product_id', 'sales_week of year']].values
y = avg_sales['avg_sales_product_quantity'].values

# Normalize features
scaler_X = MinMaxScaler()
X_normalized = scaler_X.fit_transform(X)

# Reshape X to 3D array (samples, time steps, features)
X_normalized = X_normalized.reshape((X_normalized.shape[0], X_normalized.shape[1], 1))

# Split the data into training and testing sets
print("Splitting the data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)
print(f"Data split into training and testing sets.")

# Build the CNN model
print("Building the CNN model...")
input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
x = Conv1D(filters=64, kernel_size=2, activation='relu', padding='same')(input_layer)
x = Dropout(0.3)(x)
x = Flatten()(x)
x = Dense(50, activation='relu')(x)
x = Dropout(0.3)(x)
output_layer = Dense(1)(x)  # Output layer for regression

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)
print("CNN model built.")

# Compile the model
print("Compiling the model...")
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
print("Model compiled.")

# Cross-validation
print("Performing cross-validation...")
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(X_normalized):
    X_train_cv, X_test_cv = X_normalized[train_index], X_normalized[test_index]
    y_train_cv, y_test_cv = y[train_index], y[test_index]

    # Train the model
    history = model.fit(X_train_cv, y_train_cv, epochs=100, batch_size=32, validation_data=(X_test_cv, y_test_cv),
                        verbose=1)

    # Evaluate the model
    loss, mae = model.evaluate(X_test_cv, y_test_cv)
    print(f'Fold Mean Absolute Error: {mae}')

print("Cross-validation completed.")

# Train the model on the full training data
print("Training the model...")
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
print("Model training completed.")

# Evaluate the model
print("Evaluating the model...")
loss, mae = model.evaluate(X_test, y_test)
print(f'Mean Absolute Error: {mae}')

# Predicting and calculating R-squared
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2}')



# Save the model
print("Saving the model...")
model.save('cnn_sales_model.h5')
print("Model saved.")

# Load the model (if needed)
print("Loading the model...")
model = load_model('cnn_sales_model.h5')
print("Model loaded.")


def model_predict(product_id, week, model):
    print(f"Predicting for product_id: {product_id}, week: {week}...")
    input_data = np.array([[product_id, week]])
    input_data = scaler_X.transform(input_data)  # Normalize input data
    input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))
    prediction = model.predict(input_data)
    # Introduce random noise to simulate variability
    noise = np.random.normal(0, 0.1 * prediction[0][0])
    prediction_with_noise = prediction[0][0] + noise
    print(f"Prediction for product_id: {product_id}, week: {week}: {prediction_with_noise}")
    return prediction_with_noise


def forecast_sales_for_product_rolling(product_id, start_week, product_price, model, n_weeks=52):
    print(f"Forecasting sales for product_id: {product_id} starting from week: {start_week}...")
    predictions = []
    current_week = start_week

    for _ in range(n_weeks):
        normalized_week = (current_week - 1) % 52 + 1
        prediction = model_predict(product_id, normalized_week, model)
        predictions.append(int(round(prediction)))
        current_week += 1

    forecast_df = pd.DataFrame({
        'product_id': [product_id] * n_weeks,
        'sales_week of year': np.arange(start_week, start_week + n_weeks) % 52 + 1,
        'forecasted_sales': predictions,
        'product_price': [int(round(product_price))] * n_weeks
    })

    print(f"Forecast for product_id: {product_id} completed.")
    return forecast_df


# Example usage
product_id = 242  # Replace with the actual product_id you want to forecast
start_week = 1  # Replace with the actual starting sales_week of year
product_price = 15   # Replace with the actual product price

print(f"Generating sales forecast for product_id: {product_id}...")
forecast_df = forecast_sales_for_product_rolling(product_id, start_week, product_price, model)

# Save forecast DataFrame to CSV file
forecast_df.to_csv(f'sales_forecast_52_weeks_product_{product_id}.csv', index=False)
print(f"Forecast saved to sales_forecast_52_weeks_product_{product_id}.csv")

# Optionally, display the forecast DataFrame
print(forecast_df)


def dynamic_price_interpolation(forecasted_sales, min_sales, max_sales, min_price, max_price):
    return min_price + (forecasted_sales - min_sales) * (max_price - min_price) / (max_sales - min_sales)


def optimize_prices(csv_file, model_file):
    print("Loading forecast data and model for optimization...")
    data = pd.read_csv(csv_file)
    print("Forecast data loaded. Columns:", data.columns)
    model = load_model(model_file)
    print("Model loaded.")

    columns = ["product_id", "sales_week of year", "forecasted_sales", "product_price", "dynamic_price"]
    optimization_results = pd.DataFrame(columns=columns)

    unique_products = data['product_id'].unique()

    for product_id in unique_products:
        product_data = data[data['product_id'] == product_id]
        min_sales = product_data['forecasted_sales'].min()
        max_sales = product_data['forecasted_sales'].max()
        cost = product_data.iloc[0]['product_price']
        min_price = cost - 0.15 * cost
        max_price = cost + 0.25 * cost

        for idx, row in product_data.iterrows():
            print(f"Optimizing price for product_id: {product_id}, week: {row['sales_week of year']}...")
            steady = row['forecasted_sales']

            # Calculate dynamic price using linear interpolation
            dynamic_price = dynamic_price_interpolation(steady, min_sales, max_sales, min_price, max_price)

            optimization_results = optimization_results.append({
                "product_id": product_id,
                "sales_week of year": row['sales_week of year'],
                "forecasted_sales": int(round(row['forecasted_sales'])),  # Convert forecasted_sales to integer
                "product_price": int(round(cost)),  # Convert product_price to integer
                "dynamic_price": dynamic_price
            }, ignore_index=True)
            print(
                f"Optimization for product_id: {product_id}, week: {row['sales_week of year']} completed. Dynamic price: {dynamic_price}")

    optimization_results.to_csv("optimization_results.csv", index=False)
    print("Optimization results saved to optimization_results.csv.")
    print(optimization_results)


# Example usage
csv_file = f"sales_forecast_52_weeks_product_{product_id}.csv"
model_file = "cnn_sales_model.h5"
optimize_prices(csv_file, model_file)

# Load your DataFrame
df = pd.read_csv('optimization_results.csv')

# Check column names
print("Columns in DataFrame:", df.columns)

# Ensure required columns are present
required_columns = ['forecasted_sales', 'product_price', 'dynamic_price']
for column in required_columns:
    if column not in df.columns:
        raise KeyError(f"Column '{column}' is missing from the DataFrame")

# Add missing columns
df['sales_normal_price'] = df['forecasted_sales'] * df['product_price']
df['sales_dynamic_price'] = df['forecasted_sales'] * df['dynamic_price']

# Calculate totals for the bar plot
total_sales_normal_price = df['sales_normal_price'].sum()
total_sales_dynamic_price = df['sales_dynamic_price'].sum()

# Load your DataFrame
df = pd.read_csv('optimization_results.csv')

# Check column names
print("Columns in DataFrame:", df.columns)

# Ensure required columns are present
required_columns = ['forecasted_sales', 'product_price', 'dynamic_price']
for column in required_columns:
    if column not in df.columns:
        raise KeyError(f"Column '{column}' is missing from the DataFrame")

# Add missing columns
df['sales_normal_price'] = df['forecasted_sales'] * df['product_price']
df['sales_dynamic_price'] = df['forecasted_sales'] * df['dynamic_price']

# Calculate totals for the bar plot
total_sales_normal_price = df['sales_normal_price'].sum()
total_sales_dynamic_price = df['sales_dynamic_price'].sum()


print("I'm done executing")

# Helper function to convert matplotlib figures to base64
def mpl_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# Generate figures
# Total Sales Comparison Bar Graph
plt.figure(figsize=(8, 5))
plt.bar(['Normal Price Sales', 'Dynamic Price Sales'], [total_sales_normal_price, total_sales_dynamic_price],
        color=['blue', 'green'])
plt.xlabel('Pricing Type')
plt.ylabel('Total Sales')
plt.title('Total Sales Comparison: Normal Price vs. Dynamic Price')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
total_sales_bar_img = mpl_to_base64(plt.gcf())
plt.close()

# Sales with Normal and Dynamic Prices Line Plot
plt.figure(figsize=(10, 6))
plt.plot(df['sales_week of year'], df['sales_normal_price'], label='Sales with Normal Price', color='blue')
plt.plot(df['sales_week of year'], df['sales_dynamic_price'], label='Sales with Dynamic Price', linestyle='--',
         color='green')
plt.xlabel('Week of Year')
plt.ylabel('Sales')
plt.title('Sales with Normal Price vs. Dynamic Price')
plt.legend()
sales_line_plot_img = mpl_to_base64(plt.gcf())
plt.close()

# Box Plot of Product Prices
plt.figure(figsize=(10, 6))
box = df[['product_price', 'dynamic_price']].boxplot(
    color=dict(boxes='green', whiskers='green', caps='green', medians='red'))

# Extract median values for annotations
medians = {
    'product_price': df['product_price'].median(),
    'dynamic_price': df['dynamic_price'].median()
}

# Plotting markers for median values
for i, column in enumerate(['product_price', 'dynamic_price'], start=1):
    plt.scatter([i], [medians[column]], color='red', zorder=5)
    plt.annotate(f'Median: {medians[column]:.2f}', xy=(i, medians[column]), xytext=(i + 0.1, medians[column] + 0.5),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))
plt.title('Box Plot of Product Prices')
plt.ylabel('Price')
box_plot_img = mpl_to_base64(plt.gcf())
plt.close()

# 4-Quadrant Analysis
plt.figure(figsize=(12, 6))
plt.scatter(df['dynamic_price'], df['forecasted_sales'], alpha=0.5, c='green')
plt.axhline(y=df['forecasted_sales'].median(), color='blue', linestyle='--')
plt.axvline(x=df['dynamic_price'].median(), color='blue', linestyle='--')
plt.xlabel('Optimized Price')
plt.ylabel('Forecasted Sales')
plt.title('4-Quadrant Analysis of Optimized Price vs. Sales')
plt.grid(True, linestyle='--', alpha=0.7)
quadrant_analysis_img = mpl_to_base64(plt.gcf())
plt.close()

# Stacked Bar Plot of Optimized Prices by Week and Product
stacked_data = df.pivot_table(index='sales_week of year', columns='product_id', values='dynamic_price', aggfunc='mean')

plt.figure(figsize=(12, 8))
stacked_data.plot(kind='bar', stacked=True, color=['green', 'blue'])  # Adjust colors as needed
plt.title('Stacked Bar Plot of Optimized Prices by Week and Product')
plt.xlabel('Week of Year')
plt.ylabel('Optimized Price')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Product ID', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
stacked_bar_plot_img = mpl_to_base64(plt.gcf())
plt.close()

# Initialize the Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Sales Data Visualizations"),

    html.Div([
        html.H2("Total Sales Comparison"),
        html.Img(src=f"data:image/png;base64,{total_sales_bar_img}"),
    ]),

    html.Div([
        html.H2("Sales with Normal Price vs. Dynamic Price"),
        html.Img(src=f"data:image/png;base64,{sales_line_plot_img}"),
    ]),

    html.Div([
        html.H2("Box Plot of Product Prices"),
        html.Img(src=f"data:image/png;base64,{box_plot_img}"),
    ]),

    html.Div([
        html.H2("4-Quadrant Analysis of Optimized Price vs. Sales"),
        html.Img(src=f"data:image/png;base64,{quadrant_analysis_img}"),
    ]),

    html.Div([
        html.H2("Stacked Bar Plot of Optimized Prices by Week and Product"),
        html.Img(src=f"data:image/png;base64,{stacked_bar_plot_img}"),
    ]),
])

if __name__ == '__main__':
    app.run_server(debug=False)