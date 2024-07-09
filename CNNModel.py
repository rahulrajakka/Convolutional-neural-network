from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, Flatten
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

# Reshape X to 3D array (samples, time steps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))
print("Data reshaped for the model.")

# Split the data into training and testing sets
print("Splitting the data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")

# Build the CNN model
print("Building the CNN model...")
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))  # Output layer for regression
print("CNN model built.")

# Compile the model
print("Compiling the model...")
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
print("Model compiled.")

# Train the model
print("Training the model...")
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
print("Model training completed.")

# Evaluate the model
print("Evaluating the model...")
loss, mae = model.evaluate(X_test, y_test)
print(f'Mean Absolute Error: {mae}')

# Save the model
print("Saving the model...")
model.save('cnn_sales_model.h5')
print("Model saved.")

# Load the model (if needed)
print("Loading the model...")
model = load_model('cnn_sales_model.h5')
print("Model loaded.")