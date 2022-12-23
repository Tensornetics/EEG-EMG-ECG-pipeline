import healthkit
import sqlite3
import tensorflow as tf
import torch

# Connect to Apple Health
health_store = healthkit.HealthStore()

# Set up SQLite database connection
conn = sqlite3.connect("health_data.db")
cursor = conn.cursor()

# Create table for data
cursor.execute(
    "CREATE TABLE IF NOT EXISTS health_data (timestamp INTEGER, value REAL)")

# Get data from Apple Health
data_type = healthkit.HKQuantityType.quantityType(forIdentifier: HKQuantityTypeIdentifier.stepCount)!
start_date = Date.distantPast
end_date = Date()
data = health_store.execute(HKStatisticsCollectionQuery(quantityType: data_type, quantitySamplePredicate: nil, options: .cumulativeSum, anchorDate: start_date, intervalComponents: DateComponents(day: 1))).statistics()

# Transfer data to database
for datum in data:
    timestamp = datum.startDate.timeIntervalSince1970
    value = datum.sumQuantity()?.doubleValue(for: HKUnit.count())
    cursor.execute(
        "INSERT INTO health_data (timestamp, value) VALUES (?, ?)", (timestamp, value))

# Save changes to database
conn.commit()

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Use TensorFlow Lite model to make prediction
input_tensor = interpreter.tensor(interpreter.get_input_details()[0]["index"])
output_tensor = interpreter.tensor(
    interpreter.get_output_details()[0]["index"])
input_tensor.copy_(data)
interpreter.invoke()
prediction = output_tensor.n

# Print prediction
print(f"Prediction (TensorFlow Lite): {prediction}")

# Load PyTorch model
model = torch.load("model.pt")

# Use PyTorch model to make prediction
inputs = torch.tensor(data, dtype=torch.float32)
outputs = model(inputs)
prediction = outputs.detach().numpy()

# Print prediction
print(f"Prediction (PyTorch): {prediction}")

# Close database connection
conn.close()
