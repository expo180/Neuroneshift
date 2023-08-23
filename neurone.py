import tensorflow as tf
import numpy as np

num_samples = 1000
input_data = np.random.rand(num_samples, 10)
output_data = np.random.randint(2, size=num_samples)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(input_data, output_data, epochs=10, batch_size=32)

input_sample = np.random.rand(1, 10)
prediction = model.predict(input_sample)
print(f"Input Sample: {input_sample}")
print(f"Prediction: {prediction}")

