import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Get the dataset paths
base_dir = os.path.abspath(".")  # Current directory (catdogclassification)
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")

# Verify dataset exists
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Training dataset not found: {train_dir}")
if not os.path.exists(val_dir):
    raise FileNotFoundError(f"Validation dataset not found: {val_dir}")

# Data preprocessing
train_datagen = ImageDataGenerator(rescale=1.0/255)
val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary"
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary"
)

# Define CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Save model
model.save("catdog_model.h5")
print("Model training complete! Model saved as catdog_model.h5")

# Evaluate the model on validation data
val_loss, val_accuracy = model.evaluate(validation_generator)

print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Validation Loss: {val_loss:.4f}")

