# evaluate.py - Evaluate the trained model on test dataset

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import numpy as np
import sys
import os

# Test path from argument
test_dir = sys.argv[1] if len(sys.argv) > 1 else 'dataset/test'

# Image Parameters
img_height, img_width = 224, 224
batch_size = 32

# Load model
model = tf.keras.models.load_model('gender_classification_model.h5')

# Prepare test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Predictions
predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype("int32")

# Classification report
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)
