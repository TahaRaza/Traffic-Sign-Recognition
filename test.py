import cv2
import numpy as np
from tensorflow.keras.models import load_model



# Load your trained model
model = load_model("traffic_sign_model.keras")

# Your class labels (same as training)
classes = ['Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)',
           'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)',
           'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)',
           'No Overtaking', 'No Overtaking for Heavy Vehicles', 'Right-of-Way at next Intersection',
           'Priority Road', 'Yield', 'Stop', 'No Vehicles', 'Heavy Vehicles Prohibited', 'No Entry',
           'General Caution', 'Dangerous Left Curve', 'Dangerous Right Curve', 'Double Curve',
           'Bumpy Road', 'Slippery Road', 'Narrowing Road', 'Road Work', 'Traffic Signals',
           'Pedestrian', 'Children', 'Bike', 'Snow', 'Deer', 'End of Limits', 'Turn Right Ahead',
           'Turn Left Ahead', 'Ahead Only', 'Go Straight or Right', 'Go Straight or Left',
           'Keep Right', 'Keep Left', 'Roundabout Mandatory', 'End of No Overtaking',
           'End of No Overtaking for Heavy Vehicles']

# ---- Load and preprocess test image ----
img_path = "abc.png"
img = cv2.imread(img_path)

if img is None:
    raise FileNotFoundError(f"Could not load image: {img_path}")

img_resized = cv2.resize(img, (224, 224))
img_resized = img_resized.astype("float32") / 255.0
img_input = np.expand_dims(img_resized, axis=0)

# ---- Predict ----
preds = model.predict(img_input)
top_indices = np.argsort(preds[0])[::-1][:3]  # top 3 classes

print("\nTop Predictions:")
for idx in top_indices:
    print(f"{classes[idx]} → {preds[0][idx]*100:.2f}%")



