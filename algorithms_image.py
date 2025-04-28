import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ======= PARAMETERS =======
image_size = (64, 64)  # Resize all images to 64x64
data_dir = "C:\\Users\\fayc3\\OneDrive\\Desktop\\Semestre 2\\Machine Learning\\3 Machine learning algorithm\\DATASETS\\Apple vs Orange"
predict_dir = "C:\\Users\\fayc3\\OneDrive\\Desktop\\Semestre 2\\Machine Learning\\3 Machine learning algorithm\\DATASETS\\Apple vs Orange\\predict"

# ======= LOAD AND PREPROCESS DATA =======
X = []
y = []

for label in os.listdir(data_dir):
    class_path = os.path.join(data_dir, label)
    if os.path.isdir(class_path):
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, image_size)
                X.append(img.flatten())  # Flatten the image into 1D array
                y.append(label)
            except Exception as e:
                print(f"⚠️ Skipped {img_path}: {e}")

X = np.array(X)
le = LabelEncoder()
y = le.fit_transform(y)

# ======= TRAIN / TEST SPLIT =======
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train, predict, and evaluate models
def evaluate_model(name, model, X_train, X_test, y_train, y_test, results):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model: {name}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print(f"Accuracy: {accuracy}\n")
    results[name] = accuracy

# Dictionary to store results
results = {}

# ======= TRAIN AND EVALUATE MODELS =======

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
evaluate_model('Random Forest', rf_model, X_train, X_test, y_train, y_test, results)

# SVM
svm_model = SVC(kernel='linear', random_state=42)
evaluate_model('SVM', svm_model, X_train, X_test, y_train, y_test, results)

# XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
evaluate_model('XGBoost', xgb_model, X_train, X_test, y_train, y_test, results)

# ======= PLOT RESULTS =======
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values(), color=['blue', 'green', 'orange'])
plt.title('Model Comparison: Accuracy Scores')
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.ylim(0, 1)  # Accuracy ranges from 0 to 1
plt.show()

# ======= PREDICT NEW IMAGES =======
print("\n🔮 Predictions for images in 'predict' folder:")
for filename in os.listdir(predict_dir):
    file_path = os.path.join(predict_dir, filename)
    img = cv2.imread(file_path)
    if img is not None:
        img_resized = cv2.resize(img, image_size).flatten().reshape(1, -1)
        
        print(f"\n🖼️ Predictions for '{filename}':")
        # Random Forest Prediction
        rf_pred = rf_model.predict(img_resized)
        rf_label = le.inverse_transform(rf_pred)[0]
        print(f"Random Forest predicts: {rf_label}")
        
        # SVM Prediction
        svm_pred = svm_model.predict(img_resized)
        svm_label = le.inverse_transform(svm_pred)[0]
        print(f"SVM predicts: {svm_label}")
        
        # XGBoost Prediction
        xgb_pred = xgb_model.predict(img_resized)
        xgb_label = le.inverse_transform(xgb_pred)[0]
        print(f"XGBoost predicts: {xgb_label}")
    else:
        print(f"⚠️ Could not read image: {filename}")