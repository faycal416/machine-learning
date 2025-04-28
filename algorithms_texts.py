from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import nltk

nltk.download('stopwords')

# Load the IMDB dataset from CSV
train_data = pd.read_csv("C:\\Users\\fayc3\\OneDrive\\Desktop\\Semestre 2\\Machine Learning\\3 Machine learning algorithm\\DATASETS\\IMDB Dataset.csv")
test_data = pd.read_csv("C:\\Users\\fayc3\\OneDrive\\Desktop\\Semestre 2\\Machine Learning\\3 Machine learning algorithm\\DATASETS\\aciimdb_test.csv")

# Preprocess the data
train_data = train_data.iloc[:300, :]  # Use the first 300 rows for training
train_data = train_data.dropna()
test_data = test_data.dropna()

# Separate features and labels
X_train, y_train = train_data['review'], train_data['sentiment']
X_test, y_test = test_data['review'], test_data['sentiment']

# Convert sentiment labels to numerical values (e.g., positive -> 1, negative -> 0)
y_train = y_train.map({'positive': 1, 'negative': 0})
y_test = y_test.map({'positive': 1, 'negative': 0})

# Transform text data into numerical features
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Function to train, predict, and evaluate models
def evaluate_model(name, model, X_train, X_test, y_train, y_test, results):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model: {name}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy}\n")
    results[name] = accuracy

# Dictionary to store results
results = {}

# Train and evaluate Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
evaluate_model('Decision Tree', dt_model, X_train_tfidf, X_test_tfidf, y_train, y_test, results)

# Train and evaluate SVM
svm_model = SVC(kernel='linear', random_state=42)
evaluate_model('SVM', svm_model, X_train_tfidf, X_test_tfidf, y_train, y_test, results)

# Train and evaluate Random Forest
rf_model = RandomForestClassifier(random_state=42)
evaluate_model('Random Forest', rf_model, X_train_tfidf, X_test_tfidf, y_train, y_test, results)

# Train and evaluate XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
evaluate_model('XGBoost', xgb_model, X_train_tfidf, X_test_tfidf, y_train, y_test, results)

# Plot the accuracy scores for comparison
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values(), color=['blue', 'green', 'orange', 'red'])
plt.title('Model Comparison: Accuracy Scores')
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.ylim(0, 1)  # Accuracy ranges from 0 to 1
plt.show()