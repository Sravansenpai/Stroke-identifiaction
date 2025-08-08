import joblib
from extract_features import extract_features

# Load the saved model
model = joblib.load('models/xgboost_model.pkl')

# Extract features from test data (or any new data)
X_test, y_test = extract_features('data/test')

# Predict
y_pred = model.predict(X_test)

# Evaluate
from sklearn.metrics import accuracy_score, classification_report
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
