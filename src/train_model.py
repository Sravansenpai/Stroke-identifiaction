import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
import joblib
from extract_features import extract_features
from sklearn.model_selection import train_test_split

def train():
    train_dir = '../data/train'
    test_dir = '../data/test'

    X_train, y_train = extract_features(train_dir)
    X_test, y_test = extract_features(test_dir)

    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(model, 'models/xgboost_model.pkl')

    print("Model saved!")

if __name__ == '__main__':
    train()