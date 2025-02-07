import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

# Data Loading Function
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

# Data Cleaning & Preprocessing Function
def preprocess_data(data):
    data.drop_duplicates(inplace=True)
    data.update(data.select_dtypes(include=[np.number]).median())
    for col in data.select_dtypes(include=['object', 'category']).columns:
        data[col] = data[col].fillna(data[col].mode()[0])

    label_encoders = {}
    for col in data.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    return data, label_encoders

# Feature Selection Function
def feature_selection(data):
    print(data.columns)
    X = data.drop(columns=['suicidal', 'id'], errors='ignore')
    y = data['suicidal']

    rf = RandomForestClassifier()
    rf.fit(X, y)
    feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

    selected_features = feature_importances["Feature"].values[:10]
    return X[selected_features], y, selected_features

# Model Training Function
def train_models(X_train, y_train):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models

# Model Evaluation Function
def evaluate_models(models, X_test, y_test):
    for name, model in models.items():
        y_pred = model.predict(X_test)
        print(f"Model: {name}")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")

# Prediction Function
def predict(models, user_input):
    predictions = {}
    for name, model in models.items():
        pred = model.predict(user_input)
        predictions[name] = bool(pred[0])  # Convert to True/False
    return predictions

# Main Function to Run Everything
def main1(filepath, user_input):
    data = load_data(filepath)
    data, label_encoders = preprocess_data(data)
    X, y, selected_features = feature_selection(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = train_models(X_train, y_train)
    evaluate_models(models, X_test, y_test)

    # Preprocess user input
    user_input['depressiveness'] = user_input['depressiveness'].map({'True': 1, 'False': 0})
    user_input_selected = user_input[selected_features]
    user_input_scaled = scaler.transform(user_input_selected)

    predictions = predict(models, user_input_scaled)

    return predictions

# Example usage
user_input_data = {
    'phq_score': [12],
    'depressiveness': ['True'],
    'bmi': [24.5],
    'epworth_score': [10],
    'gad_score': [15],
    'depression_severity': [3],
    'age': [22],
    'school_year': [4],
    'anxiety_severity': [2],
    'who_bmi': [25]
}

user_input_df = pd.DataFrame(user_input_data)

# Uncomment below line to run with actual data
predictions = main1("depression_anxiety_data.csv", user_input_df)
print(predictions)
