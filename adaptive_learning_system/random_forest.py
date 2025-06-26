import os
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

def train_models(df, feature_cols, target_cols, test_size=0.2, random_state=25, n_estimators=100):
    """
    Train Random Forest models for regression and classification.
    Returns trained models, feature names, and evaluation metrics.
    """
    # Extract features and targets
    X = df[feature_cols]
    y_reg = df[target_cols[0]]  # Regression target
    y_clf = df[target_cols[1]]  # Classification target

    # Split data for regression and classification
    X_train, X_test, y_train, y_test = train_test_split(
        X, df[target_cols], test_size=test_size, random_state=random_state, stratify=df[target_cols[1]]
    )
    y_reg_train = y_train[target_cols[0]]
    y_reg_test = y_test[target_cols[0]]
    y_clf_train = y_train[target_cols[1]]
    y_clf_test = y_test[target_cols[1]]

    # Train models
    reg = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    reg.fit(X_train, y_reg_train)
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_clf_train)

    # Predict and evaluate
    reg_pred = reg.predict(X_test)
    clf_pred = clf.predict(X_test)
    mse = mean_squared_error(y_reg_test, reg_pred)
    acc = accuracy_score(y_clf_test, clf_pred)

    print(f"Regression MSE (test split): {mse:.4f}")
    print(f"Classification Accuracy (test split): {acc:.4f}")

    return reg, clf, feature_cols, {'mse': mse, 'accuracy': acc}

def predict(models, feature_names, new_data_df):
    """
    Predict using trained models and a DataFrame of new data.
    """
    reg, clf = models
    X_new = new_data_df[feature_names]
    reg_pred = reg.predict(X_new)
    clf_pred = clf.predict(X_new)
    return reg_pred, clf_pred

def save_models(models, out_dir='models', prefix='rf'):
    """
    Save trained models to disk.
    """
    os.makedirs(out_dir, exist_ok=True)
    reg, clf = models
    reg_path = os.path.join(out_dir, f"{prefix}_regressor.joblib")
    clf_path = os.path.join(out_dir, f"{prefix}_classifier.joblib")
    joblib.dump(reg, reg_path)
    joblib.dump(clf, clf_path)
    print(f"Models saved to {reg_path} and {clf_path}")
