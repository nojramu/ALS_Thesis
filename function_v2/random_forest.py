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

'''Example usage:'''
from data_handling import load_csv, preprocess_data, save_csv
from function_v2.plot_utils import plot_bar_chart, save_figure_to_image_folder
if __name__ == "__main__":
    # --- Data Loading & Preprocessing ---
    csv_path = 'data/sample_training_data.csv'
    required_features = [
        'engagement_rate', 'time_on_task_s', 'hint_ratio', 'interaction_count',
        'task_completed', 'quiz_score', 'difficulty', 'error_rate',
        'task_timed_out', 'time_before_hint_used'
    ]
    target_cols = ['cognitive_load', 'engagement_level']

    # Load and preprocess data
    df = load_csv(csv_path)
    df = preprocess_data(df, required_features, is_training_data=True)
    if df is None or df.empty:
        exit("Data loading or preprocessing failed or resulted in empty DataFrame.")

    # --- Model Training ---
    reg, clf, feature_names, metrics = train_models(df, required_features, target_cols)

    # --- Model Saving (optional) ---
    save_models((reg, clf), out_dir='models', prefix='rf')

    # --- Example Prediction ---
    # Use the first row as a sample new data point
    new_data = df.iloc[[0]][required_features]
    reg_pred, clf_pred = predict((reg, clf), feature_names, new_data)
    print(f"Sample prediction (first row): Cognitive Load={reg_pred[0]:.3f}, Engagement Level={clf_pred[0]}")

    # --- Example Prediction with Custom Data ---
    import pandas as pd

    # Example new data for prediction as a DataFrame
    new_data_point = pd.DataFrame({
        'engagement_rate': [0.85],
        'time_on_task_s': [501],
        'hint_ratio': [0.67],
        'interaction_count': [14],
        'task_completed': [0],
        'quiz_score': [89.11],
        'difficulty': [2],
        'error_rate': [0.59],
        'task_timed_out': [0],
        'time_before_hint_used': [199]
    })

    # Make predictions using the trained models and feature names
    reg_pred, clf_pred = predict((reg, clf), feature_names, new_data_point)
    print(f"Example prediction (custom data): Cognitive Load={reg_pred[0]:.3f}, Engagement Level={clf_pred[0]}")

    # --- Plotting Example ---
    importances = reg.feature_importances_
    fig = plot_bar_chart(
        x=feature_names,
        y=importances,
        xlabel='Features',
        ylabel='Importance',
        title='Random Forest Feature Importances',
        show=False,
        rotation=45
    )
    save_figure_to_image_folder(fig, prefix='rf_feature_importance', image_dir='image')