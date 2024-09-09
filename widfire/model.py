import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class StackingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model

    def fit(self, X, y):
        self.base_models_ = [clone(model) for model in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        
        for model in self.base_models_:
            model.fit(X, y)
        
        meta_features = np.column_stack([
            model.predict_proba(X)[:, 1] for model in self.base_models_
        ])
        self.meta_model_.fit(meta_features, y)
        
        return self

    def predict_proba(self, X):
        meta_features = np.column_stack([
            model.predict_proba(X)[:, 1] for model in self.base_models_
        ])
        return self.meta_model_.predict_proba(meta_features)

    def predict(self, X):
        meta_features = np.column_stack([
            model.predict_proba(X)[:, 1] for model in self.base_models_
        ])
        return self.meta_model_.predict(meta_features)

def train_model():
    # Load and preprocess data
    df = pd.read_csv('nasa.csv')
    df['datetime'] = pd.to_datetime(df['acq_date'] + ' ' + df['acq_time'])

    # Feature engineering
    df['month'] = df['datetime'].dt.month
    df['hour'] = df['datetime'].dt.hour
    df['daynight_numeric'] = (df['daynight'] == 'D').astype(int)
    df['latitude_abs'] = np.abs(df['latitude'])
    df['brightness_frp_ratio'] = df['brightness'] / (df['frp'] + 1)
    df['scan_track_ratio'] = df['scan'] / (df['track'] + 1)

    # Create target variable
    df['high_confidence_fire'] = df['confidence'].apply(lambda x: 1 if x > 70 else 0)

    # Select features
    features = ['latitude', 'longitude', 'brightness', 'scan', 'track', 'bright_t31', 'frp',
                'month', 'hour', 'daynight_numeric', 'latitude_abs', 'brightness_frp_ratio', 'scan_track_ratio']
    X = df[features]
    y = df['high_confidence_fire']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create preprocessing pipeline
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
        ])

    # Define base models
    base_models = [
        RandomForestClassifier(n_estimators=200, random_state=42),
        GradientBoostingClassifier(n_estimators=200, random_state=42),
        XGBClassifier(n_estimators=200, random_state=42),
        LGBMClassifier(n_estimators=200, random_state=42)
    ]

    # Create stacked model
    stacked_model = StackingClassifier(
        base_models=base_models,
        meta_model=LogisticRegression()
    )

    # Create final pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))),
        ('classifier', stacked_model)
    ])

    # Fit the model
    pipeline.fit(X_train, y_train)

    # Print some debugging information
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # Transform the data through the pipeline up to the classifier
    X_train_transformed = pipeline.named_steps['feature_selection'].transform(
        pipeline.named_steps['preprocessor'].transform(X_train)
    )
    X_test_transformed = pipeline.named_steps['feature_selection'].transform(
        pipeline.named_steps['preprocessor'].transform(X_test)
    )
    print(f"X_train_transformed shape: {X_train_transformed.shape}")
    print(f"X_test_transformed shape: {X_test_transformed.shape}")

    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    # Evaluate the model
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Feature importance
    feature_selector = pipeline.named_steps['feature_selection']
    selected_features = feature_selector.get_support()

    # Get the feature names after preprocessing (before feature selection)
    preprocessor = pipeline.named_steps['preprocessor']
    feature_names = (preprocessor.named_transformers_['num']
                     .named_steps['imputer']
                     .feature_names_in_)

    # Select only the features that were chosen by feature selection
    selected_feature_names = [name for name, selected in zip(feature_names, selected_features) if selected]

    print(f"Number of original features: {len(X.columns)}")
    print(f"Number of features after selection: {sum(selected_features)}")
    print(f"Selected feature names: {selected_feature_names}")

    if hasattr(pipeline.named_steps['classifier'].base_models_[0], 'feature_importances_'):
        importances = pipeline.named_steps['classifier'].base_models_[0].feature_importances_
        print(f"Shape of feature importances: {importances.shape}")
        
        feature_importance = pd.DataFrame({
            'feature': selected_feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Feature Importance:")
        print(feature_importance.head(10))

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
        plt.title('Top 10 Feature Importance')
        plt.savefig('feature_importance.png')
        plt.close()
    else:
        print("\nFeature importance is not available for this model.")

    # Save the model
    joblib.dump(pipeline, 'advanced_wildfire_model.joblib')
    print("\nModel saved as 'advanced_wildfire_model.joblib'")

if __name__ == "__main__":
    train_model()