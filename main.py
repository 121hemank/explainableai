import shap
from lime import lime_tabular
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import argparse
import logging

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset(file_path):
    """Load dataset from a CSV file."""
    try:
        logger.info("Loading dataset...")
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        exit(1)

def split_data(df, target_column):
    """Split dataset into training and testing sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def create_models():
    """Create and return a dictionary of ML models."""
    logger.info("Initializing models...")
    return {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }

class XAIWrapper:
    """XAI Wrapper for explainability with SHAP and LIME."""
    
    def __init__(self):
        self.models = {}
        self.explainers = {}
        
    def fit(self, models, X_train, y_train):
        """Fit models and initialize SHAP explainers."""
        self.models = models
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            # Use SHAP KernelExplainer for interpretability
            self.explainers[model_name] = shap.KernelExplainer(model.predict, X_train[:100])

    def explain_prediction(self, model_name, user_input):
        """Explain a single prediction using SHAP values."""
        model = self.models[model_name]
        explainer = self.explainers[model_name]

        # Prepare input as a DataFrame
        input_df = pd.DataFrame([user_input])
        
        # SHAP explanation
        shap_values = explainer.shap_values(input_df)
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)
        explanation = shap_values

        return prediction, probabilities, explanation

    def analyze(self):
        """Perform global analysis on the models using SHAP summary plots."""
        logger.info("Analyzing models globally with SHAP summary plots.")
        for model_name, explainer in self.explainers.items():
            shap.summary_plot(explainer.shap_values, X_train, show=False)

    def generate_report(self):
        """Generate XAI report (this could be extended as per requirements)."""
        logger.info("Report generation functionality to be implemented.")


def generate_predictions(xai, X):
    """Prompt user for input and generate predictions."""
    while True:
        logger.info("\nEnter values for prediction (or 'q' to quit):")
        user_input = {}
        for feature in X.columns:
            value = input(f"{feature}: ")
            if value.lower() == 'q':
                return
            try:
                user_input[feature] = float(value)
            except ValueError:
                user_input[feature] = value
        
        try:
            # Choose a model for explanation (Random Forest as an example)
            model_name = 'Random Forest'
            prediction, probabilities, explanation = xai.explain_prediction(model_name, user_input)
            logger.info("\nPrediction Results:")
            logger.info(f"Prediction: {prediction}")
            logger.info(f"Probabilities: {probabilities}")
            logger.info("\nExplanation of Prediction (SHAP values):")
            logger.info(explanation)
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")

def main(file_path, target_column):
    # Load and explore the dataset
    df = load_dataset(file_path)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df, target_column)
    
    # Create models
    models = create_models()

    # XAI wrapper initialization and model fitting
    xai = XAIWrapper()
    logger.info("Fitting models and performing XAI analysis...")
    xai.fit(models, X_train, y_train)
    xai.analyze()

    # Generate report
    try:
        logger.info("Generating XAI report...")
        xai.generate_report()
    except Exception as e:
        logger.error(f"Report generation error: {str(e)}")

    # Prediction based on user input
    generate_predictions(xai, X_train)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run XAI analysis on a dataset")
    parser.add_argument("file_path", help="Path to the CSV file containing the dataset")
    parser.add_argument("target_column", help="Name of the target column in the dataset")
    args = parser.parse_args()

    logger.info("Starting the XAI program...")
    main(args.file_path, args.target_column)
