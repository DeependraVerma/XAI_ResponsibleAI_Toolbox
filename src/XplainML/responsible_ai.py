import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.reductions import ExponentiatedGradient, GridSearch, EqualizedOdds

class ResponsibleAI:
    def __init__(self, X_train, y_train, X_test, y_test, sensitive_features):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.sensitive_features = sensitive_features

    def evaluate_model(self, model):
        # Train the model
        model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = model.predict(self.X_test)
        
        # Evaluate model performance
        accuracy = accuracy_score(self.y_test, y_pred)
        confusion = confusion_matrix(self.y_test, y_pred)
        classification_rep = classification_report(self.y_test, y_pred)
        
        return accuracy, confusion, classification_rep

    def calculate_fairness_metrics(self, model):
        # Calculate demographic parity difference
        dp_diff = demographic_parity_difference(model, self.X_test, self.y_test, sensitive_features=self.sensitive_features)
        
        # Calculate equalized odds difference
        eo_diff = equalized_odds_difference(model, self.X_test, self.y_test, sensitive_features=self.sensitive_features)

        return dp_diff, eo_diff

    def mitigate_bias(self, model, constraint=None):
        # Apply fairness constraints using the Exponentiated Gradient algorithm
        exp_gradient = ExponentiatedGradient(model, constraint=constraint)
        exp_gradient.fit(self.X_train, self.y_train, sensitive_features=self.sensitive_features)
        mitigated_model = exp_gradient.predictor
        
        return mitigated_model