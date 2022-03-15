"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.17.7
"""

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest

def train_model(train_df: pd.DataFrame, contamination_value: float):
    # Initialize isolation forest classifier model
    clf = IsolationForest(random_state=42, 
                         bootstrap=True,
                         contamination=contamination_value)

    # Fit model on training dataset
    clf.fit(train_df.values)

    return clf


def predict(ml_model, test_df: pd.DataFrame):
    # Generate predictions on test dataset
    preds = ml_model.predict(test_df.values)

    # Modify predictions to match TX_FRAUD label (1 = fraud, 0 = no fraud)
    preds_mod = np.array(list(map(lambda x: 1*(x == -1), preds)))

    # Get anomaly scores that led to predictions
    anomaly_scores = ml_model.score_samples(test_df)

    # Convert anomaly scores to positive values
    anomaly_scores_mod = np.array([-x for x in anomaly_scores])

    test_df['ANOMALY_SCORE'] = anomaly_scores_mod
    test_df['ANOMALY'] = preds_mod

    return test_df


