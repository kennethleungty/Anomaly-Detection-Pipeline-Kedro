"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_model, predict


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

        node(
            func=train_model,
            inputs=["train_data", "params:contamination_value"],
            outputs="ml_model",
            name="node_train_model"
            ),

        node(
            func=predict,
            inputs=["ml_model", "test_data"],
            outputs="predictions",
            name="node_predict"
            ),

    ])
