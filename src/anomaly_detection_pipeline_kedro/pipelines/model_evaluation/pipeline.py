"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=evaluate_model,
            inputs=["predictions", "test_labels", "neptune_run"],
            outputs="evaluation_plot",
            name="node_model_evaluation"
            ),
    ])
