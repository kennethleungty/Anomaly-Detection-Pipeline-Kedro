"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 0.17.7
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import logging
import neptune.new as neptune

run = neptune.init(project='kennethleung.ty/Anomaly-Detection-Pipeline-Kedro',
                   api_token='<API_TOKEN>')

def evaluate_model(predictions: pd.DataFrame, test_labels: pd.DataFrame,
                   neptune_run: run):
    def get_auc(labels, scores):
        fpr, tpr, thr = roc_curve(labels, scores)
        auc_score = auc(fpr, tpr)
        return fpr, tpr, auc_score

    def get_aucpr(labels, scores):
        precision, recall, thr = precision_recall_curve(labels, scores)
        aucpr_score = np.trapz(recall, precision)
        return precision, recall, aucpr_score

    def plot_metric(ax, x, y, x_label, y_label, plot_label, style="-"):
        ax.plot(x, y, style, label=plot_label)
        ax.legend()
        ax.set_ylabel(x_label)
        ax.set_xlabel(y_label)

    def prediction_summary(labels, predicted_score, info, plot_baseline=True, axes=None):
        if axes is None:
            axes = [plt.subplot(1, 2, 1), plt.subplot(1, 2, 2)]

        fpr, tpr, auc_score = get_auc(labels, predicted_score)
        plot_metric(axes[0], fpr, tpr, "False positive rate",
                    "True positive rate", "{} AUC = {:.4f}".format(info, auc_score))
        if plot_baseline:
            plot_metric(axes[0], [0, 1], [0, 1], "False positive rate",
                    "True positive rate", "Baseline AUC = 0.5", "r--")

        precision, recall, aucpr_score = get_aucpr(labels, predicted_score)
        plot_metric(axes[1], recall, precision, "Recall", 
                    "Precision", "{} AUCPR = {:.4f}".format(info, aucpr_score))

        if plot_baseline:
            thr = sum(labels)/len(labels)
            plot_metric(axes[1], [0, 1], [thr, thr], "Recall",
                    "Precision", "Baseline AUCPR = {:.4f}".format(thr), "r--")

        plt.show()
        return axes
    
    _, _, auc_score = get_auc(test_labels['TX_FRAUD'].values,  predictions['ANOMALY_SCORE'].values)
    _, _, aucpr_score = get_aucpr(test_labels['TX_FRAUD'].values,  predictions['ANOMALY_SCORE'].values)

    # log = logging.getLogger(__name__)
    # log.info("AUC-ROC Score: %0.2f%%", auc_score)
    # log.info("AUC-PR Score: %0.2f%%", aucpr_score)

    # Log scores into Neptune
    neptune_run['nodes/report/auc_roc_score'].log(auc_score)
    neptune_run['nodes/report/auc_pr_score'].log(aucpr_score)

    fig = plt.figure()
    fig.set_figheight(4.5)
    fig.set_figwidth(4.5*2)
    axes = prediction_summary(test_labels['TX_FRAUD'].values, predictions['ANOMALY_SCORE'].values, "Isolation Forest")

    # Log AUC plots to Neptune
    neptune_run['nodes/report/auc_plots'].upload(fig)

    return fig