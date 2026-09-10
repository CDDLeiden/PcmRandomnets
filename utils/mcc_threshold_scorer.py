"""Calculation of MCC scores for a regression model."""
import json

import numpy as np
from qsprpred.models.assessment.metrics.base import Metric
from rdkit.ML.Scoring.Scoring import CalcBEDROC
from sklearn.metrics import matthews_corrcoef, mean_squared_error, r2_score

from utils.csv_append import append_row


def as_1d(values):
    """Flatten targets/predictions to a 1-D float array.

    QSPRpred hands targets over as an (n, 1) frame and DNNModel predicts an (n, 1)
    array, and since numpy 2 a size-1 array no longer converts to a Python scalar.
    """
    return np.asarray(values, dtype=float).ravel()

class MccThresholdScorer(Metric):
    def __init__(self, threshold, model_name: str, params, path: str, dataset: str = '',
                 fp_size: int = 256, replicate: int = 0):
        self.threshold = threshold
        self.model_name = model_name
        self.params = params
        self.path = path
        self.dataset = dataset
        self.fp_size = fp_size
        self.replicate = replicate

    def __call__(self, y_true, y_pred, sample_weight=None):
        print('CALCULATING METRICS')
        y_true, y_pred = as_1d(y_true), as_1d(y_pred)
        y_true_class = y_true > self.threshold
        y_pred_class = y_pred > self.threshold

        self.dump_metrics(y_true, y_pred)

        return matthews_corrcoef(y_true_class, y_pred_class)

    def dump_metrics(self, y_true, y_pred):
        y_true, y_pred = as_1d(y_true), as_1d(y_pred)
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)

        ys_class = y_true > 6.5
        y_hat_means_class = y_pred > 6.5

        mcc = matthews_corrcoef(ys_class, y_hat_means_class)
        #TODO: y_true or rather y_true_class?
        bedroc = CalcBEDROC(
            [[y] for _, y in sorted(zip(y_pred, ys_class), reverse=True)],
            col=0,
            alpha=20,
        )

        print(f"R2: {r2:.3f}, MSE: {mse:.3f}, MCC:{mcc:.3f}, BEDROC: {bedroc:.3f}")

        append_row(self.path, {
            'id': self.model_name,
            'replicate': self.replicate,
            'dataset': self.dataset,
            'fp_size': self.fp_size,
            'params': json.dumps(self.params),
            'r2': f"{r2:.3f}",
            'mse': f"{mse:.3f}",
            'mcc': f"{mcc:.3f}",
            'bedroc': f"{bedroc:.3f}",
        })

    def supportsTask(self, task) -> bool:
        """Return true if the scorer supports the given task.

        Args:
            task (ModelTasks): Task of the model.

        Returns:
            bool: True if the scorer supports the given task.
        """
        return True

    @property
    def needsProbasToScore(self) -> bool:
        """Return True if the scorer needs probabilities to score.

        Returns:
            bool: True if the scorer needs probabilities to score.
        """
        return False

    @property
    def needsDiscreteToScore(self) -> bool:
        """Return True if the scorer needs discrete values to score.

        Returns:
            bool: True if the scorer needs discrete values to score.
        """
        return False

    @property
    def isClassificationMetric(self) -> bool:
        """Return true if the scorer supports any type of classification tasks."""
        return False
    @property
    def isRegressionMetric(self) -> bool:
        """Return true if the scorer supports any type of regression tasks."""
        return True