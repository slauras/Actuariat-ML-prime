import numpy as np
import pandas as pd

from visualisation import (
    plot_regression_diagnostics,
    plot_classification_diagnostics
)

from sklearn.metrics import (
    r2_score, mean_absolute_error, root_mean_squared_error
)


class ModelMonitoring:
    def __init__(self, model_name:str):
        self.model_name = model_name
        self.models = []

        self.fold_metrics = []
        self.fold_loss = []
        self.fold_truth = []
        self.fold_preds = []

        self.aggregated_metrics = None
        self.aggregated_loss = None
        self.aggregated_truth = None
        self.aggregated_preds = None

    def _stack_(self, dict_:dict, elem):
        dict_[len(dict_)] = elem

        
    def _append_and_compute_metrics(self,y_true, y_pred, time):
        self.fold_truth.append(y_true)
        self.fold_preds.append(y_pred)
        self.fold_metrics.append({
            "r2": round(r2_score(y_true, y_pred), 2),
            "mae": round(mean_absolute_error(y_true, y_pred), 2),
            "rmse": round(root_mean_squared_error(y_true, y_pred), 2),
            "time": round(time, 2)
        })

    def _compute_all_aggregate(self):
        df_metrics = pd.DataFrame(self.fold_metrics)
        self.aggregated_metrics = {
            "r2_mean": round(df_metrics["r2"].mean(), 3),
            "mae_mean": round(df_metrics["mae"].mean(), 3),
            "rmse_mean": round(df_metrics["rmse"].mean(), 3),
            "time_mean": round(df_metrics["time"].mean(), 3),
            "r2_std": round(df_metrics["r2"].std(), 3),
            "mae_std": round(df_metrics["mae"].std(), 3),
            "rmse_std": round(df_metrics["rmse"].std(), 3),
            "time_std": round(df_metrics["time"].std(), 3)
        }
        self.aggregated_preds = np.concatenate(self.fold_preds)
        self.aggregated_truth = np.concatenate(self.fold_truth)

        if self.fold_loss:
            min_len = min(len(np.array(tupl[0])) for tupl in self.fold_loss)
            arrs = [(np.array(train)[:min_len], np.array(valid)[:min_len]) for train, valid in self.fold_loss]
            self.aggregated_loss = np.mean(arrs, axis=0)
        else:
            self.aggregated_loss = None

    def log(self, y_true, y_pred, time, loss=None, model=None):
        self._append_and_compute_metrics(y_true, y_pred, time)
        if loss is not None:
            self.fold_loss.append(loss)
        if model is not None:
            self.models.append(model)
        self._compute_all_aggregate()
        
    
    def plot_regression_diagnostics(self):
        if np.unique(self.aggregated_truth).size < 10:
            raise ValueError("Not a regression problem. Please use classification diagnostics.")
        plot_regression_diagnostics(
            self.aggregated_truth, 
            self.aggregated_preds, 
            self.fold_loss, 
            self.fold_metrics,
            self.model_name
        )
        
    def plot_classification_diagnostics(self):
        if np.unique(self.aggregated_truth).size >= 10:
            raise ValueError("Not a classification problem. Please use regression diagnostics.")
        plot_classification_diagnostics(
            self.aggregated_truth,
            self.aggregated_preds,
            self.model_name
        )