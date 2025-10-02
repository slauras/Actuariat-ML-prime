import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
import seaborn as sns

from scipy.stats import gaussian_kde


import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    r2_score, mean_absolute_error, root_mean_squared_error,
    classification_report, confusion_matrix, roc_curve,
    precision_recall_curve, roc_auc_score
)





def plot_law_density(data, dist_theoretical, dist_name):
    fig = px.histogram(data, nbins=200, histnorm='probability density', opacity=0.6, )
    fig.data[0].name = "CHARGE (> 0)"
    
    # Courbe de densité fittée
    x = np.linspace(data.min(), data.max(), 500)
    y = dist_theoretical.pdf(x)
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f'{dist_name} fit', line=dict(color='red', width=2)))

    fig.update_layout(title=f"Loi {dist_name} ajustée (sur notre variable d'intérêt)", yaxis_title="Densité", xaxis_title="y")
    fig.show()

def plot_law_qq(data, dist_theoretical, dist_name):
    # Trier les données
    data_sorted = np.sort(data)
    n = len(data_sorted)
    # Quantiles théoriques de la loi Gamma ajustée
    probs = (np.arange(1, n+1) - 0.5) / n
    gamma_theoretical = dist_theoretical.ppf(probs)

    fig_qq = go.Figure()
    fig_qq.add_trace(go.Scatter(x=gamma_theoretical, y=data_sorted, mode='markers', name='QQ plot'))
    fig_qq.add_trace(go.Scatter(x=gamma_theoretical, y=gamma_theoretical, mode='lines', name='y=x', line=dict(color='red', dash='dash')))
    fig_qq.update_layout(title="QQ plot (Gamma)", xaxis_title="Quantiles théoriques", yaxis_title="Quantiles empiriques")
    fig_qq.show()



def plot_classification_metrics(experience_name, y_true, y_pred, y_proba):
    
    print(f"### Rapport de classification [{experience_name}] : \n", classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
    
    precision = precision[:-1] #if precision.shape[0] > y_pred.shape[0] else precision
    recall = recall[:-1]# if recall.shape[0] > y_pred.shape[0] else recall

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axs[0],
                xticklabels=["Classe 0", "Classe 1"], yticklabels=["Classe 0", "Classe 1"])
    axs[0].set_xlabel("Prédictions")
    axs[0].set_ylabel("Vraie Classe")
    axs[0].set_title("Matrice de Confusion")

    axs[1].plot(fpr, tpr, color="blue", lw=2, label=f"Roc curve (AUC = {roc_auc:.2f})")
    axs[1].plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
    for i in range(0, len(thresholds), max(1, len(thresholds) // 10)):
        axs[1].annotate(f"{thresholds[i]:.2f}", (fpr[i], tpr[i]), fontsize=8, color="red")
    axs[1].set_xlabel("Taux de Faux Positifs (FPR)")
    axs[1].set_ylabel("Taux de Vrais Positifs (TPR)")
    axs[1].set_title("Courbe ROC avec Thresholds")
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(pr_thresholds, recall, color='orange', label='Recall')
    axs[2].plot(pr_thresholds, precision, color='blue', label='Precision')
    axs[2].axvline(x=0.5, color='grey', linestyle='--', label='Seuil à 0.5')
    axs[2].set_xlabel("Seuil de confiance")
    axs[2].set_ylabel("Recall ou Precision")
    axs[2].set_title("Courbe Rappel/Confiance")
    axs[2].legend()

    plt.tight_layout()
    plt.show()



# def qqplot_plotly(x, y, title="QQ Plot"):
#     """
#     Trace un QQ-plot entre deux séries avec plotly express.
#     x : valeurs observées
#     y : valeurs prédites
#     """
#     x_sorted = np.sort(x)
#     y_sorted = np.sort(y)
#     min_len = min(len(x_sorted), len(y_sorted))
#     df = pd.DataFrame({
#         "Observed": x_sorted[:min_len],
#         "Predicted": y_sorted[:min_len]
#     })
#     fig = px.scatter(df, x="Observed", y="Predicted", title=title)
#     fig.add_shape(
#         type="line",
#         x0=df["Observed"].min(), y0=df["Predicted"].min(),
#         x1=df["Observed"].max(), y1=df["Predicted"].max(),
#         line=dict(color="red", dash="dash"),
#         name="y=x"
#     )
#     fig.update_layout(showlegend=False)
#     fig.show()
    
    

# def qq_plot(y_true, y_pred, model_name):

#     quantiles = np.linspace(0, 1, min(len(y_true), len(y_pred)))
#     q_true = np.quantile(y_true, quantiles)
#     q_pred = np.quantile(y_pred, quantiles)

#     # Calculer la densité pour chaque point
#     xy = np.vstack([q_true, q_pred])
#     z = gaussian_kde(xy)(xy)

#     plt.figure(figsize=(4, 3))
#     sc = plt.scatter(q_true, q_pred, c=z, cmap='gist_heat', s=10) 
#     max_val = max(q_true.max(), q_pred.max())
#     plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='y=x')
#     plt.xlabel("Quantiles réels")
#     plt.ylabel("Quantiles prédits")
#     plt.title(f"QQ Plot : {model_name}")
#     plt.legend()
#     plt.grid(True)
#     plt.colorbar(sc, label='Densité', ticks=[])
#     plt.show()
    
# def qqplot_plotly2(true, pred, title="QQ-Plot"):
#     # Remove NaN and sort
#     true = np.asarray(true)
#     pred = np.asarray(pred)
#     mask = ~np.isnan(true) & ~np.isnan(pred)
#     true_sorted = np.sort(true[mask])
#     pred_sorted = np.sort(pred[mask])
#     n = min(len(true_sorted), len(pred_sorted))
#     true_sorted = true_sorted[:n]
#     pred_sorted = pred_sorted[:n]
    
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=true_sorted,
#         y=pred_sorted,
#         mode='markers',
#         name='Quantiles'
#     ))
#     fig.add_trace(go.Scatter(
#         x=[true_sorted[0], true_sorted[-1]],
#         y=[true_sorted[0], true_sorted[-1]],
#         mode='lines',
#         name='y=x',
#         line=dict(color='red', dash='dash')
#     ))
#     fig.update_layout(
#         title=title,
#         xaxis_title='Quantiles réels',
#         yaxis_title='Quantiles prédits',
#         width=700,
#         height=500
#     )
#     fig.show()
    

# def plot_scatter(y_true, y_pred, model_name):
#     # Calculer la densité pour chaque point
#     xy = np.vstack([y_true, y_pred])
#     z = gaussian_kde(xy)(xy)
    
#     plt.figure(figsize=(4, 3))
#     sc = plt.scatter(y_true, y_pred,  c=z, cmap='gist_heat', s=10)
#     plt.xlabel("Valeurs réelles")
#     plt.ylabel("Prédictions")
#     plt.title(f"Scatter plot : {model_name}")
#     plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='y = x')
#     plt.legend()
#     plt.colorbar(sc, label='Densité', ticks=[])
#     plt.show()
    

# def plot_aggregated_loss(aggregated_loss, model_name):
#     train_data, eval_data = aggregated_loss
#     plt.figure(figsize=(6, 4))
#     plt.plot(train_data, label='Mean Train RMSE')
#     plt.plot(eval_data, label='Mean Valid RMSE')
#     plt.title(f'Evolution moyenne des loss : {model_name}')
#     plt.xlabel('Itération')
#     plt.ylabel('RMSE')
#     plt.legend()
#     plt.show()

# def plot_fold_loss(fold_loss, model_name):
#     plt.figure(figsize=(6, 4))
#     colors = sns.color_palette("tab10", n_colors=len(fold_loss))
#     for i, (train_data, eval_data) in enumerate(fold_loss):
#         plt.plot(train_data, label=f'Fold {i+1}: Train', color=colors[i])
#         plt.plot(eval_data, label=f'Fold {i+1}: Valid', color=colors[i], linestyle='--', alpha=0.7)
#     plt.title(f'Evolution de la loss par fold : {model_name}')
#     plt.xlabel('Itération')
#     plt.ylabel('RMSE')
#     plt.legend()
#     plt.show()
    
# def plot_metric_table(fold_metrics, model_name):
#     df_metrics = pd.DataFrame(fold_metrics).T
#     df_metrics.columns = [f"Fold {i+1}" for i in range(df_metrics.shape[1])]
#     df_metrics["Moyenne"] = df_metrics.mean(axis=1).round(2)
#     df_metrics["Ecart-type"] = df_metrics.drop(columns=["Moyenne"]).std(axis=1).round(2)
    
#     fig, ax = plt.subplots(figsize=(5, 4 ))
#     # Définir les couleurs pour chaque colonne
#     cell_colors = []
#     for i in range(len(df_metrics)):
#         row = []
#         for j in range(df_metrics.shape[1]):
#             if j >= df_metrics.shape[1] - 2:
#                 row.append('#f8d7da')  # rouge clair pour Moyenne et Ecart-type
#             else:
#                 row.append("#dadee3")  # gris clair pour les folds
#         cell_colors.append(row)

#     # Afficher le tableau
#     table = ax.table(
#         cellText=np.round(df_metrics.values, 2),
#         rowLabels=df_metrics.index,
#         colLabels=df_metrics.columns,
#         cellColours=cell_colors,
#         loc='center'
#     )
#     table.auto_set_font_size(False)
#     table.set_fontsize(10)
#     table.scale(1.2, 1.2)
#     ax.axis('off')
#     plt.title(f"Métriques : {model_name}")
#     plt.show()

def qq_plot(y_true, y_pred, model_name=None, ax:Axes = None):
    quantiles = np.linspace(0, 1, min(len(y_true), len(y_pred)))
    q_true = np.quantile(y_true, quantiles)
    q_pred = np.quantile(y_pred, quantiles)
    xy = np.vstack([q_true, q_pred])
    z = gaussian_kde(xy)(xy)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    sc = ax.scatter(q_true, q_pred, c=z, cmap='gist_heat', s=10)
    max_val = max(q_true.max(), q_pred.max())
    ax.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='y=x')
    ax.set_xlabel("Quantiles réels")
    ax.set_ylabel("Quantiles prédits")
    ax.set_title(f"QQ Plot" + (f" : {model_name}" if model_name else ""))
    ax.legend()
    if ax is None:
        plt.colorbar(sc, label='Densité', ticks=[])
        plt.grid(True)
        plt.show()

def plot_scatter(y_true, y_pred, model_name=None, ax:Axes = None):
    xy = np.vstack([y_true, y_pred])
    z = gaussian_kde(xy)(xy)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    sc = ax.scatter(y_true, y_pred,  c=z, cmap='gist_heat', s=10)
    ax.set_xlabel("Valeurs réelles")
    ax.set_ylabel("Prédictions")
    ax.set_title(f"Scatter plot" + (f" : {model_name}" if model_name else ""))
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='y = x')
    ax.legend()
    if ax is None:
        plt.colorbar(sc, label='Densité', ticks=[])
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def plot_fold_loss(fold_loss, model_name=None, ax:Axes = None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    colors = sns.color_palette("tab10", n_colors=len(fold_loss))
    for i, (train_data, eval_data) in enumerate(fold_loss):
        ax.plot(train_data, label=f'Fold {i+1}: Train', color=colors[i])
        ax.plot(eval_data, label=f'Fold {i+1}: Valid', color=colors[i], linestyle='--', alpha=0.7)
    ax.set_title('Evolution de la loss par fold' + (f" : {model_name}" if model_name else ""))
    ax.set_xlabel('Itération')
    ax.set_ylabel('RMSE')
    ax.legend(loc='lower left')
    if ax is None:
        plt.tight_layout()
        plt.show()

def plot_metric_table(fold_metrics, model_name=None, ax:Axes = None):
    df_metrics = pd.DataFrame(fold_metrics).T
    df_metrics.columns = [f"Fold {i+1}" for i in range(df_metrics.shape[1])]
    df_metrics["Moyenne"] = df_metrics.mean(axis=1).round(2)
    df_metrics["Ecart-type"] = df_metrics.drop(columns=["Moyenne"]).std(axis=1).round(2)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    cell_colors = []
    for i in range(len(df_metrics)):
        row = []
        for j in range(df_metrics.shape[1]):
            if j >= df_metrics.shape[1] - 2:
                row.append('#f8d7da')
            else:
                row.append("#dadee3")
        cell_colors.append(row)
    table = ax.table(
        cellText=np.round(df_metrics.values, 2),
        rowLabels=df_metrics.index,
        colLabels=df_metrics.columns,
        cellColours=cell_colors,
        loc='center'
    )
    table.auto_set_font_size(True)
    # table.set_fontsize(10)
    table.scale(1.2, 1.2)
    ax.axis('off')
    ax.set_title(f"Métriques globales" + (f" : {model_name}" if model_name else ""))
    if ax is None:
        plt.show()

def plot_regression_diagnostics(y_true, y_pred, fold_loss, fold_metrics, model_name):
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    qq_plot(y_true, y_pred, ax=axs[0, 0])
    plot_scatter(y_true, y_pred, ax=axs[0, 1])
    plot_fold_loss(fold_loss, ax=axs[1, 0])
    plot_metric_table(fold_metrics, ax=axs[1, 1])
    fig.suptitle(f"Diagnostic de régression : {model_name}", fontsize=16)
    plt.tight_layout()
    plt.show()
    
