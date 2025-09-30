import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
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


def qqplot_plotly(x, y, title="QQ Plot"):
    """
    Trace un QQ-plot entre deux séries avec plotly express.
    x : valeurs observées
    y : valeurs prédites
    """
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    min_len = min(len(x_sorted), len(y_sorted))
    df = pd.DataFrame({
        "Observed": x_sorted[:min_len],
        "Predicted": y_sorted[:min_len]
    })
    fig = px.scatter(df, x="Observed", y="Predicted", title=title)
    fig.add_shape(
        type="line",
        x0=df["Observed"].min(), y0=df["Predicted"].min(),
        x1=df["Observed"].max(), y1=df["Predicted"].max(),
        line=dict(color="red", dash="dash"),
        name="y=x"
    )
    fig.update_layout(showlegend=False)
    fig.show()