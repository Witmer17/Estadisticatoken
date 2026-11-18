"""
Módulo de Evaluación y Generación de Métricas / Gráficos
=========================================================
Calcula y compara cuantitativamente el desempeño del Modelo Estocástico
frente al Rate Limiter Estático tradicional:
  - Matrices de confusión (TP, FP, TN, FN)
  - Métricas avanzadas: Precisión, Recall, F1-Score, FPR, Especificidad
  - Análisis por vector de ataque: 'Low & Slow', 'Token Replay', 'Burst'
  - Generación de gráficos de alta fidelidad:
      * matriz_confusion_comparativa.png
      * curva_roc_calibracion.png
      * resultados_metricas.csv
"""

import os
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

from traffic_generator import SimulatedEvent
from anomaly_detector import MarkovJWTAnomalyDetector
from rate_limiter import StaticRateLimiter


def compute_metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula las métricas estándar de detección de anomalías."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    return {
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1_Score": float(f1),
        "FPR": float(fpr),
        "Specificity": float(specificity),
        "Accuracy": float(accuracy)
    }


def evaluate_testbed(
    test_events: List[SimulatedEvent],
    detector: MarkovJWTAnomalyDetector,
    rate_limiter: StaticRateLimiter,
    output_dir: str = "."
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ejecuta la inferencia en streaming en el conjunto de prueba para ambos modelos,
    calcula estadísticas comparativas y genera los artefactos visuales y tabulares.
    """
    os.makedirs(output_dir, exist_ok=True)
    detector.reset_streaming_state()
    rate_limiter.reset()

    y_true: List[int] = []
    y_pred_markov: List[int] = []
    scores_markov: List[float] = []
    y_pred_rl: List[int] = []
    attack_types: List[str] = []

    # Inferencia en streaming evento a evento
    for ev in test_events:
        y_true.append(ev.is_attack)
        attack_types.append(ev.attack_type)

        # 1. Detector Markoviano Contextual
        pred_m, score_m, _ = detector.predict(ev)
        y_pred_markov.append(pred_m)
        scores_markov.append(score_m)

        # 2. Rate Limiter Estático
        pred_rl, _ = rate_limiter.inspect_request(ev.token_jti, ev.timestamp)
        y_pred_rl.append(pred_rl)

    y_true_arr = np.array(y_true)
    y_pred_markov_arr = np.array(y_pred_markov)
    scores_markov_arr = np.array(scores_markov)
    y_pred_rl_arr = np.array(y_pred_rl)
    attack_types_arr = np.array(attack_types)

    # Métricas Globales
    metrics_markov = compute_metrics_dict(y_true_arr, y_pred_markov_arr)
    metrics_rl = compute_metrics_dict(y_true_arr, y_pred_rl_arr)

    # Desglose por vector de ataque
    scenarios = ["LowAndSlow", "TokenReplay", "Burst"]
    for sc in scenarios:
        mask = (attack_types_arr == sc)
        total_sc = np.sum(mask)
        if total_sc > 0:
            rec_m = np.sum((y_pred_markov_arr[mask] == 1)) / total_sc
            rec_rl = np.sum((y_pred_rl_arr[mask] == 1)) / total_sc
        else:
            rec_m, rec_rl = 0.0, 0.0
        metrics_markov[f"Recall_{sc}"] = float(rec_m)
        metrics_rl[f"Recall_{sc}"] = float(rec_rl)

    # Crear DataFrame comparativo
    df_metrics = pd.DataFrame([
        {"Modelo": "Detector Probabilístico Estocástico (Markov + Contexto)", **metrics_markov},
        {"Modelo": "Regla Estática Tradicional (Rate Limiter: 60 req/min)", **metrics_rl}
    ])

    csv_path = os.path.join(output_dir, "resultados_metricas.csv")
    df_metrics.to_csv(csv_path, index=False)

    # =================================================================
    # 1. Gráfico de Matrices de Confusión Comparativas
    # =================================================================
    cm_markov = confusion_matrix(y_true_arr, y_pred_markov_arr, labels=[0, 1])
    cm_rl = confusion_matrix(y_true_arr, y_pred_rl_arr, labels=[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    plt.subplots_adjust(wspace=0.3)

    labels = ["Normal (0)", "Ataque (1)"]

    def plot_cm(cm, ax, title, cmap, model_type):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        annot_matrix = np.empty_like(cm).astype(str)
        for i in range(2):
            for j in range(2):
                annot_matrix[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]:.1%})"

        sns.heatmap(
            cm, annot=annot_matrix, fmt="", cmap=cmap, cbar=False,
            xticklabels=labels, yticklabels=labels, ax=ax, annot_kws={"size": 11, "weight": "bold"}
        )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Predicción del Sistema", fontsize=11, fontweight="bold")
        ax.set_ylabel("Condición Real (Ground Truth)", fontsize=11, fontweight="bold")

    plot_cm(
        cm_markov, axes[0],
        f"Modelo Probabilístico Estocástico\nF1: {metrics_markov['F1_Score']:.3f} | Recall: {metrics_markov['Recall']:.1%} | FPR: {metrics_markov['FPR']:.2%}",
        "Blues", "markov"
    )

    plot_cm(
        cm_rl, axes[1],
        f"Regla Estática (Rate Limiter: 60 req/min)\nF1: {metrics_rl['F1_Score']:.3f} | Recall: {metrics_rl['Recall']:.1%} | FPR: {metrics_rl['FPR']:.2%}",
        "Oranges", "rate_limiter"
    )

    plt.suptitle("Matriz de Confusión Comparativa: Token Anomaly Detection", fontsize=14, fontweight="bold", y=1.03)
    cm_plot_path = os.path.join(output_dir, "matriz_confusion_comparativa.png")
    fig.savefig(cm_plot_path, bbox_inches="tight")
    plt.close(fig)

    # =================================================================
    # 2. Curva ROC y Distribución de Scores Contextuales
    # =================================================================
    fpr_curve, tpr_curve, thresholds = roc_curve(y_true_arr, scores_markov_arr)
    roc_auc = auc(fpr_curve, tpr_curve)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Subplot A: Curva ROC con punto de operación
    ax_roc = axes[0]
    ax_roc.plot(fpr_curve, tpr_curve, color="#1f77b4", lw=2.5, label=f"Modelo Probabilístico (AUC = {roc_auc:.4f})")
    ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.2, label="Clasificador Aleatorio (AUC = 0.5000)")

    # Punto de operación del modelo probabilístico calibrado
    op_fpr = metrics_markov["FPR"]
    op_tpr = metrics_markov["Recall"]
    ax_roc.scatter([op_fpr], [op_tpr], color="#d62728", s=130, zorder=5, label=f"Punto Operativo τ={detector.tau:.2f} (FPR={op_fpr:.2%}, Recall={op_tpr:.1%})")

    # Punto de operación del Rate Limiter
    rl_fpr = metrics_rl["FPR"]
    rl_tpr = metrics_rl["Recall"]
    ax_roc.scatter([rl_fpr], [rl_tpr], color="#ff7f0e", marker="s", s=130, zorder=5, label=f"Rate Limiter 60 req/min (FPR={rl_fpr:.2%}, Recall={rl_tpr:.1%})")

    ax_roc.set_title("Curva ROC y Calibración del Umbral τ", fontsize=12, fontweight="bold")
    ax_roc.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=11, fontweight="bold")
    ax_roc.set_ylabel("Tasa de Verdaderos Positivos (Recall / TPR)", fontsize=11, fontweight="bold")
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.05])
    ax_roc.grid(True, linestyle=":", alpha=0.6)
    ax_roc.legend(loc="lower right", fontsize=9.5, frameon=True)

    # Subplot B: Distribución empírica del Score por Categoría de Tráfico
    ax_dist = axes[1]
    plot_df = pd.DataFrame({
        "Score": scores_markov_arr,
        "Tipo": attack_types_arr
    })
    # Mapear nombres legibles
    name_map = {
        "None": "Legítimo (Normal)",
        "LowAndSlow": "Ataque Low & Slow",
        "TokenReplay": "Token Replay / Hijack",
        "Burst": "Fuerza Bruta (Burst)"
    }
    plot_df["Categoria"] = plot_df["Tipo"].map(name_map)

    palette = {
        "Legítimo (Normal)": "#2ca02c",
        "Ataque Low & Slow": "#d62728",
        "Token Replay / Hijack": "#9467bd",
        "Fuerza Bruta (Burst)": "#ff7f0e"
    }

    sns.boxplot(
        data=plot_df, x="Categoria", y="Score", hue="Categoria", ax=ax_dist, palette=palette, legend=False, showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"}
    )
    # Línea del umbral tau
    ax_dist.axhline(detector.tau, color="red", linestyle="--", lw=2, label=f"Umbral τ = {detector.tau:.2f}")

    ax_dist.set_title("Distribución de Score de Divergencia Contextual", fontsize=12, fontweight="bold")
    ax_dist.set_xlabel("Categoría de Tráfico Evaluado", fontsize=11, fontweight="bold")
    ax_dist.set_ylabel("Score de Divergencia S(t)", fontsize=11, fontweight="bold")
    ax_dist.tick_params(axis='x', rotation=15)
    ax_dist.grid(True, linestyle=":", alpha=0.6)
    ax_dist.legend(loc="upper left", fontsize=10)

    plt.suptitle("Evaluación Estocástica: Calibración y Separabilidad de Clases", fontsize=14, fontweight="bold", y=1.02)
    roc_plot_path = os.path.join(output_dir, "curva_roc_calibracion.png")
    fig.savefig(roc_plot_path, bbox_inches="tight")
    plt.close(fig)

    return df_metrics, {
        "csv_path": csv_path,
        "cm_plot_path": cm_plot_path,
        "roc_plot_path": roc_plot_path,
        "roc_auc": roc_auc,
        "metrics_markov": metrics_markov,
        "metrics_rl": metrics_rl
    }
