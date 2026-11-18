"""
Script Reproducible Completo: Testbed de Detección de Abuso en JWT
==================================================================
Ejecución integral del experimento científico:
  1. Inicialización del API Gateway Mock con emisión de tokens JWT (HS256).
  2. Generación estocástica de tráfico (Línea Base + 3 Escenarios de Ataque).
  3. Entrenamiento de la Matriz de Transición de Markov (2,000 eventos).
  4. Calibración empírica del umbral tau (τ) con garantía FPR < 1.0%.
  5. Inferencia en streaming y comparación con Limitador Estático (Rate Limiter).
  6. Cálculo de matrices de confusión, métricas globales y por vector de ataque.
  7. Exportación de gráficos comparativos de alta resolución y reporte CSV.
"""

import sys
import os
import time

# Asegurar importaciones relativas al directorio del script
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np
import pandas as pd
from gateway_mock import APIGatewaySimulator, ENDPOINTS
from traffic_generator import TrafficGenerator
from anomaly_detector import MarkovJWTAnomalyDetector
from rate_limiter import StaticRateLimiter
from evaluator import evaluate_testbed


def run_full_experiment(random_seed: int = 42, output_dir: str = None) -> pd.DataFrame:
    if output_dir is None:
        output_dir = current_dir

    print("=" * 80)
    print("   BANCO DE PRUEBAS EXPERIMENTAL: DETECCIÓN ESTOCÁSTICA DE ABUSO EN JWT")
    print("   Arquitectura Zero-Trust | Cadenas de Markov vs Rate Limiter Estático")
    print("=" * 80)

    start_wall_time = time.time()

    # -------------------------------------------------------------
    # Paso 1: Generación de Tráfico y Escenarios
    # -------------------------------------------------------------
    print("\n[Paso 1/5] Generando flujos de tráfico y escenarios multivectoriales...")
    generator = TrafficGenerator(random_seed=random_seed)

    # Entrenar con 2,000 eventos legítimos, calibrar con 500 y testear con 4,000 mixtos
    train_events, calib_events, test_events = generator.generate_dataset(
        target_train_events=2000,
        target_calib_events=500,
        target_test_events=4000,
        attack_ratio_test=0.10  # 10% ataques, 90% legítimo
    )

    print(f"  -> Eventos de Entrenamiento (Normal): {len(train_events):,}")
    print(f"  -> Eventos de Calibración  (Normal): {len(calib_events):,}")
    print(f"  -> Eventos de Prueba Mixta (Test):   {len(test_events):,}")
    
    # Análisis del conjunto de prueba
    total_test = len(test_events)
    norm_test = sum(1 for e in test_events if e.is_attack == 0)
    atk_test = sum(1 for e in test_events if e.is_attack == 1)
    low_slow_test = sum(1 for e in test_events if e.attack_type == "LowAndSlow")
    replay_test = sum(1 for e in test_events if e.attack_type == "TokenReplay")
    burst_test = sum(1 for e in test_events if e.attack_type == "Burst")

    print(f"     * Tráfico Normal:    {norm_test:,} eventos ({norm_test/total_test:.1%})")
    print(f"     * Tráfico de Ataque: {atk_test:,} eventos ({atk_test/total_test:.1%})")
    print(f"       - Low & Slow:     {low_slow_test:,} eventos")
    print(f"       - Token Replay:   {replay_test:,} eventos")
    print(f"       - Fuerza Bruta:   {burst_test:,} eventos")

    # -------------------------------------------------------------
    # Paso 2: Entrenamiento del Modelo Estocástico de Markov
    # -------------------------------------------------------------
    print("\n[Paso 2/5] Entrenando matriz de transición de Markov de 1er orden...")
    detector = MarkovJWTAnomalyDetector(
        window_size=5,
        alpha_smoothing=0.01,
        epsilon=1e-6,
        lambda_ip=8.0,
        lambda_time=5.0
    )
    detector.fit(train_events, known_endpoints=ENDPOINTS)

    print(f"  -> Estados reconocidos ({len(detector.states)}): {detector.states}")
    print("  -> Muestra de Matriz de Transición P(e_t | e_{t-1}) aprendida:")
    df_trans = pd.DataFrame(
        detector.transition_matrix,
        index=detector.states,
        columns=detector.states
    )
    # Mostrar transiciones principales
    print(df_trans.round(4).to_string())

    # -------------------------------------------------------------
    # Paso 3: Calibración del Umbral τ en Conjunto de Validación
    # -------------------------------------------------------------
    print("\n[Paso 3/5] Calibrando umbral empírico τ para garantizar FPR < 1.0%...")
    calibrated_tau = detector.calibrate_threshold(calib_events, target_fpr=0.01)
    print(f"  -> Umbral calibrado: τ = {calibrated_tau:.4f}")

    # -------------------------------------------------------------
    # Paso 4: Inferencia en Streaming y Evaluación Comparativa
    # -------------------------------------------------------------
    print("\n[Paso 4/5] Ejecutando inferencia en streaming sobre 4,000 eventos...")
    rate_limiter = StaticRateLimiter(window_size_seconds=60.0, max_requests=60)

    df_metrics, artifacts = evaluate_testbed(
        test_events=test_events,
        detector=detector,
        rate_limiter=rate_limiter,
        output_dir=output_dir
    )

    m_markov = artifacts["metrics_markov"]
    m_rl = artifacts["metrics_rl"]

    # -------------------------------------------------------------
    # Paso 5: Reporte Científico y Validación Experimental
    # -------------------------------------------------------------
    print("\n[Paso 5/5] Síntesis de Resultados y Comparación Cuantitativa:")
    print("-" * 80)
    print(f"{'Métrica':<25} | {'Markov + Divergencia':<22} | {'Rate Limiter (60 req/min)':<25}")
    print("-" * 80)
    print(f"{'True Positives (TP)':<25} | {m_markov['TP']:<22} | {m_rl['TP']:<25}")
    print(f"{'False Positives (FP)':<25} | {m_markov['FP']:<22} | {m_rl['FP']:<25}")
    print(f"{'True Negatives (TN)':<25} | {m_markov['TN']:<22} | {m_rl['TN']:<25}")
    print(f"{'False Negatives (FN)':<25} | {m_markov['FN']:<22} | {m_rl['FN']:<25}")
    print("-" * 80)
    print(f"{'Precisión':<25} | {m_markov['Precision']:<22.4f} | {m_rl['Precision']:<25.4f}")
    print(f"{'Recall (TPR)':<25} | {m_markov['Recall']:<22.4f} | {m_rl['Recall']:<25.4f}")
    print(f"{'F1-Score':<25} | {m_markov['F1_Score']:<22.4f} | {m_rl['F1_Score']:<25.4f}")
    print(f"{'FPR (Falsas Alarmas)':<25} | {m_markov['FPR']:<22.4%} | {m_rl['FPR']:<25.4%}")
    print(f"{'Especificidad':<25} | {m_markov['Specificity']:<22.4f} | {m_rl['Specificity']:<25.4f}")
    print(f"{'ROC AUC':<25} | {artifacts['roc_auc']:<22.4f} | {'N/A (Regla Discreta)':<25}")
    print("-" * 80)
    print("DETECCIÓN POR VECTOR DE ATAQUE (RECALL):")
    print(f"  * Escenario A (Low & Slow):    Markov={m_markov['Recall_LowAndSlow']:.1%}  vs  RateLimiter={m_rl['Recall_LowAndSlow']:.1%}")
    print(f"  * Escenario B (Token Replay):  Markov={m_markov['Recall_TokenReplay']:.1%}  vs  RateLimiter={m_rl['Recall_TokenReplay']:.1%}")
    print(f"  * Escenario C (Fuerza Bruta):  Markov={m_markov['Recall_Burst']:.1%}  vs  RateLimiter={m_rl['Recall_Burst']:.1%}")
    print("-" * 80)

    # Validaciones obligatorias de la consigna
    assert m_markov["FPR"] < 0.01, f"ALERTA: FPR ({m_markov['FPR']:.4f}) no cumple la condición < 1%!"
    assert m_markov["Recall_LowAndSlow"] > 0.85, "ALERTA: Modelo estocástico no capturó el ataque Low & Slow!"
    assert m_rl["Recall_LowAndSlow"] == 0.0, "ALERTA: Rate Limiter estático no debería capturar Low & Slow!"

    print("\n[VALIDACIÓN EXPERIMENTAL SUPERADA CON ÉXITO]")
    print(f"  [✓] FPR del Modelo Estocástico: {m_markov['FPR']:.2%} (< 1.0% REQUERIDO)")
    print(f"  [✓] Detección de Ataque Low & Slow: {m_markov['Recall_LowAndSlow']:.1%} frente a {m_rl['Recall_LowAndSlow']:.1%} del Rate Limiter")
    print(f"  [✓] Detección de Token Replay:     {m_markov['Recall_TokenReplay']:.1%} frente a {m_rl['Recall_TokenReplay']:.1%} del Rate Limiter")
    print(f"  [✓] Detección de Ráfaga Masiva:    {m_markov['Recall_Burst']:.1%} frente a {m_rl['Recall_Burst']:.1%} del Rate Limiter")

    print(f"\nArchivos generados en '{output_dir}':")
    print(f"  - {os.path.basename(artifacts['cm_plot_path'])}")
    print(f"  - {os.path.basename(artifacts['roc_plot_path'])}")
    print(f"  - {os.path.basename(artifacts['csv_path'])}")

    elapsed = time.time() - start_wall_time
    print(f"\nTiempo total de ejecución: {elapsed:.2f} segundos.")
    print("=" * 80)

    return df_metrics


if __name__ == "__main__":
    run_full_experiment()

