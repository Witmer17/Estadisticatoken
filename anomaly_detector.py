"""
Módulo de Detección de Anomalías Estocástico y Cadenas de Markov
================================================================
Implementa la arquitectura probabilística contextual de detección de abuso
y secuestro de tokens JWT mediante:
  1. Aprendizaje de Matriz de Transición de Markov de primer orden P(e_t | e_{t-1}).
  2. Suavizado de Laplace y regularización estocástica.
  3. Puntuación de Divergencia Contextual con ventana deslizante W=5:
     Score(S_t) = - (1 / |W|) * ∑_{k ∈ W} log( P(e_k | e_{k-1}) + ε )
                  + λ_ip * I(ΔIP) + λ_time * S(Δt)
  4. Calibración empírica de umbral τ garantizando FPR < 1%.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from traffic_generator import SimulatedEvent


class MarkovJWTAnomalyDetector:
    """
    Detector estocástico contextual basado en Cadenas de Markov de 1er orden
    y divergencia contextual de telemetría (IP, User-Agent, Δt).
    """
    def __init__(
        self,
        window_size: int = 5,
        alpha_smoothing: float = 0.01,
        epsilon: float = 1e-6,
        lambda_ip: float = 8.0,
        lambda_time: float = 5.0,
        tau: float = 4.5
    ):
        self.window_size = window_size
        self.alpha_smoothing = alpha_smoothing
        self.epsilon = epsilon
        self.lambda_ip = lambda_ip
        self.lambda_time = lambda_time
        self.tau = tau  # Umbral de decisión

        # Espacio de estados
        self.states: List[str] = []
        self.state_to_idx: Dict[str, int] = {}
        self.idx_to_state: Dict[int, str] = {}

        # Matriz de conteos y de probabilidades de transición P(next | current)
        self.transition_counts: np.ndarray = np.empty((0, 0))
        self.transition_matrix: np.ndarray = np.empty((0, 0))

        # Estado en streaming por sesión/token: token_jti -> session_context
        self.session_states: Dict[str, Dict[str, Any]] = {}

    def fit(self, legitimate_events: List[SimulatedEvent], known_endpoints: Optional[List[str]] = None):
        """
        Fase de Entrenamiento:
        Aprende la matriz de transición de primer orden P(e_t | e_{t-1})
        usando exclusivamente eventos legítimos (Línea Base de Normalidad).
        """
        # Extraer estados únicos incluyendo el estado pseudo-inicial "START"
        endpoints_seen = set()
        if known_endpoints:
            endpoints_seen.update(known_endpoints)
        for ev in legitimate_events:
            endpoints_seen.add(ev.endpoint)

        self.states = ["START"] + sorted(list(endpoints_seen))
        self.state_to_idx = {state: idx for idx, state in enumerate(self.states)}
        self.idx_to_state = {idx: state for state, idx in self.state_to_idx.items()}

        num_states = len(self.states)
        self.transition_counts = np.zeros((num_states, num_states), dtype=np.float64)

        # Agrupar eventos por sesión para seguir las secuencias reales
        sessions: Dict[str, List[SimulatedEvent]] = defaultdict(list)
        for ev in legitimate_events:
            sessions[ev.session_id].append(ev)

        # Contar transiciones observadas
        for session_id, ev_list in sessions.items():
            ev_list.sort(key=lambda x: x.timestamp)
            prev_endpoint = "START"
            for ev in ev_list:
                curr_endpoint = ev.endpoint
                if curr_endpoint in self.state_to_idx:
                    i = self.state_to_idx[prev_endpoint]
                    j = self.state_to_idx[curr_endpoint]
                    self.transition_counts[i, j] += 1.0
                    prev_endpoint = curr_endpoint

        # Suavizado de Laplace: P(j | i) = (C_ij + alpha) / sum_k(C_ik + alpha)
        numerator = self.transition_counts + self.alpha_smoothing
        denominator = numerator.sum(axis=1, keepdims=True)
        self.transition_matrix = numerator / denominator

    def get_transition_prob(self, prev_endpoint: str, curr_endpoint: str) -> float:
        """Obtiene la probabilidad de transición P(curr | prev) suavizada."""
        i = self.state_to_idx.get(prev_endpoint)
        j = self.state_to_idx.get(curr_endpoint)
        if i is None or j is None:
            return self.alpha_smoothing / (len(self.states) * self.alpha_smoothing + 1.0)
        return float(self.transition_matrix[i, j])

    def reset_streaming_state(self):
        """Limpia el estado de sesiones en streaming."""
        self.session_states.clear()

    def _calculate_time_penalty(self, delta_t: float) -> float:
        """
        Calcula la penalización por anomalía temporal S(Δt).
        Peticiones en ráfaga (Δt < 0.5s) representan automatización o DoS masivo.
        """
        if delta_t <= 0.0:
            return 0.0
        if delta_t < 0.5:
            # Escalamiento progresivo para ráfagas extremas (Δt ~ 0.02s)
            return max(0.0, (1.0 - delta_t) / 0.5)
        return 0.0

    def compute_contextual_score(self, event: SimulatedEvent) -> Tuple[float, Dict[str, float]]:
        """
        Inferencia en Streaming:
        Mantiene ventana deslizante W de los últimos 5 eventos para cada sesión activa.
        Calcula:
          Score(S_t) = - (1 / |W|) * ∑_{k ∈ W} log( P(e_k | e_{k-1}) + ε )
                       + λ_ip * I(ΔIP) + λ_time * S(Δt)
        """
        token_key = event.token_jti

        # Inicializar contexto de sesión para este token si es nuevo
        if token_key not in self.session_states:
            self.session_states[token_key] = {
                "initial_ip": event.ip_address,
                "initial_ua": event.user_agent,
                "last_endpoint": "START",
                "last_timestamp": event.timestamp,
                "window_nll": deque(maxlen=self.window_size)
            }

        ctx = self.session_states[token_key]

        # 1. Componente Estocástico de Transición de Markov
        prev_endpoint = ctx["last_endpoint"]
        p_trans = self.get_transition_prob(prev_endpoint, event.endpoint)
        nll_k = -math.log(p_trans + self.epsilon)

        ctx["window_nll"].append(nll_k)
        markov_div = float(np.mean(ctx["window_nll"]))

        # 2. Componente de Divergencia de Red (IP / User-Agent Switch)
        ip_changed = (event.ip_address != ctx["initial_ip"]) or (event.user_agent != ctx["initial_ua"])
        i_delta_ip = 1.0 if ip_changed else 0.0
        ip_penalty = self.lambda_ip * i_delta_ip

        # 3. Componente de Frecuencia / Ráfaga Temporal S(Δt)
        delta_t = event.delta_t
        s_delta_t = self._calculate_time_penalty(delta_t)
        time_penalty = self.lambda_time * s_delta_t

        # Score global unificado
        total_score = markov_div + ip_penalty + time_penalty

        # Actualizar estado de la sesión
        ctx["last_endpoint"] = event.endpoint
        ctx["last_timestamp"] = event.timestamp

        components = {
            "markov_div": markov_div,
            "ip_penalty": ip_penalty,
            "time_penalty": time_penalty,
            "nll_instant": nll_k,
            "p_trans": p_trans
        }
        return total_score, components

    def calibrate_threshold(self, calib_events: List[SimulatedEvent], target_fpr: float = 0.01) -> float:
        """
        Calibra empíricamente el umbral tau (τ) en el conjunto de calibración normal
        para garantizar estrictamente FPR < target_fpr (1%).
        """
        self.reset_streaming_state()
        normal_scores: List[float] = []

        for ev in calib_events:
            score, _ = self.compute_contextual_score(ev)
            normal_scores.append(score)

        # Percentil correspondiente a (1.0 - target_fpr) con margen de seguridad del 0.2%
        effective_percentile = min(99.8, max(95.0, (1.0 - target_fpr + 0.002) * 100.0))
        calibrated_tau = float(np.percentile(normal_scores, effective_percentile))
        
        # Validar FPR empírico en calibración
        fps = sum(1 for s in normal_scores if s > calibrated_tau)
        empirical_fpr = fps / len(normal_scores) if normal_scores else 0.0
        
        # Ajuste fino si por discretización supera target_fpr
        while empirical_fpr >= target_fpr:
            calibrated_tau += 0.05
            fps = sum(1 for s in normal_scores if s > calibrated_tau)
            empirical_fpr = fps / len(normal_scores)

        self.tau = calibrated_tau
        self.reset_streaming_state()
        return self.tau

    def predict(self, event: SimulatedEvent) -> Tuple[int, float, Dict[str, float]]:
        """
        Clasifica una petición en streaming:
        1 si Score(S_t) > τ (Anomalía / Abuso de Token), 0 si es Normal.
        """
        score, components = self.compute_contextual_score(event)
        prediction = 1 if score > self.tau else 0
        return prediction, score, components

