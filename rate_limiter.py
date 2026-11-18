"""
Módulo de Baseline de Comparación: Limitador de Tasa Estático (Rate Limiter)
============================================================================
Implementa el mecanismo de control perimetral tradicional basado en umbral fijo:
Ventana deslizante de 60 segundos con límite de 60 peticiones/minuto por token.
"""

from collections import defaultdict, deque
from typing import Dict, Deque, Tuple


class StaticRateLimiter:
    """
    Limitador de tasa tradicional con ventana deslizante de 60 segundos.
    Regla estática: Si peticiones en ventana de 60s > 60 req/min, clasifica como Ataque (1),
    de lo contrario Normal (0).
    """
    def __init__(self, window_size_seconds: float = 60.0, max_requests: int = 60):
        self.window_size_seconds = window_size_seconds
        self.max_requests = max_requests
        # Historial de timestamps por token_jti
        self.request_history: Dict[str, Deque[float]] = defaultdict(deque)

    def reset(self):
        """Reinicia el estado del limitador."""
        self.request_history.clear()

    def inspect_request(self, token_jti: str, timestamp: float) -> Tuple[int, int]:
        """
        Evalúa una petición entrante en tiempo real.
        
        Retorna:
          (prediction, current_rate)
          - prediction: 1 si excede max_requests (Ataque detectado), 0 en caso contrario.
          - current_rate: número de peticiones en la ventana deslizante actual.
        """
        history = self.request_history[token_jti]
        
        # Eliminar peticiones fuera de la ventana de 60 segundos
        cutoff = timestamp - self.window_size_seconds
        while history and history[0] < cutoff:
            history.popleft()

        # Registrar la petición actual
        history.append(timestamp)
        current_count = len(history)

        # Regla estática: superar 60 req en 60 seg
        prediction = 1 if current_count > self.max_requests else 0
        return prediction, current_count

