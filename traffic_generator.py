"""
Módulo de Generador de Tráfico y Escenarios Sintéticos
======================================================
Simula tráfico de usuarios legítimos siguiendo un grafo de transición estocástico
y tres vectores de ataque avanzados contra tokens JWT:
  1. Escenario A: Ataque Sigiloso 'Low & Slow' (exfiltración a baja frecuencia).
  2. Escenario B: Token Replay / Secuestro de Sesión (cambio abrupto de contexto IP/UA).
  3. Escenario C: Fuerza Bruta / Ráfaga (burst masivo a 50 req/s).
"""

import random
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from gateway_mock import generate_jwt_token, ENDPOINTS, APIGatewaySimulator


# Perfiles de clientes y navegadores legítimos
LEGITIMATE_IPS = [
    f"192.168.1.{i}" for i in range(10, 50)
] + [
    f"10.0.4.{i}" for i in range(10, 40)
]

LEGITIMATE_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148",
]

# Perfiles de atacantes
ATTACKER_IPS = [
    f"198.51.100.{i}" for i in range(1, 30)
] + [
    f"203.0.113.{i}" for i in range(1, 30)
]

ATTACKER_USER_AGENTS = [
    "curl/8.4.0",
    "python-requests/2.31.0",
    "Go-http-client/1.1",
    "PostmanRuntime/7.36.0",
    "Mozilla/5.0 (compatible; EvilBot/1.0; +http://evil.com/bot)",
]

# Grafo de transición legítimo (Probabilidades condicionales humanas P(next | current))
LEGITIMATE_TRANSITIONS: Dict[str, Dict[str, float]] = {
    "START": {
        "/api/v1/auth/login": 0.95,
        "/api/v1/catalog/products": 0.05
    },
    "/api/v1/auth/login": {
        "/api/v1/catalog/products": 0.70,
        "/api/v1/user/profile": 0.30
    },
    "/api/v1/user/profile": {
        "/api/v1/catalog/products": 0.75,
        "/api/v1/user/profile": 0.15,
        "END": 0.10
    },
    "/api/v1/catalog/products": {
        "/api/v1/catalog/products": 0.40,
        "/api/v1/orders/create": 0.45,
        "/api/v1/user/profile": 0.10,
        "END": 0.05
    },
    "/api/v1/orders/create": {
        "/api/v1/payments/process": 0.90,
        "/api/v1/catalog/products": 0.10
    },
    "/api/v1/payments/process": {
        "/api/v1/user/profile": 0.65,
        "/api/v1/catalog/products": 0.15,
        "END": 0.20
    },
    "/api/v1/admin/export": {
        # Regular users never visit admin export
        "END": 1.0
    }
}


@dataclass
class SimulatedEvent:
    timestamp: float
    token_jti: str
    user_id: str
    endpoint: str
    method: str
    ip_address: str
    user_agent: str
    delta_t: float
    is_attack: int           # 0: Normal, 1: Attack
    attack_type: str         # 'None', 'LowAndSlow', 'TokenReplay', 'Burst'
    session_id: str


class TrafficGenerator:
    """
    Generador de eventos sintéticos y escenarios de ciberseguridad
    reproducibles con control temporal riguroso.
    """
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.rng = random.Random(random_seed)
        self.gateway = APIGatewaySimulator()
        self.current_sim_time = 1714500000.0  # Unix timestamp de inicio de simulación

    def _sample_next_endpoint(self, current_endpoint: str) -> str:
        """Muestrea el siguiente endpoint según el grafo de Markov legítimo."""
        transitions = LEGITIMATE_TRANSITIONS.get(current_endpoint, {"END": 1.0})
        endpoints = list(transitions.keys())
        probabilities = list(transitions.values())
        return self.rng.choices(endpoints, weights=probabilities, k=1)[0]

    def _get_http_method(self, endpoint: str) -> str:
        if endpoint in ["/api/v1/auth/login", "/api/v1/orders/create", "/api/v1/payments/process"]:
            return "POST"
        return "GET"

    def generate_legitimate_session(
        self,
        user_num: int,
        start_time: Optional[float] = None
    ) -> List[SimulatedEvent]:
        """
        Genera una sesión legítima con navegación estocástica coherente,
        intervalos Δt entre 2.0s y 8.0s, y contexto (IP/UA) invariable.
        """
        events: List[SimulatedEvent] = []
        user_id = f"user_{user_num:04d}"
        session_id = f"sess_legit_{user_num}_{int(self.current_sim_time)}"
        ip_address = self.rng.choice(LEGITIMATE_IPS)
        user_agent = self.rng.choice(LEGITIMATE_USER_AGENTS)

        token_info = generate_jwt_token(user_id=user_id, role="customer")
        token_jti = token_info["jti"]
        token_jwt = token_info["access_token"]

        sim_time = start_time if start_time is not None else self.current_sim_time

        # Petición inicial (Login o inicio de sesión)
        curr_endpoint = "/api/v1/auth/login"
        delta_t = 0.0
        events.append(SimulatedEvent(
            timestamp=sim_time,
            token_jti=token_jti,
            user_id=user_id,
            endpoint=curr_endpoint,
            method="POST",
            ip_address=ip_address,
            user_agent=user_agent,
            delta_t=delta_t,
            is_attack=0,
            attack_type="None",
            session_id=session_id
        ))

        # Secuencia de navegación
        max_steps = self.rng.randint(4, 9)
        step = 0
        while step < max_steps:
            next_endpoint = self._sample_next_endpoint(curr_endpoint)
            if next_endpoint == "END":
                break

            # Δt humano entre 2.0 y 8.0 segundos
            delta_t = round(self.rng.uniform(2.0, 8.0), 3)
            sim_time += delta_t
            method = self._get_http_method(next_endpoint)

            events.append(SimulatedEvent(
                timestamp=sim_time,
                token_jti=token_jti,
                user_id=user_id,
                endpoint=next_endpoint,
                method=method,
                ip_address=ip_address,
                user_agent=user_agent,
                delta_t=delta_t,
                is_attack=0,
                attack_type="None",
                session_id=session_id
            ))
            curr_endpoint = next_endpoint
            step += 1

        self.current_sim_time = sim_time + round(self.rng.uniform(1.0, 4.0), 3)
        return events

    def generate_low_and_slow_attack(
        self,
        victim_token_jti: str,
        victim_user_id: str,
        num_requests: int = 15,
        start_time: Optional[float] = None
    ) -> List[SimulatedEvent]:
        """
        Escenario A: Ataque Sigiloso 'Low & Slow'
        Exfiltración con token robado a baja frecuencia (Δt entre 3.0 y 5.0 s),
        navegando endpoints en orden caótico o directamente a /admin/export sin login previo.
        """
        events: List[SimulatedEvent] = []
        sim_time = start_time if start_time is not None else self.current_sim_time
        attacker_ip = self.rng.choice(ATTACKER_IPS)
        attacker_ua = self.rng.choice(ATTACKER_USER_AGENTS)
        session_id = f"sess_attack_lowslow_{int(sim_time)}"

        # Secuencia desordenada / exfiltración forzada de admin y pagos
        suspicious_sequence = [
            "/api/v1/admin/export",
            "/api/v1/payments/process",
            "/api/v1/admin/export",
            "/api/v1/orders/create",
            "/api/v1/admin/export",
            "/api/v1/admin/export",
            "/api/v1/catalog/products",
            "/api/v1/admin/export",
        ]

        for i in range(num_requests):
            endpoint = suspicious_sequence[i % len(suspicious_sequence)]
            # Δt calculado para evadir rate limiters estáticos (< 60 req/min)
            delta_t = round(self.rng.uniform(3.0, 5.0), 3)
            sim_time += delta_t
            method = self._get_http_method(endpoint)

            events.append(SimulatedEvent(
                timestamp=sim_time,
                token_jti=victim_token_jti,
                user_id=victim_user_id,
                endpoint=endpoint,
                method=method,
                ip_address=attacker_ip,
                user_agent=attacker_ua,
                delta_t=delta_t,
                is_attack=1,
                attack_type="LowAndSlow",
                session_id=session_id
            ))

        self.current_sim_time = sim_time + 5.0
        return events

    def generate_token_replay_attack(
        self,
        victim_token_jti: str,
        victim_user_id: str,
        num_requests: int = 12,
        start_time: Optional[float] = None
    ) -> List[SimulatedEvent]:
        """
        Escenario B: Token Replay / Secuestro de Sesión
        Mismo token reutilizado súbitamente desde una IP y User-Agent completamente disjuntos.
        """
        events: List[SimulatedEvent] = []
        sim_time = start_time if start_time is not None else self.current_sim_time
        attacker_ip = self.rng.choice(ATTACKER_IPS)
        attacker_ua = self.rng.choice(ATTACKER_USER_AGENTS)
        session_id = f"sess_attack_replay_{int(sim_time)}"

        # Endpoint sequence jumping around
        replay_endpoints = [
            "/api/v1/payments/process",
            "/api/v1/user/profile",
            "/api/v1/orders/create",
            "/api/v1/payments/process",
            "/api/v1/admin/export",
            "/api/v1/user/profile",
        ]

        for i in range(num_requests):
            endpoint = replay_endpoints[i % len(replay_endpoints)]
            delta_t = round(self.rng.uniform(2.0, 6.0), 3)
            sim_time += delta_t
            method = self._get_http_method(endpoint)

            events.append(SimulatedEvent(
                timestamp=sim_time,
                token_jti=victim_token_jti,
                user_id=victim_user_id,
                endpoint=endpoint,
                method=method,
                ip_address=attacker_ip,
                user_agent=attacker_ua,
                delta_t=delta_t,
                is_attack=1,
                attack_type="TokenReplay",
                session_id=session_id
            ))

        self.current_sim_time = sim_time + 4.0
        return events

    def generate_burst_attack(
        self,
        victim_token_jti: str,
        victim_user_id: str,
        num_requests: int = 50,
        start_time: Optional[float] = None
    ) -> List[SimulatedEvent]:
        """
        Escenario C: Fuerza Bruta / Ráfaga
        Solicitud masiva con token válido a 50 peticiones/segundo (Δt ~ 0.02s).
        """
        events: List[SimulatedEvent] = []
        sim_time = start_time if start_time is not None else self.current_sim_time
        attacker_ip = self.rng.choice(ATTACKER_IPS)
        attacker_ua = self.rng.choice(ATTACKER_USER_AGENTS)
        session_id = f"sess_attack_burst_{int(sim_time)}"

        target_endpoint = "/api/v1/catalog/products"

        for _ in range(num_requests):
            # 50 req/s -> Δt ~ 0.02s
            delta_t = round(self.rng.uniform(0.015, 0.025), 4)
            sim_time += delta_t

            events.append(SimulatedEvent(
                timestamp=sim_time,
                token_jti=victim_token_jti,
                user_id=victim_user_id,
                endpoint=target_endpoint,
                method="GET",
                ip_address=attacker_ip,
                user_agent=attacker_ua,
                delta_t=delta_t,
                is_attack=1,
                attack_type="Burst",
                session_id=session_id
            ))

        self.current_sim_time = sim_time + 5.0
        return events

    def generate_dataset(
        self,
        target_train_events: int = 2000,
        target_calib_events: int = 500,
        target_test_events: int = 4000,
        attack_ratio_test: float = 0.10
    ) -> Tuple[List[SimulatedEvent], List[SimulatedEvent], List[SimulatedEvent]]:
        """
        Genera los conjuntos completos:
          - Conjunto de Entrenamiento: 2,000 eventos normales exclusivamente.
          - Conjunto de Calibración: 500 eventos normales para calibrar umbral tau (FPR < 1%).
          - Conjunto Mixto de Prueba: 4,000 eventos (90% normal, 10% ataques multivectoriales).
        """
        train_events: List[SimulatedEvent] = []
        calib_events: List[SimulatedEvent] = []
        test_events: List[SimulatedEvent] = []

        user_counter = 1

        # 1. Generación de Entrenamiento Normal
        while len(train_events) < target_train_events:
            sess = self.generate_legitimate_session(user_counter)
            user_counter += 1
            train_events.extend(sess)
        train_events = train_events[:target_train_events]

        # 2. Generación de Calibración Normal
        while len(calib_events) < target_calib_events:
            sess = self.generate_legitimate_session(user_counter)
            user_counter += 1
            calib_events.extend(sess)
        calib_events = calib_events[:target_calib_events]

        # 3. Generación del Flujo Mixto de Evaluación
        num_attack_events = int(target_test_events * attack_ratio_test)
        num_normal_events = target_test_events - num_attack_events

        # Generar bloques normales
        normal_test_pool: List[SimulatedEvent] = []
        active_tokens_for_hijack = []
        while len(normal_test_pool) < num_normal_events:
            sess = self.generate_legitimate_session(user_counter)
            user_counter += 1
            normal_test_pool.extend(sess)
            if sess:
                active_tokens_for_hijack.append((sess[0].token_jti, sess[0].user_id))
        normal_test_pool = normal_test_pool[:num_normal_events]

        # Generar bloques de ataques repartidos entre los 3 escenarios
        attack_pool: List[SimulatedEvent] = []
        
        # Reparto equitativo entre Escenario A, B y C
        target_low_slow = num_attack_events // 3
        target_replay = num_attack_events // 3
        target_burst = num_attack_events - target_low_slow - target_replay

        # Escenario A: Low & Slow
        while sum(1 for e in attack_pool if e.attack_type == "LowAndSlow") < target_low_slow:
            v_token = f"stolen_jti_lowslow_{self.rng.randint(100, 999)}"
            v_user = f"victim_user_{self.rng.randint(100, 999)}"
            atk_events = self.generate_low_and_slow_attack(v_token, v_user, num_requests=10)
            attack_pool.extend(atk_events)

        # Escenario B: Token Replay (usando tokens legítimos activos)
        while sum(1 for e in attack_pool if e.attack_type == "TokenReplay") < target_replay:
            if active_tokens_for_hijack:
                v_token, v_user = self.rng.choice(active_tokens_for_hijack)
            else:
                v_token, v_user = f"replay_jti_{self.rng.randint(100, 999)}", "user_hijacked"
            atk_events = self.generate_token_replay_attack(v_token, v_user, num_requests=8)
            attack_pool.extend(atk_events)

        # Escenario C: Burst (Fuerza Bruta a 50 req/s)
        # Ráfaga masiva con el mismo token para contrastar la activación del Rate Limiter (> 60 req/min)
        v_burst_token = f"burst_token_flood_{self.rng.randint(100, 999)}"
        v_burst_user = "user_victim_burst"
        while sum(1 for e in attack_pool if e.attack_type == "Burst") < target_burst:
            needed = target_burst - sum(1 for e in attack_pool if e.attack_type == "Burst")
            batch_size = min(needed, 75)
            atk_events = self.generate_burst_attack(v_burst_token, v_burst_user, num_requests=batch_size)
            attack_pool.extend(atk_events)

        # Limitar attack_pool al número exacto deseado
        attack_pool = attack_pool[:num_attack_events]

        # Combinar e intercalar temporalmente
        all_test_raw = normal_test_pool + attack_pool
        # Ordenar estrictamente por timestamp cronológico para simulación streaming real
        all_test_raw.sort(key=lambda ev: ev.timestamp)

        # Recalcular delta_t local entre eventos consecutivos en el stream global
        test_events = all_test_raw[:target_test_events]

        return train_events, calib_events, test_events
