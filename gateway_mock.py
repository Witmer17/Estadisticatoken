"""
Módulo de Emisión y Logging (API Gateway Mock & FastAPI Service)
================================================================
Implementa la emisión criptográfica de tokens JWT (HS256), los endpoints
del sistema empresarial desacoplado y el interceptor de telemetría/logging
que registra cada transacción en tiempo real.
"""

import time
import uuid
import jwt
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Configuración Criptográfica
JWT_SECRET = "super_secret_enterprise_hs256_key_998877"
JWT_ALGORITHM = "HS256"

# Endpoints autorizados del sistema empresarial
ENDPOINTS = [
    "/api/v1/auth/login",
    "/api/v1/user/profile",
    "/api/v1/catalog/products",
    "/api/v1/orders/create",
    "/api/v1/payments/process",
    "/api/v1/admin/export",
]

# Inicialización de la aplicación FastAPI
app = FastAPI(
    title="Enterprise Zero-Trust API Gateway Mock",
    description="Gateway de autenticación e interceptor de telemetría para análisis estocástico de JWT",
    version="1.0.0"
)

security = HTTPBearer(auto_error=False)


class AuditLogger:
    """Interceptor en memoria que almacena logs estructurados de cada petición."""
    def __init__(self):
        self.logs: List[Dict[str, Any]] = []

    def log_event(
        self,
        timestamp: float,
        token_jti: Optional[str],
        user_id: Optional[str],
        endpoint: str,
        method: str,
        ip_address: str,
        user_agent: str,
        status_code: int = 200,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        event = {
            "timestamp": timestamp,
            "token_jti": token_jti or "anonymous",
            "user_id": user_id or "anonymous",
            "endpoint": endpoint,
            "method": method,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "status_code": status_code,
            "metadata": metadata or {}
        }
        self.logs.append(event)
        return event

    def clear(self):
        self.logs.clear()


audit_logger = AuditLogger()


def generate_jwt_token(
    user_id: str,
    role: str = "customer",
    sub: Optional[str] = None,
    ttl_seconds: int = 3600,
    custom_jti: Optional[str] = None
) -> Dict[str, Any]:
    """
    Genera un token JWT conforme a RFC 7519 firmado con HS256.
    Claims: sub, user_id, role, exp, iat, jti.
    """
    now = time.time()
    jti = custom_jti or str(uuid.uuid4())
    payload = {
        "sub": sub or f"user_{user_id}",
        "user_id": user_id,
        "role": role,
        "iat": int(now),
        "exp": int(now + ttl_seconds),
        "jti": jti
    }
    encoded_token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return {
        "access_token": encoded_token,
        "token_type": "bearer",
        "jti": jti,
        "payload": payload
    }


def verify_jwt_token(token: str) -> Optional[Dict[str, Any]]:
    """Valida la firma criptográfica y expiración del JWT."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except (jwt.PyJWTError, Exception):
        return None


# =====================================================================
# Definición de Endpoints REST en FastAPI
# =====================================================================

@app.post("/api/v1/auth/login")
async def login(credentials: Dict[str, Any], request: Request):
    user_id = credentials.get("user_id", "user_101")
    role = credentials.get("role", "customer")
    token_info = generate_jwt_token(user_id=user_id, role=role)
    
    audit_logger.log_event(
        timestamp=time.time(),
        token_jti=token_info["jti"],
        user_id=user_id,
        endpoint="/api/v1/auth/login",
        method="POST",
        ip_address=request.client.host if request.client else "127.0.0.1",
        user_agent=request.headers.get("user-agent", "Unknown-Client")
    )
    return token_info


@app.get("/api/v1/user/profile")
async def get_profile(request: Request, auth: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if not auth:
        raise HTTPException(status_code=401, detail="Token no proporcionado")
    payload = verify_jwt_token(auth.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Token inválido o expirado")
    
    audit_logger.log_event(
        timestamp=time.time(),
        token_jti=payload.get("jti"),
        user_id=payload.get("user_id"),
        endpoint="/api/v1/user/profile",
        method="GET",
        ip_address=request.client.host if request.client else "127.0.0.1",
        user_agent=request.headers.get("user-agent", "Unknown-Client")
    )
    return {"status": "ok", "user": payload}


@app.get("/api/v1/catalog/products")
async def get_catalog(request: Request, auth: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    payload = verify_jwt_token(auth.credentials) if auth else None
    audit_logger.log_event(
        timestamp=time.time(),
        token_jti=payload.get("jti") if payload else "anonymous",
        user_id=payload.get("user_id") if payload else "anonymous",
        endpoint="/api/v1/catalog/products",
        method="GET",
        ip_address=request.client.host if request.client else "127.0.0.1",
        user_agent=request.headers.get("user-agent", "Unknown-Client")
    )
    return {"catalog": ["Laptop ThinkPad", "Teclado Mecánico", "Monitor 4K OLED"]}


@app.post("/api/v1/orders/create")
async def create_order(request: Request, auth: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if not auth:
        raise HTTPException(status_code=401, detail="Token requerido")
    payload = verify_jwt_token(auth.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Token inválido")
    
    audit_logger.log_event(
        timestamp=time.time(),
        token_jti=payload.get("jti"),
        user_id=payload.get("user_id"),
        endpoint="/api/v1/orders/create",
        method="POST",
        ip_address=request.client.host if request.client else "127.0.0.1",
        user_agent=request.headers.get("user-agent", "Unknown-Client")
    )
    return {"status": "order_created", "order_id": str(uuid.uuid4())}


@app.post("/api/v1/payments/process")
async def process_payment(request: Request, auth: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if not auth:
        raise HTTPException(status_code=401, detail="Token requerido")
    payload = verify_jwt_token(auth.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Token inválido")
    
    audit_logger.log_event(
        timestamp=time.time(),
        token_jti=payload.get("jti"),
        user_id=payload.get("user_id"),
        endpoint="/api/v1/payments/process",
        method="POST",
        ip_address=request.client.host if request.client else "127.0.0.1",
        user_agent=request.headers.get("user-agent", "Unknown-Client")
    )
    return {"status": "payment_approved", "tx_id": str(uuid.uuid4())}


@app.get("/api/v1/admin/export")
async def admin_export(request: Request, auth: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if not auth:
        raise HTTPException(status_code=401, detail="Token requerido")
    payload = verify_jwt_token(auth.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Token inválido")
    
    audit_logger.log_event(
        timestamp=time.time(),
        token_jti=payload.get("jti"),
        user_id=payload.get("user_id"),
        endpoint="/api/v1/admin/export",
        method="GET",
        ip_address=request.client.host if request.client else "127.0.0.1",
        user_agent=request.headers.get("user-agent", "Unknown-Client")
    )
    return {"status": "export_ready", "download_url": "/downloads/db_backup.sql.gz"}


# =====================================================================
# Gateway Mock Programático de Alta Fidelidad (Simulador Desacoplado)
# =====================================================================

class APIGatewaySimulator:
    """
    Simulador desacoplado de alto rendimiento del API Gateway.
    Permite procesar miles de transacciones con verificación criptográfica
    exacta de JWT e intercepción de telemetría sin latencias de red.
    """
    def __init__(self):
        self.logger = AuditLogger()

    def dispatch(
        self,
        endpoint: str,
        method: str,
        token: Optional[str],
        ip_address: str,
        user_agent: str,
        timestamp: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Procesa y valida la petición en el Gateway mock."""
        token_jti = None
        user_id = None
        status_code = 200

        # Validación criptográfica de JWT si se incluye
        if token:
            payload = verify_jwt_token(token)
            if payload:
                token_jti = payload.get("jti")
                user_id = payload.get("user_id")
            else:
                status_code = 401
        elif endpoint != "/api/v1/auth/login":
            status_code = 401

        # Registrar log en el interceptor
        event = self.logger.log_event(
            timestamp=timestamp,
            token_jti=token_jti,
            user_id=user_id,
            endpoint=endpoint,
            method=method,
            ip_address=ip_address,
            user_agent=user_agent,
            status_code=status_code,
            metadata=metadata
        )
        return event

