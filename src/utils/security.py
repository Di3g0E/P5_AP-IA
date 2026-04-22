"""
Utilidades de seguridad para la gestión de embeddings en reposo.

Principios:
  1. Los embeddings NO se almacenan en texto plano; se cifran con Fernet
     (AES-128-CBC + HMAC-SHA256) antes de escribirse a disco.
  2. La clave de cifrado se deriva de una passphrase usando PBKDF2-HMAC-SHA256
     (310.000 iteraciones, recomendación NIST SP 800-132 para 2024).
  3. Se añade un hash de integridad (SHA-256) al registro para detectar
     manipulación de la base de datos de vectores.
  4. El contador de intentos fallidos se mantiene en memoria; en producción
     debe persistirse en una caché distribuida (Redis con TTL).
"""

from __future__ import annotations

import base64
import hashlib
import os
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from loguru import logger


# ── Derivación de clave ───────────────────────────────────────────────────────

def derive_key(passphrase: str, salt: Optional[bytes] = None) -> tuple[bytes, bytes]:
    """
    Deriva una clave Fernet de 32 bytes desde una passphrase usando PBKDF2.

    Returns:
        (fernet_key_b64, salt) — guardar el salt junto a los datos cifrados.
    """
    if salt is None:
        salt = os.urandom(16)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=310_000,   # NIST 2024 mínimo para PBKDF2-SHA256
    )
    key = base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))
    return key, salt


# ── Almacén seguro de embeddings ──────────────────────────────────────────────

@dataclass
class EncryptedEmbeddingStore:
    """
    Almacén cifrado de embeddings en disco.

    Formato del fichero:
        {
          "salt":    bytes,
          "records": { user_id: {"ciphertext": bytes, "hash": str} }
        }

    El 'hash' es SHA-256 del embedding en plano; permite detectar manipulación
    sin descifrar todos los registros (verificación rápida de integridad).
    """

    db_path: Path
    passphrase: str
    _fernet: Optional[Fernet] = field(default=None, repr=False, init=False)
    _records: Dict[str, dict] = field(default_factory=dict, repr=False, init=False)
    _salt: Optional[bytes] = field(default=None, repr=False, init=False)

    def __post_init__(self):
        self.db_path = Path(self.db_path)
        if self.db_path.exists():
            self._load()
        else:
            self._salt = os.urandom(16)
            key, _ = derive_key(self.passphrase, self._salt)
            self._fernet = Fernet(key)
            logger.info(f"Nueva base de datos de embeddings creada en {self.db_path}")

    def _load(self) -> None:
        with open(self.db_path, "rb") as f:
            data = pickle.load(f)
        self._salt = data["salt"]
        key, _ = derive_key(self.passphrase, self._salt)
        self._fernet = Fernet(key)
        self._records = data["records"]
        logger.info(
            f"Base de datos de embeddings cargada: {len(self._records)} registros."
        )

    def save(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump({"salt": self._salt, "records": self._records}, f)
        logger.debug(f"Base de datos guardada en {self.db_path}")

    def store(self, user_id: str, embedding: np.ndarray) -> None:
        """Cifra y almacena el embedding de un usuario."""
        raw = embedding.astype(np.float32).tobytes()
        ciphertext = self._fernet.encrypt(raw)
        integrity_hash = hashlib.sha256(raw).hexdigest()
        self._records[user_id] = {
            "ciphertext": ciphertext,
            "hash": integrity_hash,
        }
        logger.debug(f"Embedding almacenado para usuario '{user_id}'.")

    def retrieve(self, user_id: str) -> Optional[np.ndarray]:
        """Descifra y devuelve el embedding, verificando la integridad."""
        record = self._records.get(user_id)
        if record is None:
            return None
        try:
            raw = self._fernet.decrypt(record["ciphertext"])
        except InvalidToken:
            logger.error(
                f"Error de descifrado para '{user_id}'. "
                "¿Clave incorrecta o datos corruptos?"
            )
            return None

        # Verificación de integridad
        if hashlib.sha256(raw).hexdigest() != record["hash"]:
            logger.critical(
                f"¡Integridad comprometida para usuario '{user_id}'! "
                "El hash no coincide. Posible manipulación de la base de datos."
            )
            return None

        embedding = np.frombuffer(raw, dtype=np.float32).copy()
        return embedding

    def delete(self, user_id: str) -> bool:
        if user_id in self._records:
            del self._records[user_id]
            logger.info(f"Embedding eliminado para usuario '{user_id}'.")
            return True
        return False

    def list_users(self) -> list[str]:
        return list(self._records.keys())

    def __contains__(self, user_id: str) -> bool:
        return user_id in self._records


# ── Control de acceso por intentos fallidos ───────────────────────────────────

class AccessController:
    """
    Controla los intentos de autenticación fallidos y aplica bloqueo temporal.

    En producción sustituir el dict en memoria por Redis con TTL automático.
    """

    def __init__(self, max_attempts: int = 5, lockout_seconds: int = 300):
        self.max_attempts = max_attempts
        self.lockout_seconds = lockout_seconds
        # {user_id: {"count": int, "locked_until": float}}
        self._state: dict = defaultdict(lambda: {"count": 0, "locked_until": 0.0})

    def is_locked(self, user_id: str) -> bool:
        state = self._state[user_id]
        if state["locked_until"] > time.time():
            remaining = int(state["locked_until"] - time.time())
            logger.warning(
                f"Usuario '{user_id}' bloqueado. "
                f"Desbloqueo en {remaining} segundos."
            )
            return True
        return False

    def record_failure(self, user_id: str) -> None:
        state = self._state[user_id]
        state["count"] += 1
        logger.warning(
            f"Intento fallido para '{user_id}': "
            f"{state['count']}/{self.max_attempts}"
        )
        if state["count"] >= self.max_attempts:
            state["locked_until"] = time.time() + self.lockout_seconds
            state["count"] = 0
            logger.error(
                f"Usuario '{user_id}' bloqueado por {self.lockout_seconds} s "
                "por superar el límite de intentos."
            )

    def record_success(self, user_id: str) -> None:
        self._state[user_id] = {"count": 0, "locked_until": 0.0}

    def remaining_attempts(self, user_id: str) -> int:
        return max(0, self.max_attempts - self._state[user_id]["count"])
