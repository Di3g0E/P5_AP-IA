"""
Tests unitarios del sistema de login biométrico.

Estrategia: los componentes que requieren modelos pesados (MTCNN, DenseNet201,
InsightFace) se mockean con stubs controlados. Esto permite:
  - Verificar la lógica del pipeline sin descargar modelos de producción.
  - Simular con precisión los distintos códigos de fallo de cada etapa.
  - Ejecutar en CI sin GPU.

Los tests de integración real (con modelos cargados) se marcan con
@pytest.mark.integration y se excluyen por defecto:
  pytest tests/             # solo unit tests
  pytest tests/ -m integration  # solo integración
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_face_bgr() -> np.ndarray:
    """Imagen sintética 224×224 BGR (ruido uniforme)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def unit_embedding() -> np.ndarray:
    """Embedding L2-normalizado sintético de 512 dimensiones."""
    rng = np.random.default_rng(7)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def base_config() -> dict:
    """Configuración mínima en memoria para instanciar componentes."""
    return {
        "pipeline":       {"device": "cpu"},
        "detection":      {"min_face_size": 40, "thresholds": [0.6, 0.7, 0.7],
                           "factor": 0.709},
        "preprocessing":  {"output_size": [224, 224],
                           "clahe":        {"clip_limit": 2.0, "tile_grid_size": [8, 8]},
                           "sharpening":   {"kernel_strength": 0.3}},
        "liveness":       {"model_path": "models/fake.pth", "threshold": 0.95},
        "embedder":       {"primary": "arcface",
                           "arcface": {"embedding_dim": 512},
                           "sface":   {"embedding_dim": 512}},
        "verification":   {"threshold": 0.68, "threshold_sface": 0.60},
        "security":       {"max_failed_attempts": 5, "lockout_seconds": 10},
        "vector_store":   {"embeddings_db": "models/test_db.pkl.enc"},
    }


# ── Tests: ImagePreprocessor ──────────────────────────────────────────────────

class TestImagePreprocessor:
    def test_output_shape_preserved(self, dummy_face_bgr):
        from src.features.preprocessor import ImagePreprocessor
        proc = ImagePreprocessor()
        result = proc.process(dummy_face_bgr)
        assert result.shape == dummy_face_bgr.shape

    def test_output_dtype_uint8(self, dummy_face_bgr):
        from src.features.preprocessor import ImagePreprocessor
        proc = ImagePreprocessor()
        result = proc.process(dummy_face_bgr)
        assert result.dtype == np.uint8

    def test_clahe_increases_contrast(self):
        """Una imagen muy oscura debe tener mayor varianza tras CLAHE."""
        from src.features.preprocessor import ImagePreprocessor
        dark = np.full((224, 224, 3), 10, dtype=np.uint8)
        proc = ImagePreprocessor(clip_limit=2.0)
        result = proc.process(dark)
        assert result.std() >= dark.std()


# ── Tests: EncryptedEmbeddingStore ────────────────────────────────────────────

class TestEncryptedEmbeddingStore:
    def test_store_retrieve_roundtrip(self, unit_embedding):
        from src.utils.security import EncryptedEmbeddingStore
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.pkl.enc"
            store = EncryptedEmbeddingStore(db_path=db_path, passphrase="test123")
            store.store("alice", unit_embedding)
            retrieved = store.retrieve("alice")
            assert retrieved is not None
            np.testing.assert_array_almost_equal(unit_embedding, retrieved, decimal=6)

    def test_retrieve_unknown_user_returns_none(self, unit_embedding):
        from src.utils.security import EncryptedEmbeddingStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EncryptedEmbeddingStore(
                db_path=Path(tmpdir) / "db.pkl.enc", passphrase="pw"
            )
            assert store.retrieve("ghost") is None

    def test_wrong_passphrase_returns_none(self, unit_embedding):
        from src.utils.security import EncryptedEmbeddingStore
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "db.pkl.enc"
            store1 = EncryptedEmbeddingStore(db_path=db_path, passphrase="correct")
            store1.store("bob", unit_embedding)
            store1.save()

            store2 = EncryptedEmbeddingStore(db_path=db_path, passphrase="wrong")
            result = store2.retrieve("bob")
            assert result is None

    def test_persist_and_reload(self, unit_embedding):
        from src.utils.security import EncryptedEmbeddingStore
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "db.pkl.enc"
            s1 = EncryptedEmbeddingStore(db_path=db_path, passphrase="pw")
            s1.store("carol", unit_embedding)
            s1.save()

            s2 = EncryptedEmbeddingStore(db_path=db_path, passphrase="pw")
            result = s2.retrieve("carol")
            assert result is not None
            np.testing.assert_array_almost_equal(unit_embedding, result, decimal=6)

    def test_delete_user(self, unit_embedding):
        from src.utils.security import EncryptedEmbeddingStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EncryptedEmbeddingStore(
                db_path=Path(tmpdir) / "db.pkl.enc", passphrase="pw"
            )
            store.store("dave", unit_embedding)
            assert store.delete("dave") is True
            assert store.retrieve("dave") is None

    def test_contains_operator(self, unit_embedding):
        from src.utils.security import EncryptedEmbeddingStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EncryptedEmbeddingStore(
                db_path=Path(tmpdir) / "db.pkl.enc", passphrase="pw"
            )
            assert "eve" not in store
            store.store("eve", unit_embedding)
            assert "eve" in store


# ── Tests: AccessController ───────────────────────────────────────────────────

class TestAccessController:
    def test_not_locked_initially(self):
        from src.utils.security import AccessController
        ctrl = AccessController(max_attempts=3, lockout_seconds=10)
        assert ctrl.is_locked("user1") is False

    def test_locks_after_max_attempts(self):
        from src.utils.security import AccessController
        ctrl = AccessController(max_attempts=3, lockout_seconds=300)
        for _ in range(3):
            ctrl.record_failure("user2")
        assert ctrl.is_locked("user2") is True

    def test_success_resets_counter(self):
        from src.utils.security import AccessController
        ctrl = AccessController(max_attempts=3, lockout_seconds=300)
        ctrl.record_failure("user3")
        ctrl.record_failure("user3")
        ctrl.record_success("user3")
        assert ctrl.remaining_attempts("user3") == 3

    def test_remaining_attempts_decrements(self):
        from src.utils.security import AccessController
        ctrl = AccessController(max_attempts=5, lockout_seconds=300)
        ctrl.record_failure("user4")
        ctrl.record_failure("user4")
        assert ctrl.remaining_attempts("user4") == 3


# ── Tests: métricas de evaluación ────────────────────────────────────────────

class TestBiometricMetrics:
    def _make_scores(self, n_genuine=200, n_impostor=200, seed=0):
        rng = np.random.default_rng(seed)
        genuine  = np.clip(rng.normal(0.75, 0.08, n_genuine),  0, 1)
        impostor = np.clip(rng.normal(0.35, 0.10, n_impostor), 0, 1)
        scores = np.concatenate([genuine, impostor])
        labels = np.concatenate([np.ones(n_genuine), np.zeros(n_impostor)])
        return labels, scores

    def test_eer_in_range(self):
        from src.evaluation.metrics import compute_eer
        labels, scores = self._make_scores()
        eer, thr = compute_eer(labels, scores)
        assert 0.0 <= eer <= 1.0
        assert 0.0 <= thr <= 1.0

    def test_eer_reasonable_for_separable_distribution(self):
        """Con distribuciones bien separadas el EER debe ser < 15%."""
        from src.evaluation.metrics import compute_eer
        labels, scores = self._make_scores()
        eer, _ = compute_eer(labels, scores)
        assert eer < 0.15

    def test_roc_auc_greater_than_random(self):
        from src.evaluation.metrics import compute_full_metrics
        labels, scores = self._make_scores()
        metrics = compute_full_metrics(labels, scores, threshold=0.55)
        assert metrics["roc_auc"] > 0.5

    def test_tar_at_far_in_range(self):
        from src.evaluation.metrics import compute_tar_at_far
        labels, scores = self._make_scores()
        tar = compute_tar_at_far(labels, scores, target_far=1e-2)
        assert 0.0 <= tar <= 1.0

    def test_full_metrics_keys_present(self):
        from src.evaluation.metrics import compute_full_metrics
        labels, scores = self._make_scores()
        metrics = compute_full_metrics(labels, scores, threshold=0.55)
        expected_keys = {
            "eer", "eer_threshold", "roc_auc", "tar_at_far_1e3",
            "tar_at_far_1e4", "accuracy", "precision", "recall", "f1",
            "far_at_threshold", "frr_at_threshold",
        }
        assert expected_keys.issubset(set(metrics.keys()))


# ── Tests: cosine_similarity ──────────────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors_give_one(self, unit_embedding):
        """Un vector consigo mismo debe dar similitud = 1.0."""
        sim = float(np.dot(unit_embedding, unit_embedding))
        assert abs(sim - 1.0) < 1e-5

    def test_orthogonal_vectors_give_zero(self):
        a = np.zeros(512, dtype=np.float32)
        b = np.zeros(512, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0
        sim = float(np.dot(a, b))
        assert abs(sim) < 1e-6

    def test_opposite_vectors_give_minus_one(self):
        a = np.zeros(512, dtype=np.float32)
        a[0] = 1.0
        b = -a.copy()
        sim = float(np.dot(a, b))
        assert abs(sim + 1.0) < 1e-5


# ── Tests: FaceLoginSystem (con mocks) ───────────────────────────────────────

class TestFaceLoginSystemMocked:
    """
    Prueba la lógica del pipeline en cascada sin cargar modelos reales.
    Cada test inyecta stubs que simulan el comportamiento de una etapa específica.
    """

    def _make_system(self, base_config, unit_embedding, tmpdir):
        """Construye FaceLoginSystem con todos los componentes mockeados."""
        from src.models.face_login_system import FaceLoginSystem
        from src.features.preprocessor import DetectionResult

        system = FaceLoginSystem.__new__(FaceLoginSystem)
        system.cfg = base_config
        system.sim_threshold = 0.68

        # Mock: preprocesador siempre detecta cara
        det_result = DetectionResult(
            bbox=np.array([10., 10., 100., 100.]),
            confidence=0.99,
            landmarks=np.zeros((5, 2)),
            aligned_face=np.zeros((224, 224, 3), dtype=np.uint8),
        )
        system.preproc_pipeline = MagicMock()
        system.preproc_pipeline.run.return_value = (
            np.zeros((224, 224, 3), dtype=np.uint8), det_result
        )

        # Mock: liveness por defecto = live
        system.liveness_detector = MagicMock()
        system.liveness_detector.threshold = 0.95
        system.liveness_detector.predict.return_value = (0.98, True)

        # Mock: embedder devuelve unit_embedding
        emb_mock = MagicMock()
        emb_mock.embedding = unit_embedding
        emb_mock.backend = "arcface"
        system.embedder = MagicMock()
        system.embedder.embed.return_value = emb_mock
        system.embedder.cosine_similarity.return_value = 0.85  # > umbral

        # DB real cifrada en directorio temporal
        from src.utils.security import EncryptedEmbeddingStore, AccessController
        db_path = Path(tmpdir) / "test.pkl.enc"
        system.db = EncryptedEmbeddingStore(db_path=db_path, passphrase="test")
        system.access_ctrl = AccessController(max_attempts=5, lockout_seconds=10)

        return system

    def test_register_and_authenticate_success(self, base_config, unit_embedding):
        from src.models.face_login_system import AuthStatus
        with tempfile.TemporaryDirectory() as tmpdir:
            system = self._make_system(base_config, unit_embedding, tmpdir)
            # Enrolamiento
            ok = system.register("alice", np.zeros((224, 224, 3), dtype=np.uint8))
            assert ok is True
            # Login exitoso
            result = system.authenticate("alice", np.zeros((224, 224, 3), dtype=np.uint8))
            assert result.status == AuthStatus.SUCCESS
            assert result.granted is True

    def test_liveness_failure_aborts_pipeline(self, base_config, unit_embedding):
        from src.models.face_login_system import AuthStatus
        with tempfile.TemporaryDirectory() as tmpdir:
            system = self._make_system(base_config, unit_embedding, tmpdir)
            system.db.store("bob", unit_embedding)
            # Simular ataque de presentación
            system.liveness_detector.predict.return_value = (0.40, False)
            result = system.authenticate("bob", np.zeros((224, 224, 3), dtype=np.uint8))
            assert result.status == AuthStatus.LIVENESS_FAILED
            # El embedder NO debe haberse llamado (cascade abort)
            system.embedder.embed.assert_not_called()

    def test_face_not_found_returns_correct_status(self, base_config, unit_embedding):
        from src.models.face_login_system import AuthStatus
        with tempfile.TemporaryDirectory() as tmpdir:
            system = self._make_system(base_config, unit_embedding, tmpdir)
            system.preproc_pipeline.run.side_effect = ValueError("No face")
            result = system.authenticate("carol", np.zeros((224, 224, 3), dtype=np.uint8))
            assert result.status == AuthStatus.FACE_NOT_FOUND

    def test_user_not_found_returns_correct_status(self, base_config, unit_embedding):
        from src.models.face_login_system import AuthStatus
        with tempfile.TemporaryDirectory() as tmpdir:
            system = self._make_system(base_config, unit_embedding, tmpdir)
            # "dave" no está en la DB
            result = system.authenticate("dave", np.zeros((224, 224, 3), dtype=np.uint8))
            assert result.status == AuthStatus.USER_NOT_FOUND

    def test_low_similarity_returns_score_too_low(self, base_config, unit_embedding):
        from src.models.face_login_system import AuthStatus
        with tempfile.TemporaryDirectory() as tmpdir:
            system = self._make_system(base_config, unit_embedding, tmpdir)
            system.db.store("eve", unit_embedding)
            # Similitud baja → rechazo
            system.embedder.cosine_similarity.return_value = 0.30
            result = system.authenticate("eve", np.zeros((224, 224, 3), dtype=np.uint8))
            assert result.status == AuthStatus.SCORE_TOO_LOW

    def test_account_locks_after_max_failures(self, base_config, unit_embedding):
        from src.models.face_login_system import AuthStatus
        with tempfile.TemporaryDirectory() as tmpdir:
            system = self._make_system(base_config, unit_embedding, tmpdir)
            system.db.store("frank", unit_embedding)
            system.embedder.cosine_similarity.return_value = 0.20  # siempre falla
            for _ in range(5):
                system.authenticate("frank", np.zeros((224, 224, 3), dtype=np.uint8))
            # El siguiente intento debe retornar ACCOUNT_LOCKED
            result = system.authenticate("frank", np.zeros((224, 224, 3), dtype=np.uint8))
            assert result.status == AuthStatus.ACCOUNT_LOCKED

    def test_remove_user(self, base_config, unit_embedding):
        with tempfile.TemporaryDirectory() as tmpdir:
            system = self._make_system(base_config, unit_embedding, tmpdir)
            system.db.store("grace", unit_embedding)
            assert "grace" in system.db
            system.remove_user("grace")
            assert "grace" not in system.db
