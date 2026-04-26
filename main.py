"""
CLI principal del sistema de login biométrico facial.

Comandos disponibles:
  register    Registra un nuevo usuario con una o varias imágenes.
  login       Autentica un usuario contra la base de datos.
  list        Lista los usuarios registrados.
  remove      Elimina el registro de un usuario.
  benchmark   Ejecuta el benchmark comparativo ArcFace vs SFace.
  evaluate    Calcula EER, ROC-AUC y genera gráficos sobre un dataset de pares.

Ejemplo de uso:
  python main.py register alice data/raw/alice/
  python main.py login    alice data/raw/probe.jpg
  python main.py benchmark data/raw/test_faces/
  python main.py evaluate  data/processed/pairs.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import yaml
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ── Configuración del logger ──────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)
logger.add(
    "logs/face_login_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    level="DEBUG",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _get_passphrase() -> str:
    """Obtiene la passphrase de cifrado desde variable de entorno o input."""
    passphrase = os.environ.get("FACE_DB_PASSPHRASE", "")
    if not passphrase:
        import getpass
        passphrase = getpass.getpass("Passphrase de la base de datos: ")
    return passphrase


def _load_images_from_path(path: str) -> list:
    """Carga imagen(s) desde ruta — soporta archivo único o directorio."""
    p = Path(path)
    ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if p.is_dir():
        files = [f for f in sorted(p.iterdir()) if f.suffix.lower() in ext]
        images = [cv2.imread(str(f)) for f in files]
        images = [img for img in images if img is not None]
        logger.info(f"Cargadas {len(images)} imágenes desde {path}")
        return images
    img = cv2.imread(str(p))
    if img is None:
        logger.error(f"No se pudo leer la imagen: {path}")
        sys.exit(1)
    return [img]


# ── Subcomandos ───────────────────────────────────────────────────────────────

def cmd_register(args: argparse.Namespace) -> None:
    from src.models.face_login_system import FaceLoginSystem

    cfg = _load_config(args.config)
    system = FaceLoginSystem.from_config(
        config_path=args.config,
        passphrase=_get_passphrase(),
        db_path=cfg["vector_store"]["embeddings_db"],
    )
    images = _load_images_from_path(args.image_path)

    if len(images) == 1:
        ok = system.register(args.user_id, images[0])
    else:
        ok = system.register_from_multiple(args.user_id, images)

    if ok:
        print(f"[OK] Usuario '{args.user_id}' registrado correctamente.")
    else:
        print(f"[ERROR] No se pudo registrar '{args.user_id}'.")
        sys.exit(1)


def cmd_login(args: argparse.Namespace) -> None:
    from src.models.face_login_system import FaceLoginSystem, AuthStatus

    cfg = _load_config(args.config)
    system = FaceLoginSystem.from_config(
        config_path=args.config,
        passphrase=_get_passphrase(),
        db_path=cfg["vector_store"]["embeddings_db"],
    )
    images = _load_images_from_path(args.image_path)
    result = system.authenticate(args.user_id, images[0])

    print(f"\n{'='*50}")
    print(f"  Usuario      : {result.user_id}")
    print(f"  Estado       : {result.status.name}")
    print(f"  Mensaje      : {result.message}")
    if result.liveness_score is not None:
        print(f"  Liveness     : {result.liveness_score:.4f}")
    if result.similarity_score is not None:
        print(f"  Similitud    : {result.similarity_score:.4f}")
    print(f"{'='*50}\n")

    sys.exit(0 if result.granted else 1)


def cmd_list(args: argparse.Namespace) -> None:
    from src.models.face_login_system import FaceLoginSystem

    cfg = _load_config(args.config)
    system = FaceLoginSystem.from_config(
        config_path=args.config,
        passphrase=_get_passphrase(),
        db_path=cfg["vector_store"]["embeddings_db"],
    )
    users = system.list_users()
    if users:
        print(f"Usuarios registrados ({len(users)}):")
        for u in users:
            print(f"  - {u}")
    else:
        print("No hay usuarios registrados.")


def cmd_remove(args: argparse.Namespace) -> None:
    from src.models.face_login_system import FaceLoginSystem

    cfg = _load_config(args.config)
    system = FaceLoginSystem.from_config(
        config_path=args.config,
        passphrase=_get_passphrase(),
        db_path=cfg["vector_store"]["embeddings_db"],
    )
    ok = system.remove_user(args.user_id)
    if ok:
        print(f"[OK] Usuario '{args.user_id}' eliminado.")
    else:
        print(f"[WARN] Usuario '{args.user_id}' no encontrado.")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Benchmark de latencia y separabilidad ArcFace vs SFace."""
    from src.models.face_login_system import FaceLoginSystem

    system = FaceLoginSystem.from_config(
        config_path=args.config,
        passphrase="benchmark",
        db_path="models/benchmark_tmp.pkl.enc",
    )
    images = _load_images_from_path(args.faces_dir)
    if not images:
        print("[ERROR] No se encontraron imágenes.")
        sys.exit(1)

    results = system.embedder.benchmark(images, n_runs=min(50, len(images)))

    print("\n" + "="*60)
    print("  BENCHMARK: ArcFace vs SFace")
    print("="*60)
    for backend, metrics in results.items():
        print(f"\n  [{backend.upper()}]")
        for k, v in metrics.items():
            print(f"    {k:<25}: {v:.4f}")
    print("="*60 + "\n")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluación completa con EER, ROC-AUC y gráficos sobre pares."""
    from src.data.face_dataset import VerificationPairDataset
    from src.models.face_login_system import FaceLoginSystem
    from src.evaluation.metrics import BiometricBenchmark

    cfg = _load_config(args.config)
    system = FaceLoginSystem.from_config(
        config_path=args.config,
        passphrase="eval",
        db_path="models/eval_tmp.pkl.enc",
    )
    pair_dataset = VerificationPairDataset(args.pairs_csv)

    bench = BiometricBenchmark(pair_dataset, system)
    output_dir = cfg.get("evaluation", {}).get("output_dir", "doc/evaluation")
    report = bench.run(
        thresholds={
            "arcface":  cfg["verification"]["threshold"],
            "facenet":  cfg["verification"]["threshold_facenet"],
        },
        output_dir=output_dir,
    )
    bench.save_report(report, output_dir)


def cmd_tune(args: argparse.Namespace) -> None:
    """Optimiza los thresholds de verificación y liveness sobre los datos propios."""
    from src.data.face_dataset import VerificationPairDataset
    from src.models.face_login_system import FaceLoginSystem
    from src.evaluation.tune_thresholds import (
        tune_verification_threshold,
        tune_liveness_threshold,
        update_config,
    )

    cfg = _load_config(args.config)
    system = FaceLoginSystem.from_config(
        config_path=args.config,
        passphrase="tune",
        db_path="models/tune_tmp.pkl.enc",
    )
    output_dir = cfg.get("evaluation", {}).get("output_dir", "doc/evaluation")

    # ── Verificación ─────────────────────────────────────────────────────
    pair_dataset = VerificationPairDataset(args.pairs_csv)
    ver_result = tune_verification_threshold(
        system, pair_dataset,
        search_steps=cfg["evaluation"].get("eer_threshold_search_steps", 1000),
        output_dir=output_dir,
    )

    print("\n" + "="*60)
    print("  TUNING — Verificación")
    print("="*60)
    print(f"  Pares           : {ver_result.n_genuine} gen / {ver_result.n_impostor} imp")
    print(f"  ROC-AUC         : {ver_result.roc_auc:.4f}")
    print(f"  EER             : {ver_result.eer:.4f} @ thr={ver_result.eer_threshold:.4f}")
    print(f"  Best F1         : {ver_result.best_f1:.4f} @ thr={ver_result.best_f1_threshold:.4f}")
    print(f"  Best Accuracy   : {ver_result.best_acc:.4f} @ thr={ver_result.best_acc_threshold:.4f}")

    # ── Liveness (si se pide) ────────────────────────────────────────────
    liveness_thr = None
    if args.real_images:
        live_result = tune_liveness_threshold(
            system, args.real_images,
            target_frr=args.target_frr,
            output_dir=output_dir,
        )
        liveness_thr = live_result["recommended_threshold"]
        print("\n  TUNING — Liveness")
        print("-"*60)
        for k, v in live_result.items():
            print(f"    {k:<22}: {v:.4f}")

    print("="*60 + "\n")

    # ── Escribir en config ───────────────────────────────────────────────
    if args.apply:
        chosen = {
            "eer":  ver_result.eer_threshold,
            "f1":   ver_result.best_f1_threshold,
            "acc":  ver_result.best_acc_threshold,
        }[args.criterion]
        update_config(
            args.config,
            verification_threshold=chosen,
            liveness_threshold=liveness_thr,
            criterion=args.criterion,
        )
        print(f"[OK] config.yaml actualizado (criterio: {args.criterion}).")
    else:
        print("[INFO] --apply no indicado; config.yaml no ha sido modificado.")

def cmd_finance_add(args: argparse.Namespace) -> None:
    """Añade un registro financiero con validación interactiva de anomalías."""
    from src.models.anomaly_detector import FinancialAnomalyDetector
    from src.data.financial_data import append_transaction, parse_amount, format_amount
    import datetime

    csv_path = args.csv_path
    if not os.path.exists(csv_path):
        print(f"[ERROR] No se encuentra el archivo CSV: {csv_path}")
        sys.exit(1)

    print("\n" + "="*50)
    print("  NUEVO REGISTRO FINANCIERO")
    print("="*50)
    
    desc = input("Descripción: ").strip()
    
    hoy_str = datetime.datetime.now().strftime("%d/%m/%Y")
    date_str = input(f"Fecha (DD/MM/YYYY) [{hoy_str}]: ").strip()
    if not date_str:
        date_str = hoy_str

    amount_str = input("Cantidad (ej. 10,00€ o 10.5): ").strip()
    amount_num = parse_amount(amount_str)
    
    area = input("Área (ej. Food, Leisure, Salary...): ").strip()
    type_val = input("Tipo (Expenses/Income): ").strip()

    print("\n[INFO] Evaluando transacción con el modelo...")
    detector = FinancialAnomalyDetector(csv_path)
    is_anomalous, reasons = detector.predict(date_str, amount_num, area, type_val)

    if is_anomalous:
        print("\n[WARNING] ¡Posible anomalía detectada!")
        for r in reasons:
            print(f"  - {r}")
            
        confirm = input("\n¿Estás seguro de que quieres guardar este registro? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("Operación cancelada. El registro NO se ha guardado.\n")
            sys.exit(0)
    else:
        print("[OK] Transacción dentro de los parámetros normales.")

    # Asegurar el formato final de la cantidad
    final_amount_str = format_amount(amount_num)

    append_transaction(csv_path, desc, date_str, final_amount_str, area, type_val)
    print("[OK] Registro guardado correctamente en el histórico.\n")



# ── Punto de entrada ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sistema de Login Biométrico Facial",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", default="config/config.yaml",
        help="Ruta al archivo de configuración YAML.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # register
    p_reg = sub.add_parser("register", help="Registrar un nuevo usuario.")
    p_reg.add_argument("user_id", help="Identificador del usuario.")
    p_reg.add_argument("image_path", help="Imagen o directorio de imágenes.")
    p_reg.set_defaults(func=cmd_register)

    # login
    p_log = sub.add_parser("login", help="Autenticar un usuario.")
    p_log.add_argument("user_id", help="Identificador del usuario.")
    p_log.add_argument("image_path", help="Imagen de prueba.")
    p_log.set_defaults(func=cmd_login)

    # list
    p_lst = sub.add_parser("list", help="Listar usuarios registrados.")
    p_lst.set_defaults(func=cmd_list)

    # remove
    p_rm = sub.add_parser("remove", help="Eliminar un usuario.")
    p_rm.add_argument("user_id", help="Identificador del usuario.")
    p_rm.set_defaults(func=cmd_remove)

    # benchmark
    p_bm = sub.add_parser("benchmark", help="Benchmark ArcFace vs SFace.")
    p_bm.add_argument("faces_dir", help="Directorio con imágenes de prueba.")
    p_bm.set_defaults(func=cmd_benchmark)

    # evaluate
    p_ev = sub.add_parser("evaluate", help="Evaluación con dataset de pares.")
    p_ev.add_argument("pairs_csv", help="CSV de pares (img1,img2,label).")
    p_ev.set_defaults(func=cmd_evaluate)

    # tune
    p_tn = sub.add_parser(
        "tune",
        help="Optimiza thresholds de verificación (y liveness) sobre datos propios.",
    )
    p_tn.add_argument("pairs_csv", help="CSV de pares etiquetados.")
    p_tn.add_argument(
        "--real-images", default=None,
        help="Directorio con rostros reales para calibrar el threshold de liveness.",
    )
    p_tn.add_argument(
        "--target-frr", type=float, default=0.05,
        help="FRR operacional tolerado (percentil para threshold liveness).",
    )
    p_tn.add_argument(
        "--criterion", choices=["eer", "f1", "acc"], default="eer",
        help="Criterio de selección del threshold óptimo.",
    )
    p_tn.add_argument(
        "--apply", action="store_true",
        help="Si se indica, escribe los thresholds óptimos en config.yaml.",
    )
    p_tn.set_defaults(func=cmd_tune)

    # finance-add
    p_fin = sub.add_parser(
        "finance-add",
        help="Añade un nuevo registro financiero verificando si es anómalo.",
    )
    p_fin.add_argument(
        "--csv-path", default="data/raw/db_mod_descript.csv",
        help="Ruta al histórico CSV.",
    )
    p_fin.set_defaults(func=cmd_finance_add)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
