"""Rutas reproducibles y compatibilidad con la raíz v1."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datos"
MODELS = ROOT / "modelos"
RESULTS = ROOT / "resultados"
FIGURES = ROOT / "graficas"
REPORT = ROOT / "reporte"
DELIVERY = ROOT / "entrega"
DATASET_CANDIDATES = (
    "clientes_comercio_1000000.csv",
    "clientes_comercio_1000000_alumnos (1).csv",
    "clientes_comercio_1000000_alumnos.csv",
)

def ensure_layout():
    for path in (DATA, MODELS, RESULTS, FIGURES, REPORT, DELIVERY):
        path.mkdir(parents=True, exist_ok=True)

def first_existing(*names: str) -> Path:
    for name in names:
        for candidate in (ROOT / name, DATA / name):
            if candidate.exists():
                return candidate
    return ROOT / names[0]


def default_dataset() -> Path:
    """Devuelve el dataset local disponible, priorizando el archivo completo."""
    return first_existing(*DATASET_CANDIDATES)
