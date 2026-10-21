"""Compila el PDF y crea un ZIP de código y evidencias por pregunta.

El PDF se conserva separado en ``entrega/``. El ZIP contiene los scripts
directamente en su raíz y excluye reporte, documentación técnica y guion.
"""
from __future__ import annotations
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parent
DELIVERY_BASENAME = "1ERPARCIAL_GAELASKARYRAZOMONTAÑEZ_MD"
LATEX_TEMP_EXTENSIONS = {".aux", ".log", ".out", ".toc", ".synctex.gz"}


def clean_delivery():
    delivery = ROOT / "entrega"
    delivery.mkdir(exist_ok=True)
    for path in delivery.iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def clean_latex_artifacts():
    for directory in (ROOT / "reporte", ROOT / "entrega"):
        for path in directory.iterdir():
            if path.is_file() and (
                path.suffix in LATEX_TEMP_EXTENSIONS
                or path.name.endswith(".synctex.gz")
            ):
                path.unlink()

def compile_report():
    report = ROOT / "reporte"
    delivery = ROOT / "entrega"
    clean_delivery()
    command = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(delivery), "Reporte_1er_Parcial_Mineria_Datos.tex"]
    subprocess.run(command, cwd=report, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(command, cwd=report, check=True, stdout=subprocess.DEVNULL)
    pdf = delivery / "Reporte_1er_Parcial_Mineria_Datos.pdf"
    shutil.copy2(pdf, delivery / f"{DELIVERY_BASENAME}.pdf")
    pdf.unlink()
    clean_latex_artifacts()

def package(name: str):
    staging = ROOT / f"_staging_{name}"
    if staging.exists(): shutil.rmtree(staging)
    groups = {
        # El código se copia una sola vez en la raíz del ZIP; aquí solo van
        # los datos y evidencias asociados con cada pregunta.
        "Pregunta_1_Preprocesamiento": [
            "resultados/auditoria_preprocesamiento.json",
            "resultados/auditoria_preprocesamiento.md",
        ],
        "Pregunta_2_Muestreo": [
            "datos/train.csv", "datos/validation.csv", "datos/test.csv",
            "resultados/calculo_muestra.json",
            "resultados/distribucion_particiones.csv",
        ],
        "Pregunta_3_CrossVal": [
            "resultados/reporte_crossval.csv",
            "resultados/reporte_crossval.json",
            "graficas/02_analisis_k_crossval.png",
        ],
        "Pregunta_4_Entrenamiento": [
            "modelos/mejor_modelo_mlp.joblib",
            "resultados/resultados_gridsearch.csv",
        ],
        "Pregunta_5_y_6_Experimentos": ["resultados", "graficas"],
    }
    for group, files in groups.items():
        for rel in files:
            src = ROOT / rel
            if not src.exists(): continue
            dst = staging / group / (src.name if src.is_file() else src.name)
            if src.is_dir(): shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.pyc"))
            else:
                dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
    root_files = [
        "preprocesameinto.py", "preprocesamiento.py", "muestreo.py",
        "crossVal.py", "entrenamiento.py", "main.py",
        "empaquetar_entrega.py", "proyecto_utils.py", "requirements.txt",
    ]
    for rel in root_files:
        shutil.copy2(ROOT / rel, staging / Path(rel).name)
    shutil.copytree(ROOT / "src", staging / "src", ignore=shutil.ignore_patterns("*.pyc"))
    zip_path = ROOT / "entrega" / f"{name}.zip"
    if zip_path.exists(): zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in staging.rglob("*"):
            if file.is_file(): archive.write(file, file.relative_to(staging))
    shutil.rmtree(staging)

if __name__ == "__main__":
    compile_report()
    package(DELIVERY_BASENAME)
    print(f"Entrega creada en entrega/{DELIVERY_BASENAME}.pdf y .zip")
