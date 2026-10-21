"""Paleta maestra compartida entre Python y LaTeX."""
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

COLORS = {
    "seupurple": "#8E44AD", "lachaiseblue": "#2980B9", "seublue": "#3498DB",
    "seuteal": "#16A085", "seuamber": "#D35400", "seugray": "#7F8C8D",
    "seubg": "#F0F8FF",
}
CLASS_COLORS = {
    "Alto_valor": COLORS["seupurple"], "Frecuente": COLORS["lachaiseblue"],
    "Ocasional": COLORS["seublue"], "Promocional": COLORS["seuteal"],
    "Inactivo": COLORS["seuamber"],
}

def apply_theme():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.facecolor": COLORS["seubg"], "axes.facecolor": "white",
        "axes.edgecolor": COLORS["seugray"], "axes.spines.top": False,
        "axes.spines.right": False, "axes.titleweight": "bold",
        "axes.prop_cycle": plt.cycler(color=list(COLORS.values())[:6]),
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })

def save_figure(fig, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
