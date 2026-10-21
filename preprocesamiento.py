"""Alias compatible del módulo exigido por la rúbrica."""
from preprocesameinto import *  # noqa: F401,F403

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    from src.paths import default_dataset
    parser.add_argument("--input", default=str(default_dataset()))
    parser.add_argument("--output", default="dataset_limpio.csv")
    parser.add_argument("--report-dir", default="reportes")
    args = parser.parse_args()
    run(args.input, args.output, args.report_dir)
