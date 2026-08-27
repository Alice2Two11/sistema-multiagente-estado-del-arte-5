#!/usr/bin/env python3
"""Entrypoint productivo -- Bloque 5 (MAIN 5): invoca LangGraph
(``src.orchestration_langgraph.pipeline_graph``) como motor de
orquestación, no la máquina de estados propia
(``src.orchestration.pipeline_orchestrator``, que se conserva en el
repo temporalmente, ya sin ser el entrypoint productivo -- ver Bloque 6
pendiente para su retiro definitivo)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

VENV_PYTHON = Path(os.environ.get("ESTADO_ARTE_PYTHON", "/content/venv_estado_arte/bin/python"))
PROJECT_DIR = Path(os.environ.get("THESIS_PROJECT_DIR", "/content/proyecto_estado_arte"))

def main() -> int:
    if not VENV_PYTHON.is_file():
        print(f"ERROR: no existe {VENV_PYTHON}", file=sys.stderr)
        return 2

    if not PROJECT_DIR.is_dir():
        print(f"ERROR: no existe {PROJECT_DIR}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["THESIS_PROJECT_DIR"] = str(PROJECT_DIR)

    command = [
        str(VENV_PYTHON),
        "-m",
        "src.orchestration_langgraph.pipeline_graph",
        "--project-dir",
        str(PROJECT_DIR),
        *sys.argv[1:],
    ]

    print("Ejecutando:")
    print(" ".join(command))

    result = subprocess.run(
        command,
        cwd=str(PROJECT_DIR),
        env=env,
    )

    return result.returncode

if __name__ == "__main__":
    raise SystemExit(main())
