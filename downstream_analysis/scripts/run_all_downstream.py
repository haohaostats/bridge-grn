                     


import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

commands = [
    [sys.executable, str(HERE / "run_epiblast_annotation.py"), "--normalization", "logcpm"],
    [sys.executable, str(HERE / "run_perturbseq_scoring.py")],
    [sys.executable, str(HERE / "summarize_downstream.py"), "--output-dir", str(ROOT / "outputs")],
]

for command in commands:
    subprocess.run(command, check=True)
