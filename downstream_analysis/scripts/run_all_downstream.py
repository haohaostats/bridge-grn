                     


import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

for script in ("run_epiblast_annotation.py", "run_perturbseq_scoring.py"):
    command = [sys.executable, str(HERE / script)]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)
