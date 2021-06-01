"""
A script for running path
"""
import os
import subprocess
from pathlib import Path


output = subprocess.check_output(
    "pytest --cov=dacy --cov-config=pytest-coverage/.coveragerc --cov-report term-missing",
    shell=True,
)

save_path = "pytest-coverage"
Path(save_path).mkdir(parents=True, exist_ok=True)

save_path = os.path.join(save_path, "pytest-coverage.txt")
os.remove(save_path)
with open(save_path, "w") as f:
    f.write(output.decode("utf-8"))
