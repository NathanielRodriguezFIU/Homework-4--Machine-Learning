import os
import subprocess
import sys

SRC = os.path.join(os.path.dirname(__file__))


def run(script):
    p = subprocess.run([sys.executable, os.path.join(SRC, script)])
    if p.returncode != 0:
        raise SystemExit(f"{script} failed with code {p.returncode}")


if __name__ == "__main__":
    print("Running preprocessing.py")
    run("preprocessing.py")
    print("Running kmeans.py")
    run("kmeans.py")
    print("Running regression.py")
    run("regression.py")
    print("Running visualization.py")
    run("visualization.py")
    print("All steps finished. Results in ../results/")
