"""Run every practical in order. Figures are saved to practicals/outputs/.

Usage:
    python run_all.py                       # use the built-in sample images
    python run_all.py path/to/image.jpg     # use your own image for every practical
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = sorted(f for f in os.listdir(HERE)
                 if f[:2].isdigit() and f.endswith(".py"))


def main():
    extra_args = sys.argv[1:]
    failed = []
    for script in SCRIPTS:
        print(f"\n=== {script} ===")
        result = subprocess.run([sys.executable, os.path.join(HERE, script), *extra_args])
        if result.returncode != 0:
            failed.append(script)
    print("\nAll practicals finished." if not failed else f"\nFailed: {', '.join(failed)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
