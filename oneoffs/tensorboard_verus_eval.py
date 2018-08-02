"""Analyse if any variable correlate with model eval.

Example usage:
  python3 oneoffs/tensorboard_verus_eval.py --base_dir "v9-19x19/work_dir"
"""
import sys
sys.path.insert(0, '.')

import os.path

from absl import app, flags
import numpy as np
from tqdm import tqdm

FLAGS = flags.FLAGS



def main(unusedargv):
    get_all_events(

if __name__ == "__main__":
    app.run(main)
