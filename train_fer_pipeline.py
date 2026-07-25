"""
Facial Emotion Recognition System — Training Pipeline Root Launcher
Executes the full preprocessing, training, and TF.js conversion pipeline from scripts/train_fer_pipeline.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.train_fer_pipeline import parse_args, train_and_evaluate

if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(args)
