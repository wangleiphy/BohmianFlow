"""Shared pytest configuration: add repo root to ``sys.path``."""

import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
