"""Shared pytest configuration.

The suite got substantially slower when the panel grew from one year to eight
— 30,464 candidate rows instead of 3,400, and the ranker now fits 800 trees.
Marking the heavy tests keeps a fast subset available for the edit-run loop:

    pytest tests -q -m "not slow"     # contracts, features, rules, leakage
    pytest tests -q                   # everything, including model fitting
"""
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: fits models on the full eight-year panel")
