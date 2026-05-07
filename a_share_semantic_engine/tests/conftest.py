from pathlib import Path

import pandas as pd
import pytest

from a_share_semantic_engine.core.config import load_config


@pytest.fixture(scope="session")
def base_config():
    return load_config("configs/base.yaml")


@pytest.fixture(scope="session")
def demo_csv_path():
    return Path("examples/data/demo_records_all_20.csv")


@pytest.fixture(scope="session")
def demo_df(demo_csv_path):
    return pd.read_csv(demo_csv_path)
