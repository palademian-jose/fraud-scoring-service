from __future__ import annotations

import pandas as pd
import pytest

from fraudscore.data import Split, load_frame, time_split
from fraudscore.models import ModelBundle, fit_bundle
from fraudscore.settings import Settings


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(bootstrap_resamples=50)


@pytest.fixture(scope="session")
def frame(settings: Settings) -> pd.DataFrame:
    return load_frame(settings.sample_path)


@pytest.fixture(scope="session")
def split(frame: pd.DataFrame) -> Split:
    return time_split(frame)


@pytest.fixture(scope="session")
def champion(split: Split, settings: Settings) -> ModelBundle:
    bundle, _ = fit_bundle("champion", split, settings)
    return bundle
