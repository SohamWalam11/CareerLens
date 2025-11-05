import numpy as np
import pandas as pd
from backend.app.services.feature_pipeline import FeaturePipeline


def make_sample_df():
    return pd.DataFrame({
        "age": [22, 35, 29],
        "education_level": ["Bachelors", "Masters", "PhD"],
        "skills_count": [3, 7, 5]
    })


def test_feature_pipeline_fit_transform_shapes():
    df = make_sample_df()
    pipeline = FeaturePipeline()

    # Fit should not error
    pipeline.fit(df)

    # Transform should return numpy array or DataFrame with same number of rows
    transformed = pipeline.transform(df)
    assert transformed is not None

    if isinstance(transformed, pd.DataFrame):
        assert transformed.shape[0] == df.shape[0]
    else:
        arr = np.asarray(transformed)
        assert arr.shape[0] == df.shape[0]
