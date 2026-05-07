from a_share_semantic_engine.features.quality import build_quality_features


def test_quality_features_shape(demo_df):
    quality_df, quality_mat = build_quality_features(demo_df)
    assert len(quality_df) == len(demo_df)
    assert quality_mat.shape[0] == len(demo_df)
    assert quality_mat.shape[1] == quality_df.shape[1]
