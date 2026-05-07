from a_share_semantic_engine.data.validation import validate_required_columns


def test_required_columns(demo_df):
    validate_required_columns(demo_df)
