import pandas as pd
from pandas.testing import assert_frame_equal


def test_generate_sme_data_small_and_reproducible():
    from sme_causal.data.synth_data import SynthConfig, generate_sme_data

    cfg = SynthConfig(n_clients=8, seed=123)
    df1 = generate_sme_data(cfg)
    df2 = generate_sme_data(cfg)
    # Basic shape and required columns exist
    assert len(df1) == 8
    required = {
        "Client_ID",
        "Industry",
        "Region",
        "Business_Size",
        "Years_in_Operation",
        "Avg_Monthly_Inflow",
        "Avg_Monthly_Outflow",
        "Monthly_Transaction_Count",
        "Avg_Account_Balance",
        "New_Product_Offer",
        "Revenue_Growth_Rate",
        "Revenue_Trend",
    }
    assert required.issubset(set(df1.columns))
    # Reproducible with the same seed
    assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True), check_dtype=False)


def test_ground_truth_edges_schema():
    from sme_causal.data.synth_data import ground_truth_edges

    gt = ground_truth_edges()
    assert isinstance(gt, list)
    assert gt, "ground truth edges should not be empty"
    sample = gt[0]
    assert {"source", "target", "sign", "rationale"}.issubset(sample.keys())

