import pandas as pd

from anpr.data.split import assign_splits


def test_assign_splits_keeps_same_label_in_one_split():
    labels = [
        "AA04QZH",
        "AB12CDE",
        "LM24XYZ",
        "KT73PQR",
        "RX19NVA",
        "BC55AAA",
        "DE66BBB",
        "FG77CCC",
        "HJ88DDD",
        "KL99EEE",
    ]

    rows = []

    for label in labels:
        rows.append({"image_path": f"{label}_1.png", "label": label})
        rows.append({"image_path": f"{label}_2.png", "label": label})

    df = pd.DataFrame(rows)

    result = assign_splits(df, random_state=42)

    assert set(result["split"].unique()) == {"train", "val", "test"}

    for _, group in result.groupby("label"):
        assert group["split"].nunique() == 1