import pandas as pd

TARGET = "returned"
ID_COLUMNS = ["order_id", "customer_id"]

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model features; identifiers are excluded to prevent ID memorization."""
    X = df.drop(columns=[c for c in ID_COLUMNS + [TARGET] if c in df.columns]).copy()

    if "discount_percent" in X:
        X["discount_level"] = pd.cut(
            X["discount_percent"],
            bins=[-1, 10, 30, 100],
            labels=["Low", "Medium", "High"],
        ).astype("object")

    if "product_price" in X:
        X["high_value_order"] = (X["product_price"] >= 350).astype(int)

    if "shipping_days" in X:
        X["shipping_speed"] = pd.cut(
            X["shipping_days"],
            bins=[0, 3, 7, 100],
            labels=["Fast", "Normal", "Slow"],
        ).astype("object")

    if "order_hour" in X:
        X["order_period"] = pd.cut(
            X["order_hour"],
            bins=[-1, 5, 11, 17, 23],
            labels=["Night", "Morning", "Afternoon", "Evening"],
        ).astype("object")

    return X
