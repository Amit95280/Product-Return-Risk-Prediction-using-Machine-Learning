from pathlib import Path
import numpy as np
import pandas as pd

def generate_sample_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Generate reproducible synthetic e-commerce orders for an ML demo."""
    rng = np.random.default_rng(random_state)
    categories = np.array(["Beauty", "Clothing", "Electronics", "Home", "Sports"])
    payments = np.array(["COD", "Credit Card", "Debit Card", "UPI"])
    shipping_types = np.array(["Standard", "Express", "Same Day"])

    product_category = rng.choice(categories, n_samples)
    payment_method = rng.choice(payments, n_samples, p=[.28, .28, .20, .24])
    shipping_type = rng.choice(shipping_types, n_samples, p=[.55, .30, .15])
    product_price = rng.integers(10, 501, n_samples).astype(float)
    discount_percent = rng.integers(0, 61, n_samples)
    order_hour = rng.integers(0, 24, n_samples)
    shipping_days = rng.integers(1, 15, n_samples)
    customer_reviews = rng.integers(0, 101, n_samples)
    customer_rating = np.round(rng.uniform(1, 5, n_samples), 1)
    previous_returns = rng.integers(0, 10, n_samples)
    account_age_days = rng.integers(1, 2500, n_samples)
    customer_id = rng.integers(10000, 16000, n_samples)

    
    # Simulated return-risk logic with stronger,
    # more realistic relationships between features and returns.

    logit = (
        -2.2

        # Customer behaviour
        + 0.16 * previous_returns
        - 0.18 * (account_age_days / 365)

        # Customer satisfaction
        + 0.55 * (5 - customer_rating)

        # Delivery experience
        + 0.12 * shipping_days
        + 0.35 * (shipping_type == "Standard")

        # Payment behaviour
        + 0.75 * (payment_method == "COD")

        # Discount behaviour
        + 0.018 * discount_percent
        + 0.35 * (discount_percent >= 40)

        # Product category
        + 0.55 * (product_category == "Clothing")
        + 0.30 * (product_category == "Electronics")

        # Product price
        + 0.0008 * product_price

        # Order timing
        + 0.30 * (order_hour >= 20)

        # Interactions
        + 0.30 * (
            (payment_method == "COD") &
            (discount_percent >= 40)
        )
        + 0.35 * (
            (shipping_days >= 10) &
            (customer_rating <= 2.5)
        )
    )

    # Add moderate random variation to avoid
    # making the target perfectly predictable.
    logit += rng.normal(0, 0.45, n_samples)

    probability = 1 / (1 + np.exp(-logit))

    returned = rng.binomial(1, probability)

    return pd.DataFrame({
        "order_id": [f"ORD{i:06d}" for i in range(1, n_samples + 1)],
        "customer_id": customer_id,
        "product_category": product_category,
        "product_price": product_price,
        "discount_percent": discount_percent,
        "order_hour": order_hour,
        "shipping_days": shipping_days,
        "payment_method": payment_method,
        "shipping_type": shipping_type,
        "customer_reviews": customer_reviews,
        "customer_rating": customer_rating,
        "previous_returns": previous_returns,
        "account_age_days": account_age_days,
        "returned": returned,
    })

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    output = project_root / "data" / "product_returns.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    data = generate_sample_data()
    data.to_csv(output, index=False)
    print(f"Saved {len(data):,} orders to {output}")
    print(f"Return rate: {data['returned'].mean():.1%}")
