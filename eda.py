from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "product_returns.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
sns.set_theme(style="whitegrid")

print("Dataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nTarget distribution:\n", df["returned"].value_counts())
print("\nOverall return rate:", f"{df['returned'].mean():.2%}")

# 1. Return rate by product category
category_rate = df.groupby("product_category", observed=False)["returned"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
category_rate.plot(kind="bar", ax=ax)
ax.set(title="Return Rate by Category", xlabel="Product category", ylabel="Return rate")
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(REPORTS / "01_return_rate_by_category.png", dpi=160)
plt.close(fig)

# 2. Return rate by payment method
payment_rate = df.groupby("payment_method", observed=False)["returned"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
payment_rate.plot(kind="bar", ax=ax)
ax.set(title="Return Rate by Payment Method", xlabel="Payment method", ylabel="Return rate")
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(REPORTS / "02_return_rate_by_payment.png", dpi=160)
plt.close(fig)

# 3. Product price distribution grouped by target
fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(data=df, x="returned", y="product_price", ax=ax)
ax.set(title="Product Price Distribution by Return Status",
       xlabel="Returned (0 = No, 1 = Yes)", ylabel="Product price")
fig.tight_layout()
fig.savefig(REPORTS / "03_product_price_boxplot.png", dpi=160)
plt.close(fig)

# 4. Previous returns versus observed return rate
previous_rate = df.groupby("previous_returns", observed=False)["returned"].mean()
fig, ax = plt.subplots(figsize=(8, 5))
previous_rate.plot(marker="o", ax=ax)
ax.set(title="Previous Returns vs Return Rate",
       xlabel="Previous returns", ylabel="Observed return rate")
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(REPORTS / "04_previous_returns_vs_rate.png", dpi=160)
plt.close(fig)

# 5. Shipping days versus observed return rate
shipping_rate = df.groupby("shipping_days", observed=False)["returned"].mean()
fig, ax = plt.subplots(figsize=(8, 5))
shipping_rate.plot(marker="o", ax=ax)
ax.set(title="Shipping Days vs Return Rate",
       xlabel="Shipping days", ylabel="Observed return rate")
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(REPORTS / "05_shipping_days_vs_rate.png", dpi=160)
plt.close(fig)

# 6. Correlation heatmap (numeric columns only; IDs excluded)
numeric = df.select_dtypes(include="number").drop(
    columns=["customer_id"], errors="ignore"
)
fig, ax = plt.subplots(figsize=(11, 8))
sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=.4, ax=ax)
ax.set_title("Correlation Heatmap")
fig.tight_layout()
fig.savefig(REPORTS / "06_correlation_heatmap.png", dpi=160)
plt.close(fig)

print(f"\nAll six EDA charts saved in: {REPORTS}")
