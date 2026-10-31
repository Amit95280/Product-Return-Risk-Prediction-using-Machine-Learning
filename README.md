# Product Return Risk Prediction using Machine Learning

An end-to-end educational project that estimates whether an e-commerce order may be returned. It demonstrates synthetic data generation, exploratory data analysis, feature engineering, model comparison, evaluation, and a Streamlit interface.

> **Important:** The included dataset is synthetic. Model scores from it are only a demonstration and must not be represented as validated business performance. Do not claim a specific ROC-AUC until you run the pipeline and use the measured result.

## Tech stack
- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Streamlit, Joblib

## Project structure
```text
product-return-risk-prediction/
├── data/                 # Generated CSV dataset
├── models/               # Trained pipeline and metadata
├── reports/              # EDA charts and evaluation outputs
├── src/
│   ├── data_generation.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
├── app.py
├── eda.py
├── requirements.txt
└── README.md
```

## Setup

Python 3.10 or newer is recommended.

### Windows (PowerShell)
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the project

Run these commands from the project root, in order:

```bash
python src/data_generation.py
python eda.py
python src/train.py
streamlit run app.py
```

Open the local URL printed by Streamlit (usually http://localhost:8501).

## Dataset fields

| Field | Description |
|---|---|
| order_id | Synthetic order identifier |
| product_category | Product category |
| price | Order item price |
| discount | Discount percentage |
| delivery_days | Delivery duration |
| customer_orders | Customer's previous order count |
| shipping_type | Shipping method |
| review_score | Customer review score |
| late_delivery | Whether delivery was late |
| returned | Target: 1 = returned, 0 = not returned |

## Machine learning workflow

1. Generate reproducible synthetic orders.
2. Inspect missing values, target balance, category return rates, and numeric correlations.
3. Engineer discount level, high-value order, and delivery delay band.
4. Split data using stratification.
5. Fit preprocessing only on the training split using a Scikit-learn pipeline.
6. Compare Logistic Regression, Random Forest, and HistGradientBoosting.
7. Evaluate ROC-AUC, average precision, precision, recall, F1, and confusion matrix.
8. Save the best pipeline and use it in Streamlit.

## Evaluation

After training, inspect:
- `reports/model_comparison.csv`
- `reports/classification_report.json`
- `reports/confusion_matrix.csv`
- `models/metadata.json`

ROC-AUC is useful for ranking performance; average precision is especially useful when the positive class is less common. Choose a decision threshold based on the business cost of false positives and false negatives rather than assuming 0.50 is optimal.

## Notes and limitations

- This project uses generated data, not actual retailer transactions.
- Synthetic patterns can make a model appear more predictive than it would be on real data.
- The example uses a simple 0.50 classification threshold for demonstration.
- For production, validate on time-separated real data, monitor drift, assess calibration and fairness, and prevent data leakage. Only use information available at prediction time.
- `review_score` may not exist before an order is returned in a real system. Replace it with a valid pre-return signal if predicting risk before delivery.

## Resume description (only after running and verifying)

**Product Return Risk Prediction | Python, Pandas, Scikit-learn, Streamlit**
- Built an end-to-end classification workflow on 5,000 synthetic e-commerce orders to demonstrate product return risk prediction.
- Performed exploratory analysis and engineered discount, order-value, and delivery-delay features.
- Compared Logistic Regression, Random Forest, and gradient-boosting models using ROC-AUC, average precision, precision, recall, and F1.
- Developed a Streamlit app that displays estimated return probability from order attributes.

Replace synthetic-data wording or add a measured metric only if you later train and validate on an appropriate real dataset.


## Extended EDA analysis

Run `python eda.py` to generate six charts in `reports/`:
1. Return rate by product category
2. Return rate by payment method
3. Product-price boxplot grouped by return status
4. Previous returns vs observed return rate
5. Shipping days vs observed return rate
6. Numeric correlation heatmap

The dataset includes payment method, customer rating, customer review count,
previous returns, account age, order hour, shipping duration, and product price.
`customer_id` is intentionally excluded from model features because identifiers
should not be used as predictive signals.

The generated dataset is synthetic. Interpret the plots as demonstrations of
the relationships planted by the generator, not as real customer insights.
