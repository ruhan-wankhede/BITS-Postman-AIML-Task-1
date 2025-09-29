
## Data Understanding & Feature Engineering

- **Dataset**: Olist, a Brazilian e-commerce website  
- **Customers**: 99,441 total with 96,096 unique customers  
- **Behavior**: only a few customers have multiple orders  
- **Churn rate**: ~60% of customers are churned  
- **Granularity**: used `customer_unique_id` instead of `customer_id` (since `customer_id` changes per session)  
- **Scaling**: I've used the standard scaler since one of the models is a regression models and the features vary greatly in scale (payment will be in thousands and frequency is 1-10 mostly)[this will be similar to if i applied log to the monetary_value feature before adding it to the features]

---

## Modeling

- **Logistic Regression** implemented in **PyTorch**  
- **Random Forest Classifier** implemented in **scikit-learn**  

### Data Leakage Issue
- Initially achieved **100% accuracy** — thought it was caused by leakage (which might've been one but not the primary reason):
  - `recency` and `churn` were computed **before splitting** the dataset.
- Fixed by changing the order:
  - Split customers first, then compute features and labels relative to the training reference date.  

### Insights
- **Including recency**: churn prediction becomes trivial giving near-perfect results, as the models just learn the rule `churn = (recency > 180)`.  
- **Excluding recency**: gives more realistic results:
  - Logistic Regression ~59% accuracy  
  - Random Forest ~51% accuracy  
- **Confusion matrix**: models often misclassify active customers as churned, shows class imbalance and difficulty of predicting churn without time-based signals.  

### Next Steps
- Engineer richer features (e.g., order frequency rate, average order value, time since first purchase).
- Try better tuning.

---

##  Results

**Logistic Regression (PyTorch)**  
- Accuracy: 0.9983  
- Precision: 1.0000  
- Recall: 0.9971  
- F1 Score: 0.9985  

**Random Forest (sklearn)**  
- Accuracy: 1.0000  
- Precision: 1.0000  
- Recall: 1.0000  
- F1 Score: 1.0000  

---

## Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)

## Environment Details

- **Python**: 3.10+  
- **Key Libraries**:  
  - `torch>=2.0.0`  
  - `scikit-learn>=1.3.0`  
  - `pandas>=2.0.0`  
  - `numpy>=1.24.0`  
  - `matplotlib>=3.7.0`  
  - `seaborn>=0.12.0`  
  - `tqdm>=4.65.0`  
  - `joblib>=1.3.0`  

Install dependencies with:  
```bash
pip install -r requirements.txt
```
## Run Instructions
1. Clone this repo and install dependencies
2. Run the main.py file
