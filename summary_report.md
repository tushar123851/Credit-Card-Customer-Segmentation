
# Credit Card Customer Segmentation using Unsupervised Machine Learning

**Author:** Tushar Vala  
**Project Type:** Unsupervised Machine Learning  
**Domain:** Banking & Financial Services  
**Dataset:** CC_GENERAL.csv

---

# Project Summary

Customer segmentation is one of the most valuable applications of data science in the banking industry. Financial institutions serve customers with diverse spending habits, payment behaviors, credit usage patterns, and financial needs. Understanding these behavioral differences enables banks to offer personalized services, improve customer satisfaction, reduce financial risk, and maximize business profitability.

The objective of this project was to identify meaningful customer segments from credit card transaction data using **Unsupervised Machine Learning**. Since the dataset does not contain predefined labels, clustering algorithms were used to discover natural customer groups based on their financial behavior.

The project used the **CC_GENERAL.csv** dataset containing approximately **8,950 customer records** with information related to balances, purchases, cash advances, credit limits, payments, minimum payments, and tenure.

---

# Data Preprocessing

Several preprocessing techniques were applied before model development to improve data quality and clustering performance.

The **CUST_ID** column was removed because it is only a unique customer identifier and does not contribute to clustering.

Missing values in **CREDIT_LIMIT** and **MINIMUM_PAYMENTS** were handled using **median imputation**, which is more robust for skewed financial data.

Duplicate records were checked and removed where necessary.

Outliers were treated using the **Interquartile Range (IQR) Winsorization** technique to reduce the influence of extreme values while preserving valuable customer information.

Since many monetary variables were positively skewed, a **logarithmic (log1p) transformation** was applied to improve data distribution.

Finally, all numerical features were standardized using **StandardScaler** so that every variable contributed equally during distance-based clustering.

---

# Feature Engineering

To better represent customer financial behavior, several new features were created:

- Purchase Ratio
- Monthly Average Purchase
- Monthly Average Cash Advance
- Credit Limit Usage Ratio
- Payment-to-Minimum Payment Ratio

These engineered features improved customer representation and helped produce more meaningful clusters.

---

# Clustering Algorithms

Three different clustering algorithms were implemented and compared.

## 1. K-Means Clustering

K-Means was used as the primary clustering algorithm because of its efficiency and ability to produce compact customer groups.

**Results**

- Number of Clusters: **3**
- Silhouette Score: **0.2030**
- Davies-Bouldin Index: **1.6843**
- Calinski-Harabasz Index: **1741.34**

---

## 2. Agglomerative Hierarchical Clustering

Agglomerative clustering grouped customers hierarchically using Ward linkage.

**Results**

- Number of Clusters: **3**
- Silhouette Score: **0.1382**
- Davies-Bouldin Index: **1.9480**
- Calinski-Harabasz Index: **1186.20**

---

## 3. DBSCAN

DBSCAN was implemented to discover density-based customer groups and detect noise points.

**Best Parameters**

- eps = **1.5**
- min_samples = **10**

**Results**

- Number of Clusters: **4**
- Silhouette Score: **0.1288**
- Davies-Bouldin Index: **1.1108**
- Calinski-Harabasz Index: **22.36**

---

# Model Comparison

The clustering algorithms were evaluated using three standard internal evaluation metrics.

| Algorithm | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|------------|-----------:|---------------:|------------------:|
| K-Means | **0.2030** | **1.6843** | **1741.34** |
| Agglomerative | 0.1382 | 1.9480 | 1186.20 |
| DBSCAN | 0.1288 | **1.1108** | 22.36 |

Although DBSCAN achieved the lowest Davies-Bouldin Index, it generated many noise points and lower-quality business segmentation. K-Means achieved the highest Silhouette Score and Calinski-Harabasz Index while producing clear, interpretable customer groups. Therefore, **K-Means was selected as the final model.**

---

# Customer Segments

The K-Means algorithm identified three meaningful customer personas.

## Premium Customers

These customers maintain high balances, larger credit limits, and higher purchasing activity. They represent high-value customers for the bank.

**Recommended Strategy**

- Premium Credit Cards
- Airport Lounge Access
- Wealth Management Services
- Investment Products
- Exclusive Reward Programs

---

## Regular Customers

These customers exhibit stable purchasing behavior and consistent payment patterns.

**Recommended Strategy**

- Loyalty Programs
- Shopping Cashback Offers
- Credit Limit Upgrades
- Seasonal Promotions

---

## Cash Advance Customers

These customers frequently rely on cash advances and generally demonstrate higher financial dependency.

**Recommended Strategy**

- EMI Conversion Plans
- Debt Management Programs
- Financial Counseling
- Lower Interest Offers

---

# Business Impact

This customer segmentation model enables banks to:

- Improve personalized marketing campaigns
- Increase customer retention
- Enhance cross-selling opportunities
- Identify valuable customer segments
- Support credit risk management
- Improve business decision-making through data-driven insights

---

# Model Deployment

The trained K-Means model and preprocessing pipeline were saved using **Joblib**.

Generated deployment files:

- cc_segmentation_model.pkl
- cc_scaler.pkl

These files allow new customer records to be segmented without retraining the model.

---

# Future Improvements

Future versions of this project can be enhanced by incorporating additional customer information such as:

- Age
- Gender
- Annual Income
- Credit Score
- Occupation
- Geographic Location
- Customer Lifetime Value (CLV)
- Digital Banking Activity
- Transaction Categories
- Loan History

The project can also be deployed as a **FastAPI** or **Streamlit** application for real-time customer segmentation.

---

# Conclusion

This project successfully demonstrated the use of **Unsupervised Machine Learning** for customer segmentation in the banking domain. Three clustering algorithms were implemented and compared using multiple evaluation metrics.

Among them, **K-Means Clustering** delivered the best overall performance by producing stable, well-separated, and business-friendly customer segments. The resulting customer personas provide valuable insights that financial institutions can use to personalize services, improve customer relationships, optimize marketing strategies, and support data-driven decision-making.

The project includes a complete machine learning pipeline covering data preprocessing, feature engineering, exploratory data analysis, clustering, model evaluation, business interpretation, model persistence, and deployment-ready prediction capability, making it suitable for real-world banking applications and an excellent portfolio project.
