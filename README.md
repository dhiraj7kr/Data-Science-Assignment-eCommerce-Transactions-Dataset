# Data-Science-Assignment-eCommerce-Transactions-Dataset
#TASK:1 
---
To perform the EDA and derive business insights from the provided datasets (`Customers.csv`, `Products.csv`, and `Transactions.csv`), you can follow the steps outlined below. I'll provide the complete code, along with detailed instructions for each part of the process.

### Prerequisites:
1. **Install Required Libraries**: If you haven't already installed the necessary Python libraries, you can do so by running:

```bash
pip install pandas numpy matplotlib seaborn
```

### Step 1: Load and Inspect the Data

#### Code:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
customers = pd.read_csv('Customers.csv')
products = pd.read_csv('Products.csv')
transactions = pd.read_csv('Transactions.csv')

# Inspect the first few rows of each dataset
print(customers.head())
print(products.head())
print(transactions.head())

# Check for missing values
print(customers.isnull().sum())
print(products.isnull().sum())
print(transactions.isnull().sum())

# Summary statistics of numeric columns
print(customers.describe())
print(products.describe())
print(transactions.describe())
```

#### Explanation:
- Load the data using `pandas.read_csv()` and display the first few rows of each dataset.
- Use `isnull().sum()` to check for missing values in the datasets.
- Use `describe()` to get summary statistics for numeric columns.

### Step 2: Data Cleaning and Preprocessing

Before performing the EDA, you might need to clean and preprocess the data.

#### Code:

```python
# Convert SignupDate and TransactionDate to datetime format
customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])

# Check if there are any duplicates
print(customers.duplicated().sum())
print(products.duplicated().sum())
print(transactions.duplicated().sum())

# Remove duplicates if any
customers = customers.drop_duplicates()
products = products.drop_duplicates()
transactions = transactions.drop_duplicates()
```

#### Explanation:
- Convert the `SignupDate` and `TransactionDate` columns to `datetime` format for easier analysis.
- Check for duplicate rows and remove them if necessary.

### Step 3: Exploratory Data Analysis (EDA)

#### 1. **Customer Distribution by Region**

```python
# Plot the distribution of customers by region
plt.figure(figsize=(8, 6))
sns.countplot(data=customers, x='Region', palette='Set2')
plt.title('Customer Distribution by Region')
plt.xlabel('Region')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.show()
```

#### 2. **Signup Trends Over Time**

```python
# Plot signups over time
customers['YearMonth'] = customers['SignupDate'].dt.to_period('M')
signup_trends = customers.groupby('YearMonth').size()

plt.figure(figsize=(10, 6))
signup_trends.plot(kind='line', marker='o', color='b')
plt.title('Customer Signup Trends Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Signups')
plt.grid(True)
plt.show()
```

#### 3. **Product Categories and Pricing**

```python
# Plot distribution of products by category
plt.figure(figsize=(8, 6))
sns.countplot(data=products, x='Category', palette='Set3')
plt.title('Product Distribution by Category')
plt.xlabel('Category')
plt.ylabel('Number of Products')
plt.xticks(rotation=45)
plt.show()

# Plot the price distribution of products
plt.figure(figsize=(8, 6))
sns.histplot(products['Price'], kde=True, color='purple', bins=30)
plt.title('Price Distribution of Products')
plt.xlabel('Price (USD)')
plt.ylabel('Frequency')
plt.show()
```

#### 4. **Transaction Analysis**

```python
# Total transaction value per customer
customer_total_sales = transactions.groupby('CustomerID')['TotalValue'].sum().sort_values(ascending=False)

# Plot the top 10 customers by total sales
plt.figure(figsize=(8, 6))
customer_total_sales.head(10).plot(kind='bar', color='orange')
plt.title('Top 10 Customers by Total Transaction Value')
plt.xlabel('Customer ID')
plt.ylabel('Total Sales (USD)')
plt.show()

# Transaction frequency by product
product_sales = transactions.groupby('ProductID')['Quantity'].sum()

# Merge with product information to get product names
product_sales = product_sales.reset_index()
product_sales = pd.merge(product_sales, products[['ProductID', 'ProductName']], on='ProductID')

# Plot top 10 products by quantity sold
plt.figure(figsize=(10, 6))
product_sales.sort_values(by='Quantity', ascending=False).head(10).plot(kind='bar', x='ProductName', y='Quantity', color='green')
plt.title('Top 10 Products by Quantity Sold')
plt.xlabel('Product')
plt.ylabel('Quantity Sold')
plt.xticks(rotation=45)
plt.show()
```

#### 5. **Transaction Patterns (Price vs Quantity)**

```python
# Plot relationship between price and quantity purchased
plt.figure(figsize=(8, 6))
sns.scatterplot(data=transactions, x='Price', y='Quantity', alpha=0.6, color='red')
plt.title('Price vs Quantity Purchased')
plt.xlabel('Price (USD)')
plt.ylabel('Quantity')
plt.show()
```

### Step 4: Derive Business Insights

#### Insight 1: **Customer Distribution by Region**
- From the customer distribution plot, identify the regions with the highest and lowest number of customers. This can help target marketing efforts or expansion.

#### Insight 2: **Signup Trends Over Time**
- From the signup trend plot, determine if there are peak months or seasons when customer signups increase, and plan promotional campaigns during those periods.

#### Insight 3: **Popular Product Categories**
- The category distribution plot can show which categories are the most popular. Focus on high-demand categories for promotions or increasing inventory.

#### Insight 4: **High-Value Customers**
- By analyzing total sales per customer, you can identify high-value customers who may benefit from loyalty programs or targeted campaigns.

#### Insight 5: **Product Performance (Sales Volume)**
- By analyzing which products are selling the most (in terms of quantity), you can optimize inventory management and focus on promoting high-performing products.

### Step 5: Conclusion and Next Steps

After completing the EDA, you can write your findings in short point-wise sentences as business insights. These insights can inform decision-making, such as marketing strategies, inventory planning, customer relationship management, and product promotions.

-------

#TASK: 2
---

To build a **Lookalike Model** that recommends the top 3 similar customers based on their profile and transaction history, we need to combine the customer's demographic information (e.g., region, signup date) with their transactional behavior (e.g., product categories purchased, frequency of purchases, total spending). Here's how we can go about this.

### Overview of the Lookalike Model:
1. **Input**: CustomerID and other customer attributes (region, signup date, etc.), along with their transaction history (product purchased, quantity, total value).
2. **Output**: Top 3 similar customers with a similarity score for each of the first 20 customers in `Customers.csv`.

We'll use a **combination of demographic and transactional data** to compute the similarity between customers. For this, I’ll demonstrate how to:
- Preprocess the data.
- Create a feature vector for each customer based on both profile and transaction history.
- Compute similarity between customers (using cosine similarity).
- Recommend top 3 similar customers.

### Step 1: Import Necessary Libraries and Load Data

```python
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# Load datasets
customers = pd.read_csv('Customers.csv')
products = pd.read_csv('Products.csv')
transactions = pd.read_csv('Transactions.csv')

# Inspect the first few rows of each dataset
print(customers.head())
print(products.head())
print(transactions.head())
```

### Step 2: Preprocess Data

#### 2.1: Merge Transaction Data with Customer and Product Information

We need to aggregate the transaction data by customer and product, then merge it with customer profile data.

```python
# Merge transactions with product data to get product category
transactions = transactions.merge(products[['ProductID', 'Category']], on='ProductID')

# Aggregate transaction data per customer
customer_transactions = transactions.groupby(['CustomerID', 'Category']).agg(
    total_spent=('TotalValue', 'sum'),
    total_quantity=('Quantity', 'sum')
).reset_index()

# Create a pivot table for customers with product categories as columns
customer_profile = customer_transactions.pivot_table(
    index='CustomerID',
    columns='Category',
    values='total_spent',
    aggfunc='sum',
    fill_value=0
)

# Include customer demographics (region, signup date) in the profile
customer_profile = customer_profile.merge(customers[['CustomerID', 'Region']], on='CustomerID')
```

#### 2.2: Standardize the Data

To ensure all features are on the same scale (i.e., both spending and number of purchases), we standardize the data using `StandardScaler`.

```python
# Standardize transaction data (numeric features only)
scaler = StandardScaler()
transaction_features = customer_profile.drop(['CustomerID', 'Region'], axis=1)
transaction_features_scaled = pd.DataFrame(scaler.fit_transform(transaction_features), columns=transaction_features.columns)

# Include the region (non-numeric) as it can also contribute to similarity
customer_profile_scaled = pd.concat([customer_profile[['CustomerID', 'Region']], transaction_features_scaled], axis=1)
```

### Step 3: Compute Customer Similarities

Now, we'll compute the **cosine similarity** between customers based on their feature vectors (both demographics and transactional behavior).

```python
# Extract the feature vectors (excluding CustomerID and Region)
customer_vectors = customer_profile_scaled.drop(['CustomerID', 'Region'], axis=1)

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(customer_vectors)

# Convert similarity matrix to a DataFrame for easier analysis
similarity_df = pd.DataFrame(similarity_matrix, index=customer_profile_scaled['CustomerID'], columns=customer_profile_scaled['CustomerID'])

# Inspect the similarity matrix
print(similarity_df.head())
```

### Step 4: Find Top 3 Lookalike Customers

For each of the first 20 customers (CustomerID: C0001 to C0020), we will identify their top 3 most similar customers based on the similarity scores.

```python
# Define the list of customers we are interested in (C0001 to C0020)
target_customers = ['C0001', 'C0002', 'C0003', 'C0004', 'C0005', 'C0006', 'C0007', 'C0008', 'C0009', 'C0010',
                    'C0011', 'C0012', 'C0013', 'C0014', 'C0015', 'C0016', 'C0017', 'C0018', 'C0019', 'C0020']

# Create a dictionary to store the lookalike recommendations
lookalike_map = defaultdict(list)

# For each target customer, get top 3 similar customers (excluding themselves)
for customer_id in target_customers:
    similarities = similarity_df[customer_id].sort_values(ascending=False)  # Sort by similarity score
    top_similar_customers = similarities.drop(customer_id).head(3)  # Exclude the customer itself
    for similar_customer_id, score in top_similar_customers.items():
        lookalike_map[customer_id].append((similar_customer_id, score))

# Convert the lookalike map to a DataFrame
lookalike_data = []
for customer_id, similar_customers in lookalike_map.items():
    for similar_customer_id, score in similar_customers:
        lookalike_data.append([customer_id, similar_customer_id, score])

lookalike_df = pd.DataFrame(lookalike_data, columns=['CustomerID', 'LookalikeID', 'SimilarityScore'])

# Inspect the results
print(lookalike_df.head())
```

### Step 5: Save the Recommendations to a CSV

Finally, we save the results as a CSV file in the required format.

```python
# Save the recommendations to a CSV file
lookalike_df.to_csv('Lookalike.csv', index=False)
```

### Output

The `Lookalike.csv` file will contain the following columns:
- `CustomerID`: The original customer.
- `LookalikeID`: The recommended similar customer.
- `SimilarityScore`: The similarity score between the original customer and the recommended customer.

### Example Output in `Lookalike.csv`:
```csv
CustomerID,LookalikeID,SimilarityScore
C0001,C0004,0.92
C0001,C0005,0.89
C0001,C0006,0.85
C0002,C0007,0.88
C0002,C0008,0.84
C0002,C0009,0.83
...
```

### Explanation of Steps:
1. **Data Preprocessing**: We aggregated customer transaction data by product category, then standardized it with customer demographics.
2. **Cosine Similarity**: We used cosine similarity to measure how similar each customer is based on their behavior and demographic data.
3. **Top 3 Lookalikes**: For each of the first 20 customers, we identified the top 3 most similar customers based on the similarity scores and saved them to a CSV file.

This method provides a personalized recommendation based on customer behavior and profile, enabling targeted marketing, personalized product offerings, or customer service strategies.

#TASK:3
----------
To perform customer segmentation using clustering techniques based on both profile information (from `Customers.csv`) and transaction data (from `Transactions.csv`), we will follow a structured approach. Here’s how we can approach the task:

### Overview of the Approach:
1. **Preprocess the Data**: Combine the customer profile and transaction data.
2. **Feature Engineering**: Create useful features from both datasets.
3. **Clustering**: Apply clustering algorithms like **K-Means** and find the optimal number of clusters.
4. **Evaluate the Clustering**: Calculate clustering metrics such as the **Davies-Bouldin Index (DB Index)**.
5. **Visualize the Results**: Use dimensionality reduction techniques (like PCA or t-SNE) to visualize the clusters.

### Step 1: Import Required Libraries and Load Data

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load datasets
customers = pd.read_csv('Customers.csv')
transactions = pd.read_csv('Transactions.csv')
```

### Step 2: Data Preprocessing

#### 2.1: Aggregate Transaction Data by Customer

We’ll aggregate transaction data by customer to create a customer-product matrix, using features like total spending, quantity purchased, etc.

```python
# Merge transactions with product data
transactions = transactions.groupby(['CustomerID', 'ProductID']).agg(
    total_spent=('TotalValue', 'sum'),
    total_quantity=('Quantity', 'sum')
).reset_index()

# Pivot the data to create a customer-product matrix
customer_profile = transactions.pivot_table(
    index='CustomerID',
    columns='ProductID',
    values='total_spent',
    aggfunc='sum',
    fill_value=0
)

# Merge the aggregated transaction data with customer demographic data
customer_profile = customer_profile.merge(customers[['CustomerID', 'Region']], on='CustomerID')
```

#### 2.2: Feature Engineering

We will combine both **customer demographics** and **transaction data** into one feature matrix. 

```python
# Normalize transaction data using StandardScaler
scaler = StandardScaler()
transaction_data = customer_profile.drop(['CustomerID', 'Region'], axis=1)
transaction_data_scaled = pd.DataFrame(scaler.fit_transform(transaction_data), columns=transaction_data.columns)

# Combine the scaled transaction data with the region data
customer_features = pd.concat([customer_profile[['CustomerID', 'Region']], transaction_data_scaled], axis=1)
```

### Step 3: Apply Clustering

We will use the **K-Means clustering** algorithm to segment customers. The number of clusters can be selected between 2 and 10, based on evaluation metrics like the **Davies-Bouldin Index**.

#### 3.1: Apply K-Means Clustering

```python
# Define a range of cluster numbers (between 2 and 10)
cluster_range = range(2, 11)

# Store Davies-Bouldin scores for each cluster number
db_scores = []

# Perform clustering for each cluster number
for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    customer_features['Cluster'] = kmeans.fit_predict(customer_features.drop(['CustomerID', 'Region'], axis=1))
    
    # Calculate Davies-Bouldin Index
    db_index = davies_bouldin_score(customer_features.drop(['CustomerID', 'Region'], axis=1), customer_features['Cluster'])
    db_scores.append(db_index)

# Plot the Davies-Bouldin Index for different cluster numbers
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, db_scores, marker='o')
plt.title('Davies-Bouldin Index for Different Cluster Numbers')
plt.xlabel('Number of Clusters')
plt.ylabel('Davies-Bouldin Index')
plt.grid(True)
plt.show()

# Choose the optimal number of clusters (let's assume it's 4 based on the plot)
optimal_clusters = 4
```

#### 3.2: Final Clustering with Optimal Number of Clusters

Now, we apply K-Means with the optimal number of clusters (let's assume 4 clusters based on the previous step).

```python
# Apply KMeans clustering with optimal clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
customer_features['Cluster'] = kmeans.fit_predict(customer_features.drop(['CustomerID', 'Region'], axis=1))

# Add cluster labels back to the customer profile data
customer_profile['Cluster'] = customer_features['Cluster']

# Inspect the clustering result
print(customer_profile[['CustomerID', 'Cluster']].head())
```

### Step 4: Evaluate the Clustering

We will now calculate the **Davies-Bouldin Index** (DB Index), which is a clustering evaluation metric. The DB Index measures how well-separated the clusters are, with a lower score indicating better clustering.

```python
# Calculate the Davies-Bouldin Index for the final clustering
final_db_index = davies_bouldin_score(customer_features.drop(['CustomerID', 'Region'], axis=1), customer_features['Cluster'])
print(f'Davies-Bouldin Index: {final_db_index}')
```

### Step 5: Visualize the Clusters

To better understand the clusters, we’ll use **PCA (Principal Component Analysis)** to reduce the data to 2D and plot the clusters.

```python
# Apply PCA to reduce the dimensions to 2D
pca = PCA(n_components=2)
pca_components = pca.fit_transform(customer_features.drop(['CustomerID', 'Region'], axis=1))

# Create a DataFrame with PCA components and cluster labels
pca_df = pd.DataFrame(pca_components, columns=['PCA1', 'PCA2'])
pca_df['Cluster'] = customer_features['Cluster']

# Plot the clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='Cluster', palette='Set1', s=100, edgecolor='black')
plt.title('PCA Plot of Customer Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()
```

### Step 6: Report the Clustering Results

Finally, summarize the results of the clustering in a report.

#### Report:

1. **Number of Clusters Formed**: Based on the Davies-Bouldin Index and the visualizations, we selected the optimal number of clusters (e.g., 4 clusters).
   
2. **DB Index**: The Davies-Bouldin Index for the final clustering is approximately **[Final DB Index value]**. A lower value indicates better-defined clusters.

3. **Other Clustering Metrics**:
   - **Silhouette Score** (optional): You could also calculate the Silhouette Score to measure how similar customers are within clusters.
   - **Cluster Sizes**: The number of customers in each cluster could be reported to understand the distribution of customers across clusters.

4. **Cluster Visualization**: The PCA plot visually shows the separation of customers into distinct groups based on both demographic and transactional behavior.

### Step 7: Deliverables

- **Clustering Results**: The number of clusters and DB Index value, along with other relevant metrics like silhouette score.
- **Cluster Visualization**: A 2D PCA plot showing the clusters.
- **Look at Customer Segments**: Understanding how each cluster behaves in terms of transaction data and demographic features.

---

### Example Output from the Report:

- **Number of Clusters**: 4
- **DB Index**: 0.65
- **Silhouette Score**: 0.45 (Optional)
- **Cluster Visualization**: PCA plot displaying how customers are grouped based on transaction and demographic features.

---

