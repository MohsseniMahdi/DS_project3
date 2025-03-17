# Credit Card Fraud Detection Project

## Introduction
Credit card fraud represents a major risk to both financial institutions and consumers, causing significant financial losses annually. Detecting fraudulent transactions accurately is essential to reducing these losses and preserving trust in financial systems. This project is focused on developing a machine learning model that can identify fraudulent credit card transactions effectively. The dataset used for this project is provided by Kaggle, offering a foundation for data analysis and model building.

## Dataset Overview
The dataset features credit card transactions made by European cardholders over a two-day period in September 2013. It contains 284,807 transactions, of which only 492 are fraudulent—approximately 0.172% of the data. This notable class imbalance presents challenges during model training and evaluation, requiring careful attention to ensure fair and reliable model performance.

## Feature Description
**Time**: The time elapsed in seconds between each transaction and the first transaction in the dataset.

**V1** to **V28**: Principal components derived from a PCA transformation applied to the original features, anonymized for privacy reasons.

**Amount**: The transaction amount, which can provide insights and be utilized for cost-sensitive learning.

**Class**: The target variable, where 1 indicates a fraudulent transaction, and 0 indicates a legitimate transaction.

## Data Preprocessing
Effective preprocessing is essential to address the class imbalance and prepare the data for modeling. The following steps were undertaken:

### Loading the Data

The dataset was loaded using Pandas:

1. > **import pandas as pd**: The initial step involves loading the credit card transaction dataset from a CSV file into a pandas DataFrame named data1. The pd.read_csv() function is utilized for this purpose, reading the data from the specified file path '../raw_data/creditcard.csv'.

```python
data1 = pd.read_csv('../raw_data/creditcard.csv')
```
2. > **Verifying the absence of missing values to ensure data integrity.**:

```python
### **Data Integrity Check: Missing Values**

df.isnull().sum().sum()
```

>> - Verifying the absence of missing values to ensure data integrity.
After loading and initially inspecting the data, it's crucial to check for missing values. The code above utilizes the pandas isnull() and sum() methods to determine the total number of missing values within the DataFrame df.
>> - Specifically, df.isnull() creates a boolean mask indicating the presence of missing values (True) or their absence (False) for each element in the DataFrame. Then, .sum() is applied twice: first to sum the boolean values along each column (resulting in the count of missing values per column), and then a second time to sum those column-wise sums, yielding the grand total of missing values in the entire DataFrame.
>> - The result of this operation was 0. This 0 indicates that there are no missing values present in the DataFrame df. This is a critical verification step, as missing values can significantly impact the performance of machine learning models. Therefore, confirming the absence of missing data is a fundamental prerequisite for reliable model training and evaluation.

3. > **Identifying and removing duplicate entries to improve data quality**:
>> - This code snippet focuses on identifying and quantifying duplicate rows within the DataFrame df. The pandas duplicated() method returns a boolean Series indicating whether each row is a duplicate of a previous row. The .sum() method then counts the number of True values, effectively giving the total number of duplicate rows.

### **Data Cleaning: Duplicate Row Removal**

```python
duplicate_rows = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_rows}")

>>> Number of duplicate rows: 1081
```

>> - The output Number of duplicate rows: 1081 indicates that the original DataFrame contained 1081 duplicate rows. These duplicates can introduce bias and redundancy into the dataset, potentially affecting the performance of subsequent analysis and modeling.

>> - Following the identification of duplicate rows, the drop_duplicates() method is used to remove these entries from the DataFrame df. The reset_index(drop=True) command re-indexes the DataFrame after the removal, ensuring a contiguous index and dropping the old index.

```python
df = df.drop_duplicates().reset_index(drop=True)
print(f"Number of duplicate rows: {duplicate_rows}")

>>> Number of duplicate rows: 0
```
>> - The subsequent output Number of duplicate rows: 0 confirms that the duplicate rows have been successfully removed, leaving a clean dataset for further preprocessing and analysis. The fact that the first print statement saved the initial number of duplicates and the second print showed 0, validates the success of the removal.

4- > **Converting time-based data into cyclical features to capture temporal patterns**:
>> - This code segment focuses on transforming the 'Time' feature, which represents the seconds elapsed since the first transaction, into a more meaningful 'Hour' feature and then applying a cyclical transformation.

### **Feature Engineering: Cyclical Transformation of Time**

```python
df['Hour'] = (df['Time'] // 3600) % 24

df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

# Plot the number of transactions by Hour
plt.figure(figsize=(10, 4))
sns.countplot(x=df['Hour'])
plt.title("Transaction Count by Hour")
plt.show()

df.drop(columns=["Hour"], inplace=True)
```

>> - First, df['Hour'] = (df['Time'] // 3600) % 24 calculates the hour of the day from the 'Time' feature. The // operator performs integer division to get the number of hours, and % 24 ensures the hour wraps around to 0 after 23, effectively representing the 24-hour clock.
>> -Next, df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24) and df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24) apply a cyclical transformation using sine and cosine functions. This transformation is crucial for machine learning models to understand the cyclical nature of time (e.g., 23:00 is closer to 00:00 than to 12:00).
>> - A count plot is then generated using seaborn to visualize the distribution of transactions across different hours of the day. This plot helps to understand the transaction patterns and potential time-based trends in the data.
>> - Finally, df.drop(columns=["Hour"], inplace=True) removes the original 'Hour' feature, as its cyclical representation (Hour_sin and Hour_cos) now captures the relevant temporal information.
>> - This cyclical transformation allows the model to properly understand the relationship between time and fraudulent transactions, as it encodes the proximity of the ends of the day.

![Credit Card Fraud Detection](./images/Transaction_Count_by_Hour.png)





# Project Name
- Document your project here
- Description
- Data used
- Where your API can be accessed
- ...

# API
Document main API endpoints here

# Setup instructions
Document here for users who want to setup the package locally

# Usage
Document main functionalities of the package here


# installation

make install requirments
make run preprocess
