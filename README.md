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

- 1. import pandas as pd

data1 = pd.read_csv('../raw_data/creditcard.csv')
df = data1.copy()











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
