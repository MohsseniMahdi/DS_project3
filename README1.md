
7. > **Addressing Class Imbalance with Advanced Techniques**:

>> - One of the primary challenges when working with the Credit Card Fraud Detection dataset is the **highly imbalanced nature of the data**, where fraudulent transactions make up only 0.172% of the total. This imbalance can lead machine learning models to favor the majority class, resulting in poor performance when identifying fraudulent transactions. Such bias is harmful because failing to detect fraud accurately can lead to significant financial losses and damage to consumer trust. It also highlights the importance of employing strategies to rebalance the data, ensuring the model's fairness and reliability.

>> - Imbalanced datasets cause a disproportionate influence of the majority class during model training. This results in:
>>> - A high number of false negatives (failing to detect fraudulent transactions).
>>> - Misleading accuracy scores, as the model might predict the majority class most of the time and still appear accurate.

### Introductory Information for the Code

>> - To address the challenge of data imbalance, the project utilizes **BorderlineSMOTE**, a variation of SMOTE (Synthetic Minority Oversampling Technique). BorderlineSMOTE focuses on generating synthetic samples near the decision boundary, where misclassification is more likely. This targeted approach strengthens the model's ability to distinguish fraudulent transactions effectively.

>> - In the provided code snippet:
>>> - **BorderlineSMOTE** is applied to rebalance the training dataset by generating synthetic samples for the minority class.
>>> - After applying BorderlineSMOTE, the distribution of the target variable (`y_train_smote`) is displayed, showing the impact of oversampling.

```python
# Apply BorderlineSMOTE (instead of regular SMOTE)
smote = BorderlineSMOTE(sampling_strategy=0.3, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("\nClass distribution after SMOTE:\n", pd.Series(y_train_smote).value_counts())

#output
# Class distribution after SMOTE:
# 0    226602
# 1     67980
# Name: Class, dtype: int64
```

>> ### 🔍 Why Use BorderlineSMOTE & Tomek Links?

>> - Combining **BorderlineSMOTE** with **Tomek Links** provides a comprehensive approach to handling imbalanced data:
- **BorderlineSMOTE**:
  - Generates synthetic samples only near the decision boundary, improving robustness in distinguishing fraud from non-fraud cases.
  - Reduces the risk of overlapping classes, a common issue with regular SMOTE.
  - Works best in scenarios like this, where fraud cases are rare and lie close to the class boundary.
- **Tomek Links**:
  - Removes ambiguous samples that are close to the opposite class, further refining the dataset for better model performance.

This combined strategy ensures that the training data is both balanced and cleansed of noisy samples, resulting in a more accurate and reliable fraud detection model.

- [**Go to main page**](./README.md)

















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
