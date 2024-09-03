
# **Microsoft : Classifying Cybersecurity Incidents with Machine Learning**
## Skills take away from this Project
Data Preprocessing and Feature Engineering

Machine Learning Classification Techniques

Model Evaluation Metrics (Macro-F1 Score, Precision, Recall)

Cybersecurity Concepts and Frameworks (MITRE ATT&CK)

Handling Imbalanced Datasets

Model Benchmarking and Optimization

## Domain
Cybersecurity and Machine Learning

## Tools Used 
**IDE & Notebooks:** PyCharm, Jupyter Notebook

**Programming Language**: Python

**Libraries**: scikit-learn, Pandas, Matplotlib, Seaborn, NumPy

**Cloud Services:** BigQuery, Google Cloud Storage, Google Compute Engine

**Version Control**: Git, Github

## Dataset Overview
GUIDE_train.csv (2.43 GB)
GUIDE_test.csv (1.09 GB)

The dataset provides three hierarchies of data: (1) evidence, (2) alert, and (3) incident. At the bottom level, evidence supports an alert. For example, an alert may be associated with multiple pieces of evidence such as an IP address, email, and user details, each containing specific supporting metadata. Above that, we have alerts that consolidate multiple pieces of evidence to signify a potential security incident. These alerts provide a broader context by aggregating related evidences to present a more comprehensive picture of the potential threat. At the highest level, incidents encompass one or more alerts, representing a cohesive narrative of a security breach or threat scenario.
The primary objective of the dataset is to accurately predict incident triage grades—true positive (TP), benign positive (BP), and false positive (FP)—based on historical customer responses. To support this, we provide a training dataset containing 45 features, labels, and unique identifiers across 1M triage-annotated incidents. We divide the dataset into a train set containing 70% of the data and a test set with 30%, stratified based on triage grade ground-truth, OrgId, and DetectorId. We ensure that incidents are stratified together within the train and test sets to ensure the relevance of evidence and alert rows.

## Problem Statement:
As a data scientist at Microsoft, he/she is tasked with enhancing the efficiency of Security Operation Centers (SOCs) by developing a machine learning model that can accurately predict the triage grade of cybersecurity incidents. Utilizing the comprehensive GUIDE dataset, your goal is to create a classification model that categorizes incidents as true positive (TP), benign positive (BP), or false positive (FP) based on historical evidence and customer responses. The model should be robust enough to support guided response systems in providing SOC analysts with precise, context-rich recommendations, ultimately improving the overall security posture of enterprise environments.

## Business Use Cases:
The solution developed in this project can be implemented in various business scenarios, particularly in the field of cybersecurity. Some potential applications include:

**Security Operation Centers (SOCs)**: Automating the triage process by accurately classifying cybersecurity incidents, thereby allowing SOC analysts to prioritize their efforts and respond to critical threats more efficiently.

**Incident Response Automation**: Enabling guided response systems to automatically suggest appropriate actions for different types of incidents, leading to quicker mitigation of potential threats.

**Threat Intelligence**: Enhancing threat detection capabilities by incorporating historical evidence and customer responses into the triage process, which can lead to more accurate identification of true and false positives.

**Enterprise Security Management**: Improving the overall security posture of enterprise environments by reducing the number of false positives and ensuring that true threats are addressed promptly.

## Approach:
### Data Exploration and Understanding:

**Initial Inspection**: Start by loading the GUIDE_train.csv dataset and perform an initial inspection to understand the structure of the data, including the number of features, types of variables (categorical, numerical), and the distribution of the target variable (TP, BP, FP).

**Exploratory Data Analysis** (EDA): Use visualizations and statistical summaries to identify patterns, correlations, and potential anomalies in the data. Pay special attention to class imbalances, as they may require specific handling strategies later on.

### Data Preprocessing:

**Handling Missing Data**: Identify any missing values in the dataset and decide on an appropriate strategy, such as imputation, removing affected rows, or using models that can handle missing data inherently.

**Feature Engineering**: Create new features or modify existing ones to improve model performance. For example, combining related features, deriving new features from timestamps (like hour of the day or day of the week), or normalizing numerical variables.

**Encoding Categorical Variables**: Convert categorical features into numerical representations using techniques like one-hot encoding, label encoding, or target encoding, depending on the nature of the feature and its relationship with the target variable.

### Data Splitting:

**Train-Validation Split**: Before diving into model training, split the train.csv data into training and validation sets. This allows for tuning and evaluating the model before final testing on test.csv. Typically, a 70-30 or 80-20 split is used, but this can vary depending on the dataset's size.

**Stratification**: If the target variable is imbalanced, consider using stratified sampling to ensure that both the training and validation sets have similar class distributions.

### Model Selection and Training:

**Baseline Model**: Start with a simple baseline model, such as a logistic regression or decision tree, to establish a performance benchmark. This helps in understanding how complex the model needs to be.

**Advanced Models**: Experiment with more sophisticated models such as Random Forests, Gradient Boosting Machines (e.g., XGBoost, LightGBM), and Neural Networks. Each model should be tuned using techniques like grid search or random search over hyperparameters.

**Cross-Validation**: Implement cross-validation (e.g., k-fold cross-validation) to ensure the model's performance is consistent across different subsets of the data. This reduces the risk of overfitting and provides a more reliable estimate of the model's performance.

### Model Evaluation and Tuning:

**Performance Metrics**: Evaluate the model using the validation set, focusing on macro-F1 score, precision, and recall. Analyze these metrics across different classes (TP, BP, FP) to ensure balanced performance.

**Hyperparameter Tuning**: Based on the initial evaluation, fine-tune hyperparameters to optimize model performance. This may involve adjusting learning rates, regularization parameters, tree depths, or the number of estimators, depending on the model type.

**Handling Class Imbalance**: If class imbalance is a significant issue, consider techniques such as SMOTE (Synthetic Minority Over-sampling Technique), adjusting class weights, or using ensemble methods to boost the model's ability to handle minority classes effectively.

### Model Interpretation:

**Feature Importance**: After selecting the best model, analyze feature importance to understand which features contribute most to the predictions. This can be done using methods like SHAP values, permutation importance, or model-specific feature importance measures.

**Error Analysis**: Perform an error analysis to identify common misclassifications. This can provide insights into potential improvements, such as additional feature engineering or refining the model's complexity.

### Final Evaluation on Test Set:

**Testing**: Once the model is finalized and optimized, evaluate it on the test.csv dataset. Report the final macro-F1 score, precision, and recall to assess how well the model generalizes to unseen data.

**Comparison to Baseline**: Compare the performance on the test set to the baseline model and initial validation results to ensure consistency and improvement.

### Documentation and Reporting:

**Model Documentation**: Thoroughly document the entire process, including the rationale behind chosen methods, challenges faced, and how they were addressed. Include a summary of key findings and model performance.

**Recommendations**: Provide recommendations on how the model can be integrated into SOC workflows, potential areas for future improvement, and considerations for deployment in a real-world setting.

## Link to the notebook file
You can view the code and in depth analysis in the full notebooks

Part1: [here](./MSCyberProj-Part1.ipynb)

Part2: [here](./MSCyberProj-Part2.ipynb)

## Utilization of Cloud 

**Google Cloud**: Utilized Google Cloud Storage to store the huge data

**BigQuery**: Utlized to fetch data from Google Cloud Storage into BigQuery to then be queried in jupyter notebook on cloud instance

**Google Compute Engine**: Needed to utilize Google Cloud Compute instance due to the size of the dataset

## Results and Model Performance Analysis

**Training Dataset Performance:**

Trained using ensemble methods XGBoost and Random Forest, Random Forest performs the best

![CHEESE!](RF_train_metrics.png)
![CHEESE!](XGB_train_metrics.png)

**Test Dataset Performance:**

Thus selected Random Forest Classifier and used it on the test dataset

![CHEESE!](RF_test_metrics.png)

**Inferences:**

**High Training Performance**: The model exhibits very high performance on the training dataset, indicating it has learned the patterns in the data well.

**Good Generalization**: The model performs robustly on the test dataset, suggesting it generalizes well to new, unseen data. The slight decrease in accuracy from training to testing is typical and indicates good generalization without significant overfitting.

**Class-wise Variations**: While the model maintains strong performance across most classes, there is a noticeable drop in performance for benign positive incidents in the test set. This could be an area for further investigation and improvement.
Overall, the Random Forest model demonstrates strong capabilities in classifying cybersecurity incidents, with good generalization to real-world data. Future improvements could focus on enhancing performance for specific classes and continuing to monitor and adjust the model as more data becomes available.

## Recommendations

**Integration into SOC Workflows:**

  -**Enhanced Incident Triage:** Integrate the model into SOC workflows to automate and refine the incident triage process, providing SOC analysts with precise classifications of incidents as TP, BP, or FP.
 
  -**Real-time Analysis:** Deploy the model in real-time environments to assist in immediate incident response, helping analysts prioritize and address security threats more effectively.

**Considerations for Deployment:**

  -**Scalability**: Ensure the model can handle large volumes of data in a production environment, potentially leveraging scalable cloud infrastructure.
 
  -**Real-world Testing:** Conduct extensive testing in a real-world setting to validate model performance and address any operational challenges.
  
  -**Feedback Loop:** Implement a feedback loop to capture analyst insights and adjust the model based on real-world usage and performance metrics.
