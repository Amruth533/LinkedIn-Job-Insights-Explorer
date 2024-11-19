import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Data Loading (Using Absolute Path)
base_path = '/Users/apple/Desktop/AIT664/OneDrive_1_19-9-2023/Datasets/all/'

# Load the datasets from the specified path
job_postings = pd.read_csv(f'{base_path}job_postings.csv')
benefits = pd.read_csv(f'{base_path}benefits.csv')
job_skills = pd.read_csv(f'{base_path}job_skills.csv')
companies = pd.read_csv(f'{base_path}companies.csv')
company_industries = pd.read_csv(f'{base_path}company_industries.csv')
company_specialities = pd.read_csv(f'{base_path}company_specialities.csv')
employee_counts = pd.read_csv(f'{base_path}employee_counts.csv')

# Print confirmation messages to verify datasets are loaded successfully
print("Datasets loaded successfully:")
print(f"Job Postings: {job_postings.shape}")
print(f"Benefits: {benefits.shape}")
print(f"Job Skills: {job_skills.shape}")
print(f"Companies: {companies.shape}")
print(f"Company Industries: {company_industries.shape}")
print(f"Company Specialities: {company_specialities.shape}")
print(f"Employee Counts: {employee_counts.shape}")



# Step 2: Data Merging
# Merge job-related datasets
# Group and merge benefits and skills with job postings
benefits = benefits.groupby('job_id')['type'].agg(lambda x: ', '.join(x)).reset_index()
job_skills = job_skills.groupby('job_id')['skill_abr'].agg(lambda x: ', '.join(x)).reset_index()

job_postings = job_postings.merge(benefits, on='job_id', how='left')
job_postings = job_postings.merge(job_skills, on='job_id', how='left')

# Group and merge company-related datasets
company_industries = company_industries.groupby('company_id')['industry'].agg(lambda x: ', '.join(x)).reset_index()
company_specialities = company_specialities.groupby('company_id')['speciality'].agg(lambda x: ', '.join(x)).reset_index()
employee_counts = employee_counts.groupby('company_id')['employee_count'].max().reset_index()

companies = companies.merge(company_industries, on='company_id', how='left')
companies = companies.merge(company_specialities, on='company_id', how='left')
companies = companies.merge(employee_counts, on='company_id', how='left')

# Merge job postings with company data
linkedin_data = job_postings.merge(companies, on='company_id', how='left')

# Display the first few rows of the merged dataset
print(linkedin_data.head())


#Data Cleaning
#fill missing values and remove duplicates
linkedin_data.fillna('NA', inplace=True)
linkedin_data.drop_duplicates(inplace=True)


# Step 4: Data Analysis and Visualizations
# Filter out rows with missing company names
filtered_data = linkedin_data[linkedin_data['name'] != 'NA']

# Top 10 Companies by Job Postings (excluding 'NA')
top_companies = filtered_data['name'].value_counts().head(10)
plt.figure(figsize=(10, 6))
top_companies.plot(kind='bar', color='skyblue')
plt.title('Top 10 Companies with Most Job Postings (Excluding NA)')
plt.xlabel('Company Name')
plt.ylabel('Number of Job Postings')
plt.xticks(rotation=45)
plt.show()

# Check for missing values and print the first few rows
print(linkedin_data['skill_abr'].head(10))
print(linkedin_data['skill_abr'].isnull().sum())

# Fill missing values with 'Unknown'
linkedin_data['skill_abr'] = linkedin_data['skill_abr'].fillna('Unknown')

# Split the skills, explode the list, and get the top 10 skills
top_skills = linkedin_data['skill_abr'].str.split(', ').explode().value_counts().head(10)

# Print the top 10 skills to verify
print(top_skills)


print(linkedin_data.columns.tolist())
#top 10 job titles in the dataset
import matplotlib.pyplot as plt
import seaborn as sns

# Count the top 10 job titles
top_job_titles = linkedin_data['title'].value_counts().head(10)

# Plot the top 10 job titles
plt.figure(figsize=(12, 6))
sns.barplot(x=top_job_titles.values, y=top_job_titles.index, palette='viridis')
plt.title('Top 10 Job Titles by Frequency')
plt.xlabel('Frequency')
plt.ylabel('Job Title')
plt.show()

#distribution of salary
# Filter out rows with missing or zero median salary values
linkedin_data['med_salary'] = pd.to_numeric(linkedin_data['med_salary'], errors='coerce')
linkedin_data = linkedin_data[linkedin_data['med_salary'] > 0]

# Plot the distribution of median salary
plt.figure(figsize=(10, 6))
sns.histplot(linkedin_data['med_salary'], kde=True, color='blue')
plt.title('Distribution of Median Salary')
plt.xlabel('Median Salary')
plt.ylabel('Frequency')
plt.show()

# Count the top 10 companies by number of job postings
top_companies = linkedin_data['name'].value_counts().head(10)

# Plot the top 10 companies
plt.figure(figsize=(12, 6))
sns.barplot(x=top_companies.values, y=top_companies.index, palette='coolwarm')
plt.title('Top 10 Companies by Number of Job Postings')
plt.xlabel('Number of Job Postings')
plt.ylabel('Company Name')
plt.show()

# Count the number of remote vs on-site jobs
remote_job_counts = linkedin_data['remote_allowed'].value_counts()

# Plot the distribution of remote vs on-site jobs
plt.figure(figsize=(8, 6))
sns.barplot(x=remote_job_counts.index, y=remote_job_counts.values, palette='pastel')
plt.title('Remote vs On-Site Jobs')
plt.xlabel('Remote Allowed')
plt.ylabel('Frequency')
plt.show()

#job views vs application scatter plot
# Convert 'views' and 'applies' columns to numeric
linkedin_data['views'] = pd.to_numeric(linkedin_data['views'], errors='coerce')
linkedin_data['applies'] = pd.to_numeric(linkedin_data['applies'], errors='coerce')

# Plot a scatter plot of job views vs. applications
plt.figure(figsize=(10, 6))
sns.scatterplot(data=linkedin_data, x='views', y='applies', color='purple')
plt.title('Job Views vs Applications')
plt.xlabel('Job Views')
plt.ylabel('Number of Applications')
plt.show()

work_type_counts = linkedin_data['formatted_work_type'].value_counts()

# Plot the distribution of job postings by work type
plt.figure(figsize=(10, 6))
sns.barplot(x=work_type_counts.index, y=work_type_counts.values, palette='Set2')
plt.title('Distribution of Job Postings by Work Type')
plt.xlabel('Work Type')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# Count the number of job postings by city
top_cities = linkedin_data['city'].value_counts().head(10)

# Plot the top 10 cities by number of job postings
plt.figure(figsize=(12, 6))
sns.barplot(x=top_cities.values, y=top_cities.index, palette='Spectral')
plt.title('Top 10 Cities by Number of Job Postings')
plt.xlabel('Number of Job Postings')
plt.ylabel('City')
plt.show()

#logistic regression

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Define the Target Variable
linkedin_data['sponsored'] = pd.to_numeric(linkedin_data['sponsored'], errors='coerce').fillna(0).astype(int)

# Step 2: Select Predictor Variables (Features)
features = ['title', 'name', 'location', 'formatted_work_type', 'views', 'applies']
X = linkedin_data[features]
y = linkedin_data['sponsored']

# Step 3: Encode Categorical Features
label_encoders = {}
for feature in ['title', 'name', 'location', 'formatted_work_type']:
    le = LabelEncoder()
    X[feature] = le.fit_transform(X[feature].astype(str))
    label_encoders[feature] = le

# Step 4: Handle Missing Values for Numeric Features
X['views'] = pd.to_numeric(X['views'], errors='coerce').fillna(0)
X['applies'] = pd.to_numeric(X['applies'], errors='coerce').fillna(0)

# Step 5: Standardize Numeric Features
scaler = StandardScaler()
X[['views', 'applies']] = scaler.fit_transform(X[['views', 'applies']])

# Step 6: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Initialize and Train the Logistic Regression Model
model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)


# Step 8: Make Predictions and Evaluate the Model
predictions = model.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Logistic Regression Accuracy: {accuracy:.2f}')

# Print Classification Report
print("Classification Report:")
print(classification_report(y_test, predictions))

# Plot Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#more evaluation metrics
from sklearn.metrics import roc_curve, roc_auc_score


# Plot the ROC curve
# Calculate the predicted probabilities for the positive class
y_prob = model.predict_proba(X_test)[:, 1]

# Verify the predicted probabilities
print(y_prob[:10])

from sklearn.metrics import roc_curve, roc_auc_score

# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

# Plot the ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

print(f'AUC Score: {roc_auc:.2f}')

#random forest model

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize the Random Forest Classifier with default parameters
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
rf_predictions = rf_model.predict(X_test)

# Evaluate the model
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f'Random Forest Accuracy: {rf_accuracy:.2f}')

# Classification Report
print("Classification Report:")
print(classification_report(y_test, rf_predictions))

# Confusion Matrix
rf_conf_matrix = confusion_matrix(y_test, rf_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(rf_conf_matrix, annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix for Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve and AUC
rf_probs = rf_model.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_probs)

fpr, tpr, _ = roc_curve(y_test, rf_probs)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', label=f'ROC Curve (AUC = {rf_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title('ROC Curve for Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

print(f'Random Forest AUC Score: {rf_auc:.2f}')

#feature importance values 
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(feature_importances)

#finetuning the model 
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2, 4]
}

rf_tuned = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=10,  # Reduced iterations for speed
    scoring='roc_auc',
    cv=3,
    random_state=42
)

rf_tuned.fit(X_train, y_train)
print(f"Best Parameters: {rf_tuned.best_params_}")


