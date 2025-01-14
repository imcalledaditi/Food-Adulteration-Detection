import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.metrics import f1_score as f1_score_func
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings('ignore')
df = pd.read_csv(r'C:\Users\idey7\OneDrive\Desktop\MLDA FOOD ADULTERATION\cleaned_food_adulteration_data.csv')
df

df.drop(columns=['adulteration_id', 'detection_date'], inplace=True)

for col in df.columns:
    data = df[col].value_counts().reset_index()
    plt.figure(figsize=(10,6))
    plt.title(col)
    sns.barplot(data=data, x=col, y='count', palette='crest')
    plt.xlabel('')
    plt.ylabel('')
    plt.show()

def my_heatmap(title, data):
    plt.figure(figsize=(10,8))
    plt.title(title)
    sns.heatmap(data=data, annot=True, cmap='coolwarm')
    plt.xlabel('')
    plt.ylabel('')
    plt.yticks(rotation=0)
    plt.show()

for col in ['adulterant', 'detection_method']:
    data = pd.crosstab(df[col], df['category'])
    my_heatmap(f'Category & {col}', data)



#dataset is separated by tab, so we use seperator='\t'
df = pd.read_csv(r'C:\Users\idey7\OneDrive\Desktop\MLDA FOOD ADULTERATION\cleaned_food_adulteration_data.csv',sep=',', encoding='latin-1')

df.head()

df.info()

# Load the original dataset
file_path = r'C:\Users\idey7\OneDrive\Desktop\MLDA FOOD ADULTERATION\food_adulteration_data.csv'  # Update with your file path
data = pd.read_csv(file_path)

# Define the ranges for the new rows
products = ['Juice', 'Chicken', 'Milk', 'Bread', 'Yogurt', 'Wine', 'Beef', 'Honey', 'Cheese', 'Butter']
brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE']
adulterants = ['Coloring agents', 'Chalk', 'Artificial sweeteners', 'Melamine', 'Water']
severities = ['Minor', 'Moderate', 'Severe']
category= ['Meat', 'Dairy', 'Bakery', 'Beverages', 'Condiments']
detection_method= ['Spectroscopy', 'Microbial Analysis', 'Chemical Analysis', 'Sensory Evaluation']
health_risk= ['Low', 'Medium', 'High']
action_taken= ['Investigation Launched', 'Product Recall', 'Fine Imposed', 'Warning Issued']

# Function to generate random rows
def generate_random_row():
    return {
        'product_name': np.random.choice(products),
        'brand': np.random.choice(brands),
        'adulterant': np.random.choice(adulterants),
        'severity': np.random.choice(severities),
        'category': np.random.choice(category),
        'detection_method': np.random.choice(detection_method),
        'health_risk': np.random.choice(health_risk),
        'action_taken': np.random.choice(action_taken)

    }

# Generate new rows to triple the original dataset (in addition to the original data)
num_original_rows = data.shape[0]
new_rows_1 = [generate_random_row() for _ in range(num_original_rows)]
new_rows_2 = [generate_random_row() for _ in range(num_original_rows)]
new_rows_3 = [generate_random_row() for _ in range(num_original_rows)]

new_data_1 = pd.DataFrame(new_rows_1)
new_data_2 = pd.DataFrame(new_rows_2)
new_data_3 = pd.DataFrame(new_rows_3)

# Concatenate the original data with the new data
quadrupled_data = pd.concat([data, new_data_1, new_data_2, new_data_3], ignore_index=True)

# Save the quadrupled dataset to a new CSV file
quadrupled_data.to_csv('food_adulteration_data1.csv', index=False)

# Load the dataset
file_path = r'C:\Users\idey7\OneDrive\Desktop\MLDA FOOD ADULTERATION\cleaned_food_adulteration_data.csv'
data = pd.read_csv(file_path)

# Display basic information about the dataset
print(data.info())
print(data.describe())

# Handle missing values (if any)
data = data.dropna()  # Remove rows with missing values
# Alternatively, you can fill missing values
# data = data.fillna(data.mean())

# Assuming numerical columns to check for outliers
numerical_cols = data.select_dtypes(include=[np.number]).columns

# Calculate Z-scores
z_scores = np.abs(stats.zscore(data[numerical_cols]))

# Define a threshold for identifying outliers
threshold = 3

# Remove outliers
data_no_outliers = data[(z_scores < threshold).all(axis=1)]

# Save the cleaned dataset to a new CSV file
data_no_outliers.to_csv('cleaned_food_adulteration_data.csv', index=False)

print("Preprocessing complete. Outliers removed and data saved to 'cleaned_food_adulteration_data.csv'.")

data1= pd.read_csv(r'C:\Users\idey7\OneDrive\Desktop\MLDA FOOD ADULTERATION\cleaned_food_adulteration_data.csv')

df_filtered = data1[['product_name', 'brand', 'adulterant', 'severity']]

# Display the new table
print(df_filtered.head())

# Encode categorical variables
le_product_name = LabelEncoder()
le_brand = LabelEncoder()
le_adulterant = LabelEncoder()
le_severity = LabelEncoder()

df_filtered['product_name'] = le_product_name.fit_transform(df_filtered['product_name'])
df_filtered['brand'] = le_brand.fit_transform(df_filtered['brand'])
df_filtered['severity'] = le_severity.fit_transform(df_filtered['severity'])
df_filtered['adulterant'] = le_adulterant.fit_transform(df_filtered['adulterant'])


X = df_filtered[['product_name', 'brand', 'adulterant', ]]
y = df_filtered['severity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log_reg))
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_log_reg))

# SVM
svm_clf = SVC(kernel='linear')
svm_clf.fit(X_train, y_train)
y_pred_svm = svm_clf.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))

knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train, y_train)


y_pred_knn = knn_clf.predict(X_test)


print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("KNN Classification Report:\n", classification_report(y_test, y_pred_knn))

# Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)


y_pred_rf = rf_clf.predict(X_test)


print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

models = ['Logistic Regression', 'SVM', 'KNN', 'Random Forest']
accuracy = [ 0.19, 0.195, 0.23, 0.25]  # Example values
precision = [0.196, 0.148, 0.234, 0.262]
recall = [0.188, 0.188, 0.24, 0.264]
f1_score = [0.174, 0.16, 0.22, 0.25]

# Plotting the metrics
x = np.arange(len(models))  # Label locations
width = 0.2  # Width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width*1.5, accuracy, width, label='Accuracy')
ax.bar(x - width/2, precision, width, label='Precision')
ax.bar(x + width/2, recall, width, label='Recall')
ax.bar(x + width*1.5, f1_score, width, label='F1-Score')

# Add labels, title, and customize the plot
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Show the plot
plt.show()

# Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict the target values for the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score_func(y_test, y_pred, average='weighted')
report = classification_report(y_test, y_pred)

print("Random Forest Model Accuracy:", accuracy)
print("Random Forest Model F1 Score:", f1)
print("Random Forest Model Classification Report:\n", report)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log_reg))
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_log_reg))

# SVM
svm_clf = SVC(kernel='linear')
svm_clf.fit(X_train, y_train)
y_pred_svm = svm_clf.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))

knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train, y_train)


y_pred_knn = knn_clf.predict(X_test)


print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("KNN Classification Report:\n", classification_report(y_test, y_pred_knn))

# Load the cleaned dataset with outliers removed
df = pd.read_csv(r'C:\Users\idey7\OneDrive\Desktop\MLDA FOOD ADULTERATION\cleaned_food_adulteration_data.csv')  # or 'filtered_no_outliers_zscore.csv'

# Display the first few rows to ensure data is loaded correctly
print(df.head())

# Initialize LabelEncoders for categorical variables
le_product_name = LabelEncoder()
le_brand = LabelEncoder()
le_adulterant = LabelEncoder()

# Encode the categorical variables
df['product_name'] = le_product_name.fit_transform(df['product_name'])
df['brand'] = le_brand.fit_transform(df['brand'])
df['adulterant'] = le_adulterant.fit_transform(df['adulterant'])

# Define features (X) and target (y)
X = df[['product_name', 'brand', 'adulterant']]
y = df['severity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)

# Train the model
dt_model.fit(X_train, y_train)

# Predict the target values for the test set
y_pred = dt_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score_func(y_test, y_pred, average='weighted')
report = classification_report(y_test, y_pred)

print("Decision Tree Model Accuracy:", accuracy)
print("Decision Tree Model F1 Score:", f1)
print("Decision Tree Model Classification Report:\n", report)

# Visualize
plt.figure(figsize=(20,10))
plot_tree(dt_model, feature_names=['product_name', 'brand', 'adulterant'], class_names=['Low', 'Medium', 'High'], filled=True, rounded=True, fontsize=10)
plt.show()


# Initialize and train models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel='linear'),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB()
}

# Dictionary to store results
results = {"Model": [], "Accuracy": [], "Precision": [], "Recall": [], "F1 Score": []}

# Train, predict and evaluate each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score_func(y_test, y_pred, average='weighted')
    
    # Store results
    results["Model"].append(model_name)
    results["Accuracy"].append(accuracy)
    results["Precision"].append(precision)
    results["Recall"].append(recall)
    results["F1 Score"].append(f1)
    
    # Print classification report for each model
    print(f"\n{model_name} Classification Report:\n", classification_report(y_test, y_pred))

# Convert results to DataFrame for visualization
results_df = pd.DataFrame(results)

# Plotting the metrics
plt.figure(figsize=(12, 8))
x = np.arange(len(results_df["Model"]))  # Label locations
width = 0.2  # Width of the bars

fig, ax = plt.subplots(figsize=(12, 8))
ax.bar(x - width*1.5, results_df["Accuracy"], width, label='Accuracy')
ax.bar(x - width/2, results_df["Precision"], width, label='Precision')
ax.bar(x + width/2, results_df["Recall"], width, label='Recall')
ax.bar(x + width*1.5, results_df["F1 Score"], width, label='F1-Score')

# Add labels and title
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(results_df["Model"], rotation=45)
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()

# Plot confusion matrix for each model
for model_name, model in models.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()