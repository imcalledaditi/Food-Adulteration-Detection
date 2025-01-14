import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv(r'C:\Users\idey7\OneDrive\Desktop\MLDA FOOD ADULTERATION\cleaned_food_adulteration_data.csv')

# Encode categorical columns
le_product_name = LabelEncoder()
le_brand = LabelEncoder()
le_adulterant = LabelEncoder()

df['product_name'] = le_product_name.fit_transform(df['product_name'])
df['brand'] = le_brand.fit_transform(df['brand'])
df['adulterant'] = le_adulterant.fit_transform(df['adulterant'])

# Define features (X) and target (y)
X = df[['product_name', 'brand', 'adulterant']]
y = df['severity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(kernel='linear'),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB()
}

# Train models and evaluate their accuracy
model_performance = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_performance[model_name] = accuracy

# Get the best model based on accuracy
best_model_name = max(model_performance, key=model_performance.get)
best_model = models[best_model_name]
best_model_accuracy = model_performance[best_model_name]

# Create the Streamlit app
st.title("Food Adulteration Detection with Multiple Models")

# Input widgets for user input
product_name = st.selectbox("Select Product Name", le_product_name.classes_)
brand = st.selectbox("Select Brand", le_brand.classes_)
adulterant = st.selectbox("Select Adulterant", le_adulterant.classes_)

# Encode the input
input_data = pd.DataFrame({
    'product_name': [le_product_name.transform([product_name])[0]],
    'brand': [le_brand.transform([brand])[0]],
    'adulterant': [le_adulterant.transform([adulterant])[0]]
})

# Predict and display results using the best model
if st.button("Predict Severity"):
    prediction = best_model.predict(input_data)
    st.write(f"Predicted Severity: {prediction[0]}")

