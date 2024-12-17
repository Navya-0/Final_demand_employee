import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load and Train Model
@st.cache_resource  # Cache the model training process
def load_and_train_model():
    file_path = 'synthetic_final_mapping (1).csv'
    data = pd.read_csv(file_path)

    # Select relevant columns for the model
    relevant_columns = [
        "Role Status", "Region", "Project Type", "Track", "Location Shore", 
        "Primary Skill (Must have)", "Grade", "Employment ID", "First Name", 
        "Last Name", "Work Region", "Designation", "Email"
    ]
    data = data[relevant_columns]

    # Preprocess data
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].fillna("Unknown"))
        # Add "Unknown" to the encoder's classes if it's not already there
        if "Unknown" not in le.classes_:
            le.classes_ = list(le.classes_) + ["Unknown"]
        label_encoders[column] = le

    # Train the model
    feature_columns = [
        "Role Status", "Region", "Project Type", "Track", "Location Shore", 
        "Primary Skill (Must have)", "Grade"
    ]
    X = data[feature_columns]
    y = data["Employment ID"]
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    return model, data, label_encoders, feature_columns

# Load Test Case Dataset
@st.cache_data  # Cache the test case loading process
def load_test_case_dataset():
    test_case_file = 'selected_demand_1.xlsx'
    test_case_data = pd.read_excel(test_case_file)

    # Clean column names by stripping whitespace and special characters
    test_case_data.columns = test_case_data.columns.str.strip().str.replace('\t', '', regex=False)
    return test_case_data

# Recommend Employees
def recommend_employees(model, input_data, data):
    predictions = model.predict_proba([input_data])[0]
    employee_indices = predictions.argsort()[-3:][::-1]
    employee_ids = data["Employment ID"].unique()
    top_employees = [employee_ids[i] for i in employee_indices]
    return top_employees

# Decode employee details
def decode_employee_details(employee_ids, data, label_encoders):
    # Get unique employee records
    unique_employees = data[data["Employment ID"].isin(employee_ids)].drop_duplicates(subset="Employment ID")

    # Decode categorical columns back to their original values
    decoded_data = unique_employees.copy()
    for column, encoder in label_encoders.items():
        if column in decoded_data.columns:
            decoded_data[column] = encoder.inverse_transform(decoded_data[column])
    
    # Select and return required columns
    return decoded_data[["Employment ID", "First Name", "Last Name", "Work Region", "Designation", "Email"]]

# Streamlit App Styling
st.markdown(
    """
    <style>
        .title {text-align: center; font-size: 36px; color: #4F8BF9;}
        .header {color: #333333; text-decoration: underline;}
        .button {background-color: #4CAF50; color: white; font-size: 16px; padding: 10px;}
        .result-table {background-color: #f8f9fa; border-radius: 10px; padding: 10px;}
    </style>
    """, 
    unsafe_allow_html=True
)

st.markdown("<h1 class='title'>🚀 Demand to Talent Recommendation System</h1>", unsafe_allow_html=True)

# Load model and datasets
model, data, label_encoders, feature_columns = load_and_train_model()
test_case_data = load_test_case_dataset()

# Project ID Section
st.markdown("### 📋 Select Test Case Project ID")
project_ids = test_case_data["Demand ID"].unique()
selected_project_id = st.selectbox("**Project ID:**", project_ids)

# Retrieve the selected project
selected_project = test_case_data[test_case_data["Demand ID"] == selected_project_id]

# Handle empty selected project
if selected_project.empty:
    st.error(f"No project found with Demand ID {selected_project_id}. Please check your dataset.")
else:
    selected_project = selected_project.iloc[0]  # Get the first matching row

    # Auto-Populated Attributes (Read-only on UI)
    st.markdown("### 🛠️ Auto-Populated Project Attributes")
    col1, col2 = st.columns(2)
    user_input = []

    # Prepare user input to match training features
    for idx, column in enumerate(feature_columns):
        with col1 if idx % 2 == 0 else col2:
            if column in selected_project:
                value = selected_project[column]
                st.text_input(f"**{column}**", value, disabled=True)  # Read-only field
                if column in label_encoders:
                    # Transform the value using LabelEncoder
                    user_input.append(label_encoders[column].transform([value])[0])
                else:
                    user_input.append(value)
            else:
                # Handle missing columns with default values
                default_value = "Unknown" if column in label_encoders else 0
                if column in label_encoders:
                    # Check if the default value exists in the encoder's classes
                    if default_value in label_encoders[column].classes_:
                        user_input.append(label_encoders[column].transform([default_value])[0])
                    else:
                        # If default value is not present, use the first class (e.g., 0)
                        user_input.append(0)
                else:
                    user_input.append(default_value)

    # Prediction Button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🎯 Get Suitable Employees", help="Click to fetch top employees based on project attributes."):
        try:
            st.markdown("### 🏆 Top 3 Recommended Employees")
            recommendations = recommend_employees(model, user_input, data)

            # Fetch details of top recommended employees
            employee_details = decode_employee_details(recommendations, data, label_encoders)

            # Display results in a styled table
            st.markdown("<div class='result-table'>", unsafe_allow_html=True)
            st.table(employee_details)
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
