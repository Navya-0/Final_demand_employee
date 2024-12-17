import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load Dataset and Train Model
@st.cache_resource
def load_and_train_model():
    file_path = 'synthetic_final_mapping (1).csv'
    data = pd.read_csv(file_path)

    # Select relevant columns for the model
    relevant_columns = [
        "Role Status", "Region", "Project Type", "Track", "Location Shore", 
        "Primary Skill (Must have)", "Grade", "Employment ID", "Email", 
        "First Name", "Last Name", "Designation", "Pay Grade"
    ]
    data = data[relevant_columns]

    # Preprocess data
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        if column not in ["Employment ID", "Email", "First Name", "Last Name", "Designation", "Pay Grade"]:
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column].fillna("Unknown"))
            label_encoders[column] = le

    # Train the model
    X = data.drop(["Employment ID", "Email", "First Name", "Last Name", "Designation", "Pay Grade"], axis=1)
    y = data["Employment ID"]
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    return model, data, label_encoders, relevant_columns[:-6]

# Recommend Employees
def recommend_employees(model, input_data, data):
    predictions = model.predict_proba([input_data])[0]
    employee_indices = predictions.argsort()[-3:][::-1]
    employee_ids = data["Employment ID"].unique()
    top_employees = [employee_ids[i] for i in employee_indices]
    return top_employees

# Streamlit App
st.set_page_config(page_title="Demand to Talent", layout="wide", page_icon=":briefcase:")
st.title("🌟 Demand to Talent Matchmaker")
st.markdown("---")

# Load and train model
model, data, label_encoders, feature_columns = load_and_train_model()

# Load the user-provided CSV file
uploaded_file = 'sample_demand_data.csv'
demand_data = pd.read_csv(uploaded_file)

# Layout: Input Section
st.header("📋 Enter Demand Details")
st.markdown("Please select a demand ID to auto-populate required attributes.")

# User selects ID from dropdown
col1, col2 = st.columns([1, 3])
with col1:
    demand_id = st.selectbox("**Select Demand ID**", demand_data['ID'].unique(), help="Choose a demand ID to populate attributes.")
    
with col2:
    st.write("")

# Auto-populate fields based on selected ID
selected_row = demand_data[demand_data['ID'] == demand_id].iloc[0]
user_input = []


auto_populated_section = st.container()
with auto_populated_section:
    col1, col2 = st.columns(2)
    for idx, column in enumerate(feature_columns):
        with col1 if idx % 2 == 0 else col2:
            if column in selected_row.index:
                value = selected_row[column]
                st.text_input(f"**{column}:**", value, key=column, disabled=True)
                if column in label_encoders:
                    user_input.append(label_encoders[column].transform([value])[0])
                else:
                    user_input.append(value)

# Ensure input features are complete
if len(user_input) != len(feature_columns):
    st.error(f"Error: Input features are incomplete. Expected {len(feature_columns)}, but got {len(user_input)}.")
else:
    if st.button("🚀 Get Suitable Employees"):
        try:
            recommendations = recommend_employees(model, user_input, data)

            # Layout: Recommended Employees Section
            st.markdown("---")
            st.header("🏆 Top 3 Recommended Employees")
            for i, employee_id in enumerate(recommendations, 1):
                employee_details = data[data["Employment ID"] == employee_id][
                    ["Employment ID", "First Name", "Last Name", "Email", "Designation", "Pay Grade"]
                ].iloc[0].to_dict()

                # Display each employee in a styled "card"
                st.markdown(f"""
                <div style="border: 2px solid #4CAF50; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                    <h3 style="color: #2E8B57;">{i}. {employee_details['First Name']} {employee_details['Last Name']}</h3>
                    <p><strong>Employee ID:</strong> {employee_details['Employment ID']}</p>
                    <p><strong>Email:</strong> {employee_details['Email']}</p>
                    <p><strong>Designation:</strong> {employee_details['Designation']}</p>
                    <p><strong>Pay Grade:</strong> {employee_details['Pay Grade']}</p>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
