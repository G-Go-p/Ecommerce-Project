import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import numpy as np

# Set page title and layout
st.set_page_config(page_title="Customer Spending Predictor", layout="wide")

# Custom CSS for background image
st.markdown(
    """
    <style>
    .main {
        background-image: url('https://th.bing.com/th/id/OIP.6rHjdwhwrL_VCpWWTh1m_gHaHa?pid=ImgDet&w=172&h=172&c=7&dpr=1.1');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Load the data
df = pd.read_csv("Ecommerce_Customers (2).csv")

df = df.drop(columns=['Email','Address','Avatar'])

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define the bounds for filtering
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter the DataFrame
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers for specified columns
for col in df.columns:
    df = remove_outliers_iqr(df, col)

scaler = StandardScaler()

columns_to_scale = ['Avg Session Length', 'Time on App', 'Time on Website', 'Length of Membership']
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

X = df[['Avg Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
#print("Linear Regression - MSE:", mean_squared_error(y_test, y_pred_lr), "R²:", r2_score(y_test, y_pred_lr))


st.title("Customer Spending Prediction")
st.markdown("""
This application predicts the *Yearly Amount Spent* by customers based on their behavior metrics. 
Please use the sidebar to enter customer details for prediction.
""")

# Sidebar inputs
st.sidebar.header("Customer Input")
avg_session_length = st.sidebar.number_input("Average Session Length", min_value=0.0, step=0.1)
time_on_app = st.sidebar.number_input("Time on App", min_value=0.0, step=0.1)
time_on_website = st.sidebar.number_input("Time on Website", min_value=0.0, step=0.1)
length_of_membership = st.sidebar.number_input("Length of Membership", min_value=0.0, step=0.1)

# Model performance metrics
st.subheader("Model Performance")
col1, col2, col3,col4= st.columns(4)
with col1:
    st.metric(label="Mean Absolute Error", value=f"{mean_absolute_error(y_test, y_pred):.2f}")
with col2:
    st.metric(label="Mean Squared Error", value=f"{mean_squared_error(y_test, y_pred):.2f}")
with col3:
    st.metric(label="Root Mean Squared Error", value=f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
with col4:
    st.metric(label="R2_Score", value=f"{r2_score(y_test, y_pred):.2f}")

# Prediction
if st.sidebar.button("Predict"):
    # Collect and scale the user input data
    new_data = np.array([[avg_session_length, time_on_app, time_on_website, length_of_membership]])
    new_data_scaled = scaler.transform(new_data)  # Apply the same scaler used on training data
    
    # Predict using the scaled data
    prediction = lr.predict(new_data_scaled)
    
    # Display the result
    st.subheader("Prediction Result")
    st.success(f"Predicted Yearly Amount Spent: *${prediction[0]:.2f}*")

