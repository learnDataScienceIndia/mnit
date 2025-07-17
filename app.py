import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import re # Import the regular expression module

# --- Page Configuration ---
st.set_page_config(
    page_title="Sound Reduction Predictor",
    page_icon="🔊",
    layout="centered"
)

# --- Data Processing Function ---
def preprocess_for_model(df):
    """Converts 'Floor' column into a purely numerical feature."""
    df_processed = df.copy()

    def convert_floor_to_numeric(floor_val):
        if isinstance(floor_val, str) and floor_val.lower() == 'g':
            return 0
        try:
            num_str = ''.join(re.findall(r'\d+', str(floor_val)))
            return int(num_str) if num_str else 0
        except (ValueError, TypeError):
            return 0

    df_processed['floor_no'] = df_processed['Floor'].apply(convert_floor_to_numeric)
    return df_processed

# --- Main Application ---
st.title('🔊 Sound Reduction Predictor')
st.write(
    "Upload your CSV to train a model that predicts **Reduction_Leq** "
    "based on **any Floor Number** and **Sound_Type**."
)

# --- NEW: Sample Data Expander ---
with st.expander("Click to see an example of the required data format"):
    st.write(
        "Your CSV must contain these three columns: `Floor`, `Sound_Type`, and `Reduction_Leq`. "
        "Other columns can be included but will be ignored by the model."
    )
    # Create a sample DataFrame to display
    sample_data = {
        'Floor': ['G', 'F1', 'F2', 'F3', 'G', 'F1'],
        'Sound_Type': ['Traffic Sound', 'Traffic Sound', 'Traffic Sound', 'Grinder Sound', 'Grinder Sound', 'Grinder Sound'],
        'FF1': [85.7, 75.7, 73.6, 75.7, 92.3, 82.4],
        'Average': [85.98, 75.28, 70.64, 75.04, 91.82, 82.86],
        'Reduction_Leq': [0.00, 10.70, 15.34, 16.78, 0.00, 9.02]
    }
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df)

# --- File Uploader ---
st.subheader("Upload Your Data")
uploaded_file = st.file_uploader(
    "Choose your CSV file",
    type="csv",
    label_visibility='collapsed' # Hides the label as we have a subheader
)

# This block runs only if a file has been uploaded
if uploaded_file is not None:
    try:
        # Load data
        df_original = pd.read_csv(uploaded_file)

        # --- Data Validation ---
        required_cols = ['Floor', 'Sound_Type', 'Reduction_Leq']
        if not all(col in df_original.columns for col in required_cols):
            st.error(f"⚠️ Error: CSV must contain the columns: {', '.join(required_cols)}")
        else:
            st.success("✅ File uploaded successfully!")
            
            # Preprocess the data for the model
            df_processed = preprocess_for_model(df_original)
            df_processed = pd.get_dummies(df_processed, columns=['Sound_Type'], prefix='Sound')

            # --- Model Training Section ---
            st.write("---")
            st.header("1. Model Training")
            
            # Define features (X) and target (y)
            features_for_model = [col for col in df_processed.columns if 'floor_no' in col or 'Sound_' in col]
            X_train = df_processed[features_for_model]
            y_train = df_processed['Reduction_Leq']
            
            # Train the Linear Regression model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            st.success("✅ Model training complete!")

            # --- Prediction Interface ---
            st.write("---")
            st.header("2. Make a Prediction")

            col1, col2 = st.columns(2)
            with col1:
                # Use a number input for the floor
                selected_floor_num = st.number_input("Enter a Floor Number:", min_value=0, value=10, step=1)
            with col2:
                # Use a selectbox for the sound type
                sound_options = sorted(df_original['Sound_Type'].unique())
                selected_sound = st.selectbox("Select a Sound Type:", options=sound_options)
            
            if st.button("Predict Reduction", type="primary"):
                # Prepare the input data for prediction
                prediction_input = pd.DataFrame(np.zeros((1, len(features_for_model))), columns=features_for_model)
                
                # 1. Set the floor number from the user's input
                prediction_input['floor_no'] = selected_floor_num
                
                # 2. Set the sound type column to 1
                sound_column_name = f'Sound_{selected_sound}'
                if sound_column_name in prediction_input.columns:
                    prediction_input[sound_column_name] = 1
                
                # Make the prediction
                predicted_value = model.predict(prediction_input)
                
                # Display the result
                st.success(f"**Predicted Reduction_Leq for Floor {selected_floor_num}:** `{predicted_value[0]:.2f}`")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Upload a CSV file to begin.")