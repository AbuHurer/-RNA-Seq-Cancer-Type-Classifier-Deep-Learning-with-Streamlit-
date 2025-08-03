import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
# Load model and encoder
model = load_model(r"C:\Users\saeem\Desktop\RNA_SEQ_for _Cancer\model\model.h5")
label_encoder = joblib.load(r"C:\Users\saeem\Desktop\RNA_SEQ_for _Cancer\model\label_encoder.pkl")

st.set_page_config(page_title="RNA-Seq Cancer Type Classifier", layout="centered")

st.title("🧬 RNA-Seq Cancer Type Classifier")
st.markdown("""
Upload gene expression data or manually enter the expression levels for prediction.
""")

# Option 1: Upload CSV file
uploaded_file = st.file_uploader("📄 Upload a CSV file with gene expression data", type=["csv"])

# Option 2: Manual input
manual_input = st.checkbox("Or, enter expression values manually-NOT RECCOMMENDED AS it requires MANUAL INPUT OF 16383 FEATURES", value=False)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data:")
        st.dataframe(df.head())

        if st.button("Predict"):
            preds_proba = model.predict(df)  # shape: (n_samples, num_classes)
            preds = np.argmax(preds_proba, axis=1)  # shape: (n_samples,)
            decoded_preds = label_encoder.inverse_transform(preds)
            st.markdown(f"""
                <div style='
                    background-color: #1f4c2e;
                    padding: 2rem;
                    border-radius: 1rem;
                    text-align: center;
                    margin-top: 3rem;
                    margin-bottom: 2rem;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                '>
                    <img src='C:\\Users\\saeem\\Desktop\\RNA_SEQ_for _Cancer\\rna1.png' width='80' style='margin-bottom: 1rem;'/>
                    <h2 style='color: #E6FFE6;'>🧬 Predicted Cancer Type: <span style='color: #00FF9C;'>{decoded_preds}</span></h2>
                </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"⚠️ Error reading file: {e}")


elif manual_input:
    num_features = model.input_shape[1]
  # Automatically get number of features
    st.markdown(f"Model expects **{num_features} gene expression values**.")

    values = []
    for i in range(num_features):
        val = st.number_input(f"Gene Expression Value {i+1}", value=0.0)
        values.append(val)

    if st.button("Predict"):
        input_array = np.array(values).reshape(1, -1)
        prediction = model.predict(input_array)
        cancer_type = label_encoder.inverse_transform(prediction)[0]
        # Set a flag to show prediction
        prediction_available = True

        # Show stylized result
        st.markdown(f"""
            <div style='
                background-color: #1f4c2e;
                padding: 2rem;
                border-radius: 1rem;
                text-align: center;
                margin-top: 3rem;
                margin-bottom: 2rem;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            '>
                <img src='https://cdn-icons-png.flaticon.com/512/616/616408.png' width='80' style='margin-bottom: 1rem;'/>
                <h2 style='color: #E6FFE6;'>🧬 Predicted Cancer Type: <span style='color: #00FF9C;'>{cancer_type}</span></h2>
            </div>
        """, unsafe_allow_html=True)

else:
    st.info("👈 Upload a CSV file or check the box to input values manually.")

st.caption("Built by Mohammed Abu Hurer with ❤️ using Streamlit")
