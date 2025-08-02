# Trigger rebuild
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import gradio as gr
import shap
import matplotlib.pyplot as plt

# Load model
model = joblib.load("fraud_xgb_model.pkl")

# SHAP explainer
explainer = shap.TreeExplainer(model)

# Prediction function with auto-calculation and SHAP
def predict_fraud(step, trans_type, amount, old_orig, new_orig, old_dest, new_dest):
    # Encode transaction type
    type_mapping = {"TRANSFER": 0, "CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3}
    type_encoded = type_mapping.get(trans_type.upper(), -1)

    # Auto-calculate error balances
    errorBalanceOrig = new_orig + amount - old_orig
    errorBalanceDest = old_dest + amount - new_dest

    # Build dataframe
    input_data = {
        "step": step,
        "type": type_encoded,
        "amount": amount,
        "oldbalanceOrg": old_orig,
        "newbalanceOrig": new_orig,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
        "errorBalanceOrig": errorBalanceOrig,
        "errorBalanceDest": errorBalanceDest
    }
    input_df = pd.DataFrame([input_data])

    # Predict
    dmatrix = xgb.DMatrix(input_df, feature_names=input_df.columns.tolist())
    pred_prob = model.predict(dmatrix)[0]
    label = "🚨 Fraud" if pred_prob > 0.5 else "✅ Legitimate"

    # SHAP values
    shap_values = explainer.shap_values(input_df)
    plt.figure(figsize=(8, 5))
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], input_df.iloc[0], show=False)
    plt.tight_layout()
    plt.savefig("shap_output.png")
    plt.close()

    return label, "shap_output.png"

# Gradio UI
demo = gr.Interface(
    fn=predict_fraud,
    inputs=[
        gr.Number(label="Step"),
        gr.Dropdown(["TRANSFER", "CASH_OUT", "DEBIT", "PAYMENT"], label="Type"),
        gr.Number(label="Amount"),
        gr.Number(label="Old Balance Orig"),
        gr.Number(label="New Balance Orig"),
        gr.Number(label="Old Balance Dest"),
        gr.Number(label="New Balance Dest"),
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Image(label="SHAP Explanation")
    ],
    title="Fraud Detector with SHAP 💡",
    description="Enter transaction details to predict fraud and understand what influenced the model."
)

demo.launch()
