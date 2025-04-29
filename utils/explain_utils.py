
import shap
import matplotlib.pyplot as plt

def explain_xgboost_model(model, X_sample):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)
    shap.summary_plot(shap_values, X_sample)
    shap.plots.waterfall(shap_values[0])