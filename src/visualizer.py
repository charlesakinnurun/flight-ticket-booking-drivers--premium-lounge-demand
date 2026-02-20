import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def plot_target_dist(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x="booking_complete", data=df, palette="viridis")
    plt.title("Target Distribution")
    plt.show()

def plot_correlation(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

def plot_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        imps = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=imps.head(15), y=imps.head(15).index, palette="mako")
        plt.title("Top 15 Feature Importances")
        plt.show()

def plot_confusion(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()