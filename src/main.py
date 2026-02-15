# %% [markdown]
# ## Predictive modeling of customer bookings via Buying Behaviour

# %% [markdown]
# ### Exploratory data analysis
# 
# First, we must explore the data in order to better understand what we have and the statistical properties of the dataset.

# %%
import pandas as pd

# %%
df = pd.read_csv("customer_booking.csv", encoding="ISO-8859-1")
df.head()

# %% [markdown]
# The `.head()` method allows us to view the first 5 rows in the dataset, this is useful for visual inspection of our columns

# %%
df.info()

# %% [markdown]
# The `.info()` method gives us a data description, telling us the names of the columns, their data types and how many null values we have. Fortunately, we have no null values. It looks like some of these columns should be converted into different data types, e.g. flight_day.
# 
# To provide more context, below is a more detailed data description, explaining exactly what each column means:
# 
# - `num_passengers` = number of passengers travelling
# - `sales_channel` = sales channel booking was made on
# - `trip_type` = trip Type (Round Trip, One Way, Circle Trip)
# - `purchase_lead` = number of days between travel date and booking date
# - `length_of_stay` = number of days spent at destination
# - `flight_hour` = hour of flight departure
# - `flight_day` = day of week of flight departure
# - `route` = origin -> destination flight route
# - `booking_origin` = country from where booking was made
# - `wants_extra_baggage` = if the customer wanted extra baggage in the booking
# - `wants_preferred_seat` = if the customer wanted a preferred seat in the booking
# - `wants_in_flight_meals` = if the customer wanted in-flight meals in the booking
# - `flight_duration` = total duration of flight (in hours)
# - `booking_complete` = flag indicating if the customer completed the booking
# 
# Before we compute any statistics on the data, lets do any necessary data conversion

# %%
df["flight_day"].unique()

# %%
mapping = {
    "Mon": 1,
    "Tue": 2,
    "Wed": 3,
    "Thu": 4,
    "Fri": 5,
    "Sat": 6,
    "Sun": 7,
}

df["flight_day"] = df["flight_day"].map(mapping)

# %%
df["flight_day"].unique()

# %%
df.describe()

# %% [markdown]
# The `.describe()` method gives us a summary of descriptive statistics over the entire dataset (only works for numeric columns). This gives us a quick overview of a few things such as the mean, min, max and overall distribution of each column.
# 
# From this point, you should continue exploring the dataset with some visualisations and other metrics that you think may be useful. Then, you should prepare your dataset for predictive modelling. Finally, you should train your machine learning model, evaluate it with performance metrics and output visualisations for the contributing variables. All of this analysis should be summarised in your single slide.

# %% [markdown]
# 

# %%
df

# %% [markdown]
# ### Import Libraries

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# Sklearn modules for preprocessing and model selection
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,StratifiedKFold
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,classification_report,roc_curve)

# %%
# Classification Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier

# %%
import warnings
warnings.filterwarnings("ignore") # Ignore warnings for cleaner output

# %% [markdown]
# ### Data Loading

# %%
def load_and_explore_data(df):
    """
    Loads and performs initial exploration
    """

    print("----- Loading Data -----")
    # Using "ISO-8859-1" encoding as some datasets contain special characters

    try:
        df = pd.read_csv("customer_booking.csv", encoding="ISO-8859-1")
    except:
        df = pd.read_csv("customer_booking.csv") # Falls back to default


    print(f"Data Shape: {df.shape}")
    print("---- Data Information -----")
    print(df.info())


    print("----- First 5 rows -----")
    print(df.head())


    return df

# %%
load_data = load_and_explore_data(df)
print(load_data)

# %% [markdown]
# ### Exploartory Data Analysis

# %%
def visualize_data(df):
    # Create viualization to understand data distribution

    print("----- Generating Visualizations (Before Processing) -----")

    # 1. Target Variable Distribution
    plt.figure(figsize=(6,4))
    sns.countplot(x="booking_complete", data=df, palette="viridis")
    plt.title("Distribution of Target Variable (Booking Complete)")
    plt.xlabel("Booking Complete (0=No, 1=Yes)")
    plt.ylabel("Count")
    plt.show()

    # 2. Numerical Features Histograms
    numerical_cols = df.select_dtypes(include=["int64","float64"]).columns
    df[numerical_cols].hist(figsize=(12,10),bins=20,color="skyblue",edgecolor="black")
    plt.suptitle("Histograms of Numerical Features")
    plt.show()

    # 3. Correlation Matrix (Numerical Only)
    plt.figure(figsize=(12,10))
    # Select only numeric columns for correlation
    numerical_df = df.select_dtypes(include=[np.number])
    corr_matrix = numerical_df.corr()
    sns.heatmap(corr_matrix,annot=True,cmap="coolwarm", fmt=".2f",linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()

    # 4. Flight Day vs Booking
    plt.figure(figsize=(10,5))
    sns.countplot(x='flight_day', hue='booking_complete', data=df, 
                  order=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.title("Booking Completion by Flight Day")
    plt.show()

# %%
visualizations = visualize_data(df)
print(visualizations)

# %% [markdown]
# ### Data Cleaning

# %%
df

# %%
# Check for missing values
df_missing = df.isnull().sum()
print("----- Missing Values -----")
print(df_missing)

# %%
# Check for duplicated rows
df_duplicated = df.duplicated().sum()
print("----- Duplicated Values -----")
print(df_duplicated)

# %%
# Drop duplicated rows
df_duplicates = df.drop_duplicates(inplace=True)
print(df_duplicates)

# %%
# Check for duplicated rows again
df_duplicated = df.duplicated().sum()
print("----- Duplicated Values -----")
print(df_duplicated)

# %%
df

# %%
df_cleaned = df.to_csv("cleaned_customer_boking.csv",index=False)
print("Data Saved Successfully...............")

# %% [markdown]
# ### Data Preprocessing and Feature Engineering

# %%
def preprocess_data(df):
    # Cleans data, encodes categoricals, and scae features

    print("----- Preprocessing Data -----")

    # Make a copy to avoid SettingWithCopy warnings
    data = df.copy()


    # ----- Feature Engineering -----
    # Convert flight_day to numerical to ordinal
    day_mapping = {
        "Mon": 1,
        "Tue": 2,
        "Wed": 3,
        "Thu": 4,
        "Fri": 5,
        "Sat": 6,
        "Sun": 7
    }

    data["flight_day"] = data["flight_day"].map(day_mapping)
    
    # Handle High Cardinality Categoricals (Route, Booking Origin)
    # Strategy: Frequency Encoding (Replace category with its count/frequncy)
    # This captures the popularity of a route/origin without thousands of columns

    for col in ["route","booking_origin"]:
        freq_encoding = data[col].value_counts(normalize=True)
        data[col + "_freq"] = data[col].map(freq_encoding)


    # Drop the high cardinality columns
    data = data.drop(["route","booking_origin"],axis=1)

    # One-Hot Encoding for low cardinalitu categorical columns
    categorical_cols = ["sales_channel", "trip_type"]
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    # ----- Splitting Features and Target -----
    X = data.drop("booking_complete", axis=1)
    y = data["booking_complete"]

    # ----- Scaling -----
    #  Standardize features (Mean=0, std=1) - vital for KNN, SVM and Logistic Regression
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


    return X_scaled, y, scaler,  X.columns

# %% [markdown]
# ### Model Comparison, Training and Comparison

# %%
def train_and_compare_models(X_train,X_test,y_train,y_test):
    # Trains multiple models and compare their performance

    print("-----Training and Comparing Models -----")

    # Define the dictionary of models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        #"Support Vector Machine": SVC(probability=True, random_state=42), # Probability=True for ROC-AUC
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42)
    }

    results = []

    for name, model in models.items():
        print(f"Training {name}......")

        # Train the model
        model.fit(X_train,y_train)

        # Predict 
        y_pred = model.predict(X_test)

        # Probability for ROC-AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decison_function(X_test) # For SVM if probability=False

        # Metrics
        acc = accuracy_score(y_test,y_pred)
        prec = precision_score(y_test,y_pred,zero_division=0)
        rec = recall_score(y_test,y_pred,zero_division=0)
        f1 = f1_score(y_test,y_pred,zero_division=0)
        roc = roc_auc_score(y_test,y_prob)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precison": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC AUC": roc
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Accuracy", ascending=False) # Sort by F1 (good for imbalance data)

    print("----- Model Comparison Results -----")
    print(results_df)


    # Plot Comparison
    plt.figure(figsize=(12,6))
    sns.barplot(x="Accuracy", y="Model",data=results_df, palette="magma")
    plt.title("Model Comparison by Accuracy")
    plt.show()


    return results_df, models

# %% [markdown]
# ### Hyperparamter Tuning (Random Forest)

# %%
def tune_best_model(X_train,y_train):
    # Performs Grid Search on Random Forest (usually a strong performer)

    print("----- Hyperparameter Tuning (Random Forest) -----")

    param_grid = {
        "n_estimators": [50,100,200],
        "max_depth": [10,20, None],
        "min_samples_split": [2,5,10],
        "min_samples_leaf": [1,2,4] 
    }

    rf = RandomForestClassifier(random_state=42)

    # 5-Fold Cross Validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2, scoring="f1")
    
    grid_search.fit(X_train,y_train)

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation F1 Score: {grid_search.best_score_:.4f}")


    return grid_search.best_estimator_

# %% [markdown]
# ### Model Evaluation & Interpretation

# %%
def evaluate_final_model(model, X_test, y_test, feature_names):
    # Detailed evaluation of the tuned model

    print("----- Final Model Evaluation -----")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    # Classification Report
    print("Classification Report")
    print(classification_report(y_test,y_pred))

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Comfusion Matrix (Tuned Random Foest)")
    plt.show()

    # Feature Importance Visualization (Basic)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12,6))
        plt.title("Feature Importances (Basic Plot)")
        plt.bar(range(X_test.shape[1]), importances[indices], align="center")
        plt.xticks(range(X_test.shape[1]), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ### Feature Importance

# %%
def explain_feature_importance(model,feature_names):
    # Analyze and prints the contriution of features to the model
    print("="*40)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*40)

    if not hasattr(model, "feature_importances_"):
        print("This model type does not support built-in importance")
        return
    

    # Create a DataFrame for better handling
    importances = model.feature_importances_
    feature_imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance",ascending=False)

    # 1. Top Contributors
    print("TOP 5 PREDICTORS")
    for index, row in feature_imp_df.head(5).iterrows():
        print(f"    -  {row["Feature"]}: {row["Importance"]:.4f}")


    # 2. Least Contributors
    print("BOTTOM 5 PREDICTORS")
    for index, row in feature_imp_df.tail(5).iterrows():
        print(f"    - {row["Feature"]}: {row["Importance"]:.4f}")

    
    # 3. Visual Analysis
    plt.figure(figsize=(12,8))
    sns.barplot(x="Importance", y="Feature", data=feature_imp_df.head(15), palette="mako")
    plt.title("Feature Importance: Top 15 Variables")
    plt.xlabel("Importance Score (Mean Decrease in Impurity)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    # 4. Narrative
    top_feature = feature_imp_df.iloc[0]["Feature"]
    print(f"ANALYSIS SUMMARY:")
    print(f"The most influential variable is '{top_feature}'.")
    print("This suggests that this factor is the primary driver in distingushing between")
    print("customers who book and those who don't")

# %% [markdown]
# ### New Prediction Input

# %%
def predict_new_customer(model, scaler, feature_columns):
    print("Enter customer booking details:\n")

    new_customer = {
        'num_passengers': int(input("Number of passengers: ")),
        'purchase_lead': int(input("Days before flight booking was made: ")),
        'length_of_stay': int(input("Length of stay (days): ")),
        'flight_hour': int(input("Flight hour (0–23): ")),
        'flight_day': int(input("Flight day (0=Sun, 6=Sat): ")),
        'wants_extra_baggage': int(input("Wants extra baggage? (1=yes, 0=no): ")),
        'wants_preferred_seat': int(input("Wants preferred seat? (1=yes, 0=no): ")),
        'wants_in_flight_meals': int(input("Wants in-flight meals? (1=yes, 0=no): ")),
        'flight_duration': float(input("Flight duration (hours): ")),
        'route_freq': float(input("Route frequency (e.g., 0.005): ")),
        'booking_origin_freq': float(input("Booking origin frequency (e.g., 0.1): ")),
        'sales_channel_Mobile': int(input("Booked via Mobile? (1=yes, 0=no): ")),
        'sales_channel_Internet': int(input("Booked via Internet? (1=yes, 0=no): ")),
        'trip_type_OneWay': int(input("Trip type OneWay? (1=yes, 0=no): ")),
        'trip_type_RoundTrip': int(input("Trip type RoundTrip? (1=yes, 0=no): "))
    }


    # IMPORTANT: ENsure the input list matches the EXACT order of X_train columns
    # Create a zero-filled array
    input_data = np.zeros ((1, len(feature_columns)))
    input_df = pd.DataFrame(input_data, columns=feature_columns)

    # Fill in known values (this logic simplifies the mapping for demonstration)
    # In a production app, you would have a more robust pipeline transformer
    
    # Manually mapping for the example (Assuming columns match keys roughly)
    # Note: In production, use the same Pipeline used for training

    input_df['num_passengers'] = new_customer['num_passengers']
    input_df['purchase_lead'] = new_customer['purchase_lead']
    input_df['length_of_stay'] = new_customer['length_of_stay']
    input_df['flight_hour'] = new_customer['flight_hour']
    input_df['flight_day'] = new_customer['flight_day']
    input_df['wants_extra_baggage'] = new_customer['wants_extra_baggage']
    input_df['wants_preferred_seat'] = new_customer['wants_preferred_seat']
    input_df['wants_in_flight_meals'] = new_customer['wants_in_flight_meals']
    input_df['flight_duration'] = new_customer['flight_duration']

    # Handle the generated frequency columns and one-hot columns
    # We try to fill by name if they exist in the trained features 
    for col in feature_columns:
        if col in new_customer:
            input_df[col] = new_customer[col]


    # Scale the input 
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]


    print(f"New Customer Input: {new_customer}")
    print(f"Prediction: {'Booking will be COMPLETED' if prediction[0] == 1 else 'Booking will NOT be completed'}")
    print(f"Probability of Booking: {probability:.2%}")

# %% [markdown]
# ### Main Execution

# %%
if __name__ == "__main__":
    
    # 1. Load Data
    # Ensure 'customer_booking.csv' is in the same directory
    df = load_and_explore_data('customer_booking.csv')
    
    # 2. EDA
    visualize_data(df)
    
    # 3. Preprocessing
    X, y, scaler, feature_names = preprocess_data(df)
    
    # Train/Test Split (80% Train, 20% Test)
    # stratify=y ensures the class balance is maintained in splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training Features Shape: {X_train.shape}")
    print(f"Testing Features Shape: {X_test.shape}")

    # 4. Compare All Models
    results_df, model_dict = train_and_compare_models(X_train, X_test, y_train, y_test)
    
    # 5. Hyperparameter Tuning (Focusing on Random Forest as it usually performs well on this type of data)
    best_rf_model = tune_best_model(X_train, y_train)
    
    # 6. Evaluate Final Model
    evaluate_final_model(best_rf_model, X_test, y_test, feature_names)

    # 7. Detailed Feature Importance Analysis (Added)
    explain_feature_importance(best_rf_model, feature_names)


