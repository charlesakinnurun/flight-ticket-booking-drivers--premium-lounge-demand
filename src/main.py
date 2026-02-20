import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# Import our custom modules
import data_loader
import preprocessing
import model_engine
import visualizer

# Filter warnings for a cleaner console output
warnings.filterwarnings("ignore")

def run_pipeline(csv_path=None):
    """
    Orchestrates the end-to-end machine learning pipeline.
    """
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "customer_booking.csv")
    
    print("Starting Customer Booking Predictive Pipeline...")

    # 1. Data Acquisition & Initial Check
    # Handled by data_loader.py
    df = data_loader.load_data(csv_path)
    data_loader.check_missing(df)
    
    # 2. Exploratory Data Analysis (EDA)
    # Handled by visualizer.py
    print("\n[Step 1/5] Performing Exploratory Data Analysis...")
    visualizer.plot_target_dist(df)
    
    # 3. Feature Engineering & Preprocessing
    # Handled by preprocessing.py
    print("\n[Step 2/5] Engineering Features and Scaling Data...")
    processed_df = preprocessing.engineer_features(df)
    
    # Visualize correlations after feature engineering
    visualizer.plot_correlation(processed_df)
    
    X, y, scaler = preprocessing.prepare_xy(processed_df)
    feature_names = X.columns
    
    # 4. Data Splitting
    # 80% Training, 20% Testing with stratification to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Dataset split completed. Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # 5. Model Comparison
    # Handled by model_engine.py
    print("\n[Step 3/5] Benchmarking Multiple Classifiers...")
    base_models = model_engine.get_base_models()
    comparison_results = model_engine.compare_models(base_models, X_train, y_train, X_test, y_test)
    
    print("\nModel Benchmark Results (Sorted by Accuracy):")
    print(comparison_results)
    
    # 6. Hyperparameter Tuning
    # We focus on tuning the Random Forest model
    print("\n[Step 4/5] Tuning Random Forest Hyperparameters...")
    tuned_rf = model_engine.tune_random_forest(X_train, y_train)
    
    # 7. Final Model Evaluation
    print("\n[Step 5/5] Final Model Evaluation on Test Set...")
    y_pred = tuned_rf.predict(X_test)
    
    print("\nFinal Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # 8. Post-Training Visualizations
    # Handled by visualizer.py
    visualizer.plot_confusion(y_test, y_pred)
    visualizer.plot_importance(tuned_rf, feature_names)
    
    print("\nPipeline Execution Complete.")

if __name__ == "__main__":
    run_pipeline()