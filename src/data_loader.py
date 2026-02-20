import pandas as pd


def load_data(filepath, encoding="ISO-8859-1"):
    """Loads customer booking data and performs initial cleaning."""
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath, encoding=encoding)
    except Exception as e:
        print(f"Error loading with primary encoding: {e}. Trying default.")
        df = pd.read_csv(filepath)  # Fallback to default encoding
    
    # Drop duplicates as identified in EDA
    original_len = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Loaded {len(df)} rows (Dropped {original_len - len(df)} duplicates).")
    return df

def check_missing(df):
    """Checks for null values in the dataset."""
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("Missing values found:\n", missing[missing > 0])
    else:
        print("No missing values found.")
    return missing