### 2. main.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.impute import SimpleImputer

def hour_extractor(df):
    temp_datetime = pd.to_datetime(df["timestamp"], errors='coerce')
    df["hour"] = temp_datetime.dt.hour
    imputer = SimpleImputer(strategy='most_frequent')
    df["hour"] = imputer.fit_transform(df[["hour"]])
    df["hour"] = df["hour"].astype(int)
    return df

def add_cyclic_hour(df):
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df

def convert_text(df, source_weight=3, title_weight=2):
    df['text'] = (
        ("__source__" + df["source"].astype(str) + " ") * source_weight
        + (df["title"].fillna("") + " ") * title_weight
        + df["article"].fillna("")
    )
    return df

def main():
    print("Loading data...")
    df = pd.read_csv("development.csv")
    dfeval = pd.read_csv("evaluation.csv")

    # Feature Engineering
    for data in [df, dfeval]:
        data = hour_extractor(data)
        data = add_cyclic_hour(data)
        data = convert_text(data)

    X = df[['text', 'page_rank', 'hour_sin', 'hour_cos']]
    y = df['label']

    # Define Text Features
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9, 
                                 max_features=30000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,6), 
                                 min_df=3, max_features=35000, sublinear_tf=True))
    ])

    # Preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ("text", features, "text"),
        ("hour", "passthrough", ["hour_sin", "hour_cos"]),
        ("pagerank", "passthrough", ["page_rank"])
    ])

    # Model Configuration
    class_weights = {0: 1.0, 5: 1.34, 2: 1.45, 1: 1.49, 3: 1.54, 4: 1.65, 6: 2.76}
    lsvc = LinearSVC(C=0.095, class_weight=class_weights, dual=False, max_iter=700)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', lsvc)
    ])

    print("Training model...")
    pipeline.fit(X, y)

    print("Generating predictions...")
    ypred = pipeline.predict(dfeval)

    output = pd.DataFrame({
        "Id": range(len(ypred)),
        "Predicted": ypred
    })
    
    output.to_csv("final_output.csv", index=False)
    print("Success: final_output.csv generated.")

if __name__ == "__main__":
    main()