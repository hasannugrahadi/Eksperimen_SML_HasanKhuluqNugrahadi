import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COLUMNS_TO_DROP = [
    "BROKERTITLE",
    "ADDRESS",
    "MAIN_ADDRESS",
    "LONG_NAME",
    "STATE",
    "FORMATTED_ADDRESS",
]

NUMERIC_FEATURES = [
    "BEDS",
    "BATH",
    "PROPERTYSQFT",
    "LATITUDE",
    "LONGITUDE",
]

CATEGORICAL_FEATURES = [
    "TYPE",
    "LOCALITY",
    "SUBLOCALITY",
    "ADMINISTRATIVE_AREA_LEVEL_2",
]


def remove_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=COLUMNS_TO_DROP)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = remove_unused_columns(df)
    df = remove_duplicates(df)

    df = df.reset_index(drop=True)

    X = df.drop(columns=["PRICE"])
    y = df["PRICE"]

    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    feature_names = preprocessor.get_feature_names_out()

    X_processed = pd.DataFrame(
        X_processed,
        columns=feature_names,
    )
    processed_df = pd.concat(
        [
            X_processed,
            y,
        ],
        axis=1,
    )
    return processed_df


if __name__ == "__main__":
    INPUT_PATH = "preprocessing/ny_house_preprocessing.csv"
    OUTPUT_PATH = "ny_house_preprocessing.csv"

    df = pd.read_csv(INPUT_PATH)
    processed_df = preprocess(df)
    processed_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )
