"""One training-fitted feature pipeline shared by trees and the network."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# The data dictionary defines NA as absence for these categorical features.
ABSENCE_COLUMNS = (
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish",
    "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature",
)


def prepare_features(frame):
    """Apply fixed semantic rules; no statistics are learned here."""
    features = frame.drop(columns=["Id", "SalePrice"], errors="ignore").copy()
    # Numeric-looking codes are nominal dwelling types, not measurements.
    features["MSSubClass"] = features["MSSubClass"].map(
        lambda value: str(int(value)) if not np.isnan(value) else np.nan
    )
    for column in ABSENCE_COLUMNS:
        features[column] = features[column].fillna("Absent")
    return features


def build_preprocessor(training_features):
    """Determine column roles using training data, then return an unfitted pipeline."""
    categorical = training_features.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric = [column for column in training_features if column not in categorical]
    return ColumnTransformer(
        [
            ("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]), numeric),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                         dtype=np.float32)),
            ]), categorical),
        ],
        remainder="drop",
        sparse_threshold=0,
    )
