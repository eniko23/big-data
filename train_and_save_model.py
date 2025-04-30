
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X = train.drop(["Id", "SalePrice"], axis=1)
y = train["SalePrice"]
X_test = test.drop("Id", axis=1)
test_ids = test["Id"]

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X, y)

predictions = model.predict(X_test)

submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": predictions
})
submission.to_csv("submission.csv", index=False)
print(" 'submission.csv' başarıyla oluşturuldu.")

joblib.dump(model, "house_price_model.pkl")
print(" 'house_price_model.pkl' başarıyla kaydedildi.")
