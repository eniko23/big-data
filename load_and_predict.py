# load_and_predict.py

import pandas as pd
import joblib

model = joblib.load("house_price_model.pkl")

test = pd.read_csv("test.csv")
test_ids = test["Id"]
X_test = test.drop("Id", axis=1)

predictions = model.predict(X_test)

submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": predictions
})
submission.to_csv("submission.csv", index=False)

print(" 'submission.csv' başarıyla oluşturuldu (model dosyasından tahmin).")

print(submission.head())
