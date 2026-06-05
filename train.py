import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
df = pd.read_csv("bank_churn.csv")
df = df.drop(['customer_id', 'nome', 'email', 'numero_conta', 'data_nascimento'], axis=1)
median_salary = df["estimated_salary"].median()
df["estimated_salary"].fillna(df["estimated_salary"].median(), inplace=True)
df = pd.get_dummies(df, columns=['cidade', 'country', 'gender'])
x = df.drop("churn", axis=1)
y = df["churn"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter= 1000, class_weight="balanced")
model.fit(X_train, y_train)
joblib.dump(model, "model.pkl")
joblib.dump(x.columns.to_list(), "features.pkl")
print("congratulations you trained your model")
print(df.isnull().sum())