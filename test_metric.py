import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
df = pd.read_csv("bank_churn.csv")
df = df.drop(['customer_id', 'nome', 'email', 'numero_conta', 'data_nascimento'], axis=1)
median_salary = df["estimated_salary"].median()
df["estimated_salary"].fillna(df["estimated_salary"].median(), inplace=True)
df = pd.get_dummies(df, columns=['cidade', 'country', 'gender'])
x = df.drop("churn", axis=1)
y = df["churn"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = joblib.load("model.pkl")
prediction = model.predict(X_test)
print("classification report")
print(classification_report(y_test, prediction))
print("confusion matrix")
print(confusion_matrix(y_test, prediction))

