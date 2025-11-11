# inference
import mlflow
import mlflow.pyfunc
import pathlib
import pandas as pd
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://ec2-65-2-178-74.ap-south-1.compute.amazonaws.com:8080/")

home_dir = pathlib.Path(__file__).parent.parent.parent
data_path = home_dir / "data" / "features" / "test.csv"
data = pd.read_csv(data_path).dropna()
X = data.drop(columns=["sentiment","content"])[:5]

model_name = "bagging_clf"
model_version = 1

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

y = data["sentiment"][:5]
print(accuracy_score(y, model.predict(X)))