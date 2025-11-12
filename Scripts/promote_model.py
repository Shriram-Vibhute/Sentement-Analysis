import os
import mlflow


def promote_model():
    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(f'http://ec2-13-234-110-125.ap-south-1.compute.amazonaws.com:8080/')

    # Get the latest version in staging
    client = mlflow.MlflowClient()
    model_name = "bagging_classifier"
    
    try: 
        if len(client.get_model_version_by_alias(model_name, "production")) >= 1:
            previous_version_staging = client.get_model_version_by_alias(model_name, "production")[0].version
            client.delete_registered_model_alias(
                name=model_name,
                alias="production"
            )
            client.set_registered_model_alias(
                name=model_name,
                alias="archived",
                version=previous_version_staging
            )
        else:
            previous_version_staging = None
    
    except Exception as e:
        print(f"There are no any models in Production stage.")

    latest_version_staging = client.get_latest_versions(model_name, stages=["None"])[0].version

    try:
        if "staging" in client.get_latest_versions(model_name, stages=["None"])[0].aliases:
            client.delete_registered_model_alias(
                name=model_name,
                alias="staging"
            )
    except Exception as e:
        print(f"Model version {latest_version_staging} does not have staging alias.")

    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=latest_version_staging
    )

    print(f"Model version {latest_version_staging} promoted to Production")

if __name__ == "__main__":
    promote_model()