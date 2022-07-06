import os
import pdb

import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

def load(model_name, version):
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio-service.kubeflow.svc:9000"
    client = MlflowClient("http://mlflow-server-service.mlflow-system.svc:5000")


    filter_string = "name='{}'".format(model_name)
    results = client.search_model_versions(filter_string)  # 버전별로 따로 나옴
    # print(results)
    for res in results:
        if res.version == str(version):
            model_uri = res.source
            break

    reconstructed_model = mlflow.sklearn.load_model(model_uri)
    return reconstructed_model


if __name__ == '__main__':
    model = load(model_name='Random-Forest', version=1)
    sample = pd.read_csv('/Users/hong-eunpyo/PycharmProjects/spaceship-titanic/raw_data/test.csv')
    output = model.predict(sample)
    print(output)