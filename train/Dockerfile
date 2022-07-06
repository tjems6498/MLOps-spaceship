FROM python:3.8

RUN pip install -U numpy pandas scikit-learn lazypredict mlflow boto3
RUN mkdir -p /app

ARG DISABLE_CACHE
ADD train.py /app/

WORKDIR /app
ENTRYPOINT ["python", "train.py" ]


# CUR_TIME=$(date +%s)
# docker build --build-arg DISABLE_CACHE=$CUR_TIME -t tjems6498/spaceship-titanic-train:v0.0.1 .
# docker push tjems6498/spaceship-titanic-train:v0.0.1