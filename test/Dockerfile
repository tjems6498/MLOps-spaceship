FROM python:3.8

RUN pip install -U pandas mlflow boto3 lightgbm
RUN mkdir -p /app

ARG DISABLE_CACHE
ADD test.py /app/

WORKDIR /app
ENTRYPOINT ["python", "test.py" ]


# CUR_TIME=$(date +%s)
# docker build --build-arg DISABLE_CACHE=$CUR_TIME -t tjems6498/spaceship-titanic-test:v0.3 .
# docker push tjems6498/spaceship-titanic-test:v0.3