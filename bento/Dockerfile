FROM python:3.8

RUN pip install -U pandas scikit-learn mlflow boto3 bentoml protobuf~=3.19.0
RUN mkdir -p /app

ARG DISABLE_CACHE
ADD bento_packer.py /app/
ADD bento_service.py /app/
ADD load_model.py /app/


WORKDIR /app
ENTRYPOINT ["python", "bento_packer.py" ]




# CUR_TIME=$(date +%s)
# docker build --build-arg DISABLE_CACHE=$CUR_TIME -t tjems6498/spaceship-titanic-bento:v0.0.1 .
# docker push tjems6498/spaceship-titanic-bento:v0.0.1

