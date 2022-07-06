FROM python:3.8

RUN pip install -U pandas scikit-learn
RUN mkdir -p /app

ARG DISABLE_CACHE
ADD preprocessing.py /app/

WORKDIR /app
ENTRYPOINT ["python", "preprocessing.py" ]


# CUR_TIME=$(date +%s)
# docker build --build-arg DISABLE_CACHE=$CUR_TIME -t tjems6498/spaceship-titanic-preprocess:v0.1 .
# docker push tjems6498/spaceship-titanic-preprocess:v0.1