FROM python:3.8

RUN pip install -U numpy pandas scikit-learn wandb
RUN mkdir -p /app

ARG DISABLE_CACHE
ADD sweep.py /app/

WORKDIR /app
ENTRYPOINT ["python", "sweep.py" ]


# CUR_TIME=$(date +%s)
# docker build --build-arg DISABLE_CACHE=$CUR_TIME -t tjems6498/spaceship-titanic-hyp:v0.0.1 .
# docker push tjems6498/spaceship-titanic-hyp:v0.0.1