FROM python:3.10-slim

WORKDIR /app

COPY ./arrhythmia/api ./arrhythmia/api
COPY ./requirements_prod.txt .
COPY ./service-account.json .

RUN pip install --upgrade pip
RUN pip install --no-cache -r requirements_prod.txt

EXPOSE $PORT

CMD uvicorn arrhythmia.api.fast:app --host 0.0.0.0 --port $PORT
