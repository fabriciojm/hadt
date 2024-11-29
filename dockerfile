FROM python:3.10.6-buster

COPY requirements.txt /requirements.txt

COPY arrhythmia /arrhythmia

RUN pip install -r requirements.txt

CMD TBD
