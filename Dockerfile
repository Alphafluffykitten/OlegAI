FROM python:3.9
WORKDIR /app/code
RUN mkdir tgstat_logs
COPY . .
RUN pip3 install -r requirements.txt
