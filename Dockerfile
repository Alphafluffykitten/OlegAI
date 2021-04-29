FROM python:3.9
WORKDIR /app/code
COPY . .
RUN pip3 install -r requirements.txt
