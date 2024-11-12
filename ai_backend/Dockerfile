FROM python:3.9-slim

WORKDIR /app
COPY . /app


RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

RUN pip install -r requirements.txt

COPY deploy.prototxt mobilenet_iter_73000.caffemodel /app/

EXPOSE 5001
CMD ["python", "app.py"]
