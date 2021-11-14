FROM python:3.7-slim

COPY setup.py setup.py
COPY requirements.txt requirements.txt
COPY Makefile Makefile
COPY src src
COPY streamlit streamlit

RUN apt-get update &&\
    apt-get install -y wget &&\
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip &&\
	pip install --no-cache-dir -r requirements.txt &&\
	pip install --no-cache-dir -e .

RUN wget https://github.com/stephenllh/mnist-mlops/releases/latest/download/mnist_cnn.pt

EXPOSE 8080

CMD ["streamlit", "run", "streamlit/st_app.py"]