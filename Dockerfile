FROM python:3.7-slim

COPY setup.py setup.py
COPY requirements.txt requirements.txt
COPY Makefile Makefile
COPY src src
COPY streamlit streamlit

RUN make install
RUN wget https://github.com/stephenllh/mnist-mlops/releases/latest/download/mnist_cnn.pt

EXPOSE 8080

CMD ["streamlit", "streamlit/st_app.py"]