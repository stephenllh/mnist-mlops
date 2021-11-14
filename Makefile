env:
	python3 -m venv ~/.venv &&\
	source ~/.venv/bin/activate


install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt &&\
	pip install -e .


style:
	black .
	flake8 .


test:
	pytest


deploy:
	git push heroku master


.PHONY: streamlit
streamlit:
	streamlit run streamlit/st_app.py


all: env install style test