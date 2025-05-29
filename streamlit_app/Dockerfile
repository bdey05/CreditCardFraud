FROM python:3.11-slim 

RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app 

COPY requirements.txt ./requirements.txt 

RUN GIT_TERMINAL_PROMPT=0 pip install -r requirements.txt 

COPY ./streamlit_app/ . 

EXPOSE 8501 

CMD ["streamlit", "run", "app.py"]



