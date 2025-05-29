FROM python:3.11-slim 

RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app 

COPY requirements.txt ./requirements.txt 

RUN GIT_TERMINAL_PROMPT=0 pip install -r requirements.txt 

COPY ./backend/ . 

ENV FLASK_APP=api 
ENV FLASK_ENV=development 
ENV FLASK_DEBUG=1 

EXPOSE 5000 

CMD ["python", "-m", "flask", "--app", "api", "run", "--host=0.0.0.0", "--reload"]



