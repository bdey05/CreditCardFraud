FROM python:3.11-slim 

WORKDIR /usr/src/app 

COPY requirements.txt ./requirements.txt 

RUN pip install -r requirements.txt 

COPY ./backend/ . 

ENV FLASK_APP=api 
ENV FLASK_ENV=development 
ENV FLASK_DEBUG=1 

EXPOSE 5000 

CMD ["python", "-m", "flask", "--app", "api", "run", "--host=0.0.0.0", "--reload"]



