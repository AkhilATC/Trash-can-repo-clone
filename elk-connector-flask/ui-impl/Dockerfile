FROM python:3.7.3-slim
# We copy just the requirements.txt first to leverage Docker cache
WORKDIR /app
RUN pip install --upgrade pip
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

EXPOSE 5000
COPY . /app
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
