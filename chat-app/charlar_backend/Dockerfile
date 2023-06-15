FROM python:3.7.3-slim
# We copy just the requirements.txt first to leverage Docker cache
WORKDIR /apps
COPY requirements.txt ./requirements.txt
RUN pip3 --no-cache-dir install -r requirements.txt
COPY . /apps
ENTRYPOINT [ "python3", "./run.py" ]

