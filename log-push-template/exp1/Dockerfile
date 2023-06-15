#Specifying the base image
FROM python:3.10
#here the dockerfile is pulling the python 3.10 from docker hub which already has python installed so we have all the things we need to have python in our container.

ADD log-gen.py .
ADD requirements.txt .
#Here we added the python file that we want to run in docker and define its location.

RUN pip install -r requirements.txt
#Here we installed the dependencies, we are using the pygame library in our main.py file so we have to use the pip command for installing the library

CMD [ "python3","./log-gen.py" ]