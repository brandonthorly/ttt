# base image
FROM python:3.7-slim

RUN apt-get update && apt-get install -y cmake build-essential gcc g++ git wget libgl1-mesa-glx

# set working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# add and install requirements
COPY ./requirements.txt /usr/src/app/requirements.txt
RUN pip install -r requirements.txt

# add app
COPY . /usr/src/app

#CMD ["/bin/bash"]
CMD ["gunicorn", "-b", "0.0.0.0:5000", "wsgi"]