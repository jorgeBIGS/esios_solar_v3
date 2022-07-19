FROM tensorflow/tensorflow:latest-gpu

MAINTAINER Tomás Cabello López "tomascl1998@gmail.com"

ENV VIRTUAL_ENV=/opt/venv

RUN apt-get update

RUN apt-get -y install python3.8-venv

RUN python3 -m venv $VIRTUAL_ENV

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip3 install wheel

RUN mkdir /esios_solar

WORKDIR /esios_solar

COPY requirements.txt /esios_solar

RUN pip install -r requirements.txt

COPY ./esios_solar_v3 /esios_solar

CMD ["python3", "run.py"]
