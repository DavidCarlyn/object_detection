# syntax=docker/dockerfile:1
FROM ubuntu

WORKDIR /home

RUN apt-get update -y && apt-get install -y \
    lsb-core \
    git \
    python3-pip

RUN git clone https://github.com/DavidCarlyn/object_detection.git
RUN pip install -r object_detection/requirements.txt

CMD ["chmod +x ./data/train.sh; ./data/train.sh"]


