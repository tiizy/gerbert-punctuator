FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt