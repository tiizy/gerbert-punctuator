FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt