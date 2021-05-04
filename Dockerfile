FROM python:3.8.9

WORKDIR /usr/src

RUN apt install git

RUN python3 -m pip install torch torchvision google-cloud-storage numpy pytorch-lightning

RUN git clone http://github.com/jhonasiv/mpnet.git /usr/src/mpnet && cd /usr/src/mpnet && git checkout gcloud

ENV PYTHONPATH $PYTHONPATH:/usr/src/mpnet

ENTRYPOINT ["python3", "/usr/src/mpnet/MPNet/enet/statistics.py"]
CMD ["--num_gpus", "1", "--workers", "0", "--itt", "40", "--gcloud_project","avid-battery-312014", "--bucket", "mpnet-bucket", "--model_id", "0", "--log_path", "data/cae", "--batch_size", "1000"]