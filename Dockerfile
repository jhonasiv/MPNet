FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04

# Installs necessary dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
         wget \
         curl \
         python3-dev \
         python3-pip && \
     rm -rf /var/lib/apt/lists/*


WORKDIR /usr/src

RUN apt update && apt install -y --no-install-recommends git

RUN python3.8 -m pip install torch torchvision google-cloud-storage numpy pytorch-lightning

RUN git clone http://github.com/jhonasiv/mpnet.git /usr/src/mpnet && cd /usr/src/mpnet && git checkout gcloud

ENV PYTHONPATH $PYTHONPATH:/usr/src/mpnet
# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

ENTRYPOINT ["python3.8", "/usr/src/mpnet/MPNet/enet/statistics.py","--num_gpus", "1", "--itt", "40",\
 "--gcloud_project","avid-battery-312014", "--bucket", "mpnet-bucket", "--log_path", "data/cae" ]
CMD ["--model_id", "0","--batch_size", "1000", "--learning_rate", "1e-2", "--workers", "0"]