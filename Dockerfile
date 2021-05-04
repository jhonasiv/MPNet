FROM python:3.8.9

WORKDIR /usr/src

RUN apt install git

RUN python3 -m pip install torch torchvision google-cloud-storage numpy pytorch-lightning

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

ENTRYPOINT ["python3", "/usr/src/mpnet/MPNet/enet/statistics.py"]
CMD ["--num_gpus", "1", "--workers", "0", "--itt", "40", "--gcloud_project","avid-battery-312014", "--bucket", \
 "mpnet-bucket", "--model_id", "0", "--log_path", "data/cae", "--batch_size", \
 "1000"]