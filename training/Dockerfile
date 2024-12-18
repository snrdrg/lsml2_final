FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
EXPOSE 8888
RUN mkdir /home/train
ADD . /home/train
RUN apt-get update
RUN apt-get install -y tzdata
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y bash
RUN apt-get install -y python3.11
RUN apt-get install python3-pip -y
RUN pip3 install --upgrade setuptools
RUN pip3 install wheel
RUN pip3 install argon2-cffi==20.1.0 
RUN pip3 install torch==2.4.1 torchvision==0.19.0 torchaudio --index-url https://download.pytorch.org/whl/test/cu124
RUN pip3 install jupyter
RUN pip3 install --no-cache-dir -r /home/train/requirements.txt
WORKDIR /home/train
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]