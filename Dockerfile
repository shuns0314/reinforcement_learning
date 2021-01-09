FROM nvidia/cuda:11.1-runtime-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git zsh python-opengl ffmpeg python3 python3-pip\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && curl -L https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh | sh
RUN ln -s /usr/bin/python3.8 /usr/bin/python \
    && ln -s /usr/bin/pip3 /usr/bin/pip
WORKDIR /code
COPY . /code
RUN pip install -r requirements.txt
ENV PYTHONPATH='/code'
ENV DISPLAY=':0.0'
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics