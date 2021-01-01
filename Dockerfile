FROM python:3.8-buster
RUN apt update && apt install -y zsh python-opengl \
    && apt clean \
    && rm -rf /var/lib/apt/lists/* \
    && curl -L https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh | sh
WORKDIR /code
COPY . /code
RUN pip install -r requirements.txt \
    && jt -t monokai -f inconsolata -N -T -fs 11 -nfs 11 -cellw 90% -lineh 140
ENV PYTHONPATH='/code'
ENV DISPLAY=':0.0'
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics