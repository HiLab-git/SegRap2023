# Pull the docker image | 拉取镜像
# FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

FROM python:3.10-slim

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"
COPY --chown=user:user requirements.txt /opt/app/

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools
RUN python -m piptools sync requirements.txt

# RUN python -m pip install --user -r requirements.txt
RUN python -m pip install --user torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
RUN python -m pip install --no-deps --user nnunet==1.7.1

COPY --chown=user:user process.py /opt/app/
COPY --chown=user:user inference_code.py /opt/app/
COPY --chown=user:user post_processing.py /opt/app/
# COPY --chown=user:user images/ /opt/app/images/
COPY --chown=user:user weight/ /opt/app/weight/
COPY --chown=user:user output/ /opt/app/


ENTRYPOINT [ "python", "-m", "process" ]
