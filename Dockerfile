# syntax=docker/dockerfile:1
FROM tensorflow/tensorflow:2.10.0-gpu

# RUN rm -rf /usr/local/cuda/lib64/stubs

# These deps are large, so put them in their own layer to save rebuild time
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu116 \
    diffusers==0.6.0 \
    torch==1.12.1+cu116

COPY <<-EOF /requirements.txt
diffusers==0.6.0
torch==1.12.1+cu116
transformers==4.23.1
discord-py-interactions==4.3.4
pathvalidate==2.5.2
EOF

RUN pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu116

RUN pip install -U future

RUN useradd -m huggingface
USER huggingface
WORKDIR /home/huggingface

RUN <<EOF
mkdir -p /home/huggingface/.cache/huggingface
mkdir -p /home/huggingface/output
EOF

COPY config.py .
COPY diffusion.py .
COPY dto.py .
COPY scheduler.py .
COPY entrypoint.py .

ENTRYPOINT [ "python", "/home/huggingface/entrypoint.py" ]
