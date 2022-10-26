# syntax=docker/dockerfile:1
FROM tensorflow/tensorflow:2.10.0-gpu

# RUN rm -rf /usr/local/cuda/lib64/stubs

COPY <<-EOF /requirements.txt
diffusers==0.6.0
torch==1.12.1+cu116
transformers==4.23.1
discord-py-interactions==4.3.4
EOF

RUN pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu116

RUN useradd -m huggingface
USER huggingface
WORKDIR /home/huggingface

RUN <<EOF
mkdir -p /home/huggingface/.cache/huggingface
mkdir -p /home/huggingface/output
EOF

RUN pip install -U future

COPY diffusion.py .
COPY workerqueue.py .
COPY entrypoint.py .

ENTRYPOINT [ "python", "/home/huggingface/entrypoint.py" ]
