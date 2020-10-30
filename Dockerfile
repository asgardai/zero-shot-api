FROM python:3.8-slim

ENV PIP_NO_CACHE_DIR=off

RUN apt-get update && \
    apt-get install -y libopenblas-dev gcc g++ git wget && \
    pip install --no-cache-dir -U pip && \
    pip install poetry

COPY pyproject.toml /zero-shot/pyproject.toml
WORKDIR /zero-shot

RUN poetry config virtualenvs.create false && poetry install --no-root && \
    pip install --no-cache-dir https://download.pytorch.org/whl/cpu/torch-1.7.0%2Bcpu-cp38-cp38-linux_x86_64.whl

COPY . /zero-shot
RUN poetry install

CMD gunicorn -k uvicorn.workers.UvicornWorker -w 1 main:app -b :5000 --timeout=600

# Open port & add env variable
EXPOSE 5000
ENV PORT=5000

# clean stuff
RUN apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
