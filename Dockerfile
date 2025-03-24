# Base CUDA image
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

LABEL maintainer="breakstring@hotmail.com"
LABEL version="dev-20240209"
LABEL description="Docker image for GPT-SoVITS"


# Install 3rd party apps
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt-get update && \
    apt-get install -y --no-install-recommends tzdata ffmpeg libsox-dev parallel aria2 git git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# Copy only requirements.txt initially to leverage Docker cache
WORKDIR /workspace
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

COPY ./Docker/LangSegment/__init__.py /usr/local/lib/python3.10/dist-packages/LangSegment/__init__.py

# Define a build-time argument for image type
ARG IMAGE_TYPE=full

# Conditional logic based on the IMAGE_TYPE argument
# Always copy the Docker directory, but only use it if IMAGE_TYPE is not "elite"
COPY ./Docker /workspace/Docker
# elite 类型的镜像里面不包含额外的模型
RUN if [ "$IMAGE_TYPE" != "elite" ]; then \
        chmod +x /workspace/Docker/download.sh && \
        /workspace/Docker/download.sh && \
        python /workspace/Docker/download.py && \
        python -m nltk.downloader averaged_perceptron_tagger averaged_perceptron_tagger_eng cmudict && \
        rm -rf /workspace/GPT_SoVITS/pretrained_models/ && \
        git clone https://huggingface.co/lj1995/GPT-SoVITS /workspace/GPT_SoVITS/pretrained_models; \
    fi


# Copy the rest of the application
COPY . /workspace

EXPOSE 9871 9872 9873 9874 9880

CMD ["python", "runpod_handler.py"]
