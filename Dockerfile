ARG PYTORCH="1.8.1"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# To fix GPG key error when running apt-get update
RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install system dependencies for opencv-python
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV FORCE_CUDA="1"
ENV MMCV_WITH_OPS="1"

# Install mmcv-full
ARG MMCV="1.3.3"
RUN if [ "${MMCV}" = "" ]; then pip install -U openmim && mim install mmcv-full; else pip install -U openmim && mim install mmcv-full==${MMCV}; fi

# Verify the installation
RUN python -c 'import mmcv;print(mmcv.__version__)'

# Install SCRFD env
RUN apt-get update && apt-get install -y git && git clone https://github.com/giacnguyenbmt/SCRFD-LPD.git \
    && cd SCRFD-LPD \
    && pip install -r requirements/build.txt \
    && pip install -v -e . \
    && pip install -r requirements/optional.txt \
    && pip install scipy \
    && pip install pillow
