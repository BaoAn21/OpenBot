# Self-contained training image for OpenBot's driving policy.
#
# Build once (on either machine, GPU not required to build):
#   docker build -t openbot-train .
#
# Run on a machine with an NVIDIA GPU + nvidia-container-toolkit installed:
#   ./docker-run-train.sh --create_tf_record --model pilot_net --batch_size 128 --num_epochs 100 --batch_norm --flip_aug
#
# dataset/ and models/ are NOT baked into the image - they are bind-mounted
# at run time (see docker-run-train.sh), so re-training with new data or a
# different machine never requires rebuilding the image.

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget bzip2 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

WORKDIR /workspace

# Install the exact env pinned by environment_linux.yml (cudatoolkit + cudnn
# come from conda-forge, so only the host NVIDIA driver is needed at run
# time - no CUDA base image required here).
COPY environment_linux.yml requirements.txt ./
RUN conda config --set auto_activate false && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda env create -f environment_linux.yml && \
    conda run -n openbot pip install --no-cache-dir -r requirements.txt && \
    conda clean -afy

# Only the training package is needed for headless training (no notebook,
# no frontend, no web server).
COPY openbot ./openbot

# Matches the LD_LIBRARY_PATH tweak the README has you add manually for a
# conda install, so the conda-forge cudatoolkit/cudnn libs are found.
ENV LD_LIBRARY_PATH=/opt/conda/envs/openbot/lib

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "openbot", "python", "-m", "openbot.train"]
