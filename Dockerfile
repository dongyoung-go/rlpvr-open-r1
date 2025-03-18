# Start from the base image
FROM reg.navercorp.com/base-nvidia/nvidia/pytorch:24.07-py3

# Switch to the root user to install packages
USER root

# Install packages and configure SSH
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y openssh-server vim tmux htop libgl1-mesa-glx fping\
    && sed -i 's/#Port 22/Port 8822/g' /etc/ssh/sshd_config \
    && sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/g' /etc/ssh/sshd_config \
    && service ssh restart

RUN apt-get update && apt-get install -y git
WORKDIR /workspace
# RUN git clone https://oss.navercorp.com/dongyoung-go/rlpvr-open-r1.git
RUN git clone https://github.com/dongyoung-go/rlpvr-open-r1.git
WORKDIR /workspace/rlpvr-open-r1

# kinit
# hdfs-connector
RUN apt update && apt install -y krb5-user pdsh \
    && wget http://dist.navercorp.com/repos/release/c3s-hdfs-connector/c3s-hdfs-connector-0.7.tar.gz -O /root/c3s-hdfs-connector-0.7.tar.gz \
    && tar xzvf /root/c3s-hdfs-connector-0.7.tar.gz -C /root \
    && rm /root/c3s-hdfs-connector-0.7.tar.gz \
    && echo "source /root/c3s-hdfs-connector-0.7/bin/source.me" >> /root/.bashrc \
    && echo "alias hdfs='/root/c3s-hdfs-connector-0.7/bin/hdfs-connector'" >> /root/.bashrc

# Copy the keytab into the Docker image
COPY c3s.search-gpt.keytab /root/c3s.search-gpt.keytab

# Install UV package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
# Ensure the UV binary is available in PATH
ENV PATH="/root/.local/bin:${PATH}"
# Create and activate a virtual environment
RUN uv venv openr1 --python 3.11 \
    && source openr1/bin/activate \
    && uv pip install --upgrade pip --link-mode=copy \
    && uv pip install vllm==0.7.2 --link-mode=copy \
    && uv pip install setuptools \
    && uv pip install flash-attn --no-build-isolation \
    && uv pip install wandb --link-mode=copy \
    && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ".[dev]" --link-mode=copy

# Ensure that the "adllm" environment is activated by default
# ENV PATH="/root/miniconda3/envs/adllm/bin:$PATH"
# ENV CONDA_DEFAULT_ENV=adllm

# Run a persistent process to keep the container alive
CMD ["sh", "-c", "while true; do sleep 1; done"]
