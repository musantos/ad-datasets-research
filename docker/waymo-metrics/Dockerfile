# # Usar Python 3.8 - a versão de ouro para o ecossistema TensorFlow 2.11
# FROM python:3.8-slim

# ENV DEBIAN_FRONTEND=noninteractive

# # Instalar dependências de sistema
# RUN apt-get update && apt-get install -y \
#     git build-essential wget \
#     && apt-get clean && rm -rf /var/lib/apt/lists/*

# # 1. Atualizar pip e instalar o TensorFlow CPU 2.11
# RUN pip install --no-cache-dir --upgrade pip
# RUN pip install --no-cache-dir tensorflow-cpu==2.11.0

# # 2. Instalar o Waymo Open Dataset (versão para TF 2.11)
# # Esta versão costuma ser mais resiliente com as dependências de JAX
# RUN pip install --no-cache-dir waymo-open-dataset-tf-2-11-0==1.6.1

# # Configurar diretório e caminhos
# WORKDIR /home/murilo/autonomous_drive
# ENV PYTHONPATH="/home/murilo/autonomous_drive/src/waymo-open-dataset/src"

# CMD ["bash"]

FROM python:3.8-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git build-essential wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir tensorflow-cpu==2.11.0
RUN pip install --no-cache-dir waymo-open-dataset-tf-2-11-0==1.6.1

# Definir um diretório de trabalho curto e limpo
WORKDIR /workspace

# Garantir que o PYTHONPATH não interfira nos binários do Pip
ENV PYTHONPATH=""

CMD ["bash"]
