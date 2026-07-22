# Usando a imagem mais recente do PyTorch com CUDA 13.0 (compatível com sua RTX 50 e host CUDA 13.0)
FROM pytorch/pytorch:2.9.0-cuda13.0-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependências de sistema para OpenCV, Open3D e ferramentas de rede
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Instalar bibliotecas essenciais para Perception e Autodrive
# Fixando NumPy para evitar conflitos com PyTorch
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir numpy==1.26.4 \
    opencv-python \
    open3d \
    nuscenes-devkit \
    pyquaternion \
    scikit-learn \
    matplotlib \
    trimesh \
    tqdm \
    pandas \
    scipy

# Definir o diretório de trabalho raiz
WORKDIR /workspace

# O PYTHONPATH aponta para a sua pasta de código fonte
ENV PYTHONPATH="/workspace/src"

CMD ["bash"]
