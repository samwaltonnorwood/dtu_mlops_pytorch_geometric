FROM python:3.8-slim

RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt install -y git && \
apt clean && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/hviidhenrik/dtu_mlops_pytorch_geometric.git
WORKDIR "/dtu_mlops_pytorch_geometric"

RUN pip install torch==1.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html --no-cache-dir && \
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cpu.html --no-cache-dir && \
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cpu.html --no-cache-dir && \
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.10.0+cpu.html --no-cache-dir && \
pip install torch-geometric --no-cache-dir && \
pip install -e . --no-cache-dir && \
pip install -r requirements.txt --no-cache-dir

RUN wandb login daf73ad785871e297c736ddfb67826aa1663e305 #service account API key

ENTRYPOINT ["python", "-u", "src/models/train.py", "--conf", "config/example.yaml", "--dataset", "QM9", "--log-dir", "output/"]
