DTU MLOps project - PyTorch Geometrics
==============================
*Bjarke Hastrup, Sam Norwood, Henrik Hansen*

## Modelling objective
For our project, we have chosen to work with the PyTorch-Geometrics framework to construct graph neural networks (GNN) that can predict the potential energy of small molecules. For this exact purpose Pytorch-Geometric already comes with out-of-the-box GNN implementations but here we will use the TorchMD-net Equivariant Transformer (ET) architecture [1], as this is considered state-of-the-art in 2022. This architecture comprises a series of multihead attention layers operating on embedded graph data, with an output layer which is equivariant to rotation of the graph. Rotationally equivariant graph neural networks have shown state-of-the-art accuracy and data efficiency in modeling the properties of atomistic systems.

## Dataset
We will be using the QM9 dataset [2] which contains geometric, energetic, electronic, and thermodynamic properties for 134k stable small organic molecules made up of Carbon, Hydrogen, Oxygen, Nitrogen and Fluorine, with each molecular structure consisting of 9 atoms at the most. We aim to predict the potential energy given the molecular structure.

## Online data scheme
In certain settings, we wish to model data which is costly to generate and/or is produced continuously by an external resource; a motivating example is the integration of learned predictive models with a queryable experimental or computational workflow. In our project, we emulate this setting by dividing the QM9 dataset into small subsets and create an online data scheme in which these subsets only become available gradually. Our system will keep track of the expanding dataset, triggering a retrain of the model when the total available data has grown by a prespecified fraction.

## ML-Ops tools
We will use version control to keep track of our code base as it evolves. We will record the parameters of the model and its performance as it is retrained. Data version control will be used to record changes to the dataset as new data is obtained. Record will also be kept of which states of the dataset are used for training the model. Additionally, we will use version control to keep track of all trained model objects.

A priori, we expect to use especially the following frameworks taught in the course (subject to change):
 - Version control and deployment: Git + Github, DVC, gcp
 - Structure: Cookiecutter
 - Formatting: black, isort, mypy, typing
 - Reproducibility: hydra, docker
 - Code robustness: Unit testing on data and model construction, coverage, profiling
 - Deep learning: Pytorch-Geometric + Pytorch-Lightning

## References
[1] https://ml4physicalsciences.github.io/2021/files/NeurIPS_ML4PS_2021_54.pdf

[2] http://quantum-machine.org/datasets/

# Instructions
## Setup
The following instructions are heavily inspired by the original repository: 
https://github.com/torchmd/torchmd-net.

Set up a new conda environment:

     conda create --name dtu_mlops_pytorch_geometric python=3.8
     conda activate dtu_mlops_pytorch_geometric

After this, install PyTorch according to your hardware. The correct install can be 
found here
[PyTorch Installation](https://pytorch.org/get-started/locally/#start-locally). Paste
the appropriate install command in from there. Below is shown an example command 
required to install a basic PyTorch version without CUDA on Windows: 

     conda install pytorch torchvision torchaudio cpuonly -c pytorch

Also, install PyTorch Geometric: 

     conda install pytorch-geometric -c rusty1s -c conda-forge

Finally, clone the repo:

     git clone https://github.com/hviidhenrik/dtu_mlops_pytorch_geometric.git
     pip install -e .
     pip install -r requirements.txt
      
The repository should now be ready to run training as described below.

## Run training
For the purpose of the present project, the model is only configured to run on the QM9 
dataset. Specific hyperparameters are set in a yaml file which can be pointed to at the
command line. To run model training with the example configuration:

     python src/models/train.py --conf config/train_hparams.yaml

Which will save the model along with some output metrics to the `models/` directory.

View training diagnostics:

	https://wandb.ai/ml-ops-awesome-25

# TO DO
 - make unit tests run and check coverage :heavy_check_mark:
 - set up github actions workflow :heavy_check_mark:
     - unit tests run on pull request to master branch 
 - do profiling
 - make train.py save model to models/ folder
 - make a predict_model file and some data for proof of concept
 - set up wandb logging :heavy_check_mark:

## Deployment
 - Write predict.py
 - Write handler.py ???
 
 Cloud:
 - Create gcloud project: equivariant-transformer :heavy_check_mark:
 - Create artifact-repo: gnn-artifact-repo :heavy_check_mark:
 - Write dockerfile creates .mar model object using "torch-model-archiver"
   and serves this using torchserve. Like the MNIST example:
 
-----------------------  Dockerfile   ---------------------------------
FROM pytorch/torchserve:0.3.0-cpu

COPY mnist.py mnist_cnn.pt mnist_handler.py /home/model-server/

USER root
RUN printf "\nservice_envelope=json" >> /home/model-server/config.properties
USER model-server

RUN torch-model-archiver \
  --model-name=mnist \
  --version=1.0 \
  --model-file=/home/model-server/mnist.py \
  --serialized-file=/home/model-server/mnist_cnn.pt \
  --handler=/home/model-server/mnist_handler.py \
  --export-path=/home/model-server/model-store

CMD ["torchserve", \
     "--start", \
     "--ts-config=/home/model-server/config.properties", \
     "--models", \
     "mnist=mnist.mar"]
END
-----------------------------------------------------------------------

 - Build image like this

 	docker build \
  	  --tag=us-central1-docker.pkg.dev/equivariant-transformer/gnn-artifact-repo/serve-energy-predictor \
  	  .
  
 - Authenticate docker

 	gcloud auth configure-docker us-central1-docker.pkg.dev

 - and push it to container registry

	docker push us-central1-docker.pkg.dev/equivariant-transformer/gnn-artifact-repo/serve-energy-predictor
	
 - Create model version resource
 
	gcloud beta ai-platform models create gnn-model \
	  --region=us-central1 \
	  --enable-logging \
	  --enable-console-logging
	  
 - and deploy 

	gcloud beta ai-platform versions create v1 \
	  --region=us-central1 \
	  --model=gnn-model \
	  --machine-type=n1-standard-4 \
	  --image=us-central1-docker.pkg.dev/equivariant-transformer/gnn-artifact-repo/serve-energy-predictor \
	  --ports=8080 \
	  --health-route=/ping \
	  --predict-route=/predictions/ ...?
