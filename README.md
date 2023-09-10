# Generatice Convolutional Occupancy Networks

Our project aims to generate 3D synthetic data using a Variational Autoencoder. 

We build upon the [Convolutional Occupancy Netwrok](https://github.com/autonomousvision/convolutional_occupancy_networks) framework, which is proficient at detailed 3D reconstruction, ranging from individual objects to entire 3D scenes. However, the original framework produces deterministic results. In our project, we extend it with a Bayesian probabilistic approach. This extension allows our model to generate synthetic 3D data with variability. You can find more details in the accompanying report within this repository.

## Environment setup

1. First create an anaconda environment using:
```
conda create -n conv_vae python=3.8
conda activate conv_vae
```

2. Install numpy and cython: 
```
pip install numpy cython 
```

3. Compile the extension modules:
```
python setup.py build_ext --inplace
```

4. Install PyTorch with cuda 11.7 following the official instruction:
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

5. Install torch-scatter:
```
conda install pytorch-scatter -c pyg
```

6. Install remaining dependencies with pip:
```
pip install scipy pandas scikit-image matplotlib tensorboardx trimesh pyyaml tqdm plyfile
```

## Dataset

We use two datasets in our project. You can download the preprocessed ShapeNet dataset directly. However, for the IntrA dataset, please follow the instructions below for preprocessing.

### ShapeNet
You can download the dataset(73.4 GB) by running the [script](https://github.com/autonomousvision/occupancy_networks#preprocessed-data) from Occupancy Networks. Once downloaded, you will find the dataset in the `data/ShapeNet` folder.

### IntrA
Download IntrA dataset from the [official github](https://github.com/intra3d2019/IntrA). 

Process the downloaded dataset using [mesh-fusin](https://github.com/davidstutz/mesh-fusion) following the instructions in the ```external/mesh-fusion``` folder. This allow to generate pointclouds with occupancy score from ```.off``` mesh.

## Usage

### Training
To train a new network from scratch, run:
```
python train.py CONFIG.yaml
```
and replace ```CONFIG.yaml``` with your config file.

For example, for training on ShapeNet dataset, run:
```
python train.py configs/pointcloud/shapenet_3plane.yaml
```

### Mesh Generation
To generate meshes using a trained model, use: 
```
python generate.py CONFIG.yaml
```

### Evaluation
To evaluate the model, run `eval_meshes.py`:
```
python eval_meshes.py CONFIG.yaml
```