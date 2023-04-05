
# Russian Sign Language Classification Report
## Introduction
There is a training procedure and preparing for deployment of the model from [our tech report](https://arxiv.org/abs/2302.07693).
We will release model checkpoints later.

## Finetuning procedure
![alt text](https://i.imgur.com/DkgCqpV.png)

## Datasets used for training:
1) WLASL - https://dxli94.github.io/WLASL/
2) AUTSL http://cvml.ankara.edu.tr/datasets/
3) RSL - comming soon

## Environment preparation

### Install MMaction2 dev-1.x
Create new environment:
```bash
conda create -n mmaction2dev python=3.8
conda activate mmaction2dev
```

### Install PyTorch:
```bash
conda install pytorch=={pytorch_version} torchvision=={torchvision_version} cudatoolkit={cudatoolkit_version} -c pytorch -c conda-forge
# Configuration we used:
conda install pytorch=1.10.1 torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```
### Install MMEngine
```
pip install -U openmim
mim install mmengine 'mmcv>=2.0.0rc1'
```
### Install MMAction2
Please note we use dev-1.x branch
```
cd mmaction2
pip install -v -e .
pip install -r requirements.txt
cd ..
```
### Install MMDeploy and dependencies
```
# install MMDeploy
wget https://github.com/open-mmlab/mmdeploy/releases/download/v1.0.0rc1/mmdeploy-1.0.0rc1-linux-x86_64-onnxruntime1.8.1.tar.gz
tar -zxvf mmdeploy-1.0.0rc1-linux-x86_64-onnxruntime1.8.1.tar.gz
cd mmdeploy-1.0.0rc1-linux-x86_64-onnxruntime1.8.1
pip install dist/mmdeploy-1.0.0rc1-py3-none-linux_x86_64.whl
pip install sdk/python/mmdeploy_python-1.0.0rc1-cp38-none-linux_x86_64.whl
cd ..
# install inference engine: ONNX Runtime
pip install onnxruntime==1.8.1
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz
tar -zxvf onnxruntime-linux-x64-1.8.1.tgz
export ONNXRUNTIME_DIR=$(pwd)/onnxruntime-linux-x64-1.8.1
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
```
## Training procedure

```
cd mmaction2
bash tools/dist_train.sh ${CONFIG} ${GPUS} [PY_ARGS]
```
Example:
```
bash tools/dist_train.sh configs/recognition/mvit/mvit-small-p244_64x1x1_kinetics400-rgb-RSL2AUTSL2WLASL.py 8 --amp --auto-scale-lr
```
References: https://mmaction2.readthedocs.io/en/dev-1.x/user_guides/4_train_test.html

Feel free to change an order of datasets for training and fine-tuning of a model.

## Convert model to ONNX format:

```
cd mmdeploy
python torch2onnx.py [DEPLOY_CONFIG] [MODEL_CONFIG] [CHECKPOINT] [VIDEO.mp4] [OPTIONAL_PY_ARGS]
```
References: https://mmaction2.readthedocs.io/en/dev-1.x/user_guides/4_train_test.html

## Citation
If you find this repository useful, please consider giving a star ⭐ to this repo and citation:
```
@misc{novopoltsev2023finetuning,
      title={Fine-tuning of sign language recognition models: a technical report},
      author={Maxim Novopoltsev and Leonid Verkhovtsev and Ruslan Murtazin and Dmitriy Milevich and Iuliia Zemtsova},
      year={2023},
      eprint={2302.07693},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
