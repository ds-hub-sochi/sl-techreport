# This repository provide possibility for reproductioning results in technical report

### Install MMaction2 dev-1.x

Create new environment:
```
conda create -n mmaction2dev python=3.9
conda activate mmaction2dev
```

### Install PyTorch:
```
conda install pytorch=={pytorch_version} torchvision=={torchvision_version} cudatoolkit={cudatoolkit_version} -c pytorch -c conda-forge
```
### Install MMEngine
```
pip install -U openmim
mim install mmengine 'mmcv>=2.0.0rc1'
```
### Install MMAction2 and dependencies
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
### Training

#### Run the command to start training
```
cd mmaction2
bash tools/dist_train.sh ${CONFIG} ${GPUS} [PY_ARGS]
```
Example:
```
bash tools/dist_train.sh configs/recognition/mvit/mvit-small-p244_64x1x1_kinetics400-rgb-RSL2AUTSL2WLASL.py 8 --amp --auto-scale-lr
```
References: https://mmaction2.readthedocs.io/en/dev-1.x/user_guides/4_train_test.html

### Convert model:
#### Run the command to start model conversion
```
cd mmdeploy
python torch2onnx.py [DEPLOY_CONFIG] [MODEL_CONFIG] [CHECKPOINT] [VIDEO.mp4] [OPTIONAL_PY_ARGS]
```