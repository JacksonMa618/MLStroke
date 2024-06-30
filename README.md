# MLStroke
### Set up the python environment

```
conda create -n e2ec python=3.7
conda activate e2ec

# install pytorch, the cuda version is 11.1
# You can also install other versions of cuda and pytorch, but please make sure # that the pytorch cuda is consistent with the system cuda

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

pip install Cython==0.28.2
pip install -r requirements.txt


### Set up datasets

#### CCSE

1. Download the CCSE datasets from  https://pan.baidu.com/s/1ri-nBygvbxKG4rvFO18HeA?pwd=7jt3
2. Organize the dataset as the following structure:
    ```
    ├── /path/to/CCSE
    │   ├── annotations
    │   ├── images
    ```
    
4. Create a soft link:
    ```
    ROOT=/path/to/e2ec
    cd $ROOT/data
    ln -s /path/to/CCSE CCSE
    ```
#### CSSCD

1. Download the CSSCD datasets from https://pan.baidu.com/s/14N2zfWZpQpa-BvgBktsVsg?pwd=njmt and  https://pan.baidu.com/s/1IDvLieE1xf0tHkrUo5OheQ?pwd=4fvj
2. Organize the dataset as the following structure:
    ```
    ├── /path/to/CSSCD
    │   ├── annotations
    │   ├── images
    ```
    
4. Create a soft link:
    ```
    ROOT=/path/to/e2ec
    cd $ROOT/data
    ln -s /path/to/CSSCD CSSCD
