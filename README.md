# Tensorflow 代码示例
You have just find the code for Tensoflow r1.0 with Chinese annotations</br>  
## Tensorflow 安装教程  
该示例代码均基于Tensorflow 1.0版本。由于1.0和以前好多API函数的改变，网上好多教程都是基于0.12版本的，升级1.0以后的需要改动代码才能使用。为此做了这个基于1.0api的示例代码。1.0版本以后是主流，建议大家尝试学习。
每个部分详细的介绍请参考文件夹里面的readme文件
[Tensorflow官网](https://www.tensorflow.org)
### Linux安装  
  详细的安装过程可以查看官网给出的示例，这里仅提供简单的pip安装方式
  ```bash
 $ pip install tensorflow      # Python 2.7; CPU support (no GPU support)
 $ pip3 install tensorflow     # Python 3.n; CPU support (no GPU support)
 $ pip install tensorflow-gpu  # Python 2.7;  GPU support
 $ pip3 install tensorflow-gpu # Python 3.n; GPU support
 ```
 如果第一步失败，可以使用以下的命令安装最新版的tensorflow
 ```bash
 $ sudo pip  install --upgrade TF_PYTHON_URL   # Python 2.7
 $ sudo pip3 install --upgrade TF_PYTHON_URL   # Python 3.N 
 ```
### Windows安装
  Windows使用pip安装或者使用anaconda安装均可，但需要python3.0以上版本
  ```bash
  C:\> pip3 install --upgrade tensorflow
To install the GPU version of TensorFlow, enter the following command:
  C:\> pip3 install --upgrade tensorflow-gpu
  ```
  * installing with Anaconda
  ```
  1. 首先去anaconda官网，下载安装anaconda。[下载地址]https://www.continuum.io/downloads
  2.使用以下命令，创建一个conda的环境  
    C:> conda create -n tensorflow 
  3.通过下面的命令，激活这个环境  
    C:> activate tensorflow    
    (tensorflow)C:>  # Your prompt should change  
  4.安装仅支持cpu版本的tensorflow:

    (tensorflow)C:> pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-1.0.1-cp35-cp35m-win_amd64.whl 

  5.安装支持GPU的tensorflow，需要配置cuda和cudnn等gpu运行环境:

    (tensorflow)C:> pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-1.0.1-cp35-cp35m-win_amd64.whl 
    ```
    


  
