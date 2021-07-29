# 一、人脸情绪识别挑战赛介绍
aistudio项目地址：[https://aistudio.baidu.com/aistudio/projectdetail/2199083](https://aistudio.baidu.com/aistudio/projectdetail/2199083)

链接 ：[http://challenge.xfyun.cn/h5/invite?invitaCode=8zCBfV](http://challenge.xfyun.cn/h5/invite?invitaCode=8zCBfV)

![](https://ai-studio-static-online.cdn.bcebos.com/2cc3daaed35c4d8d81ba3fc88cf8a82caae7e771b73f4fcd9b56a03f36e9c0bf)


## 1.赛事背景
人脸表情是传播人类情感信息与协调人际关系的重要方式，表情识别是指从静态照片或视频序列中选择出表情状态，从而确定对人物的情绪与心理变化。在日常生活中人类习惯从面部表情中吸收非言语暗示，那么计算机可以完成类似任务吗？答案是肯定的，但是需要训练它学会识别情绪。



## 2.赛事任务
给定人脸照片完成具体的情绪识别，选手需要根据训练集数据构建情绪识别任务，并对测试集图像进行预测，识别人脸的7种情绪。

## 3. 数据说明
赛题数据由训练集和测试集组成，训练集数据集按照不同情绪的文件夹进行存放。其中：

训练集：2.8W张人脸图像；

测试集：7K张人脸图像；

为了简化任务赛题图像只包含单张人脸，所有图像的尺寸为48*48像素。数据集包括的情绪标签包括以下7类：

* angry
* disgusted
* fearful
* happy
* neutral
* sad
* surprised


# 二、数据处理

## 1.paddlex环境准备
各版本如下所示：

* (from versions: 0.1.0, 0.1.1, 0.1.2, 0.1.3, 0.1.4, 0.1.5, 0.1.6, 0.1.7, 0.1.8, 0.1.9, 0.2.0, 0.2.1, 0.3.0, 1.0.0, 1.0.1, 1.0.2, 1.0.3, 1.0.4, 1.0.5, 1.0.6, 1.0.7, 1.0.8, 1.1.0, 1.1.1, 1.1.5, 1.1.6, 1.2.0, 1.2.1, 1.2.2, 1.2.3, 1.2.4, 1.2.5, 1.2.6, 1.2.7, 1.2.8, 1.3.0, 1.3.1, 1.3.2, 1.3.3, 1.3.4, 1.3.5, 1.3.6, 1.3.7, 1.3.8, 1.3.9, 1.3.10, 1.3.11, 2.0.0rc0, 2.0.0rc3)
* 在此使用2.0.0rc3，此版本和以前的api略有不同，一定要注意，不然会出现莫名的错误！


```python
! pip install paddlex==2.0.0rc3 -i https://mirror.baidu.com/pypi/simple
```

    Looking in indexes: https://mirror.baidu.com/pypi/simple
    Requirement already satisfied: paddlex==2.0.0rc3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (2.0.0rc3)
    Requirement already satisfied: colorama in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex==2.0.0rc3) (0.4.4)
    Requirement already satisfied: shapely>=1.7.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex==2.0.0rc3) (1.7.1)
    Requirement already satisfied: visualdl>=2.1.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex==2.0.0rc3) (2.2.0)
    Requirement already satisfied: tqdm in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex==2.0.0rc3) (4.36.1)
    Requirement already satisfied: scikit-learn==0.23.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex==2.0.0rc3) (0.23.2)
    Requirement already satisfied: scipy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex==2.0.0rc3) (1.6.3)
    Requirement already satisfied: motmetrics in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex==2.0.0rc3) (1.2.0)
    Requirement already satisfied: paddleslim==2.1.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex==2.0.0rc3) (2.1.0)
    Requirement already satisfied: lap in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex==2.0.0rc3) (0.4.0)
    Requirement already satisfied: opencv-python in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex==2.0.0rc3) (4.1.1.26)
    Requirement already satisfied: pycocotools; platform_system != "Windows" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex==2.0.0rc3) (2.0.2)
    Requirement already satisfied: pyyaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex==2.0.0rc3) (5.1.2)
    Requirement already satisfied: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.1.1->paddlex==2.0.0rc3) (1.21.0)
    Requirement already satisfied: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.1.1->paddlex==2.0.0rc3) (2.22.0)
    Requirement already satisfied: Flask-Babel>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.1.1->paddlex==2.0.0rc3) (1.0.0)
    Requirement already satisfied: pandas in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.1.1->paddlex==2.0.0rc3) (1.1.5)
    Requirement already satisfied: Pillow>=7.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.1.1->paddlex==2.0.0rc3) (7.1.2)
    Requirement already satisfied: matplotlib in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.1.1->paddlex==2.0.0rc3) (2.2.3)
    Requirement already satisfied: flask>=1.1.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.1.1->paddlex==2.0.0rc3) (1.1.1)
    Requirement already satisfied: bce-python-sdk in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.1.1->paddlex==2.0.0rc3) (0.8.53)
    Requirement already satisfied: six>=1.14.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.1.1->paddlex==2.0.0rc3) (1.15.0)
    Requirement already satisfied: shellcheck-py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.1.1->paddlex==2.0.0rc3) (0.7.1.1)
    Requirement already satisfied: protobuf>=3.11.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.1.1->paddlex==2.0.0rc3) (3.14.0)
    Requirement already satisfied: numpy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.1.1->paddlex==2.0.0rc3) (1.20.3)
    Requirement already satisfied: flake8>=3.7.9 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.1.1->paddlex==2.0.0rc3) (3.8.2)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn==0.23.2->paddlex==2.0.0rc3) (2.1.0)
    Requirement already satisfied: joblib>=0.11 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn==0.23.2->paddlex==2.0.0rc3) (0.14.1)
    Requirement already satisfied: pytest in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from motmetrics->paddlex==2.0.0rc3) (6.2.4)
    Requirement already satisfied: pytest-benchmark in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from motmetrics->paddlex==2.0.0rc3) (3.4.1)
    Requirement already satisfied: xmltodict>=0.12.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from motmetrics->paddlex==2.0.0rc3) (0.12.0)
    Requirement already satisfied: flake8-import-order in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from motmetrics->paddlex==2.0.0rc3) (0.18.1)
    Requirement already satisfied: pyzmq in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddleslim==2.1.0->paddlex==2.0.0rc3) (18.1.1)
    Requirement already satisfied: setuptools>=18.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pycocotools; platform_system != "Windows"->paddlex==2.0.0rc3) (56.2.0)
    Requirement already satisfied: cython>=0.27.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pycocotools; platform_system != "Windows"->paddlex==2.0.0rc3) (0.29)
    Requirement already satisfied: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.1.1->paddlex==2.0.0rc3) (0.10.0)
    Requirement already satisfied: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.1.1->paddlex==2.0.0rc3) (16.7.9)
    Requirement already satisfied: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.1.1->paddlex==2.0.0rc3) (1.3.0)
    Requirement already satisfied: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.1.1->paddlex==2.0.0rc3) (1.4.10)
    Requirement already satisfied: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.1.1->paddlex==2.0.0rc3) (2.0.1)
    Requirement already satisfied: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.1.1->paddlex==2.0.0rc3) (1.3.4)
    Requirement already satisfied: importlib-metadata; python_version < "3.8" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.1.1->paddlex==2.0.0rc3) (0.23)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.1.1->paddlex==2.0.0rc3) (2019.9.11)
    Requirement already satisfied: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.1.1->paddlex==2.0.0rc3) (2.8)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.1.1->paddlex==2.0.0rc3) (3.0.4)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.1.1->paddlex==2.0.0rc3) (1.25.6)
    Requirement already satisfied: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.1.1->paddlex==2.0.0rc3) (2019.3)
    Requirement already satisfied: Jinja2>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.1.1->paddlex==2.0.0rc3) (2.10.1)
    Requirement already satisfied: Babel>=2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.1.1->paddlex==2.0.0rc3) (2.8.0)
    Requirement already satisfied: python-dateutil>=2.7.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pandas->visualdl>=2.1.1->paddlex==2.0.0rc3) (2.8.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.1.1->paddlex==2.0.0rc3) (2.4.2)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.1.1->paddlex==2.0.0rc3) (1.1.0)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.1.1->paddlex==2.0.0rc3) (0.10.0)
    Requirement already satisfied: click>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.1.1->paddlex==2.0.0rc3) (7.0)
    Requirement already satisfied: Werkzeug>=0.15 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.1.1->paddlex==2.0.0rc3) (0.16.0)
    Requirement already satisfied: itsdangerous>=0.24 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.1.1->paddlex==2.0.0rc3) (1.1.0)
    Requirement already satisfied: future>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl>=2.1.1->paddlex==2.0.0rc3) (0.18.0)
    Requirement already satisfied: pycryptodome>=3.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl>=2.1.1->paddlex==2.0.0rc3) (3.9.9)
    Requirement already satisfied: pyflakes<2.3.0,>=2.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.1.1->paddlex==2.0.0rc3) (2.2.0)
    Requirement already satisfied: mccabe<0.7.0,>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.1.1->paddlex==2.0.0rc3) (0.6.1)
    Requirement already satisfied: pycodestyle<2.7.0,>=2.6.0a1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.1.1->paddlex==2.0.0rc3) (2.6.0)
    Requirement already satisfied: packaging in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pytest->motmetrics->paddlex==2.0.0rc3) (20.9)
    Requirement already satisfied: pluggy<1.0.0a1,>=0.12 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pytest->motmetrics->paddlex==2.0.0rc3) (0.13.1)
    Requirement already satisfied: py>=1.8.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pytest->motmetrics->paddlex==2.0.0rc3) (1.10.0)
    Requirement already satisfied: attrs>=19.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pytest->motmetrics->paddlex==2.0.0rc3) (19.2.0)
    Requirement already satisfied: iniconfig in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pytest->motmetrics->paddlex==2.0.0rc3) (1.1.1)
    Requirement already satisfied: py-cpuinfo in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pytest-benchmark->motmetrics->paddlex==2.0.0rc3) (8.0.0)
    Requirement already satisfied: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata; python_version < "3.8"->pre-commit->visualdl>=2.1.1->paddlex==2.0.0rc3) (0.6.0)
    Requirement already satisfied: MarkupSafe>=0.23 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Jinja2>=2.5->Flask-Babel>=1.0.0->visualdl>=2.1.1->paddlex==2.0.0rc3) (1.1.1)
    Requirement already satisfied: more-itertools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from zipp>=0.5->importlib-metadata; python_version < "3.8"->pre-commit->visualdl>=2.1.1->paddlex==2.0.0rc3) (7.2.0)


## 2.数据解压缩


```python
# %cd ~
# !unzip data/data100671/Datawhale_人脸情绪识别_数据集.zip -d data
```


```python
# !unzip -oq /home/aistudio/data/Datawhale_人脸情绪识别_数据集/test.zip
# !unzip -oq /home/aistudio/data/Datawhale_人脸情绪识别_数据集/train.zip
# !cp data/Datawhale_人脸情绪识别_数据集/sample_submit.csv .
```

## 3.数据均衡
disgusted仅仅只有436，数据极其不均衡，计划均衡下
```
/home/aistudio/train/angry
3995
/home/aistudio/train/disgusted
436
/home/aistudio/train/fearful
4097
/home/aistudio/train/neutral
4965
/home/aistudio/train/sad
4830
/home/aistudio/train/surprised
3171
```


```python
%cd ~/train/angry
!ls -lR | grep "^-"| wc -l
%cd ~/train/disgusted/
!ls -lR | grep "^-"| wc -l
%cd ~/train/fearful/
!ls -lR | grep "^-"| wc -l
%cd ~/train/neutral/
!ls -lR | grep "^-"| wc -l
%cd ~/train/sad/
!ls -lR | grep "^-"| wc -l
%cd ~/train/surprised/
!ls -lR | grep "^-"| wc -l
```

    /home/aistudio/train/angry
    4965
    /home/aistudio/train/disgusted
    4965
    /home/aistudio/train/fearful
    4965
    /home/aistudio/train/neutral
    4965
    /home/aistudio/train/sad
    4965
    /home/aistudio/train/surprised
    4965



```python
# # 数据均衡到4965，仅执行一次
# import os
# import shutil

# def get_file_list(target_path):
#     img_list = os.listdir(target_path)
#     img_list=[os.path.join(target_path, item) for item in img_list]
#     return img_list


# def cpfile(file_list, max_num):
#     current_num=len(file_list)
#     while(current_num<max_num):
#         i=current_num%current_num
#         current_path= '/'.join(file_list[i].split('/')[:-1])
#         filename=file_list[i].split('/')[-1]
#         new_path = os.path.join(str(current_path), 'new_' + str(current_num) + filename)
#         shutil.copy(file_list[i], new_path)
#         current_num=current_num+1


# train_dir=os.listdir('/home/aistudio/train')
# train_dir.remove('.DS_Store')
# img_paths=[os.path.join('/home/aistudio/train', item) for item in train_dir]
# print(img_paths)
# for item in img_paths:
#     img_list=get_file_list(item)
#     cpfile(img_list, 4965)
# print("数据以均衡，各分类均为4965张！")
```


```python
%cd ~/train/angry
!ls -lR | grep "^-"| wc -l
%cd ~/train/disgusted/
!ls -lR | grep "^-"| wc -l
%cd ~/train/fearful/
!ls -lR | grep "^-"| wc -l
%cd ~/train/neutral/
!ls -lR | grep "^-"| wc -l
%cd ~/train/sad/
!ls -lR | grep "^-"| wc -l
%cd ~/train/surprised/
!ls -lR | grep "^-"| wc -l
```

    /home/aistudio/train/angry
    4965
    /home/aistudio/train/disgusted
    4965
    /home/aistudio/train/fearful
    4965
    /home/aistudio/train/neutral
    4965
    /home/aistudio/train/sad
    4965
    /home/aistudio/train/surprised
    4965


## 3.生成数据列表
切分为train、eval数据集
```
2021-07-18 21:53:18 [INFO]	Dataset split starts...
2021-07-18 21:53:18 [INFO]	Dataset split done.
2021-07-18 21:53:18 [INFO]	Train samples: 22968
2021-07-18 21:53:18 [INFO]	Eval samples: 5741
2021-07-18 21:53:18 [INFO]	Test samples: 0
2021-07-18 21:53:18 [INFO]	Split files saved in ./train
```


```python
%cd ~/
!paddlex --split_dataset --format ImageNet --dataset_dir ./train --val_value 0.2
```

## 4.定义transforms


```python
import matplotlib
matplotlib.use('Agg') 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import paddlex as pdx
```


```python
from paddlex import transforms as T
dir(T)
```




    ['ArrangeClassifier',
     'ArrangeDetector',
     'ArrangeSegmenter',
     'BatchRandomResize',
     'BatchRandomResizeByShort',
     'CenterCrop',
     'Compose',
     'Decode',
     'MixupImage',
     'Normalize',
     'Padding',
     'RandomBlur',
     'RandomCrop',
     'RandomDistort',
     'RandomExpand',
     'RandomHorizontalFlip',
     'RandomResize',
     'RandomResizeByShort',
     'RandomScaleAspect',
     'RandomVerticalFlip',
     'Resize',
     'ResizeByLong',
     'ResizeByShort',
     'T',
     '_BatchPadding',
     '__builtins__',
     '__cached__',
     '__doc__',
     '__file__',
     '__loader__',
     '__name__',
     '__package__',
     '__path__',
     '__spec__',
     'arrange_transforms',
     'batch_operators',
     'box_utils',
     'build_transforms',
     'functions',
     'operators']




```python
from paddlex import transforms as T

train_transforms = T.Compose([T.RandomBlur(), 
                            T.RandomHorizontalFlip(), T.Normalize()])
# train_transforms = T.Compose([T.RandomBlur(), T.RandomDistort(brightness_range=0.9, brightness_prob=0.4, contrast_range=0.4, contrast_prob=0.5, saturation_range=0.4),
#                             T.RandomHorizontalFlip(), T.Normalize()])                            
eval_transforms = T.Compose([T.Normalize()])
```

## 4.定义dataset


```python
train_dataset = pdx.datasets.ImageNet(
    data_dir='train',
    file_list='train/train_list.txt',
    label_list='train/labels.txt',
    transforms=train_transforms,
    shuffle=True)
    
eval_dataset = pdx.datasets.ImageNet(
    data_dir='train',
    file_list='train/val_list.txt',
    label_list='train/labels.txt',
    transforms=eval_transforms)
```

    2021-07-19 21:37:07 [INFO]	Starting to read file list from dataset...
    2021-07-19 21:37:08 [INFO]	29604 samples in file train/train_list.txt
    2021-07-19 21:37:08 [INFO]	Starting to read file list from dataset...
    2021-07-19 21:37:08 [INFO]	7401 samples in file train/val_list.txt


# 三、选PaddleX模型并训练

## 1.模型选择 && 训练



```python
num_classes = len(train_dataset.labels)
model = pdx.cls.ResNet101_vd_ssld(num_classes=num_classes)
model.train(num_epochs=10,
            train_dataset=train_dataset,
            train_batch_size=128,
            eval_dataset=eval_dataset,
            lr_decay_epochs=[4, 6, 8],
            save_interval_epochs=1,
            learning_rate=0.025,
            save_dir='output/ResNet101_vd_ssld',
            use_vdl=True)
```

## 2.训练日志

![](https://ai-studio-static-online.cdn.bcebos.com/5017a5c5b6834dcdb807a8dabec7697162e035cdba1a46ea8c3d40c355aa0b32)



```
2021-07-18 22:01:55 [INFO]	[TRAIN] Epoch=10/10, Step=7/57, loss=0.612496, acc1=0.797500, acc5=1.000000, lr=0.000063, time_each_step=0.13s, eta=0:0:6
2021-07-18 22:01:56 [INFO]	[TRAIN] Epoch=10/10, Step=17/57, loss=0.707318, acc1=0.740000, acc5=0.997500, lr=0.000063, time_each_step=0.08s, eta=0:0:3
2021-07-18 22:01:57 [INFO]	[TRAIN] Epoch=10/10, Step=27/57, loss=0.733735, acc1=0.735000, acc5=0.997500, lr=0.000063, time_each_step=0.07s, eta=0:0:2
2021-07-18 22:01:57 [INFO]	[TRAIN] Epoch=10/10, Step=37/57, loss=0.822799, acc1=0.682500, acc5=0.992500, lr=0.000063, time_each_step=0.07s, eta=0:0:1
2021-07-18 22:01:58 [INFO]	[TRAIN] Epoch=10/10, Step=47/57, loss=0.722330, acc1=0.722500, acc5=0.997500, lr=0.000063, time_each_step=0.07s, eta=0:0:0
2021-07-18 22:01:59 [INFO]	[TRAIN] Epoch=10/10, Step=57/57, loss=0.781480, acc1=0.730000, acc5=0.985000, lr=0.000063, time_each_step=0.07s, eta=0:0:0
2021-07-18 22:01:59 [INFO]	[TRAIN] Epoch 10 finished, loss=0.72172904, acc1=0.74149126, acc5=0.99293864 .
2021-07-18 22:01:59 [INFO]	Start to evaluate(total_samples=5741, total_steps=15)...
2021-07-18 22:02:01 [INFO]	[EVAL] Finished, Epoch=10, acc1=0.554805, acc5=0.973047 .
2021-07-18 22:02:01 [INFO]	Current evaluated best model on eval_dataset is epoch_8, acc1=0.5554988384246826
2021-07-18 22:02:01 [INFO]	Model saved in output/mobilenetv3_large_ssld/epoch_10.
```

# 四、预测

## 1.生成预测列表


```python
import pandas as pd

test=pd.read_csv('sample_submit.csv')
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00001.png</td>
      <td>sad</td>
    </tr>
    <tr>
      <th>1</th>
      <td>00002.png</td>
      <td>sad</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00003.png</td>
      <td>sad</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00004.png</td>
      <td>sad</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00005.png</td>
      <td>sad</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.info
```

## 2.预测


```python
import paddlex as pdx
import os
model = pdx.load_model('0.7377796769142151/')
labels=[]
for index, item in test.iterrows():  
    image_name = os.path.join('test', item['name'])
    label = model.predict(image_name)
    labels.append(label)
print("Predict Done:", len(labels))
```

    2021-07-19 21:47:03 [INFO]	Model[ResNet101_vd_ssld] loaded.


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/tensor/creation.py:125: DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`. To silence this warning, use `object` by itself. Doing this will not modify any behavior and is safe. 
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if data.dtype == np.object:


    Predict Done: 7178


## 3.保存并提交


```python
test['label']=labels
```


```python
for i in range(len(labels)):
    test['label'][i]=test['label'][i][0]['category']
```


```python
print(test['label'][0])
```

    angry



```python
# 不要index
test.to_csv('result.csv', index=False)
```


```python
!zip result.zip result.csv
```

      adding: result.csv (deflated 82%)


## 4.结果
```
返回分数	0.65534	result.csv		livingbody	2021-07-19 01:52:09
```
