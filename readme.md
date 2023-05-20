# ASR CTC Model
基于[连接时序分类(Connectionist Temporal Classification, CTC)](https://dl.acm.org/doi/10.1145/1143844.1143891)实现端到端学习方法的中文语音识别模型，声学模型部分使用BiGRU和WaveNet构建  

## 训练数据集 THCHS-30
THCHS-30 是清华大学语音与语言技术中心(CSLT)发布的开放式中文语音数据库  
该数据集在安静的办公室环境下通过单个碳粒麦克风录取，总时长超过 30 个⼩时  
共 13388 个文件，采样频率 16kHz，采样大小 16bits

**本项目使用了数据集data文件夹下的所有语音数据进行训练**，将该文件夹解压到项目根目录下即可
> 下载地址：[http://www.openslr.org/18/](http://www.openslr.org/18/)

## 训练/测试流程
* `1.data_handle.ipynb` 语音数据预处理，使用MFCC构建音频特征数据
* `2.model_bigru.py`和`2.model_wavenet.py`  
    双向GRU和WaveNet声学模型构建，CTC模型构建及训练  
    可在`run/`下执行`train_2.sh`后台运行模型训练，在`run.log`中可查看训练过程  
* `3.predict.ipynb`     模型预测，使用CTC解码函数预测模型输出，默认为贪婪法(greedy)
* `4.diagrams.ipynb`    模型训练过程图，绘制CTC Loss曲线
* `5.wer.ipynb`         模型评价指标，计算词错率(WER)及实时率(RTF)，默认在无GPU加速下测试

## 模型应用
可以参见另外一个Project: [中文语音识别实验室](https://github.com/SzLeaves/asr-webapp)

## Tips
* 有问题或发现了bug，欢迎提交PR和issue，如果对你有帮助的话就给个star吧~QAQ
* 本项目有多种模型结构分支，可以`checkout`查看
* 提高模型的语音识别能力需要更多的数据集训练
* 音频数据特征部分使用了[SpecAugment](https://arxiv.org/abs/1904.08779)数据增强方法扩充等量数据
* **本项目仅供学习参考，不承担任何因实际使用和生产环境部署等因素导致的不可预知的后果**
