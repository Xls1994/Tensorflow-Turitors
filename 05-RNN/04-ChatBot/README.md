# 聊天机器人

- 这是一个简单的基于seq2seq的聊天机器人

- Seq2Seq 可以进行序列到序列的生成，用于对话系统

### 数据集

使用config.py 进行你的数据配置

- data/chat.log : 输入数据
- data/chat.txt : 词汇表

### 命令行使用

```
python chat.py
```



### 训练模型

``` 
python train.py --train
```



可视化

```
tensorboard --logdir=./logs
```

### 测试

测试你的模型

```
python train.py --test
```

### 生成的词汇

使用数据集建立词汇表，你可以通过设置config中的参数来查看词汇表。并且通过改变参数来检查数据的正确于错误

```
python dialog.py --voc_build
```
