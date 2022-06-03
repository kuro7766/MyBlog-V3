---
title: PatentPhrase比赛记录
date: 2022-06-03
sidebar: "auto"
categories:
- kaggle
  
tags:
- kaggle
---

<!-- more -->

[PatentPhrase Competition](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching) 



## 数据集的特点

观察数据析发现，人为标注的分数其实是离散的，因此可以用一个**分类模型**来预测分数，目前想的是5分类或者6分类，分类数目如果太多，一旦错误可能会导致误差很大。

![](http://kuroweb.tk/picture/16542561862046630.jpg)

同时可以观察到，训练集的socore分布**偏差比较大**，只有很少的一部分靠于1，由于不知道最终数据，这里也许可以尝试一下





对于 context 字段，根据别人的baseline，每一个编号有对应语言的含义，直接将他们预先保存在字典中，然后展开成为句子，输入到语言模型中，训练他们的句子向量。

```
{'A01': 'HUMAN NECESSITIES. GRICULTURE; FORESTRY; ANIMAL HUSBANDRY; HUNTING; TRAPPING; FISHING',
 'A21': 'HUMAN NECESSITIES. BAKING; EDIBLE DOUGHS',
 'A22': 'HUMAN NECESSITIES. BUTCHERING; MEAT TREATMENT; PROCESSING POULTRY OR FISH',
 'A23': 'HUMAN NECESSITIES. FOODS OR FOODSTUFFS; TREATMENT THEREOF, NOT COVERED BY OTHER CLASSES',
 'A24': "HUMAN NECESSITIES. TOBACCO; CIGARS; CIGARETTES; SIMULATED SMOKING DEVICES; SMOKERS' REQUISITES",
 'A41': 'HUMAN NECESSITIES. WEARING APPAREL',
 'A42': 'HUMAN NECESSITIES. HEADWEAR',
 'A43': 'HUMAN NECESSITIES. FOOTWEAR',
 'A44': 'HUMAN NECESSITIES. HABERDASHERY; JEWELLERY',
 'A45': 'HUMAN NECESSITIES. HAND OR TRAVELLING ARTICLES',
```

对于这类的词语，也许还可以精心设计——就像代码编号转字符串一样，**继续把词语展开**，找到更多分类代码之间**词语级别上更直接的关系**，**或者手动处理**。



不过这个可能没有什么用？因为用于训练模型的话，也相当于是把外部词语拿过来用了。但是从A01->人类必需品的转换，是必须的。即使如此，上述的方法也值得尝试。

A01,G02这种分类应该可以用机器学习在原始的代码分类上操作。



## baseline的做法

```
test['anchor'] + '[SEP]' + test['target'] + '[SEP]'  + test['context_text']
```

将上述相关的句子全部直接作为输入，所有需要的信息都在这里的上下文中，然后训练他们的输出分数。完全端到端的处理。目前上面这部分代码，堆积了三个模型，一共花费大约两个小时，平均下来每隔**大概半个多小时**。这里的大部分时间依然在模型推理上。

## ML

此后接上ml的方法，应该不需要重新训练，因为已经在本数据集上训练过。



$$ml花费的时间 = pretrained预处理时间 + ml分类时间 $$

预训练大模型处理的时间比较长，所以大概会和之前的模型持平，也在半个小时左右。



## Model building

使用多种不同的预训练模型向量化文本，然后将其输入到不同的机器学习模型中。

代码编写的时候要注意可以容易替换，可以省下不少工作。



![](http://kuroweb.tk/picture/16542568140600260.jpg)



在剩下的一些操作，比如第一列用vectorizor1，第2列用vectorizor2，然后做分类

ml model的组件:

xgboost , lightgbm , svr , mlp , cnn

## 还是关于专利分类编号处理的问题

```python
def prepare_input(cfg, text):
    adb(text)
    inputs = cfg.tokenizer(text,
                           max_length=cfg.max_len,
                           padding="max_length",
                           truncation=True)
    
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
        
    return inputs
```

在预训练向量化的过程中，他使用了截断处理，并且设置max_len = 130，然而，有相当一部分数值超过了130。针对truncate参数的[讲解](https://discuss.huggingface.co/t/purpose-of-padding-and-truncating/412/5)，可见超过的部分根本没有利用上这部分句子的信息，而是应该由用户来把握句子的长度。因此有两种办法，一种是把130提高，另一种是手动把句子才减到130以内。作者设置成130，可能是因为时间的考量，**因此手动截取句子可能比较好**，去掉那些不太重要的含义，而句子长度不够的，也可以补充一些词汇。

即使如此，句子的长度依然是不定长的，因此**把不重要的关键词放在后面**，即使剔除了也无所谓的那种。同时**把说明文字少的分类补充补充，避免token的浪费**。

## 一种提速方法

由于目前的baseline用的是这种方法

```
test['anchor'] + '[SEP]' + test['target'] + '[SEP]'  + test['context_text']

```

对于每个句子的上下文信息都要重复推断一次，造成了时间上的浪费。应该可以拆解成如下的形式



```
1.test['anchor'] + '[SEP]' + test['target']   

2.test['context_text']

```

然后将 1.2  融合 ， 2 的部分直接**提前存储成map形式**。1的部分允许deberta减小tokenizer max_len，可以提高速度。**2的部分精心训练**

## cos sim

目前用的是端到端的处理，最后的分数计算用的是mlp，可以改成余弦相似度



![](http://kuroweb.tk/picture/16542706444440894.jpg)



## fake data cleaning

人为清除掉训练集中错误的数据

## vectorizer选择

https://www.kaggle.com/general/201825

## 参考文章

https://www.kaggle.com/competitions/petfinder-pawpularity-score/discussion/288896