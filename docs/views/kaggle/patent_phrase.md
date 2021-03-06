---
title: PatentPhrase比赛记录
date: 2022-06-03
sidebar: "auto"
categories:
- kaggle
  
tags:
- kaggle
- nlp
- 深度学习
---

<!-- more -->

[PatentPhrase Competition](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching) 

## 比赛总结

这次打的还是非常差，主要有以下几点。

- 一个是没有阅读足够多的模型。数据的操作上也不够熟练。

- 另一个是时间投入比较少，其实应该尽早的练习，早期应该多搞一些最强的单模型，以及记录一些挑战的技巧，这些东西大部分别人是有可能想不到的。在最后一周，即使有人公开了高分笔记本也可以用上。



- 合并的时候经常遇到变量冲突，所以需要一个saver和restorer，用exec实现



- 另外，使用别人的数据集非常的危险， 别人一旦删除我这就没法提交了，所以建议拷贝到自己的笔记本上变更为version1

- 为了方便模型的调试，通常需要减少数据集。数据集减少应该用修改文件的方式，这样做是因为简单方便；另一个原因是如果别的多个融合文件用py的形式执行（这样比ipynb节省空间，及时free内存），无需修改py的代码。

- 融合的笔记本记得带上版本号，防止搞混了，养成修改前新建一个版本文件的习惯。



- 在训练多个模型的时候，为了让后面的模型融合，可以顺利的进行，尽量保存信息量比较多的那些数据，比如分类模型输出的logits要保存下来，而直接保存模型，通常来说不经济，能做到保存logits大概就可以了。



- 大道至简，首先泡一个还不错的baseline，然后根据这个去一点一点的调试。而不是花里胡哨的大刀阔斧直接改模型，这样会迷失eval loss的下降方向，优秀的大模型也应该是一点一点过来的。

## 日志记录

### 揭榜

榜单的变化很大，比赛失败了，甚至连铜牌也没有，非常失落。

### day1

融了electra，突然冲进了银牌区。

### day2

又融合了一个笔记本，8484

### day3

达到铜牌吊车尾了，但是很危险

这次也是第1次尝试，堆叠了很多机器学习模型。有些机器学习模型效果很差，可能是自己使用方面的不到位。这个方面的还需要继续学习一下。

这次比赛之后我发现自己非常需要写一个爬虫，用来扒取比较重要的笔记本所对应的数据集是否完整，以及历史分数可视化等等。包括他使用的是什么样的模型。爬虫数据可视化之后，我只需要负责合并就行了，这样也可以在早期合并一些有潜力的笔记本，减少后期的工作量，因为很多人都是改一下模型就推断了。对于数据集来说，必须要及时拉取下来，防止别人在比赛的最后关闭。

### day4

前一段时间忙着改毕业论文和毕业典礼了，忘了搞这个。今天重新开始搞吧。发现了一个870笔记本，看来只能基于这个修改了。

我发现这个比赛不同于以往的知识，都是用的是transformer模型，但是对超参数非常敏感，因此也可以同时融合，然后得到一个比较好的分数。以往我认为用CNN + transformer模型才可以得到一个更好的分数，因为他们是不同的结构。

所以接下来的事情就是把不同的笔记本模型融合一下。

希望能调参数拿个铜牌吧。

最后的这几天要珍惜每一次的提交机会。



犯了个大错误：没有用merge on id的方式，导致分数一直出错误



总结了一个经验：数据集减少直接修改文件就行，可以减少计算量



我能想到的，别人基本都实现过了

### day14

预计从以下几个方面入手

- encoder vector生成

- ml models

- score stacking策略

  

  模型融合方法 https://www.6aiq.com/article/1536427413103

### day16

#### 修复了"吸附"运行失败的问题

#### 公开模型更好的一个

https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/324330

[0.846](https://www.kaggle.com/code/tanlikesmath/pretrained-sentence-transformer-model-baseline)，额外采用的两种模型

https://huggingface.co/anferico/bert-for-patents

https://huggingface.co/microsoft/cocolm-large



编写了mlp submission版本



### day 17

跑通、调试了bert encoder cosine sim的模型，**直接匹配相似度输出结果**，得到一个还不错的分数，pca展示roberta，deberta，cosine sim分数cluster，但是合并之后分数降低了



- 完成数据加载器，把所有的特征提取出来

- 完成了mlp训练、保存

#### 特征提取:

 三组cos sim、anchor向量、context向量、target向量

不同的距离编码

#### 未完成

naive baiyes，svr，xgboost、mlp对接

##### 异常点清理

clustering，numeric outlier方法

z-score应该也可以

Isolation Forest应该比较快、DBScan，Isolation Forest

#### 基于分类任务的改进

都拿不准的分大类和小类，0.5 +-，然后把别的剔除，剩下的做回归。
用多个分类器猜区间
分两个类和分三个类。
拿不准的，用回归，拿得准的，用分类无损

#### 吸附功能

加规则 ： 0.26->0.25，**预计hidden dataset也只是分类任务**，如果比较准确的就直接吸附，误差为0，等待结果。和上述分类任务改进结合是**最有希望拿牌子的方法**







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

## 丑陋的技巧
训练一个简单的回归器，利用你的预测向量，试图预测你提交这个内核所得到LB分数。 关于这一点，不再多说了！;)

## 如何得到一个好的向量表示?

必然要finetune，但是根据什么?

## 更多的信息?

如何在[此表格](https://www.kaggle.com/datasets/xhlulu/cpc-codes)的基础上挖掘更多的信息，比如给**类做pca**。加一个聚类表示



## 模型挑选



## 参考文章

https://www.kaggle.com/competitions/petfinder-pawpularity-score/discussion/288896

树模型：

https://zhuanlan.zhihu.com/p/453866197
https://zhuanlan.zhihu.com/p/405981292