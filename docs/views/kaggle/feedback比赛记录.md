---
title: FeedBack比赛记录
date: 2022-08-03
sidebar: "auto"
categories:
- kaggle
  
tags:
- kaggle
---

<!-- more -->

[Feedback Prize - Predicting Effective Arguments](https://www.kaggle.com/competitions/feedback-prize-effectiveness) 

因为我手里卡不多，只能冻结一部分。

![](http://kuroweb.tk/picture/16591075317336250.jpg)



不加预训练的模型都要差。其中两层的模型和4层的模型效果差不多。

- v39-shuffle_nopreload: 无预训练模型
- v46：无预训练模型，仅训练后2层

- v39-shuffle_nopreload: 预训练模型

![](http://kuroweb.tk/picture/16591073049762930.jpg)

所以后续的尝试在预训练模型上面，在cls上做分类，因为与训练任务的特性。预训练有两个任务，一个是判断两段文本是否连接，一个是mlm。CLS位置为了捕获两段文本的上下文信息，需要有全局的关注力。而每个位置的token经过bert最终得到的输出可以看成是某种embedding，适合做token ner任务。



![](http://kuroweb.tk/picture/16591075317336250.jpg)



## 模型学习了什么

### 打分可视化



我在榜单上找到了一个高分的笔记本。使用前端界面把打分[可视化](https://kuro7766.github.io/FeedbackEda/build/web/)后，可以发现两个重要的事实。



- 1.全文的分数几乎都一样
- 2.个别分数与全文分数不同的部分，置信度全都比较低。



第1点正如deberta v3 large1024的效果相符合，更长的的句子长度可以帮助模型捕获整篇文章的信息。



第2点是难以学习的，因为这部分句子往往不超过整个句子的10%，模型难以关注这部分。如何才能让模型更好的关注到这部分的信息？

请问闲暇时间的翻阅之后，我发现比较差的文章比较喜欢用一些短语。

对于短语的提取可以使用text Cnn来实现，在bert的最后一层加上。

那么是加全局还是仅在ShortSentence呢？可能各有优势

- 1.全局：增加文章整体分数判断的能力。但是局部分类可能不行。

- 2.ShortSentence

  

  本次的任务是段文本分类，长文本作为上下文来使用。由于bert是微调的，输出位置的短文本位置保留了短文本这部分的信息，如果短文本需要一些额外信息，他可以自己去长文本里查询——如果这一点成立，那么不一定需要用text CNN，作用在局部的mean pooling也可以试一试。

![](http://kuroweb.tk/picture/16591082895515714.jpg)

### 部分作用的cnn技术上的实现

为了验证上述的想法，自然是需要做实验，一个个尝试。

在预处理的时候计算mask，然后直接mask_fill作用于最后一层

![](http://kuroweb.tk/picture/16591089156903596.jpg)