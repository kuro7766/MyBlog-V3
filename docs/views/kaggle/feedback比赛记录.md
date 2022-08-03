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





## baseline & improve

- baseline

 CV大概在0.6左右，由于是 straight K fold，所以每个fold的CV差不多。



找到了一个别人的base，运行了两遍，分数几乎一样

![](http://kuroweb.tk/picture/16594198833918976.jpg)

后续的模型改进都和这个曲线做对比。

- baseline后两层相加

![](C:\Users\1\AppData\Roaming\Typora\typora-user-images\image-20220803191354913.png)



把bert最后两层直接加起来，可以实现更好的效果，低了0.004，当然也可能是抖动随机的效果。为了探究其中的原因，我们来看一下train loss。

![](http://kuroweb.tk/picture/16595254938246282.jpg)

我们以后半部分作为观察对象，可见在绝大部分的位置上，后两层相加的模型的loss显然更优，这也说明网络在这样的结构下，更容易找到梯度。因此后续的工作应该集中在如何找到更快下降的loss上，如果发现loss明显下降的更快，则有更大的概率是eval也更优的模型。



- 后两层+1个可学习权重

![](http://kuroweb.tk/picture/16595367798825914.jpg)

和baseline模型相比。后两层相加得到了更好的效果。实现了在早期的时间达到了一个最低的loss。0.4492优于之前的0.4574，是一个比较明显的提升，可见模型的收敛较快。

![](http://kuroweb.tk/picture/16595371488447532.jpg)

同时eval loss 也在第三个epoch的时候达到了和 Baseline模型相持平的水平，后续改进有较大的可能性。

## 其他尝试

- deberta base 512

  

似乎不work

长度增长可见有明显的问题，很可能是淹没了短句子的效果。

loss图像，train loss偏高，且更早的达到了过拟合。。

![](http://kuroweb.tk/picture/16595261404163210.jpg)



- deberta large 384

没有得到更好的效果，loss下降变得困难，可能是因为搜索空间太大了。

训练loss偏高，甚至和上面的base512也有一定的差距。且过拟合发生的时间较早。猜测模型可能关注更深的信息，而针对本数据集没有那么多的信息要学习。

![](http://kuroweb.tk/picture/16595262821585440.jpg)



- deberta base 360

两者几乎持平，说明长度变少并未变弱模型的效果。后面可以进一步尝试，继续减少长度。

![](http://kuroweb.tk/picture/16595267143843442.jpg)

eval loss最优大概减少了0.0003个点，继续来看一下train loss后半段

![](http://kuroweb.tk/picture/16595274436582224.jpg)

max_len 360的train loss最低点比baseline更优，同时最优eval loss也是在这个点左右完成的。可见模型train loss下降的越快、更低，越能找到更真正的东西。train loss和eval loss是一致的



可参考的最优loss为 **0.456~0.457**



- 最后三层相加

1.求平均

这种办法效果会变得特别差，效果不好。

2.直接相加，但是权重等于变成了三倍

同样的来看train loss后半段。

![](http://kuroweb.tk/picture/16595288088231332.jpg)

后三层相加的结果要稍微的差一些。与最后两层直接相加相比，没有达到更好的效果。可见，适合本数据集的层数大概在一层上下的差距。



同时，1）和2）相差悬殊，这让我想到了另一个提高模型的点的办法，那就是给后两层相加的模型添加一个可学习的权重。

