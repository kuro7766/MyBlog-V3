---
title: FeedBack比赛记录
date: 2022-08-03
sidebar: "auto"
categories:
- kaggle
  
tags:
- kaggle

---





---



<!-- more -->

[Feedback Prize - Predicting Effective Arguments](https://www.kaggle.com/competitions/feedback-prize-effectiveness) 



因为我手里卡不多，只能冻结一部分。

![](http://kuroweb.tk/picture/16591075317336250.jpg)



不加预训练的模型都要差。其中两层的模型和4层的模型效果差不多。

- v39-shuffle_nopreload: 无预训练模型
- v46：无预训练模型，仅训练后2层
- v39-shuffle_nopreload: 预训练模型



---



![](http://kuroweb.tk/picture/16591073049762930.jpg)

所以后续的尝试在预训练模型上面，在cls上做分类，因为与训练任务的特性。预训练有两个任务，一个是判断两段文本是否连接，一个是mlm。CLS位置为了捕获两段文本的上下文信息，需要有全局的关注力。而每个位置的token经过bert最终得到的输出可以看成是某种embedding，适合做token ner任务。



![](http://kuroweb.tk/picture/16591075317336250.jpg)

---



## 模型学习了什么

---



### 打分可视化



我在榜单上找到了一个高分的笔记本。使用前端界面把打分[可视化](https://kuro7766.github.io/FeedbackEda/build/web/)后，可以发现两个重要的事实。

![](http://kuroweb.tk/picture/16725838813280564.png)

- 1.全文的分数几乎都一样
- 2.个别分数与全文分数不同的部分，置信度全都比较低。



第1点正如deberta v3 large1024的效果相符合，更长的的句子长度可以帮助模型捕获整篇文章的信息。

第2点是难以学习的，因为这部分句子往往不超过整个句子的10%，模型难以关注这部分。如何才能让模型更好的关注到这部分的信息？

---



## baseline & improve

### baseline

 CV大概在0.6左右，由于是 straight K fold，所以每个fold的CV差不多。



找到了一个别人的base，运行了两遍，分数几乎一样

![](http://kuroweb.tk/picture/16594198833918976.jpg)

后续的模型改进都和这个曲线做对比。

---



### baseline后两层相加

![](http://kuroweb.tk/picture/16622917010122740.png)



把bert最后两层直接加起来，可以实现更好的效果，低了0.004，当然也可能是抖动随机的效果。为了探究其中的原因，我们来看一下train loss。

![](http://kuroweb.tk/picture/16595254938246282.jpg)

我们以后半部分作为观察对象，可见在绝大部分的位置上，后两层相加的模型的loss显然更优，这也说明网络在这样的结构下，更容易找到梯度。因此后续的工作应该集中在如何找到更快下降的loss上，如果发现loss明显下降的更快，则有更大的概率是eval也更优的模型。

---



### <a name="1"></a> 后两层+1个可学习权重

![](http://kuroweb.tk/picture/16595367798825914.jpg)

和baseline模型相比。后两层相加得到了更好的效果。实现了在早期的时间达到了一个最低的loss。0.4492优于之前的0.4574，是一个比较明显的提升，可见模型的收敛较快。

![](http://kuroweb.tk/picture/16595371488447532.jpg)

同时eval loss 也在第三个epoch的时候达到了和 Baseline模型相持平的水平，后续改进有较大的可能性。



### 后两层两个动态权重

![](http://kuroweb.tk/picture/16595443671352084.png)



非常有效，train loss达到了0.43，在3.26 epoch，降低了0.02，前所未有的低



由于eval的频次有点低，导致最低点处没有eval，所以没有体现出来。这里显示的是动态权重不如baseline。

![](http://kuroweb.tk/picture/16595444895616866.png)

所以这个方法和[后两层+1个可学习权重](#1)都需要设置eval为3 or 4/per epoch左右才可以显示出来效果

经过修改eval per epoch为4后，可见确实出现了更优的结果

![](http://kuroweb.tk/picture/16596026399109478.png)

在2.25 epoch出现了best loss

best eval loss: **0.5958 -> 0.5857**

![](http://kuroweb.tk/picture/16596030745607114.png)

意外的是，最优的eval loss不是在train loss 3.25 epoch最低点这里出现的，而是在2.25，出现的更早了。但这也不是坏事，说明模型有非常快的收敛速度。



同样的我把baseline也调整为4 eval per epoch



### 添加外部数据和特征

仿照[bert topic](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/333277)的思路，我添加了另一个模型。

基于[phrasebank数据集](https://www.phrasebank.manchester.ac.uk/)的分类模型，该数据集质量比较高，并且和写作的相关性比较大，我们可以提取它的分类前的logits作为本任务的预处理特征

![](http://kuroweb.tk/picture/16602927485311630.jpg)

v29在v19（baseline）上面有了小幅度的提升，效果和berttopic类似。可见，添加和数据集相关的数据是比较稳定的上分方法。根据No free lunch theory，没有一种在任何任务上表现的都很好的模型。

## 其他尝试

### deberta base 512


似乎不work

长度增长可见有明显的问题，很可能是淹没了短句子的效果。

loss图像，train loss偏高，且更早的达到了过拟合。。

![](http://kuroweb.tk/picture/16595261404163210.jpg)



### deberta large 384

没有得到更好的效果，loss下降变得困难，可能是因为搜索空间太大了。

训练loss偏高，甚至和上面的base512也有一定的差距。且过拟合发生的时间较早。猜测模型可能关注更深的信息，而针对本数据集没有那么多的信息要学习。

![](http://kuroweb.tk/picture/16595262821585440.jpg)



### deberta base 360

两者几乎持平，说明长度变少并未变弱模型的效果。后面可以进一步尝试，继续减少长度。

![](http://kuroweb.tk/picture/16595267143843442.jpg)

eval loss最优大概减少了0.0003个点，继续来看一下train loss后半段

![](http://kuroweb.tk/picture/16595274436582224.jpg)

max_len 360的train loss最低点比baseline更优，同时最优eval loss也是在这个点左右完成的。可见模型train loss下降的越快、更低，越能找到更真正的东西。train loss和eval loss是一致的



可参考的最优loss为 **0.456~0.457**



### 最后三层相加

- 1.求平均

这种办法效果会变得特别差，效果不好。

- 2.直接相加，但是权重等于变成了三倍

同样的来看train loss后半段。

![](http://kuroweb.tk/picture/16595288088231332.jpg)

后三层相加的结果要稍微的差一些。与最后两层直接相加相比，没有达到更好的效果。可见，适合本数据集的层数大概在一层上下的差距。



同时，1）和2）相差悬殊，这让我想到了另一个提高模型的点的办法，那就是给后两层相加的模型添加一个可学习的权重。



### top layer reinitialization

因为语言模型的浅层保存的是通用的特征，靠近输出的地方是更加 task specific特征。之前的mlm任务保留的权重可能会对当前任务造成一些的限制

![](http://kuroweb.tk/picture/16597132571626208.png)

但是运行之后发现这么做反而是最差的，如图中v16两个曲线（红蓝），在所有曲线的最上层

### 加入人工抽取的关键词

有略微的提升，当然也可能是偶然出现的，提升不大。如果能像上次patent针对每个类别加入动态的文本比较好

![](http://kuroweb.tk/picture/16597140998333988.jpg)

```
you your they we i think but also these some many people because there are
```

是我认为写的比较差的作文里常用的词汇。经尝试提升不明显。



### TextCNN

闲暇时间的翻阅之后，我发现比较差的文章比较喜欢用一些短语。

对于短语的提取可以使用text Cnn来实现，在bert的最后一层加上。

那么是加全局还是仅在ShortSentence呢？可能各有优势

- 1.全局：增加文章整体分数判断的能力。但是局部分类可能不行。

- 2.ShortSentence

  本次的任务是段文本分类，长文本作为上下文来使用。由于bert是微调的，输出位置的短文本位置保留了短文本这部分的信息，如果短文本需要一些额外信息，他可以自己去长文本里查询——如果这一点成立，那么不一定需要用text CNN，作用在局部的mean pooling也可以试一试。

![](http://kuroweb.tk/picture/16591082895515714.jpg)

- 部分作用的cnn技术上的实现

为了验证上述的想法，自然是需要做实验，一个个尝试。

在预处理的时候计算mask，然后直接mask_fill作用于最后一层

![](http://kuroweb.tk/picture/16591089156903596.jpg)



经过尝试，效果不佳。



### Longformer相关的尝试

因为比赛的举办方曾经用这次比赛数据集的另一部分举办过一次比赛。并且在那次比赛中longformer是sota，所以我在这里尝试但是似乎并不适合。



- 把作文尾句放在前面

大家都知道写作的时候，作文的尾句是比较重要的东西，一般都是总结文章重新重申主题，并且语言通常是比较干净的。

![](http://kuroweb.tk/picture/16602940912477520.jpg)

cv在0.76



- 浅层隐藏层权重

语言模型的浅层保存的是通用的特征，深层是更加抽象的特征，但是如果数据的量不够，深层的模型反而对训练会造成阻碍。

已知 deberta base->deberta large 会 cv+0.02，但是从12层->25层这样粗暴的调整它，肯定还有某些层是更优的，就是这样的调参是非常粗略的。关于模型的深浅度只有两种情况，一种是模型太深，一种是太浅。

太浅的解决办法就是向后面添加更多的层，fcl，rnn，attention均可。剩下的部分就是世界当前算力的限制，大部分人都难以突破。

太深的解决办法，就是只看某浅层hidden layer的输出，在梯度下降的时候忽视深层网络里面的输出，阻止其梯度下降。因为我们知道在数据的分布比较简单的数据集中，如果用太深的网络，反而会导致收敛慢精度低的问题，浅层的神经网络在此方面反而表现的更好。

让NLP深度模型变浅有几种办法：

1.冻结前几层

2.丢弃后几层



但是这个几层是怎么确定的呢？如果不断的调参是非常麻烦的。可以给每个隐藏层加权加起来，作为最后的隐藏层输出，模型梯度下降会自己去找，下降速度最快的那一层。

实验证明，这样效果不错，是比较通用且稳定的上分办法。

- Attention Pooling

```python
class AttentionHead(nn.Module):
    def __init__(self, in_size: int = 768, hidden_size: int = 512) -> None:
        super(AttentionHead,self).__init__()
        self.W = nn.Linear(in_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, features):
        att = torch.tanh(self.W(features))
        score = self.V(att)
        # sdb(score)
        # adb(score)
        attention_weights = torch.softmax(score, dim=1)
        # adb(attention_weights)
        # sdb(features)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)
        output = self.dropout(context_vector)
        return output
```

attention pooling 可以作为一种选择，和cnn、lstm、meanpooling类似。

- 冻结+多层lstm pooling

❌效果很差



- Shuffle vs Order

经过[EDA](https://kuro7766.github.io/FeedbackEda/build/web/)观察，我发现同一篇文章中，那些得分和本文大部分评分不一样的地方，比如说全文都是effective，只有一个句子是ineffective，那么这个句子打分的置信度会非常低。经过观察这是非常普遍的现象。可以尝试一下用一个单独的分类器来分类是否为置信度低的样本，然后把logits送给fcl。



### 其他还未尝试想法

加入ML book的特征

加外部数据