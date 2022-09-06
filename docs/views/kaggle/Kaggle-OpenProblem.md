---
# try also 'default' to start simple
theme: seriph
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
background: https://source.unsplash.com/collection/94734566/1920x1080
# apply any windi css classes to the current slide
class: 'text-center'
# https://sli.dev/custom/highlighters.html
highlighter: shiki
# some information about the slides, markdown enabled
info: |
  ## Slidev Starter Template
  Presentation slides for developers.

  Learn more at [Sli.dev](https://sli.dev)
title: Kaggle-OpenProblem
date: 2022-09-06
sidebar: "auto"
categories:
- kaggle
tags:
- kaggle
---

# Kaggle-OpenProblem

---

**目录**

<toc/>

---

## 数据集


<style>
  h2 {
    text-align: center;
    margin:200px;
    background: #aaaaaa;
    text-shadow: 0 0 10px #000000;
    color: #ffffff;
  }
  </style>

---

**数据集**

- evaluation_ids.csv

- metadata.csv

- sample_submission.csv

- test_cite_inputs.h5
- test_multi_inputs.h5
- train_cite_inputs.h5
- train_cite_targets.h5
- train_multi_inputs.h5
- train_multi_targets.h5

---


- evaluation_ids.csv


<img src="http://kuroweb.tk/picture/16624378953676982.jpg"  class="h-100 mx-auto" />

---

- metadata.csv

<!-- ![](http://kuroweb.tk/picture/16624379455884908.jpg) -->
描述了测量天数，捐赠者，细胞类型（不准确），测量技术
<img src="http://kuroweb.tk/picture/16624379455884908.jpg"  class="h-90 mx-auto my-10" />

---

- sample_submission.csv

和`evaluation_ids.csv`一一对应

<!-- ![](http://kuroweb.tk/picture/16624380043696516.jpg) -->

<!-- <img src="http://kuroweb.tk/picture/16624380043696516.jpg"  class="h-100 mx-auto" /> -->


| evaluation_id | sample_id |
| :---: | :---: |
| <img src="http://kuroweb.tk/picture/16624378953676982.jpg"  class="h-80 mx-auto" /> | <img src="http://kuroweb.tk/picture/16624380043696516.jpg"  class="h-80 mx-auto" /> |

---

本次比赛有两个任务，一个是citeseq，一个是multiome，可以看成两个比赛

cite和multi分别对应citeseq和multiome

>For the Multiome samples: given chromatin accessibility, predict gene expression. **DNA->RNA**

>For the CITEseq samples: given gene expression, predict protein levels. **RNA->Protein**

0.743 for SITE and 0.257 for MULTI

Cite有48663个测试行，Multi有55935个行（行=单元格的数量）。Multi只是使用了30%的行，意味着它实际上有16780个单元格。

---

- test_cite_inputs.h5
- test_multi_inputs.h5
- train_cite_inputs.h5

---

- train_cite_targets.h5

<img src="http://kuroweb.tk/picture/16624673725193884.jpg" class="h-50 ">
<div class="text-xs">
70988 cell × 140 

140列为已被dsb归一化的相同细胞的表面蛋白水平。

</div>

重要的列
<div class="text-xs">
Important_cols is the set of all features whose name matches the name of a target protein. If a gene is named 'ENSG00000114013_CD86', it should be related to a protein named 'CD86'. These features will be used for the model unchanged, that is, they don't undergo dimensionality reduction.
</div>


> 所有的140 column如下



<div class="overflow-auto h-100 text-xs">
CD86 CD274 CD270 CD155 CD112 CD47 CD48 CD40 CD154 CD52 CD3 CD8 CD56 CD19 CD33 CD11c HLA-A-B-C CD45RA CD123 CD7 CD105 CD49f CD194 CD4 CD44 CD14 CD16 CD25 CD45RO CD279 TIGIT Mouse-IgG1 Mouse-IgG2a Mouse-IgG2b Rat-IgG2b CD20 CD335 CD31 Podoplanin CD146 IgM CD5 CD195 CD32 CD196 CD185 CD103 CD69 CD62L CD161 CD152 CD223 KLRG1 CD27 CD107a CD95 CD134 HLA-DR CD1c CD11b CD64 CD141 CD1d CD314 CD35 CD57 CD272 CD278 CD58 CD39 CX3CR1 CD24 CD21 CD11a CD79b CD244 CD169 integrinB7 CD268 CD42b CD54 CD62P CD119 TCR Rat-IgG1 Rat-IgG2a CD192 CD122 FceRIa CD41 CD137 CD163 CD83 CD124 CD13 CD2 CD226 CD29 CD303 CD49b CD81 IgD CD18 CD28 CD38 CD127 CD45 CD22 CD71 CD26 CD115 CD63 CD304 CD36 CD172a CD72 CD158 CD93 CD49a CD49d CD73 CD9 TCRVa7.2 TCRVd2 LOX-1 CD158b CD158e1 CD142 CD319 CD352 CD94 CD162 CD85j CD23 CD328 HLA-E CD82 CD101 CD88 CD224
</div>

---

- train_multi_inputs.h5
- train_multi_targets.h5

---


## 训练相关



<style>
  h2{
    text-align: center;
    margin:200px;
    background: #aaaaaa;
    text-shadow: 0 0 10px #000000;
    color: #ffffff;
  }
  </style>

--- 

### 比赛特点

本次比赛只需要提交submission.csv，也就是纯表格赛


训练时间无限，训练资源无限


---

### CV划分

<br/>

```
kf.split(X, groups=meta.donor)
```


特征和天数也有关系，随着时间有固定方向的偏移

<img src="http://kuroweb.tk/picture/16624607353931628.jpg" class="h-75"/>

> 注意：ensemble要统一cv

---

### 基于TensorCSR编写Pytorch模型

```
config = dict(
    layers = [128, 128, 128],
...
class MLP(nn.Module):
    def __init__(self, layer_size_lst, add_final_activation=False):
        super().__init__()
        
        assert len(layer_size_lst) > 2
        
        layer_lst = []
        for i in range(len(layer_size_lst)-1):
            sz1 = layer_size_lst[i]
            sz2 = layer_size_lst[i+1]
            layer_lst += [nn.Linear(sz1, sz2)]
            if i != len(layer_size_lst)-2 or add_final_activation:
                 layer_lst += [nn.ReLU()]
        self.mlp = nn.Sequential(*layer_lst)
```

在Kaggle 16G GPU，13G RAM中，可以传入进来一个22万维的tensor稀疏向量`torch.Size([512, 228942])`

> 缺点：
只能使用max归一化;减去平均值，会破坏这里的tensorCSR稀疏性。操作受限。

---

### 人工特征+PCA/SVD降维+树模型/MLP


编码器解码器


target 是 140 ，140个机器学习器


--- 

### Loss

<table class="my-10" style="table-layout:fixed">
  <tr>
    <th>Pearson</th>
    <th>MSE</th>
  </tr>
  <tr>
    <td><pre class="slidev-code shiki shiki-light">
class NegativeCorrLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, preds, targets):
        my = torch.mean(preds, dim=1)
        my = torch.tile(torch.unsqueeze(my, dim=1), (1, targets.shape[1]))
        ym = preds - my
        r_num = torch.sum(torch.multiply(targets, ym), dim=1)
        r_den = torch.sqrt(
            torch.sum(torch.square(ym), dim=1) * float(targets.shape[-1])
        )
        r = torch.mean(r_num / r_den)
        return -r
</pre></td>
    <td>
    <pre  class="slidev-code shiki shiki-light">def criterion(outputs, labels):
    """ MSE Loss function"""
    return nn.MSELoss()(outputs, labels)</pre>
    </td>
 </tr>
</table>


> 两者可以都尝试一下

---

## ensemble策略



- Statement 1. Correlation loss is <font class="text-red-600">insensitive to linear transformations</font> of predictions

- Statement 2. Per-cell_id standardization helps to rescale base submissions
Under assumption that two base submissions are similar and demonstrate similar performance we could rescale them in the way that they become comparable and weighting in a regular way becomes adequate:

<img src="http://kuroweb.tk/picture/16624614231804902.jpg" class="h-50 pl-10"/>

- Statement 3. Weighting coefficients don't have to add up to 1!
This is one of the benefit of the loss function that is agnostic to linear transformations. You don't have to weight base submissions as usual with $\sum_i w_i=1$. Any coefficients will do the job!


---

作者做了一个实验

<img src="http://kuroweb.tk/picture/16624668586941400.jpg" class="h-35 mx-auto">

<br/>
<br/>
<br/>

|std前|std后|
|:--:|:--:|
0.92417836 | 0.94238122

