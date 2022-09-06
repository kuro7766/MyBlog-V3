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
date: 2022-05-06
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
    background: #00ff00;
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

<img src="http://kuroweb.tk/picture/16624379455884908.jpg"  class="h-100 mx-auto" />

---

- sample_submission.csv

和`evaluation_ids.csv`一一对应

<!-- ![](http://kuroweb.tk/picture/16624380043696516.jpg) -->

<!-- <img src="http://kuroweb.tk/picture/16624380043696516.jpg"  class="h-100 mx-auto" /> -->


| evaluation_id | sample_id |
| :---: | :---: |
| <img src="http://kuroweb.tk/picture/16624378953676982.jpg"  class="h-80 mx-auto" /> | <img src="http://kuroweb.tk/picture/16624380043696516.jpg"  class="h-80 mx-auto" /> |

---

本次比赛有两个任务，一个是citeseq，一个是multiome

- test_cite_inputs.h5
- test_multi_inputs.h5
- train_cite_inputs.h5
- train_cite_targets.h5
- train_multi_inputs.h5
- train_multi_targets.h5

---

<div class="top-60 left-100 absolute">

## 训练相关

</div>

---

### cv划分