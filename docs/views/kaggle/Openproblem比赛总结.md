---
title: Openproblem比赛总结
date: 2022-11-14
sidebar: "auto"
categories:
- kaggle
  
tags:
- kaggle
---

<!-- more -->



## hyperopt调参

hyperopt调参会出现一个问题，就是best参数的返回值对于np.choice对单独返回一个索引，issue见[此处](https://github.com/hyperopt/hyperopt/issues/284)。

```python
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

print space_eval(tuner_space, best)
```

[保存训练历史状态](https://github.com/hyperopt/hyperopt/issues/671)

调参的时候，如果觉得参数传递和dict的写法很别扭，可以试一下globals()全局变量，例如：

`globals().update({'HP_LEARNING_RATE': False})`

加上字符串前缀过滤的时候也会比较容易找到

## 数据集创建

```python
upd = f'cnn1d-bestparam-sub-cite-predstd-mlp12-seed{my_seed}'
!mkdir -p {upd}
!kaggle datasets init -p {upd}
assert (len(upd) < 50 and re.findall(r'[^a-zA-Z0-9-]',upd) == []),f'upd name {upd} is not valid'
with open(f'{upd}/dataset-metadata.json','w') as f:
  f.write('''
  {
    "title": "%s",
    "id": "galegale05/%s",
    "licenses": [
      {
        "name": "CC0-1.0"
      }
    ]
  }
  ''' % (upd,upd))

!cat {upd}/dataset-metadata.json

```

## kaggle token
```python
!pip install kaggle
!mkdir .kaggle
!mkdir ~/.kaggle/
import json
token = {"username":"galegale05","key":""}
with open('kaggle.json', 'w') as file:
    json.dump(token, file)
!cp kaggle.json ~/.kaggle/kaggle.json
!chmod 600 /root/.kaggle/kaggle.json
# !kaggle config set -n path -v ./input
!kaggle datasets list
```
