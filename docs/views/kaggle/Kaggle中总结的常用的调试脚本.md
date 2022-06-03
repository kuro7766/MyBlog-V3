---
title: Kaggle中常用的调试脚本
date: 2022-05-06
sidebar: "auto"
categories:
- kaggle
  
tags:
- kaggle
---

![](http://kuroweb.tk/picture/16519227147870658.jpg)

<!-- more -->

## imports 



## 动态执行语句

注意这个语句不可以写在其他文件中

```python
exec(compile(open("动态执行的代码", "rb").read(), "tmp.py", 'exec'))
```

## Imports

```python
import os
import warnings
from pprint import pprint
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as T

from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset

import tensorflow as tf
import gc
```



## 常用函数

```python
import inspect, re

from contextlib import contextmanager
import time

class g:
#     d=False
    d=True
    
    @classmethod
    def tqdm(cls,iterable):
        if g.d:
            return iterable[0:g.lslice:1000]
        return iterable
    
def dbg(*args, **kwargs):
#     return
    print(*args, **kwargs)

class NamePrinter:
  def __init__(self,funcname,print_fun = print,argprint_lambda = lambda x: x):
    self.funcname = funcname
    self.print_fun = print_fun
    self.argprint_lambda = argprint_lambda
  def adb(self,p):
#      return 
      funcname= self.funcname
      argument_real_name = None
      for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
          m = re.search(r'\b%s\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)' %funcname, line)
      if m:
          argument_real_name = m.group(1)

      self.print_fun('\n>>>',argument_real_name,f'\n{self.argprint_lambda(p)}',)
      self.print_fun('<<< --------------','\n')

adb = NamePrinter('adb').adb
sdb = NamePrinter('sdb',argprint_lambda=lambda x : x.shape).adb
    
class classproperty(property):
    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()
    
    
def monkeypatch_method_to_class(cls): #为torch tensor等挂载一些函数
    def decorator(func):
        setattr(cls, func.__name__, func)
        return func
    return decorator


# 时间测试工具
@contextmanager
def timed(label="NoLabel",enabled = True):
    start = time.time()  # Setup - __enter__
    if enabled:
        print(f"[{label}] time benchmark started")
    try:
        yield  # yield to body of `with` statement
    finally:  # Teardown - __exit__
        if enabled:
            end = time.time()
            print(f"[{label}] used {end - start} s")
        

    
def iv(func,*args,**kwargs):
#     print(f'{func.__name__}')
    return func(*args,**kwargs)
```

## 需要pip安装的工具

```python
!pip install py-heat-magic #https://github.com/csurfer/pyheatmagic

```

## colabcode远程连接

```python
import time
from threading import Thread
import os,re
! pip install git+https://github.com/kuro7766/colabcode.git
!ngrok authtoken 27xqdUZec8gJCpJ2g8maHKgQAuA_6uT52UntAv25GP48JzA4?
from colabcode import ColabCode
import random
def a():
    ColabCode(port=random.randint(10000,12000),lab=True)
Thread(target=a,name='a').start()
import pickle,re
while not os.path.exists("_ng_url.pkl"):
    time.sleep(1)
    print('waiting for url')
with open("_ng_url.pkl", "rb") as f:
    url = pickle.load(f)
    print('>>> url')
    print(re.findall('https://.*.ngrok.io',str(url))[0]+'/?token=123456')

```



## Numpy拼接

```python
import numpy as np


class GrowableNumpyArray:

    def __init__(self, dtype=np.float, grow_speed=4):
        self.data = np.zeros((100,), dtype=dtype)
        self.capacity = 100
        self.size = 0
        self.grow_speed = grow_speed

    def update(self, row):
        for r in row:
            self.add(r)

    def add(self, x):
        if self.size == self.capacity:
            self.capacity *= self.grow_speed
            newdata = np.zeros((self.capacity,))
            newdata[:self.size] = self.data
            self.data = newdata

        self.data[self.size] = x
        self.size += 1

    def finalize(self):
        data = self.data[:self.size]
        return data

```

## 可以计数的上下文


```python
import sys
import inspect

class CounterExec(object):
    counter = 0
    
    def __init__(self,enabled=False,every=50):
        """
        if mode = 0, proceed as normal
        if mode = 1, may do not execute block
        """
        self.mode=enabled
        self.every = every
    def __enter__(self):
        self.__class__.counter += 1
        
        exec_flag = False
        if self.mode == 1:
            if self.__class__.counter%self.every==0:
                exec_flag = True
        elif self.mode ==0:
            exec_flag=True
            
        if exec_flag:
            pass
        else:
            print('Skipping Context ... ')
            sys.settrace(lambda *args, **keys: None)
            frame = sys._getframe(1)
            frame.f_trace = self.trace
            return 'SET BY TRICKY CONTEXT MANAGER!!'
    def trace(self, frame, event, arg):
        raise
    def __exit__(self, type, value, traceback):
        print('Exiting context ...')
        return True
```



## 随机数&可复现性

```python
def set_seed(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')
    
```