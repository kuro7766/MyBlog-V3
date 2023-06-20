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
from torch.utils.data import DataLoader
from datasets import Dataset,DatasetDict

import tensorflow as tf
import gc
```

## v2ray代理

用于解决huggingface，wandb等网络问题。Idea系列的代理等也是这个端口。Proxychains系列是Sock5 10808端口。

```python
os.environ['http_proxy'] = "http://127.0.0.1:10809" 
os.environ['https_proxy'] = "http://127.0.0.1:10809"
```



## g-Config 常用函数

```python
from __future__ import print_function
import inspect, re,os,pickle

from contextlib import contextmanager
import time

# os.environ["WANDB_DISABLED"] = "true"
class g: 
    debug=False
    # debug=True
    
    # d1 = False
    d1 = True # explanation -> 
    ''' '''

    # d2 = False
    d2 = True # explanation -> 
    ''' '''

    # d3 = False
    d3 = True # explanation -> 
    ''' '''

    # d4 = False
    d4 = True # explanation -> 
    ''' '''

    # d5 = False
    d5 = True # explanation -> 
    ''' '''
    
    # d6 = False
    d6 = True # explanation -> 
    ''' '''

    # d7 = False
    d7 = True # explanation -> 
    ''' '''

    if not debug:
        d1,d2,d3,d4,d5,d6,d7 = False,False,False,False,False,False,False,False
    
    seed = 42
import uuid

class EnvHelper:
    def new_name(self):
        return uuid.uuid4().__str__().replace('-','_')
    def __init__(self):
        self.data = {}
    def save(self,*names):
        commands = []
        for n in names:
            commands.append(f'global {n}')
            if (not self.data) or not self.data.__contains__(n):
                print(n)
                self.data[n] = []
        for n in names:
            new_name = n + self.new_name()
            self.data[n].append(new_name)
            commands.append(f'global {new_name}')
            commands.append(f'{new_name} = {n}')
            commands.append(f'{n} = None')
        exec(compile('\n'.join(commands), "tmp.py", 'exec'))
    def restore(self,*names):
        commands = []
        for n in names:
            commands.append(f'global {n}')
            if not self.data.__contains__(n):
                raise f"{n} not saved !"
        for n in names:
            shadow_name = self.data[n].pop()
            commands.append(f'global {shadow_name}')
            commands.append(f'{n} = {shadow_name}')
        exec(compile('\n'.join(commands), "tmp.py", 'exec'))
    
def dbg(*args, **kwargs):
#     return
    print(*args, **kwargs)

class NamePrinter:
  def __init__(self,funcname,print_fun = print,argprint_lambda = lambda x: x):
    self.funcname = funcname
    self.print_fun = print_fun
    self.argprint_lambda = argprint_lambda
  def adb(self,p=''):
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
tdb = NamePrinter('tdb',argprint_lambda=lambda x : (time.sleep(1),x)[1]).adb

if locals().get('get_ipython',None):

    from IPython.core.magic import Magics, magics_class, line_magic,cell_magic

    @magics_class
    class MyMagics(Magics):

        @cell_magic
        def loop(self, line, cell):
            # get cmagic args
            args = line.split(' ')
            for i in range(int(args[0])):
                print('>>> loop',i+1,'of',args[0])
                self.shell.run_cell(cell, store_history=False)
                
    get_ipython().register_magics(MyMagics)


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

def write_obj(path, obj3):
    obj3 = pickle.dumps(obj3)
    with open(path, 'wb')as f:
        f.write(obj3)

def read_obj(path,default=''):
    if not os.path.exists(path):
        return default
    f = open(path, "rb")
    obj = default
    try:
        obj = pickle.load(f)
    except:
        obj = default
    f.close()
    return obj
    
def iv(func,*args,**kwargs):
#     print(f'{func.__name__}')
    return func(*args,**kwargs)

```

## 需要pip安装的工具

```python
!pip install py-heat-magic 

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
os.remove('_ng_url.pkl')
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

## 调参用

```
def param_picker(dict_of_list,constrant = lambda c: True):
    retry = 0
    while True:
        params = {}
        import random
        for key in dict_of_list.keys():
            lst = dict_of_list[key]
            elem = random.choice(lst)
            params[key] = elem
        if constrant(params):
            return params
        retry += 1
        
        assert retry < 1000,f'param_picker retry too many times with failure'
{**param_picker({'a':[1,2,3],'b':[4,5,6]}),**{'ext_dict':10}}
```

## Kaggle初始化&文件上传



```
!pip install kaggle
!mkdir .kaggle
!mkdir ~/.kaggle/
import json
token = {"username":"galegale05","key":"keykey"}
with open('kaggle.json', 'w') as file:
    json.dump(token, file)
!cp kaggle.json ~/.kaggle/kaggle.json
!chmod 600 /root/.kaggle/kaggle.json
!kaggle config set -n path -v ./autodl-tmp
!kaggle datasets list
```



```
upd = 'path-to-dir'
!kaggle datasets init -p {upd}
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
!kaggle datasets create -p {upd} --dir-mode zip
```

```

```

