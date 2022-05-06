---
title: Kaggle中常用的调试脚本
date: 2022-05-06
sidebar: "auto"
categories:
- kaggle
  
tags:
- kaggle
---

<!-- more -->

# 动态执行语句

注意这个语句不可以写在其他文件中

```python
exec(compile(open("动态执行的语句", "rb").read(), "tmp.py", 'exec'))
```

# 常用函数

```python
class classproperty(property):
    def __get__(self, cls, owner):
        return classmethod(self.fget).__get__(None, owner)()
    
def monkeypatch_method_to_class(cls): #为torch tensor等挂载一些函数
    def decorator(func):
        setattr(cls, func.__name__, func)
        return func
    return decorator

class g:
#     d=False
    d=True
    lslice=1000000
    sample_image=200
    @classmethod
    def tqdm(cls,iterable):
        if g.d:
            return iterable[0:g.lslice:1000]
        return iterable
    
def iv(func,*args,**kwargs):
#     print(f'{func.__name__}')
    return func(*args,**kwargs)
```

