###该项目是原HIT模型的优化版本
###增加了相对位置编码
###增加了数据增强策略
###采用了动态图结构学习
###优化的主要目的是用尽量少的显存使得模型性能尽量强
###优化后使用全部数据即便用4060 8g显存也可以运行


### How to install requirements
```sh
$ pip install -r requirements.txt
```

### Run HITmajorization
```
$ python exp.py
```





