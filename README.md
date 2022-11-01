# bert_socre_test
测试 transformers 库对bert_score计算的影响

环境安装：


```
conda create python=3.8 -n bert_score
conda activate bert_score
pip install bert_score
pip install evaluate
```

运行test.py，得到 bert score 的结果为 0.5038

之后，降级 transformers 版本，其他保持不变

```
pip install transformers~=4.15.0
```

再次运行，结果为 0.45
