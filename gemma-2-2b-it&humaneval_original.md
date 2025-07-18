
# HumanEval 错误日志报告

## ❌ HumanEval/162
**BLEU**: 0.0603
```python
Traceback (most recent call last): 
  File "C:\Users\13915\AppData\Local\Temp\tmpoyliorr9.py", line 27, in <module>
    check(string_to_md5)
  File "C:\Users\13915\AppData\Local\Temp\tmpoyliorr9.py", line 17, in check
    assert candidate('Hello world') == '3e25960a79dbc69b674cd4ec67a72c62'
  File "C:\Users\13915\AppData\Local\Temp\tmpoyliorr9.py", line 12, in string_to_md5
    return hashlib.md5(text.encode()).hexdigest()
NameError: name 'hashlib' is not defined
```

## ❌ HumanEval/163
**BLEU**: 0.3106
```python
Traceback (most recent call last):
  File "C:\Users\13915\AppData\Local\Temp\tmpz0eq6w9r.py", line 27, in <module>
    check(generate_integers)
  File "C:\Users\13915\AppData\Local\Temp\tmpz0eq6w9r.py", line 17, in check
    assert candidate(2, 10) == [2, 4, 6, 8], "Test 1"
AssertionError: Test 1
```

## 📊 评估结果
- K1 score: **0.2170**
- 平均 BLEU: **0.1854**
