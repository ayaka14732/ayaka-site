---
title: 入聲字情感分析
lang: zh-HK
keywords:
- 入聲字
- 情感分析
- 音韻學
- 自然語言處理
- Hugging Face Transformers
math: |-
  <meta name="description" content="TL;DR: 入聲字的情感與舒聲字無明顯差別。"/>
---

> 同時發佈於[知乎](https://zhuanlan.zhihu.com/p/501762638)

TL;DR: 入聲字的情感與舒聲字無明顯差別。

# 簡介

上次羣友提出了一個問題：「有沒有一種説法，入聲字有很多冷酷、消極、負能量的字？」

![](1.jpg)

為了解決這個問題，我決定使用情感分析模型。

在 NLP 中，情感分析是文本分類的一個分支，是對帶有情感色彩（褒義貶義/正向負向）的主觀性文本進行分析，以確定該文本的觀點、喜好、情感傾向^[<https://houbb.github.io/2020/01/20/nlp-qingganfenxi-01-overview>]。目前非常流行的 NLP 函式庫 [Hugging Face Transformers](https://huggingface.co/docs/transformers/quicktour) 中提供了預訓練的 DistilBERT 模型作為情感分析模型。

解決問題的思路是：首先從漢英詞典 [Heptagon196/Dict](https://github.com/Heptagon196/Dict) 中選出所有單字條目及對應的英文解釋^[如果某個條目對應多個解釋，則只選擇第一個。例如「[贴]{lang=zh-Hans}」有 to stick; to paste; to keep close to; to fit snugly; allowance 五個解釋，分析時僅選擇 to stick 一個]。
然後，對於所有單字，使用 [ToJyutping](https://github.com/CanCLID/ToJyutping) 函式庫為單字標註粵語拼音^[對於多音字，ToJyutping 函式庫只返回頻率最高的一個讀音]，
然後據粵語拼音判斷是否該字是否為入聲字^[有少數字在中古並非入聲字，但在粵語中變為入聲，如「[值]{lang=zh-Hans}」。但這類字較少，因此對結果影響不大]。接下來，對於所有單字，根據其對應的英文解釋，使用預訓練的 DistilBERT 模型判斷該字的情感傾向得分^[單字與對應的英文解釋的含義可能不完全一致，從而導致情感分析的結果不準確。例如「[挽]{lang=zh-Hans}」的英文解釋為 draw，但 draw 還有「繪畫」的意思，而「[挽]{lang=zh-Hans}」字無。不過這類字較少，因此對結果影響不大]。最後，比較舒聲字和入聲字的情感傾向平均分。若得分相近，則説明入聲字的情感與舒聲字無明顯差別，否則有明顯差別。

# 實驗過程

本次實驗使用 NumPy 1.22.3，TensorFlow 2.10.0 及 Hugging Face Transformers 4.18.0 版本。

從漢英詞典中選出所有單字條目及對應的英文解釋的程式較為簡單，此處略去。

情感分析部分的程式如下。

1\. 引入函式庫

```python
import ToJyutping
from tqdm import tqdm
from transformers import pipeline
import numpy as np
import re
```

2\. 載入情感分析模型

```python
classifier = pipeline('sentiment-analysis')
```

3\. 定義情感傾向得分函式

```python
def handle_one_result(result):
    if result['label'] == 'POSITIVE':
        score = result['score'] * 100
        assert score > 50.
    else:
        score = 100. - result['score'] * 100
        assert score < 50.
    return score
```

4\. 定義輔助函式

```python
def 分块(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

def 根据粤拼判断入声(粵拼: str) -> bool:
    return bool(re.search('[ptk]\d$', 粵拼))
```

5\. 讀入單字及其英文解釋，並分塊

```python
汉字_英文列表 = []
英文列表 = []

with open('词表.txt', encoding='utf-8') as f:
    for line in f:
        汉字, 英文 = line.rstrip('\n').split('\t')
        汉字_英文列表.append((汉字, 英文))
        英文列表.append(英文)

分块英文列表 = 分块(英文列表, 64)
```

6\. 使用情感分析模型得出情感傾向得分

```python
情感分析结果列表 = []

for 英文块 in tqdm(分块英文列表):
    情感分析结果块 = classifier(英文块)
    情感分析结果块 = list(map(handle_one_result, 情感分析结果块))
    情感分析结果列表.extend(情感分析结果块)
```

7\. 判斷入聲字，並分別統計舒聲字和入聲字的得分

```python
舒声字得分列表 = []
入声字得分列表 = []

with open('结果.txt', 'w', encoding='utf-8') as f:
    for (汉字, 英文), 情感分析结果 in zip(汉字_英文列表, 情感分析结果列表):
        粵拼 = ToJyutping.get_jyutping_text(汉字)
        if not 粵拼:
            continue  # 不处理没有读音的情况
        是入声字 = 根据粤拼判断入声(粵拼)
        if not 是入声字:
            舒声字得分列表.append(情感分析结果)
        else:
            入声字得分列表.append(情感分析结果)
        print(汉字, 粵拼, '舒' if not 是入声字 else '入', 英文, 情感分析结果, sep='\t', file=f)
```

8\. 計算舒聲字和入聲字的情感傾向平均分，並輸出

```python
舒声字平均分 = np.array(舒声字得分列表).mean()
入声字平均分 = np.array(入声字得分列表).mean()

print(f'舒声字平均分：{舒声字平均分:.2f}')
print(f'入声字平均分：{入声字平均分:.2f}')
```

# 實驗結果

程式輸出：

```
舒声字平均分：60.90
入声字平均分：59.42
```

因此，入聲字的情感與舒聲字無明顯差別。

# 附錄

得分最高的前 10 個字：

- [恺]{lang=zh-Hans} joyful 舒 99.9882459640503
- [嫣]{lang=zh-Hans} captivating 舒 99.98819828033447
- [绮]{lang=zh-Hans} beautiful 舒 99.98807907104492
- [粲]{lang=zh-Hans} beautiful 舒 99.98807907104492
- [琼]{lang=zh-Hans} beautiful 舒 99.98807907104492
- [婺]{lang=zh-Hans} beautiful 舒 99.98807907104492
- [婵]{lang=zh-Hans} beautiful 舒 99.98807907104492
- [娟]{lang=zh-Hans} beautiful 舒 99.98807907104492
- [娈]{lang=zh-Hans} beautiful 舒 99.98807907104492
- [妍]{lang=zh-Hans} beautiful 舒 99.98807907104492

得分最低的前 10 個字：

- [累]{lang=zh-Hans} cumbersome 舒 0.019413232803344727
- [髡]{lang=zh-Hans} make the head bald 舒 0.019556283950805664
- [滞]{lang=zh-Hans} sluggish 舒 0.0196993350982666
- [镌]{lang=zh-Hans} degrade 舒 0.019878149032592773
- [颓]{lang=zh-Hans} become bald 舒 0.020331144332885742
- [鲁]{lang=zh-Hans} crass 舒 0.020509958267211914
- [朽]{lang=zh-Hans} rotten 舒 0.020557641983032227
- [摞]{lang=zh-Hans} pile up 舒 0.020569562911987305
- [旷]{lang=zh-Hans} waste 舒 0.02065300941467285
- [懈]{lang=zh-Hans} lax 舒 0.020700693130493164

可以看到得分最高和最低的前 10 個字都是舒聲字。

得分最高的前 10 個入聲字：

- [赫]{lang=zh-Hans} awe-inspiring 入 99.98770952224731
- [阔]{lang=zh-Hans} rich 入 99.9876856803894
- [倜]{lang=zh-Hans} energetic 入 99.98763799667358
- [实]{lang=zh-Hans} real 入 99.98708963394165
- [诺]{lang=zh-Hans} to promise 入 99.98689889907837
- [燠]{lang=zh-Hans} warm 入 99.98680353164673
- [煜]{lang=zh-Hans} brilliant 入 99.98660087585449
- [灼]{lang=zh-Hans} brilliant 入 99.98660087585449
- [铄]{lang=zh-Hans} bright 入 99.98644590377808
- [翌]{lang=zh-Hans} bright 入 99.98644590377808

得分最低的前 10 個入聲字：

- [蛭]{lang=zh-Hans} fluke 入 0.02079606056213379
- [恝]{lang=zh-Hans} indifferent 入 0.020998716354370117
- [辱]{lang=zh-Hans} disgrace 入 0.021058320999145508
- [谪]{lang=zh-Hans} disgrace (an official) 入 0.02148747444152832
- [碌]{lang=zh-Hans} laborious 入 0.02244114875793457
- [僻]{lang=zh-Hans} low 入 0.02256035804748535
- [隰]{lang=zh-Hans} low 入 0.02256035804748535
- [俗]{lang=zh-Hans} vulgar 入 0.022727251052856445
- [弱]{lang=zh-Hans} weak 入 0.023156404495239258
- [蹩]{lang=zh-Hans} limp 入 0.023239850997924805

本次實驗的完整程式碼見 [Gist](https://gist.github.com/ayaka14732/d4527a028e5e96e54de67305eb2967f1)。

（作於 2022&#8239;年&#8239;4&#8239;月&#8239;17&#8239;日，發佈於 2022&#8239;年&#8239;4&#8239;月&#8239;20&#8239;日）
