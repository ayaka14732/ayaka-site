---
title: 挖空算法及其 Python 实现
lang: zh-CN
keywords:
- 挖空
- wakong
- BART
- 大规模语言模型
- 模型预训练
---

> 同时发布于知乎：[挖空算法及其 Python 实现](https://zhuanlan.zhihu.com/p/572321312)

# 简介

给句子挖空是深度学习预训练中的一个常见操作。例如，在 BART 模型的预训练中，text infilling 这一训练目标是让模型习得完形填空的能力，这就需要设计挖空算法，以生成用于模型训练的句子。然而，目前的论文对实验中采取的挖空方法只有模糊的介绍，只具有理论意义，无法真正实现为算法；而大多数实现也只是设计了大致可用的算法，没有进行深入的分析。对此，本文提出了一个恰当且健壮的挖空算法 (Wakong Algorithm)，并发布了 Python 库，可以在生产环境中直接使用。

本文作为我的大规模语言模型预训练项目的一部分，得到了谷歌 [TPU Research Cloud](https://sites.research.google/trc/about/) (TRC) 的云 TPU 支持，为项目节省了大量时间。

# 问题定义

## 算法输入与输出

挖空算法是一个用于给固定长度的句子挖空的算法。算法的输入为句子的长度 `seq_len`，输出为一个由二元组组成的列表，其中每个二元组表示在句子中挖的一个空，第一个元素为起始位置，第二个元素为长度。

例如，当输入句子长度 40 时，算法的输出可能为：

```
[(5, 4), (23, 2)]
```

这一输出表示句子被挖了两个空，第一个空的起始位置为 5，长度为 4；第二个空的起始位置为 23，长度为 2。用图示可以表示如下：

.....(xxxx)..............(xx)...............

## 算法要求

在 BART 论文的基础上，本文认为一个合理的挖空算法需要满足以下四个要求：

1. 挖空算法是一个随机算法，即在输入相同时，算法可能会输出不同的结果。这是为了使挖空的方式更为多样化，使模型更好地习得完形填空的知识；
1. 挖空算法平均挖去的词应当占句子总词数的 15%。选择 15% 这个数字较为适中，不致使预训练目标过于简单，同时又保留一定量的语义信息，可以完成完形填空的任务；
1. 挖空算法中每一个空的长度不固定，在 0-10 之间，长度为 3 的空出现频率最高，频率由 0-3 递增，由 3-10 递减。可以出现长度为 0 的空，这是为了使模型学到完形填空中可能不需要填写任何词，增加训练目标的难度，从而使模型学到更多的语义知识；
1. 挖空算法中任意两个空不能重叠，也不能直接相邻，间隔至少为 1。这是为了保证算法输出的格式合理。

# 算法流程

1\. 常数

$\mathsf{proposedMaskRate} = 0.188$

$\mathsf{poissonRate} = 4.2$

$\mathsf{maxSpanLen} = 10$

2\. `probsList`

$\mathsf{probsList} = \left[ \mathrm{normalise} \left(  \mathsf{probs} \left[ {:}\,i \right] \right) \mathrm{for} \; i \; \mathrm{in} \left[2, \; .. , \; \mathsf{maxSpanLen} + 1 \right] \right]$

$\mathsf{probs} = \left[ \Pr(X=i) \; \mathrm{for} \; i \; \mathrm{in} \left[ 0, \; .., \; \mathsf{maxSpanLen} + 1 \right] \right]$

$X \sim \mathrm{Pois}(\mathsf{poissonRate})$

3\. `determineShouldMaskLen`

$\mathsf{determineShouldMaskLen} \left( \mathsf{seqLen} \right) =
\begin{cases}
    \lceil x \rceil, & \text{if} \; \omega < p \\
    \lfloor x \rfloor, & \text{otherwise} \\
\end{cases}$

$\omega \sim \mathrm{U} \left( 0, 1 \right)$

$x = \mathsf{seqLen} * \mathsf{proposedMaskRate}$

$p = x - \lfloor x \rfloor$

4\. `generateSpans`

$\mathsf{generateSpans} \left( m \right) = \mathrm{shuffle} \left( \mathrm{anamorphism} \left( f \right) \left( m \right) \right)$

$f \left( \mathsf{remainedLen} \right) =
\begin{cases}
    \mathrm{Nothing}, & \text{if} \; \mathsf{remainedLen} \leq 0 \\
    \left( \mathsf{span}, \; \mathrm{Just} \left( \mathsf{remainedLen} - \mathsf{span} - 1 \right) \right), & \text{otherwise}
\end{cases}$

$\mathsf{span} \sim \mathrm{Categorical} \left( [0, \; .., \; n + 1], \; \mathsf{probsList} \left[ n - 1 \right] \right)$

$n = \min \left( \mathsf{maxSpanLen}, \; \mathsf{remainedLen} \right)$

5\. `distributeInsertPoses`

$\mathsf{distributeInsertPoses} \left( \mathsf{xs} \right) = f \left( \mathsf{xs}, \; 0 \right)$

$f \left( n, \; \mathsf{xs} \right) =
\begin{cases}
    \mathsf{\left[ \, \right]}, & \text{if} \; \mathrm{empty} \left( \mathsf{xs} \right) \\
    \left[ \left( p + n, \; s \right) \right] + f \left(n + s + 1, \; \mathsf{ys} \right), & \text{otherwise} \\
\end{cases}$

$\left[ \left( p, s \right) \right] + \mathsf{ys} \leftarrow \mathsf{xs}$

6\. `randomAddOne`

$\mathsf{randomAddOne} \left( \mathsf{xs} \right) = \begin{cases}
    \mathsf{xs}, & \text{if} \; \omega < 0.5 \\
    \left[ (p + 1, s) \; \mathrm{for} \; (p, s) \; \mathrm{in} \; \mathsf{xs} \right], & \text{otherwise} \\
\end{cases}$

$\omega \sim \mathrm{U} \left( 0, 1 \right)$

7\. `wakong`

$\mathsf{wakong} \left( \mathsf{seqLen} \right) = \mathsf{randomAddOne} \left( \mathsf{distributeInsertPoses} \left( \mathrm{zip} \left( \mathsf{absInsertPoses}, \; \mathsf{spans} \right) \right) \right)$

$\mathsf{absInsertPoses} = \mathrm{sort} \left( X \right)$

$X = X_{1, \; .., \; \mathsf{nSpans}} \sim \mathrm{DiscreteUniform} \left[ 0, \; \mathsf{nPossibleInsertPoses} - 1 \right]$

$\left( \forall \; i, j \in \left\{ 1, \; .., \; \mathsf{nSpans} \right\}, X_i \ne X_j \right)$

$\mathsf{nPossibleInsertPoses} = \mathsf{seqLen} - \mathrm{sum} \left( \mathsf{spans} \right) - \mathsf{nSpans} + 1$

$\mathsf{nSpans} = \mathrm{len} \left( \mathsf{spans} \right)$

$\mathsf{spans} = \mathsf{generateSpans} \left( \mathsf{shouldMaskLen} \right)$

$\mathsf{shouldMaskLen} = \mathsf{determineShouldMaskLen} \left( \mathsf{seqLen} \right)$

# 时间复杂度

算法中时间复杂度最高的步骤是对随机生成的 $kn$ 个空排序，因此总体时间复杂度为 $O \left( n \log n \right)$。

# 算法设计中的难点

## 确定从句子中挖去的词数

根据算法要求，平均挖去的词应当占句子总词数的 15%，但这样计算有时会出现小数。为此，本文设定在出现小数时，根据小数位随机决定向下取整或向上取整。例如，若计算得出挖去的词为 3.3 个，则随机生成一次 0-1 之间均匀分布的随机数，若该数小于 0.3 则向上取整为 4，否则向下取整为 3。

## 随机选取挖空的长度

根据 BART 论文，本文从泊松分布中采样，随机生成挖空的长度。本文没有按照 BART 论文将泊松分布的参数设定为 4，而是设定为 3.5，以使长度为 3 的空出现频率最高（但是，在后续步骤中会提到，本文将这一参数修订为 4.2）。对于大于 10 的值，本文将其概率设定为 0，并将 0-10 的值的概率标准化至求和为 1。这样生成的分布的累积分布函数为 [0.0151 0.0783 0.2111 0.3970 0.5922 0.7562 0.8710 0.9399 0.9760 0.9929 1.0000]。

## 生成挖空的长度列表

从上述分布中采样多次，即可生成挖空的长度列表。采样在挖空的长度之和达到要挖去的词数时停止。

若挖空的长度之和没有达到目标词数，但采样结果加挖空的长度之和大于目标词数（如目标词数为 10，当前长度之和为 9，但采样结果为 5，9 加 5 大于 10），则舍弃该次采样结果，重新进行采样，直到采样结果加挖空的长度之和在目标词数的范围内为止。在实际实现时，为了保证算法效率，不应该采样失败重采，而是应当首先根据目标词数与挖空的长度之和计算出预期采样结果的范围，再根据上述分布计算出除去超出预期采样结果范围的值的新分布，将概率标准化至求和为 1 后从新分布中采样。

由于算法要求任意两个空不能直接相邻，长度为 _k_ 的空实际也占据了它右侧的一个位置，即实际长度为 _k_+1，因此在计算挖空的长度之和时，每一个空的长度需要额外加 1，即挖空的长度之和加挖空的个数。这样很好地避免了两个空直接相邻的问题，但会导致最终平均挖去的词数比预期的 15% 小（为此，在后续步骤中会将平均挖去的词数调整为 18.8%，使得最终结果接近 15%）。

由于在采样起始时可能出现长度为 0 的样本，而采样终止的条件为达到目标长度，不可能出现长度为 0 的样本，因此会产生不对称的现象。为此，在采样结束后应当对挖空的长度列表随机打乱，使挖空的长度随机分布。

## 将要挖的空均匀分布在句子中

设句子长度为 _m_，挖空的长度之和为 _K_，挖空的个数为 _n_，则可选的起始位置共有 _m_-_K_-_n_+1 个，要从这些起始位置中随机选出 _n_ 个作为起始位置。其中，减 _n_ 的原因如同上文所述，是因为长度为 _k_ 的空实际也占据了它右侧的一个位置，所以 _n_ 个空会额外占据 _n_ 个位置。

但是，这样做会导致句子的最后一个单词永远不会被挖空。为此，在完成上述步骤后，再随机生成一次 0-1 之间均匀分布的随机数，若该数小于 0.5，则将挖的所有空向右移动一位，也就是假设预留的空位在左边，由此保证了算法的对称性。

## 调整算法参数

在实现算法后，发现平均挖去的词数小于 15%，这是由于上文提到的原因，在算法的执行过程中，计算挖空的长度之和时，每一个空的长度需要额外加 1，导致实际挖去的词数小于目标词数。为此，经过试验发现，将算法参数中平均挖去的词数调整为 18.8%，最终结果接近 15.17%，接近 15%。

另外，算法生成长度较短的空的频率高于预期，这是因为当采样接近尾声时，预期的采样结果只能出现较小的值。这种现象是为算法所允许的，因为只需要保证长度为 3 的空出现频率最高即可。但是，为了使算法生成长度较长的空的频率增加，将泊松分布的参数由 3.5 修改为 4.2。

# Python 实现

```python
import jax.numpy as np
import numpyro.distributions as dist
from random import Random

proposed_mask_rate = 0.188  # resulting mask rate would be approximately 0.15
poisson_rate = 4.2  # span length = 3 would be the most frequent in the resulting distribution
max_span_len = 10

def normalise_probs(a: np.ndarray) -> np.ndarray:
    return a / a.sum()

def generate_probs_list() -> list[list[float]]:
    probs_list = []

    poisson = dist.Poisson(rate=poisson_rate)
    probs = np.exp(poisson.log_prob(np.arange(max_span_len + 1)))

    probs_ = normalise_probs(probs)
    probs_list.append(probs_.cumsum().tolist())

    for i in range(max_span_len - 1):
        probs_ = normalise_probs(probs[:-i-1])
        probs_list.append(probs_.cumsum().tolist())

    return probs_list[::-1]

probs_list = generate_probs_list()

MaskScheme = list[tuple[int, int]]

def determine_should_mask_len(rng: Random, seq_len: int) -> int:
    x = seq_len * proposed_mask_rate
    integer_part = int(x)
    fractional_part = x - float(integer_part)
    should_add = rng.random() < fractional_part
    should_mask_len = integer_part + should_add
    return should_mask_len

def generate_spans(rng: Random, should_mask_len: int) -> list[int]:
    spans = []
    while should_mask_len > 0:
        current_max_span_len = min(max_span_len, should_mask_len)
        probs = probs_list[current_max_span_len - 1]
        span_len = rng.choices(range(current_max_span_len + 1), cum_weights=probs)[0]
        spans.append(span_len)
        should_mask_len -= span_len + 1
    rng.shuffle(spans)
    return spans

def distribute_insert_poses(abs_insert_poses: list[int], spans: list[int]) -> MaskScheme:
    offset = 0
    mask_scheme = []
    for abs_insert_pos, span in zip(abs_insert_poses, spans):
        insert_pos = abs_insert_pos + offset
        mask_scheme.append((insert_pos, span))
        offset += span + 1
    return mask_scheme

def random_add_one(rng: Random, mask_scheme: MaskScheme) -> MaskScheme:
    should_add_one = rng.random() < 0.5
    if should_add_one:
        mask_scheme = [(insert_pos + 1, span) for insert_pos, span in mask_scheme]
    return mask_scheme

def wakong(rng: Random, seq_len: int) -> MaskScheme:
    should_mask_len = determine_should_mask_len(rng, seq_len)
    spans = generate_spans(rng, should_mask_len)

    n_spans = len(spans)
    n_possible_insert_poses = seq_len - sum(spans) - n_spans + 1
    abs_insert_poses = sorted(rng.sample(range(n_possible_insert_poses), n_spans))

    mask_scheme = distribute_insert_poses(abs_insert_poses, spans)
    mask_scheme = random_add_one(rng, mask_scheme)
    return mask_scheme

def test():
    seed = 42
    rng = Random(seed)
    mask_scheme = wakong(rng, 100)
    print(mask_scheme)

if __name__ == '__main__':
    test()
```

# 后记

虽然挖空算法本身并不复杂，却有大量边缘情况需要考虑。在写作本文的过程中，我感到我与其说是在设计算法，不如说是在不断说服自己必须要对一个个边缘情况特殊处理，没有更美的解法了。好在最终的算法保证了数学上的对称性，满意地解决了这一问题。

（发布于 2022&#8239;年&#8239;10&#8239;月&#8239;10&#8239;日）
