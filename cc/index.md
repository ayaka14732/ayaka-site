---
title: 繁简中文转换概说
lang: zh-CN
math: |-
  <style>
  :lang(zh-Hant-CN) { font-language-override: 'ZHS'; }
  td { vertical-align: middle !important; }
  </style>
---

> 同時發佈於[知乎](https://zhuanlan.zhihu.com/p/104314323)

# 预备知识

## Unicode 字符集

<img src="encode.png" alt="图中是计算机存储和显示字符的过程。Unicode 字符集的字符组成文本，使用 UTF-8 编码储存；而文本使用各地区字体显示，效果不同"/>

在计算机中储存的字符，都要经过 **编码**。目前通用的编码是 **UTF-8 编码**。

UTF-8 编码的对象是 **Unicode 字符集**，换言之，只有 Unicode 字符集收录的字符才能被编码。

Unicode 字符集的每个字符都有唯一的 **Unicode 码位**。一段连续的 Unicode 码位称为 **区**。

Unicode 中收录的汉字分布在以下几个区：

* 中日韩统一表意文字 (U+4E00-U+9FFF)
* 中日韩统一表意文字扩充区A (U+3400-U+4DBF)
* 中日韩统一表意文字扩充区B (U+20000-U+2A6DF)
* 中日韩统一表意文字扩充区C (U+2A700-U+2B73F)
* 中日韩统一表意文字扩充区D (U+2B740-U+2B81F)
* 中日韩统一表意文字扩充区E (U+2B820-U+2CEAF)
* 中日韩统一表意文字扩充区F (U+2CEB0-U+2EBEF)
* 中日韩统一表意文字扩充区G (U+30000-U+3134F)

从码位并不能直接判断简繁体。

有一些区是为兼容其他字符集而设的，称为 **兼容区**；还有一些区域是为用户自己造字而设的，称为 **私人使用区**（PUA），一般情况下不应使用。但是，这些区的字符在计算机上的显示效果有时与普通的字符并无差异，在处理来源不明的数据时需要特别留意。

## 中文的地区差异

中国大陆、香港、台湾、日本、韩国、新加坡等地各有不同的汉字标准（或规范，以下通称标准），详见维基百科 [汉字标准列表](https://zh.wikipedia.org/zh-hk/%E6%BC%A2%E5%AD%97%E6%A8%99%E6%BA%96%E5%88%97%E8%A1%A8)。

中文环境的汉字标准共有五种：

* <span lang="zh-CN">简体中文（中国大陆）</span>，简称「<span lang="zh-CN">大陆简体</span>」
* <span lang="zh-SG">简体中文（新加坡）</span>，与大陆简体相同，仅用词有别
* <span lang="zh-HK">繁體中文（香港）</span>，简称「<span lang="zh-HK">香港繁體</span>」
* <span lang="zh-TW">繁體中文（臺灣）</span>，简称「<span lang="zh-TW">臺灣繁體</span>」或「<span lang="zh-TW">臺灣正體</span>」
* <span lang="zh-Hant-CN">繁體中文（中國大陸）</span>，简称「<span lang="zh-Hant-CN">大陸繁體</span>」

实际上大陆繁体并不常见，因为大陆出版的古籍亦需要保留原文样态，不会随意根据现代标准修改。

地区差异分为以下几类：

1. 句子的表达方式不同（最难，本文暂不讨论）
1. 句子大致相同，用词不同（称「用词差异」）
1. 用词相同，但汉字的形态可能不同。这分为三种情况：
    1. 汉字形态不同，但 Unicode 码位相同（称「字形差异」）
    1. 汉字形态不同，且 Unicode 码位也不同，但是同一汉字（称「编码差异」）
    1. 汉字形态不同，且 Unicode 码位也不同，但是不同汉字（称「用字差异」）

以下分别讲解。

# 字形差异

例如「削弱」一词，香港与台湾均用「削（U+524A）弱」，但是分别使用香港与台湾的字体显示，会有明显差异。

* 香港：<span lang="zh-HK"><img src="soek-HK.png" alt="削"/><img src="joek.png" alt="弱"/></span>
* 台湾：<span lang="zh-TW"><img src="soek-TW.png" alt="削"/><img src="joek.png" alt="弱"/></span>

此类差异，需要用户选用合适的字体，方能正确显示，例如 [思源黑体](https://github.com/adobe-fonts/source-han-sans) 支持陆、港、台、日、韩五地的汉字字形。

作为开发者，通常是通过在代码中指定语言来解决，例如「育」字：

::: {lang="en-x-code"}
```html
<ul>
  <li>大陆：<span lang="zh-CN">育</span></li>
  <li>台湾：<span lang="zh-TW">育</span></li>
</ul>
```
:::

效果如下：

* 大陆：<span lang="zh-CN">育</span>
* 台湾：<span lang="zh-TW">育</span>

# 编码差异

对于「说明」一词中的「说」字，香港用「<span lang="zh-HK">説</span>」(U+8AAC)，台湾用「<span lang="zh-TW">說</span>」(U+8AAA)，码位本身就不相同。但是，尽管「説」、「說」码位不同，却明显是同一汉字，这样的关系称为 **微差异码**。

* 香港：<span lang="zh-HK"><img src="syut-HK.png" alt="説"/><img src="ming.png" alt="明"/></span>
* 台湾：<span lang="zh-TW"><img src="syut-TW.png" alt="說"/><img src="ming.png" alt="明"/></span>

# 用字差异

对于「化妝」一词中的「妝」字，香港用「<span lang="zh-HK">粧</span>」(U+7CA7)，台湾用「<span lang="zh-TW">妝</span>」(U+599D)，码位不相同，且为不同汉字。

* 香港：<span lang="zh-HK"><img src="faa-HK.png" alt="化"/><img src="zong-HK.png" alt="粧"/></span>
* 台湾：<span lang="zh-TW"><img src="faa-TW.png" alt="化"/><img src="zong-TW.png" alt="妝"/></span>

## OpenCC 标准

[OpenCC 项目](https://github.com/BYVoid/OpenCC) 总结了古籍、书法、文字学等领域传统的繁体汉字用字，据此确定的用字标准称为 OpenCC 标准。

<img src="opencc2all.png" alt="图为 OpenCC 标准基本杜绝了下面提到的几类「一对多」问题的示意"/>

OpenCC 用字标准制定的原则之一是「能分则不合」，基本杜绝了下面提到的几类「一对多」问题。

## 一简对多繁

「一简对多繁」是指同一个简体汉字可能对应多个繁体汉字的现象。由于一简对多繁，在将简体汉字转换为繁体汉字时，不能简单地逐字转换，需要考虑上下文。

<table><thead>
<tr><th>简体</th><th>繁体</th></tr>
</thead><tbody>
<tr><td lang="zh-CN" rowspan="2">发</td><td lang="zh-HK">發（發行）</td></tr>
<tr><td lang="zh-HK">髮（頭髮）</td></tr>
</tbody></table>

<table><thead>
<tr><th>简体</th><th>繁体</th></tr>
</thead><tbody>
<tr><td lang="zh-CN" rowspan="2">面</td><td lang="zh-HK">面（下面）</td></tr>
<tr><td lang="zh-HK">麪（下麪條）</td></tr>
</tbody></table>

常有人将简体的「<span lang="zh-CN">头发</span>」转换为繁体的「<del lang="zh-HK">頭發</del>」，就是因为一简对多繁的影响。

（在台湾标准中「麪」写作「麵」）

## 一繁对多简

与「一简对多繁」类似，还有「一繁对多简」问题，但这种情况远少于「一简对多繁」。

<table><thead>
<tr><th>繁体</th><th>简体</th></tr>
</thead><tbody>
<tr><td lang="zh-HK" rowspan="2">乾</td><td lang="zh-CN">乾（乾坤）</td></tr>
<tr><td lang="zh-CN">干（干燥）</td></tr>
</tbody></table>

## 一繁对多繁

繁体汉字的不同标准之间也存在一对多的情况，称为「一繁对多繁」，但这种情况也远少于「一简对多繁」。

**一台湾繁对多香港繁**

<table><thead>
<tr><th>台湾繁体</th><th>香港繁体</th></tr>
</thead><tbody>
<tr><td lang="zh-TW" rowspan="2">著</td><td lang="zh-HK">著（著作）</td></tr>
<tr><td lang="zh-HK">着（拿着）</td></tr>
</tbody></table>

**一香港繁对多台湾繁**

<table><thead>
<tr><th>香港繁体</th><th>台湾繁体</th></tr>
</thead><tbody>
<tr><td lang="zh-HK" rowspan="2">台</td><td lang="zh-TW">臺（臺灣）</td></tr>
<tr><td lang="zh-TW">台（普通话 <span lang="cmn-Latn">tāi</span>，天台山）</td></tr>
</tbody></table>

（现在台湾民间也常用「<span lang="zh-TW">台灣</span>」，与香港同）

**一香港、台湾繁对多 OpenCC 繁**

<table><thead>
<tr><th>香港、台湾繁体</th><th>OpenCC 繁体</th></tr>
</thead><tbody>
<tr><td lang="zh-HK" rowspan="2">才</td><td lang="zh-Hant-x-opencc">才（組織才能十分出衆）</td></tr>
<tr><td lang="zh-Hant-x-opencc">纔（實幹纔能夢想成真）</td></tr>
</tbody></table>

## 例外

OpenCC 基本杜绝了「一对多问题」，换言之，如果文本使用 OpenCC 用字标准，转换为其他用字标准时通常不会出现「一对多」，故不需要根据上下文进行额外的判断。

但是，OpenCC 标准也有少数例外：

**一、「湧」、「涌」问题**

<table><thead>
<tr><th>OpenCC 繁体</th><th>香港繁体</th></tr>
</thead><tbody>
<tr><td lang="zh-Hant-x-opencc" rowspan="2">涌</td><td lang="zh-HK">湧（湧起）</td></tr>
<tr><td lang="zh-HK">涌（普通话 <span lang="cmn-Latn">chōng</span>，粤语 <span lang="yue-Latn">cung¹</span>，東涌）</td></tr>
</tbody></table>

「涌」是粤语区常见的地名用字。OpenCC 标准依据的是古代传统典籍，根据《说文解字》确定「涌」为「湧」的本字，因此将「湧」写为「涌」，并不考虑粤语区的情况。

**二、「无线」、「有线」问题**

实际使用时，由于历史原因，并不一定遵守标准。如香港公司名：「<span lang="zh-HK">無綫</span>」、「<span lang="zh-HK">有線</span>」。

# 用词差异

不同地区的用词存在差异。例如：

* 中国大陆和台湾称草莓，香港称为士多啤梨
* 变压器在香港及中国大陆部分使用粤语的地方称为「火牛」
* 在中国大陆，通常称为「自行车」；台湾称「脚踏车」，江浙、上海、福建等地亦有「脚踏车」一称；在香港、澳门、广东、广西、湖南等中国南方地区则更常称其为「单车」；在新加坡、马来西亚、广东潮汕地区则称之「脚车」，江西赣语更称之为「线车（嘚）」、「钢丝车」、「脚踏车嘚」

出于本地化的需要，繁简转换通常至少需要考虑中国大陆、台湾、香港的用词差异。

用字差异、编码差异与用词差异无关，是相互独立的：

* 不是用字差异或编码差异、是用词差异：大陆繁、台湾用「<span lang="zh-TW">隱私</span>」，香港用「<span lang="zh-HK">私隱</span>」
* 是用字差异、不是用词差异：大陆繁、台湾用「<span lang="zh-TW">床前</span>」，香港用「<span lang="zh-HK">牀前</span>」
* 是编码差异、不是用词差异：大陆繁、香港用「<span lang="zh-HK">用户</span>」，台湾用「<span lang="zh-TW">用戶</span>」

# OpenCC 程序

## OpenCC 原理

OpenCC 是用于繁简转换的程序，许多 Linux 发行版已内置该程序，Windows 系统亦可自行编译安装。

由于「一对多」现象的存在，繁简转换不是简单地逐字对应，而是需要以词为单位考虑。因此，繁简转换分为两步：分词与词汇转换。

OpenCC 默认采用「正向最长匹配」的分词算法。例如，若词库中同时存在「<span lang="zh-Hant">快取</span>」、「<span lang="zh-Hant">記憶體</span>」和「<span lang="zh-Hant">快取記憶體</span>」，则「<span lang="zh-Hant">快取記憶體</span>」的切分结果为「<span lang="zh-Hant">快取記憶體</span>」而不是「<span lang="zh-Hant">快取/記憶體</span>」。

词汇转换就是将分词后的词汇逐一查找转换表。若有多个结果，就返回第一个。

若用户编译 OpenCC 不成功，也可以自行实现。下面是使用 57 行 Python 代码的一种简易实现（仅为示例，不做性能优化，且命令行参数有一定差异）。

首先将 [词典数据](https://github.com/sgalal/opencc-data/tree/master/data) 置于 `data`{lang="en-x-code"} 文件夹中，然后编写代码如下：

::: {lang="en-x-code"}
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pygtrie
import sys

def build_trie(ds):  # build a trie
    t = pygtrie.CharTrie()
    for d in ds:  # read a list of dictionaries
        with open(os.path.join('dict', d + '.txt')) as f:
            for line in f:
                if line != '\n' and line[0] != '#':  # ignore empty and commented lines
                    l, r = line.rstrip().split('\t')  # split the line by TAB
                    t[l] = r  # put words into the trie
    return t

def replace_words(s, t):
    l = []  # list of coverted words
    while s:
        longest_prefix = t.longest_prefix(s)  # match the longest prefix
        if not longest_prefix:  # if the prefix does not exist
            l.append(s[0])  # append the first character
            s = s[1:]  # remove the first character from the string
        else:  # if exists
            l.append(longest_prefix.value.split(' ')[0])  # append the first converted word
            s = s[len(longest_prefix.key):]  # remove the word from the string
    return ''.join(l)

DICT_FROM = \
    { 'cn': ('STCharacters', 'STPhrases')
    , 'hk': ('HKVariantsRev', 'HKVariantsRevPhrases')
    , 'tw': ('TWVariantsRev', 'TWVariantsRevPhrases')
    , 'twp': ('TWVariantsRev', 'TWVariantsRevPhrases', 'TWPhrasesRev')
    , 'jp': ('JPVariantsRev',)
    }

DICT_TO = \
    { 'cn': ('TSCharacters', 'TSPhrases')
    , 'hk': ('HKVariants', 'HKVariantsPhrases')
    , 'tw': ('TWVariants',)
    , 'twp': ('TWVariants', 'TWPhrasesIT', 'TWPhrasesName', 'TWPhrasesOther')
    , 'jp': ('JPVariants',)
    }

parser = argparse.ArgumentParser(description='Open Chinese Convert (OpenCC) Command Line Tool')
parser.add_argument('-f', '--from', default='cn', dest='from_region', help='from region')
parser.add_argument('-t', '--to', default='twp', dest='to_region', help='to region')
args = parser.parse_args()

s = sys.stdin.read()
if args.from_region != 't':
    s = replace_words(s, build_trie(DICT_FROM[args.from_region]))
if args.to_region != 't':
    s = replace_words(s, build_trie(DICT_TO[args.to_region]))
sys.stdout.write(s)
```
:::

## OpenCC 应用

OpenCC 可在命令行中使用，亦可通过 API 在程序中调用。

**命令行用法**

将繁体中文（OpenCC 标准）转换为简体中文（中国大陆），不转换用词：

::: {lang="en-x-code"}
```sh
$ echo '嘗試' | opencc -c t2s
尝试
```
:::

将简体中文（中国大陆）转换为繁体中文（香港），不转换用词：

::: {lang="en-x-code"}
```sh
$ echo '地球仪' | opencc -c s2hk
地球儀
```
:::

将简体中文（中国大陆）转换为繁体中文（台湾），转换用词：

::: {lang="en-x-code"}
```sh
$ echo '内存' | opencc -c s2twp
記憶體
```
:::

**应用举例：输入法**

* 字形上，应能根据用户语言自动调整字体。作为开发者，通常是通过在代码中指定语言
* 用字上，
    * 词条应只收繁体字（OpenCC 标准），然后通过转换模块完成繁简转换
    * 单字应酌情收录异体字，以确保在某个标准下可以打出非标准用字，如「<span lang="zh-Hant-CN">綫</span>」，但应把字频调低
* 用词上，应多地兼收，如「<span lang="zh-HK">網絡</span>」、「<span lang="zh-TW">網路</span>」

# 问与答

## 如何查看汉字的 Unicode 码位？

使用 Python 查看汉字的 Unicode 码位：

::: {lang="en-x-code"}
```python
>>> print(hex(ord('一')))
0x4e00
```
:::

将 `0x`{lang="en-x-code"} 换为 `U+`{lang="en-x-code"}，即可得到汉字「一」的 Unicode 码位为 U+4E00。

## 如何查看 Unicode 码位对应的字符？

使用 Python 查看 Unicode 码位对应的字符：

::: {lang="en-x-code"}
```python
>>> print(chr(0x9fa5))
龥
```
:::

这说明 Unicode 码位 U+9FA5 对应汉字「龥」。

## Unicode 和 UTF-8 有什么区别？

Unicode 是 **字符集**，UTF-8 是对 Unicode 字符集的一种 **编码方式**。

::: {lang="en-x-code"}
```sh
$ python -c 'print(hex(ord("湫")))'
0x6e6b
$ echo -n 湫 | xxd -ps
e6b9ab
```
:::

前两行表明「湫」字的 Unicode 码位是一个抽象的十六进制数字 U+6E6B，不涉及具体编码；后两行表明「湫」字在计算机中实际上是以十六进制的 UTF-8 编码 `e6b9ab`{lang="en-x-code"} 存储的。
