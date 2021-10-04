---
title: 正确实现简转繁字体
lang: zh-CN
keywords:
- 简繁转换
- 字体
- 简转繁字体
- OpenType 特性
- GSUB 规则
math: |-
  <link rel="alternate" hreflang="zh-HK" href="hant/"/>
---

> [繁體中文版《[正確實現簡轉繁字型](hant/){hreflang=zh-HK}]{lang=zh-HK}》，同时发布于知乎：[正确实现简转繁字体 - 三日月 綾香的文章](https://zhuanlan.zhihu.com/p/166089642){hreflang=zh-CN}
>
> 简转繁字体下载页面：[繁媛明朝](https://github.com/ayaka14732/FanWunMing/releases)、[繁媛黑体](https://github.com/ayaka14732/FanWunHak/releases)

# 简介

简转繁字体是将简体文本以繁体字显示的字体。对于习惯阅读繁体字的用户，在某些没有简繁转换的平台（如电子书阅读设备），可以利用简转繁字体将简体文本以繁体字显示，从而减轻阅读时的负担。

![](https://raw.githubusercontent.com/ayaka14732/FanWunMing/f0f2f9bd0d8f76a853248ec03e3f54296dc85b80/demo.png)

然而，现有的简转繁字体是简单地将简体字的码位对应繁体字形，不能处理简体字与繁体字之间存在的「一简对多繁」现象，因此使用效果并不理想，也使许多人产生了「字体不能正确处理简转繁」的错误印象。

事实上，目前通行的 OpenType 字体格式提供了大量字体特性。使用其中的 GSUB（glyph substitution，字图替换）规则，可以实现能处理「一简对多繁」的简转繁字体。

本文介绍了简繁转换工具 [OpenCC](https://github.com/BYVoid/OpenCC) 与 OpenType 字体的 GSUB 规则，并介绍了将二者结合，在字体层面实现简繁转换的方法。利用同一方法，也可以实现能处理多音字的拼音字体及注音字体。

# 「一简对多繁」现象

「一简对多繁」是指一个简体字对应多个繁体字的现象。例如，简体字「后」同时对应「[後]{lang=zh-HK}」（[後面]{lang=zh-HK}）与「[后]{lang=zh-HK}」（[太后]{lang=zh-HK}）两个繁体字。

由于「一简对多繁」现象，在简转繁时需要考虑上下文。

# OpenCC 的简繁转换算法

OpenCC 是开源的中文简繁转换工具，可以利用词语处理「一简对多繁」现象。

OpenCC 词库除了定义单字的简繁对应关系外，还定义了数万个词语的简繁对应关系。OpenCC 程序运行时根据「正向最大匹配」算法匹配文章中的简体词语，然后转换为繁体。

例如，在转换句子「太后的头发很干燥」时，程序根据词库中存在的词语，从左至右依次匹配「太后」、「的」、「头发」、「很」、「干燥」，然后分别转换为「[太后]{lang=zh-HK}」、「[的]{lang=zh-HK}」、「[頭髮]{lang=zh-HK}」、「[很]{lang=zh-HK}」、「[乾燥]{lang=zh-HK}」。

# OpenType 字体的 GSUB 规则

OpenType 字体提供了 GSUB 规则，用于替换字体中的字图。

GSUB 规则可以分为以下六类：

1. single substitution（一对一替换）
1. ligature substitution（多对一替换）
1. multiple substitution（一对多替换）
1. alternate substitution（一对一替换，但提供多种选择）
1. chaining contextual substitution（带上下文的一对一/多对一/一对多替换）
1. reverse chaining contextual substitution（逆向、带上下文的一对一替换）

# 使用 GSUB 规则实现简繁转换算法

简繁转换需要实现多对多替换，例如将「错综复杂」转换为「[錯綜複雜]{lang=zh-HK}」，四个字均发生了变化。然而，以上六种 GSUB 规则仅包括一对一、多对一、一对多替换，却不包括多对多替换。

为了实现多对多替换，可以采用「伪字图」的方法。首先使用多对一替换，将简体词语替换为一个伪字图；然后使用一对多替换，将这个伪字图替换为繁体词语。

示例如下：

<table>
<tr><th>输入字符串</th><td>U+97E9</td><td>U+5267</td><td>U+5947</td><td>U+7687</td><td>U+540E</td><td>U+64AD</td><td>U+51FA</td><td>U+540E</td></tr>
<tr><th>对应简体字图</th><td>韩</td><td>剧</td><td>奇</td><td>皇</td><td>后</td><td>播</td><td>出</td><td>后</td></tr>
<tr><th>第一步替换</th><td colspan="2">伪字图 0</td><td>奇</td><td colspan="2">伪字图 1</td><td colspan="2">伪字图 2</td><td>后</td></tr>
<tr><th>第二步替换</th><td colspan="2">伪字图 0</td><td lang="zh-HK">奇</td><td colspan="2">伪字图 1</td><td colspan="2">伪字图 2</td><td lang="zh-HK">後</td></tr>
<tr><th>第三步替换</th><td lang="zh-HK">韓</td><td lang="zh-HK">劇</td><td lang="zh-HK">奇</td><td lang="zh-HK">皇</td><td lang="zh-HK">后</td><td lang="zh-HK">播</td><td lang="zh-HK">出</td><td lang="zh-HK">後</td></tr>
</table>

- 第一步替换（多对一替换）将简体词语「韩剧」、「皇后」、「播出」分别替换为三个伪字图。
- 第二步替换（一对一替换）将不成词的单字「后」替换为繁体字「[後]{lang=zh-HK}」，简繁同形的「奇」保持不变。
- 第三步替换（一对多替换）将三个伪字图替换为对应的繁体词语「[韓劇]{lang=zh-HK}」、「[皇后]{lang=zh-HK}」、「[播出]{lang=zh-HK}」。

注意「皇后」、「播出」是繁简同形的词语，但仍要在第一步替换为伪字图，否则会影响第二步替换。

经过简单的试验可以发现，OpenType 字体在执行第一步替换时遵循「正向最大匹配」原则，这与 OpenCC 的简繁转换算法相同，因此二者是等效的。

# 局限性

## 句子本身存在歧义

简转繁不可能做到完全准确。例如「时间不准差一分钟」这个句子本身存在歧义，对应的繁体既可能是「[時間不準差一分鐘]{lang=zh-HK}」（不准确），也可能是「[時間不准差一分鐘]{lang=zh-HK}」（不允许）。

## OpenCC 简繁转换算法的局限性

「正向最大匹配」算法存在一定的局限性。例如，对句子「拥有 116 年历史」执行正向最大匹配，当匹配到「年」时，由于「年历」是一个词，会被匹配，从而错误地转换为「<del lang="zh-HK" style="text-decoration: none;">擁有 116 年曆史</del>」；正确的写法应该为「[擁有 116 年歷史]{lang=zh-HK}」。

## OpenCC 词库的局限性

OpenCC 词库不可能囊括所有词语，例如新词不一定能及时收录于 OpenCC 词库中。

## OpenType 字体的局限性

在上述方法中，一条简繁词语的对应关系需要使用一个伪字图，而伪字图也会占据字图数量。OpenType 字体的字图数量上限为 65535 个。

为了防止字图数量超过上限，可以采取两种方法：

第一：可以从字体中删除一些生僻字的字图。

第二：可以适当删除一些简繁词语的对应关系。应优先删除繁简同形、且在逐字转换时不会产生错误的词语，例如上面例子中的「播出」。

# 功能扩展

## OpenType 字体的 trad 特性

OpenType 字体提供了 trad 特性。在排版引擎中打开 trad 特性时，字体可以将简体字显示为对应的繁体字。但是，目前支持 trad 特性的字体并不多。

只要将上述替换表放入字体的 trad 表中，就可以制作支持 trad 特性、且能处理「一简对多繁」的字体。

## 地区字词转换

OpenCC 支持中国大陆、台湾、香港的习惯用字和用词转换。如果在上述简繁词语的对应关系中加入地区字词的对应关系，就可以处理不同地区的字词转换，例如将「内存」轉換為「[記憶體]{lang=zh-TW}」。

另外，OpenType 字体可以根据语言指定不同的替换表。只要根据 OpenCC 词库，在 `ZHS`, `ZHT`, `ZHH` 表中指定不同的替换表，就可以使同一字体在不同语言环境下使用不同的字词转换。

# 字体下载

根据上述方法制作了简转繁字体[繁媛明朝](https://github.com/ayaka14732/FanWunMing)。

（作于 2020&#8239;年&#8239;7&#8239;月&#8239;31&#8239;日，修改于 2020&#8239;年&#8239;10&#8239;月&#8239;1&#8239;日）
