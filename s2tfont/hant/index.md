---
title: 正確實現簡轉繁字型
lang: zh-HK
keywords:
- 簡繁轉換
- 字型
- 簡轉繁字型
- OpenType 特性
- GSUB 規則
math: |-
  <link rel="alternate" hreflang="zh-CN" href="../"/>
---

> 簡轉繁字型下載頁：[繁媛明朝](https://github.com/ayaka14732/FanWunMing/releases)、[繁媛黑體](https://github.com/ayaka14732/FanWunHak/releases)
>
> [简体中文版](../){lang=zh-CN hreflang=zh-CN}

# 簡介

簡轉繁字型是將簡體文字以繁體字顯示的字型。對於習慣閲讀繁體字的用户，在某些不提供簡繁轉換的平台（如電子書閲讀裝置），可以利用簡轉繁字型令簡體文字以繁體字顯示，從而減少閲讀時的負擔。

![](https://raw.githubusercontent.com/ayaka14732/FanWunMing/f0f2f9bd0d8f76a853248ec03e3f54296dc85b80/demo.png)

然而，現時的簡轉繁字型是簡單地將簡體字的碼位對應繁體字形，不能處理簡體字與繁體字之間存在的「一簡對多繁」現象，因此使用效果並不理想，也使許多人產生了「字型不能正確處理簡轉繁」的錯誤印象。

事實上，目前通行的 OpenType 字型格式提供了大量字型特性。使用其中的 GSUB（glyph substitution，字圖替換）規則，可以實現能處理「一簡對多繁」的簡轉繁字型。

本文介紹了簡繁轉換工具 [OpenCC](https://github.com/BYVoid/OpenCC) 與 OpenType 字型的 GSUB 規則，並介紹了將二者結合，在字型層面實現簡繁轉換的方法。利用同一方法，也可以實現能處理破音字的拼音字型及注音字型。

# 「一簡對多繁」現象

「一簡對多繁」是指一個簡體字對應多個繁體字的現象。例如，簡體字「[后]{lang=zh-CN}」同時對應「後」（後面）與「后」（太后）兩個繁體字。

由於「一簡對多繁」現象，在簡轉繁時需要考慮上下文。

# OpenCC 的簡繁轉換演算法

OpenCC 是開放原始碼的中文簡繁轉換工具，可以利用詞語處理「一簡對多繁」現象。

OpenCC 詞庫不僅定義了單字的簡繁對應關係，還定義了數萬個詞語的簡繁對應關係。OpenCC 程式執行時根據「正向最大匹配」演算法匹配文章中的簡體詞語，然後轉換為繁體。

例如，在轉換句子「[太后的头发很干燥]{lang=zh-CN}」時，程式根據詞庫中存在的詞語，從左至右依次匹配「[太后]{lang=zh-CN}」、「[的]{lang=zh-CN}」、「[头发]{lang=zh-CN}」、「[很]{lang=zh-CN}」、「[干燥]{lang=zh-CN}」，然後分別轉換為「太后」、「的」、「頭髮」、「很」、「乾燥」。

# OpenType 字型的 GSUB 規則

OpenType 字型提供了 GSUB 規則，用於替換字型中的字圖。

GSUB 規則可以分為以下六類：

1. single substitution（一對一替換）
1. ligature substitution（多對一替換）
1. multiple substitution（一對多替換）
1. alternate substitution（一對一替換，但提供多種選擇）
1. chaining contextual substitution（帶上下文的一對一/多對一/一對多替換）
1. reverse chaining contextual substitution（逆向、帶上下文的一對一替換）

# 使用 GSUB 規則實現簡繁轉換演算法

簡繁轉換需要實現多對多替換，例如將「[错综复杂]{lang=zh-CN}」轉換為「錯綜複雜」，四個字均發生了變化。然而，以上六種 GSUB 規則僅包括一對一、多對一、一對多替換，卻不包括多對多替換。

為了實現多對多替換，可以採用「偽字圖」的方法。首先使用多對一替換，將簡體詞語替換為一個偽字圖；然後使用一對多替換，將這個偽字圖替換為繁體詞語。

示例如下：

<table>
<tr><th>輸入字串</th><td>U+97E9</td><td>U+5267</td><td>U+5947</td><td>U+7687</td><td>U+540E</td><td>U+64AD</td><td>U+51FA</td><td>U+540E</td></tr>
<tr><th>對應簡體字圖</th><td lang="zh-CN">韩</td><td lang="zh-CN">剧</td><td lang="zh-CN">奇</td><td lang="zh-CN">皇</td><td lang="zh-CN">后</td><td lang="zh-CN">播</td><td lang="zh-CN">出</td><td lang="zh-CN">后</td></tr>
<tr><th>第一步替換</th><td colspan="2">偽字圖 0</td><td lang="zh-CN">奇</td><td colspan="2">偽字圖 1</td><td colspan="2">偽字圖 2</td><td lang="zh-CN">后</td></tr>
<tr><th>第二步替換</th><td colspan="2">偽字圖 0</td><td>奇</td><td colspan="2">偽字圖 1</td><td colspan="2">偽字圖 2</td><td>後</td></tr>
<tr><th>第三步替換</th><td>韓</td><td>劇</td><td>奇</td><td>皇</td><td>后</td><td>播</td><td>出</td><td>後</td></tr>
</table>

- 第一步替換（多對一替換）將簡體詞語「[韩剧]{lang=zh-CN}」、「[皇后]{lang=zh-CN}」、「[播出]{lang=zh-CN}」分別替換為三個偽字圖。
- 第二步替換（一對一替換）將不成詞的單字「[后]{lang=zh-CN}」替換為繁體字「後」，簡繁同形的「奇」保持不變。
- 第三步替換（一對多替換）將三個偽字圖替換為對應的繁體詞語「韓劇」、「皇后」、「播出」。

注意「皇后」、「播出」是繁簡同形的詞語，但仍要在第一步替換為偽字圖，否則會影響第二步替換。

經過簡單的試驗可以得出，OpenType 字型在執行第一步替換時遵循「正向最大匹配」原則，與 OpenCC 的簡繁轉換演算法相同，因此二者是等效的。

# 侷限性

## 句子本身存在歧義

簡轉繁不可能做到完全準確。例如「[时间不准差一分钟]{lang=zh-CN}」這個句子本身存在歧義，對應的繁體既可能是「時間不準差一分鐘」（不準確），也可能是「時間不准差一分鐘」（不允許）。

## OpenCC 簡繁轉換演算法的侷限性

「正向最大匹配」演算法存在一定的侷限性。例如，對句子「[拥有 116 年历史]{lang=zh-CN}」執行正向最大匹配，當匹配到「[年]{lang=zh-CN}」時，由於「[年历]{lang=zh-CN}」是一個詞，會被匹配，從而錯誤地轉換為「<del style="text-decoration: none;">擁有 116 年曆史</del>」；正確的寫法應為「擁有 116 年歷史」。

## OpenCC 詞庫的侷限性

OpenCC 詞庫不可能囊括所有詞語，例如新詞未必能及時收錄於 OpenCC 詞庫中。

## OpenType 字型的侷限性

在上述方法中，一條簡繁詞語的對應關係需要使用一個偽字圖，而偽字圖也會佔據字圖數量。OpenType 字型的字圖數量上限為 65535 個。

為了防止字圖數量超過上限，可以採取兩種方法：

第一：可以從字型中刪除一些生僻字的字圖。

第二：可以適當刪除一些簡繁詞語的對應關係。應優先刪除繁簡同形、且在逐字轉換時不會產生錯誤的詞語，例如上面例子中的「播出」。

# 功能擴展

## OpenType 字型的 trad 特性

OpenType 字型提供了 trad 特性。在排版引擎中開啟 trad 特性時，字型可以將簡體字顯示為對應的繁體字。但是，目前支援 trad 特性的字型並不多。

只要將上述替換表放入字型的 trad 表中，就可以製作支援 trad 特性、且能處理「一簡對多繁」的字型。

## 地區字詞轉換

OpenCC 支援中國大陸、台灣、香港的習慣用字和用詞轉換。如果在上述簡繁詞語的對應關係中加入地區字詞的對應關係，就可以處理不同地區的字詞轉換，例如將「[内存]{lang=zh-CN}」轉換為「[記憶體]{lang=zh-TW}」。

另外，OpenType 字型可以根據語言指定不同的替換表。只要根據 OpenCC 詞庫，在 `ZHS`, `ZHT`, `ZHH` 表中指定不同的替換表，就可以令同一字型在不同語言環境下使用不同的字詞轉換。

# 字型下載

根據上述方法制作了簡轉繁字型[繁媛明朝](https://github.com/ayaka14732/FanWunMing)。

（作於 2020&#8239;年&#8239;7&#8239;月&#8239;31&#8239;日，修改於 2020&#8239;年&#8239;10&#8239;月&#8239;1&#8239;日）
