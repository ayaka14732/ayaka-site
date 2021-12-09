---
title: 結巴分詞處理粵語
lang: zh-HK
keywords:
- 計算語言學
- 粵語
- 分詞
- 結巴分詞
- 詞性標註
math: |-
  <link rel="alternate" hreflang="zh-CN" href="../"/>
---

> [简体中文版《[结巴分词处理粤语](../){hreflang=zh-CN}》]{lang=zh-CN}

# 方法

分詞使用的詞庫是結合結巴分詞大詞庫與 PyCantonese 詞庫而成，不可直接使用結巴分詞預設詞庫。

## 為何不可直接使用結巴分詞預設詞庫？

結巴分詞詞庫中的詞頻與詞性是基於官話文體的，不可直接用於粵文。

```python
>>> import jieba.posseg as pseg
>>> text = '但是，到了吃晚饭的时候，奶牛跑回农场了，福思也迷路了。他很晚才到姐姐的婚礼上，客人们已经在吃饭了。'
>>> print(' '.join('%s/%s' % (word, flag) for word, flag in pseg.cut(text)))
但是/c ，/x 到/v 了/ul 吃晚饭/l 的/uj 时候/n ，/x 奶牛/n 跑/v 回/v 农场/n 了/ul ，/x 福思/nr 也/d 迷路/n 了/ul 。/x 他/r 很/d 晚/tg 才/d 到/v 姐姐/n 的/uj 婚礼/n 上/f ，/x 客人/n 们/k 已经/d 在/p 吃饭/v 了/ul 。/x
>>> text = '但系，到咗食晚饭嘅时候，只乳牛走返咗农场，夫思亦都系途中荡失路。佢好夜先至到家姐嘅婚礼度，𠮶阵时啲客都已经喺度食紧饭啰。'
>>> print(' '.join('%s/%s' % (word, flag) for word, flag in pseg.cut(text)))
但/c 系/n ，/x 到/v 咗/zg 食/v 晚饭/n 嘅/x 时候/n ，/x 只/d 乳牛/n 走/v 返/v 咗/zg 农场/n ，/x 夫思/n 亦/d 都/d 系/v 途中/s 荡失路/i 。/x 佢/yg 好/a 夜/tg 先/d 至/p 到/v 家/q 姐/n 嘅/zg 婚礼/n 度/zg ，/x 𠮶/x 阵/ng 时/ng 啲/zg 客/ng 都/d 已经/d 喺/yg 度/q 食/v 紧/d 饭/n 啰/zg 。/x
```

（注：詞性標記可以參考 [ICTCLAS 漢語詞性標註集](https://gist.github.com/luw2007/6016931#ictclas-%E6%B1%89%E8%AF%AD%E8%AF%8D%E6%80%A7%E6%A0%87%E6%B3%A8%E9%9B%86)）

可以看出，結巴分詞對粵語詞語的識別效果並不理想。

## 為何不可直接用 PyCantonese 詞庫？

PyCantonese 專案的語料庫以對話語料為主，體裁較不正式，而粵文維基的條目有許多正式的書面詞語。

為此，可以將結巴分詞預設詞庫與從 PyCantonese 專案匯出的詞庫結合起來，將二者合併為新的詞庫。

在合併兩個詞庫前，需要為二者的詞頻賦予一定的權重，再將結果相加。採用的方法是調整 PyCantonese 的詞頻，使 PyCantonese 中「嘅」字的詞頻與結巴分詞詞庫中「的」字的詞頻一致。這僅是一種近似的方法，並非是將二者等比例合併。

## 為何需要使用結巴分詞的大詞庫？

雖然[目前結巴分詞的首頁](https://github.com/fxsjy/jieba/blob/67fa2e36e72f69d9134b8a1037b83fbb070b9775/README.md)稱「支援繁體分詞」，但使用預設詞庫時，分詞效果並不理想：

```python
>>> import jieba.posseg as pseg
>>> text = '英超榜首的利物浦主場將會迎戰般尼茅夫，領隊高普賽前表示，門將艾利臣在足總盃對車路士的比賽中臀部肌肉受傷，今場以及下周中歐聯16強次回合對馬德里體育會的賽事，都肯定無法上陣。'
>>> print(' '.join('%s/%s' % (word, flag) for word, flag in pseg.cut(text)))
英超/nr 榜首/n 的/uj 利物浦/ns 主場/n 將/d 會/v 迎戰/v 般/u 尼茅夫/nrt ，/x 領隊/n 高普賽/nr 前/f 表示/v ，/x 門/n 將/d 艾利臣在足總/nr 盃/yg 對/p 車/n 路/n 士/ng 的/uj 比/p 賽/vn 中/f 臀部/n 肌肉/n 受傷/v ，/x 今/tg 場/q 以及/c 下周/t 中歐/ns 聯/v 16/m 強/a 次/q 回合/v 對/p 馬/nr 德里/ns 體育會/n 的/uj 賽事/n ，/x 都/d 肯定/v 無法上/l 陣/n 。/x
>>> text = '英超榜首的利物浦主场将会迎战般尼茅夫，领队高普赛前表示，门将艾利臣在足总杯对车路士的比赛中臀部肌肉受伤，今场以及下周中欧联16强次回合对马德里体育会的赛事，都肯定无法上阵。'
>>> print(' '.join('%s/%s' % (word, flag) for word, flag in pseg.cut(text)))
英超/nr 榜首/n 的/uj 利物浦/ns 主场/n 将/d 会/v 迎战/v 般尼茅夫/nz ，/x 领队/n 高普/nr 赛前/t 表示/v ，/x 门将/n 艾利臣/nr 在/p 足总杯/n 对/p 车路/n 士/ng 的/uj 比赛/vn 中/f 臀部/n 肌肉/n 受伤/v ，/x 今/tg 场/q 以及/c 下周/t 中欧/ns 联/v 16/m 强/a 次/q 回合/v 对/p 马德里/nr 体育/vn 会/v 的/uj 赛事/n ，/x 都/d 肯定/v 无法/n 上阵/v 。/x
```

可以看出，結巴分詞預設詞庫對繁體文字的分詞能力明顯弱於簡體分詞，而結巴分詞大詞庫則同時支援繁體與簡體。

# 實現

> 程式碼見 [ayaka14732/cantoseg](https://github.com/ayaka14732/cantoseg/tree/main/build)。

1. 首先安裝 PyCantonese，然後編寫程式匯出 PyCantonese 詞庫。執行後生成 `dict_cantonese.txt`。
1. 下載最新的[結巴分詞大詞庫](https://raw.githubusercontent.com/fxsjy/jieba/master/extra_dict/dict.txt.big)，存儲為 `dict.big.txt`。
1. 編寫程式合併詞庫。執行後生成 `merged_dict.txt`。

使用：

```python
import jieba
jieba.set_dictionary('merged_dict.txt')
jieba.initialize()
import jieba.posseg as pseg
...
```

# 附錄：詞庫對結巴分詞的影響

原文：

> 考古證明，香港喺舊石器時代就有人住。喺西貢半島北岸黃地峒，有石器同整石器嘅架生。喺新界同大嶼山好多地方，都搵到新石器時代嘅遺址。旺角（以前叫「芒角」）就曾經係東漢、隋朝同唐朝人做陶器嘅地方。李鄭屋古墓亦證明香港喺東漢嗰時，畀番禺管。喺唐朝嗰陣，香港由廣州寶安縣管，嗰陣軍隊就係駐喺而家嘅屯門（屯兵之門），而呢區個名就係咁嚟㗎喇。

## 結巴分詞預設詞庫

```
考古/vn 證明/n ，/x 香港/ns 喺/yg 舊/a 石器/n 時代/n 就/d 有人/r 住/v 。/x 喺/y 西/f 貢/vg 半/m 島/n 北岸/f 黃地峒/ns ，/x 有/v 石器/n 同/p 整/b 石器/n 嘅/yg 架/q 生/v 。/x 喺/x 新界/n 同/p 大嶼山/ns 好多/m 地方/n ，/x 都/d 搵/yg 到/v 新石器/n 時代/nr 嘅/zg 遺/vg 址/ng 。/x 旺角/n （/x 以前/f 叫/v 「/x 芒角/n 」/x ）/x 就/d 曾/d 經/p 係/zg 東/f 漢/j 、/x 隋朝/t 同/p 唐朝/t 人/n 做/v 陶器/n 嘅/x 地方/n 。/x 李鄭/nr 屋/s 古墓/n 亦/d 證明/v 香港/ns 喺東漢/nr 嗰/zg 時/ng ，/x 畀/g 番禺/ns 管/vn 。/x 喺/x 唐朝/t 嗰/yg 陣/ng ，/x 香港/ns 由/p 廣州/ns 寶安縣/ns 管/vn ，/x 嗰陣軍隊就/nr 係/zg 駐/v 喺而家/nr 嘅/zg 屯/v 門/n （/x 屯兵/n 之門/ns ）/x ，/x 而呢/y 區/n 個/q 名就/nr 係/zg 咁/x 嚟/zg 㗎/x 喇/n 。/x
```

可以看出分詞效果並不理想。

## 結巴分詞大詞庫

```
考古/vn 證明/n ，/x 香港/ns 喺/x 舊石器時代/t 就/d 有人/r 住/v 。/x 喺/x 西貢/ns 半島/n 北岸/f 黃地峒/ns ，/x 有/v 石器/n 同/p 整/b 石器/n 嘅/yg 架/q 生/v 。/x 喺/x 新界/n 同/p 大嶼山/ns 好多/m 地方/n ，/x 都/d 搵/yg 到/v 新石器/n 時代/n 嘅/x 遺址/n 。/x 旺角/n （/x 以前/f 叫/v 「/x 芒角/n 」/x ）/x 就/d 曾經/d 係/zg 東漢/t 、/x 隋朝/t 同/p 唐朝/t 人/n 做/v 陶器/n 嘅/x 地方/n 。/x 李鄭/nr 屋/s 古墓/n 亦/d 證明/n 香港/ns 喺/x 東漢/t 嗰/yg 時/ng ，/x 畀/g 番禺/ns 管/vn 。/x 喺/x 唐朝/t 嗰/yg 陣/ng ，/x 香港/ns 由/p 廣州/ns 寶安縣/ns 管/vn ，/x 嗰/yg 陣/ng 軍隊/n 就/d 係/yg 駐/v 喺而家/nr 嘅/zg 屯門/ns （/x 屯兵/n 之門/ns ）/x ，/x 而呢/y 區/n 個/q 名就/nr 係/zg 咁/x 嚟/zg 㗎/x 喇/n 。/x
```

與結巴分詞預設詞庫相比，結巴分詞大詞庫效果更佳，但仍不能正確處理粵語詞（如「喺」）。

## PyCantonese 詞庫

```
考古/vn 證明/v ，/x 香港/ns 喺/yg 舊/a 石/ng 器/n 時代/n 就/u 有人/r 住/v 。/x 喺/y 西/f 貢/vg 半/m 島/n 北/f 岸/n 黃/a 地/uv峒/g ，/x 有/v 石器/n 同/p 整石/n 器/n 嘅/zg 架/q 生/v 。/x 喺/r 新界/ns 同/p 大嶼山/ns 好多/m 地方/n ，/x 都/d 搵/yg 到/v 新/a 石/ng 器/n 時代/n 嘅/y 遺/vg 址/ng 。/x 旺角/ns （/x 以前/f 叫/v 「/x 芒角/n 」/x ）/x 就/u 曾經/d 係/y 東/f 漢/j 、/x 隋朝/t 同/p 唐朝/t 人/n 做/v 陶/nr 器/n 嘅/zg 地方/n 。/x 李/nr 鄭屋/ns 古墓/n 亦/d 證明/v 香港/ns 喺/y 東/f 漢/j 嗰時/t ，/x畀/x 番禺/ns 管/v 。/x 喺/yg 唐/tg 朝/p 嗰陣/r ，/x 香港/ns 由/v 廣州/ns 寶安縣/ns 管/vn ，/x 嗰陣/r 軍隊/n 就係/d 駐/v 喺/zg 而家/t 嘅/e 屯門/n （/x 屯兵/n 之/u 門/n ）/x ，/x 而呢/y 區/n 個/q 名/q 就係/d 咁/x 嚟/zg 㗎/x 喇/u 。/x
```

PyCantonese 詞庫以對話語料為主，不能識別專有名詞。

## 合併後的詞庫

```
考古/vn 證明/v ，/x 香港/ns 喺/r 舊石器時代/t 就/u 有人/r 住/v 。/x 喺/r 西貢/ns 半島/n 北岸/f 黃地峒/ns ，/x 有/v 石器/n 同/p 整/b 石器/n 嘅/yg 架/q 生/v 。/x 喺/r 新界/n 同/cg 大嶼山/ns 好多/m 地方/n ，/x 都/d 搵/yg 到/v 新石器/n 時代/n 嘅/e 遺址/n。/x 旺角/n （/x 以前/t 叫/v 「/x 芒角/n 」/x ）/x 就/u 曾經/d 係/zg 東漢/t 、/x 隋朝/t 同/cg 唐朝/t 人/n 做/v 陶器/n 嘅/e 地方/n 。/x 李鄭/nr 屋/s 古墓/n 亦/d 證明/v 香港/ns 喺/r 東漢/t 嗰時/t ，/x 畀/g 番禺/ns 管/v 。/x 喺/r 唐朝/t 嗰陣/r ，/x 香港/ns 由/v 廣州/ns 寶安縣/ns 管/v ，/x 嗰陣/r 軍隊/n 就係/d 駐/v 喺/zg 而家/t 嘅/e 屯門/n （/x 屯兵/n 之門/ns ）/x ，/x 而呢/y 區/n 個/q 名/q 就係/d 咁/x 嚟/zg 㗎/x 喇/u 。/x
```

合併後既能處理粵語詞，又能處理專有名詞，是幾種方法中最好的。

然而，這種方法的效果仍然不夠理想，未來需要建設規模更大的粵語語料庫。

（初稿作於 2020&#8239;年&#8239;3&#8239;月&#8239;7&#8239;日，發佈於 2020&#8239;年&#8239;7&#8239;月&#8239;17&#8239;日，修改於 2020&#8239;年&#8239;8&#8239;月&#8239;22&#8239;日，原為計算語言學課程實驗作）
