---
title: 正則表達式匹配漢字
lang: zh-HK
keywords:
- 正則表達式
- 漢字
- 中日韓越統一表意文字
- Unicode
---

**匹配漢字的正則表達式如下 / [The regex matching Chinese characters is as follows]{lang=en-HK}**

```
[\u3006\u3007\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002b73f\U0002b740-\U0002b81f\U0002b820-\U0002ceaf\U0002ceb0-\U0002ebef\U00030000-\U0003134f]
```

**解釋 / [Explanation]{lang=en-HK}**

:::{lang=en-HK}
- `U+3006`: Character 〆 (often regarded as a Chinese character)
- `U+3007`: Character 〇 (often regarded as a Chinese character)
- `U+4E00-U+9FFF`: CJK Unified Ideographs
- `U+3400-U+4DBF`: CJK Unified Ideographs Extension A
- `U+20000-U+2A6DF`: CJK Unified Ideographs Extension B
- `U+2A700-U+2B73F`: CJK Unified Ideographs Extension C
- `U+2B740-U+2B81F`: CJK Unified Ideographs Extension D
- `U+2B820-U+2CEAF`: CJK Unified Ideographs Extension E
- `U+2CEB0-U+2EBEF`: CJK Unified Ideographs Extension F
- `U+30000-U+3134F`: CJK Unified Ideographs Extension G
:::

**Python 程式碼示例 / [Python code example]{lang=en-HK}**

```python
>>> import re
>>> han_regex = re.compile(r'[\u3006\u3007\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002b73f\U0002b740-\U0002b81f\U0002b820-\U0002ceaf\U0002ceb0-\U0002ebef\U00030000-\U0003134f]')
>>> bool(han_regex.match('m'))
False
>>> bool(han_regex.match('！'))
False
>>> bool(han_regex.match('文'))
True
>>> bool(han_regex.match('𦫖'))
True
```
