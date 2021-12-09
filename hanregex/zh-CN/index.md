---
title: 如何正确使用正则表达式匹配汉字
lang: zh-CN
keywords:
- 正则表达式
- 汉字
- 中日韩越统一表意文字
- Unicode
---

> [English Version: [_How to Properly Match Chinese Characters With Regular Expression_](../){hreflang=en}]{lang=en}

# tl;dr

```
[\u3006\u3007\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002ebef\U00030000-\U0003134f]
```

# 解释

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

# Python 示例

```python
>>> import re
>>> han_regex = re.compile(r'[\u3006\u3007\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002ebef\U00030000-\U0003134f]')
>>> is_han = lambda c: bool(han_regex.fullmatch(c))
>>> print([is_han(c) for c in 'm！文𦫖〇〆'])
[False, False, True, True, True, True]
```

# JavaScript 示例

如果可以用 ES6 的 [Unicode point escapes](https://caniuse.com/mdn-javascript_grammar_unicode_point_escapes) (`\u{...}`)：

```javascript
> const isHan = (c) => /^[\u3006\u3007\u4e00-\u9fff\u3400-\u4dbf\u{20000}-\u{2a6df}\u{2a700}-\u{2ebef}\u{30000}-\u{3134f}]$/u.test(c);
> console.log([...'m！文𦫖〇〆'].map(isHan));
[ false, false, true, true, true, true ]
```

如果不能用 ES6，就必须使用[代理对](http://russellcottrell.com/greek/utilities/SurrogatePairCalculator.htm)：

```javascript
> const isHan = (c) => /^[\u3006\u3007\u4e00-\u9fff\u3400-\u4dbf]|[\ud840-\ud868\ud86a-\ud879\ud880-\ud883][\udc00-\udfff]|\ud869[\udc00-\udedf\udf00-\udfff]|\ud87a[\udc00-\udfef]|\ud884[\udc00-\udf4f]$/.test(c);
> console.log([...'m！文𦫖〇〆'].map(isHan));
[ false, false, true, true, true, true ]
```

# 为什么不能用 `\p{sc=Han}`

`\p{...}` 这种语法称为 [Unicode property escapes](https://tc39.es/proposal-regexp-unicode-property-escapes/)。第一部分是 Unicode property name，`sc` 表示 script；第二部分 `Han` 是 Unicode property value。

要查看哪些字符属于 Han script，可以查看 UCD 中的 [`Scripts.txt`](https://www.unicode.org/Public/UCD/latest/ucd/Scripts.txt)。

- `U+2E80-U+2E99`: CJK Radicals
- `U+2E9B-U+2EF3`: CJK Radicals
- `U+2F00-U+2FD5`: Kangxi Radicals
- `U+3005`: Ideographic Iteration Mark
- `U+3007`: Ideographic Number Zero
- `U+3021-U+3029`: Suzhou Numerals
- `U+3038-U+303A`: Suzhou Numerals
- `U+303B`: Vertical Ideographic Iteration Mark
- `U+3400-U+4DBF`: **CJK Unified Ideographs Extension A**
- `U+4E00-U+9FFF`: **CJK Unified Ideographs**
- `U+F900-U+FA6D`: CJK Compatibility Ideographs
- `U+FA70-U+FAD9`: CJK Compatibility Ideographs
- `U+16FE2`: Old Chinese Hook Mark
- `U+16FE3`: Old Chinese Iteration Mark
- `U+16FF0-U+16FF1`: Vietnamese Alternate Reading Marks
- `U+20000-U+2A6DF`: **CJK Unified Ideographs Extension B**
- `U+2A700-U+2B738`: **CJK Unified Ideographs Extension C**
- `U+2B740-U+2B81D`: **CJK Unified Ideographs Extension D**
- `U+2B820-U+2CEA1`: **CJK Unified Ideographs Extension E**
- `U+2CEB0-U+2EBE0`: **CJK Unified Ideographs Extension F**
- `U+2F800-U+2FA1D`: CJK Compatibility Ideographs Supplements
- `U+30000-U+3134A`: **CJK Unified Ideographs Extension G**

可以看出它不只包括汉字。

# 为什么不能用 `\p{Ideo}`

类似地，这里的 `Ideo` 是 Ideograph 的缩写。要查看哪些字符属于 Ideograph，可以查看 UCD 中的 [`PropList.txt`](https://www.unicode.org/Public/UCD/latest/ucd/PropList.txt)。

- `U+3006`: Ideographic Closing Mark
- `U+3007`: Ideographic Number Zero
- `U+3021-U+3029`: Suzhou Numerals
- `U+3038-U+303A`: Suzhou Numerals
- `U+3400-U+4DBF`: **CJK Unified Ideographs Extension A**
- `U+4E00-U+9FFF`: **CJK Unified Ideographs**
- `U+F900-U+FA6D`: CJK Compatibility Ideographs
- `U+FA70-U+FAD9`: CJK Compatibility Ideographs
- `U+16FE4`: Khitan Small Script Filler
- `U+17000-U+187F7`: Tangut Ideographs
- `U+18800-U+18AFF`: Tangut Components
- `U+18B00-U+18CD5`: Khitan Small Script Characters
- `U+18D00-U+18D08`: Tangut Ideographs Supplement
- `U+1B170-U+1B2FB`: Nushu Characters
- `U+20000-U+2A6DF`: **CJK Unified Ideographs Extension B**
- `U+2A700-U+2B738`: **CJK Unified Ideographs Extension C**
- `U+2B740-U+2B81D`: **CJK Unified Ideographs Extension D**
- `U+2B820-U+2CEA1`: **CJK Unified Ideographs Extension E**
- `U+2CEB0-U+2EBE0`: **CJK Unified Ideographs Extension F**
- `U+2F800-U+2FA1D`: CJK Compatibility Ideographs Supplements
- `U+30000-U+3134A`: **CJK Unified Ideographs Extension G**

可以看出它不只包括汉字。
