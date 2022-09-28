---
title: How to Properly Match Chinese Characters With Regular Expression
lang: en
keywords:
- regular expression
- Chinese character
- regex
- CJK ideograph
- Unicode
---

> [简体中文版](zh-CN/){lang=zh-CN hreflang=zh-CN}

# tl;dr

## Python

```
[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002ebef\U00030000-\U000323af\ufa0e\ufa0f\ufa11\ufa13\ufa14\ufa1f\ufa21\ufa23\ufa24\ufa27\ufa28\ufa29\u3006\u3007][\ufe00-\ufe0f\U000e0100-\U000e01ef]?
```

## Python (need `pip install regex`)

```
[\p{Unified_Ideograph}\u3006\u3007][\ufe00-\ufe0f\U000e0100-\U000e01ef]?
```

## JavaScript (RegExp: Unicode property escapes)

[Can I use RegExp: Unicode property escapes](https://caniuse.com/mdn-javascript_builtins_regexp_property_escapes)

```
[\p{Unified_Ideograph}\u3006\u3007][\ufe00-\ufe0f\u{e0100}-\u{e01ef}]?
```

## JavaScript (RegExp: Unicode)

[Can I use RegExp: Unicode](https://caniuse.com/mdn-javascript_builtins_regexp_unicode)

```
[\u4e00-\u9fff\u3400-\u4dbf\u{20000}-\u{2a6df}\u{2a700}-\u{2ebef}\u{30000}-\u{323af}\ufa0e\ufa0f\ufa11\ufa13\ufa14\ufa1f\ufa21\ufa23\ufa24\ufa27\ufa28\ufa29\u3006\u3007][\ufe00-\ufe0f\u{e0100}-\u{e01ef}]?
```

## JavaScript

```
([\u4e00-\u9fff\u3400-\u4dbf\ufa0e\ufa0f\ufa11\ufa13\ufa14\ufa1f\ufa21\ufa23\ufa24\ufa27\ufa28\ufa29\u3006\u3007]|[\ud840-\ud868\ud86a-\ud879\ud880-\ud887][\udc00-\udfff]|\ud869[\udc00-\udedf\udf00-\udfff]|\ud87a[\udc00-\udfef]|\ud888[\udc00-\udfaf])([\ufe00-\ufe0f]|\udb40[\udd00-\uddef])?
```

# Examples

## Python


```python
import json
import re

pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002ebef\U00030000-\U000323af\ufa0e\ufa0f\ufa11\ufa13\ufa14\ufa1f\ufa21\ufa23\ufa24\ufa27\ufa28\ufa29\u3006\u3007][\ufe00-\ufe0f\U000e0100-\U000e01ef]?')

for i, match in enumerate(pattern.finditer('a〆文𦫖﨑禰󠄀')):
    print(f'Match {i}:', match[0], json.dumps(match[0]))

# Match 0: 〆 "\u3006"
# Match 1: 文 "\u6587"
# Match 2: 𦫖 "\ud85a\uded6"
# Match 3: 﨑 "\ufa11"
# Match 4: 禰󠄀 "\u79b0\udb40\udd00"
```

## Python (need `pip install regex`)

```python
import json
import regex as re

pattern = re.compile(r'[\p{Unified_Ideograph}\u3006\u3007][\ufe00-\ufe0f\U000e0100-\U000e01ef]?')

for i, match in enumerate(pattern.finditer('a〆文𦫖﨑禰󠄀')):
    print(f'Match {i}:', match[0], json.dumps(match[0]))

# Match 0: 〆 "\u3006"
# Match 1: 文 "\u6587"
# Match 2: 𦫖 "\ud85a\uded6"
# Match 3: 﨑 "\ufa11"
# Match 4: 禰󠄀 "\u79b0\udb40\udd00"
```

## JavaScript (RegExp: Unicode property escapes)

[Can I use RegExp: Unicode property escapes](https://caniuse.com/mdn-javascript_builtins_regexp_property_escapes)

```javascript
const pattern = /[\p{Unified_Ideograph}\u3006\u3007][\ufe00-\ufe0f\u{e0100}-\u{e01ef}]?/gmu;

'a〆文𦫖﨑禰󠄀'.match(pattern).forEach((match, i) => {
   console.log(`Match ${i}: ${match}, length: ${match.length}`);
});
// Match 0: 〆, length: 1
// Match 1: 文, length: 1
// Match 2: 𦫖, length: 2
// Match 3: 﨑, length: 1
// Match 4: 禰󠄀, length: 3
```

## JavaScript (RegExp: Unicode)

[Can I use RegExp: Unicode](https://caniuse.com/mdn-javascript_builtins_regexp_unicode)

```javascript
const pattern = /[\u4e00-\u9fff\u3400-\u4dbf\u{20000}-\u{2a6df}\u{2a700}-\u{2ebef}\u{30000}-\u{323af}\ufa0e\ufa0f\ufa11\ufa13\ufa14\ufa1f\ufa21\ufa23\ufa24\ufa27\ufa28\ufa29\u3006\u3007][\ufe00-\ufe0f\u{e0100}-\u{e01ef}]?/gmu;

'a〆文𦫖﨑禰󠄀'.match(pattern).forEach((match, i) => {
   console.log(`Match ${i}: ${match}, length: ${match.length}`);
});
// Match 0: 〆, length: 1
// Match 1: 文, length: 1
// Match 2: 𦫖, length: 2
// Match 3: 﨑, length: 1
// Match 4: 禰󠄀, length: 3
```

## JavaScript

```javascript
const pattern = /([\u4e00-\u9fff\u3400-\u4dbf\ufa0e\ufa0f\ufa11\ufa13\ufa14\ufa1f\ufa21\ufa23\ufa24\ufa27\ufa28\ufa29\u3006\u3007]|[\ud840-\ud868\ud86a-\ud879\ud880-\ud887][\udc00-\udfff]|\ud869[\udc00-\udedf\udf00-\udfff]|\ud87a[\udc00-\udfef]|\ud888[\udc00-\udfaf])([\ufe00-\ufe0f]|\udb40[\udd00-\uddef])?/gm;

'a〆文𦫖﨑禰󠄀'.match(pattern).forEach((match, i) => {
   console.log(`Match ${i}: ${match}, length: ${match.length}`);
});
// Match 0: 〆, length: 1
// Match 1: 文, length: 1
// Match 2: 𦫖, length: 2
// Match 3: 﨑, length: 1
// Match 4: 禰󠄀, length: 3
```

# Explanation

**CJK Unified Ideographs:**

- `U+4E00-U+9FFF`: CJK Unified Ideographs
- `U+3400-U+4DBF`: CJK Unified Ideographs Extension A
- `U+20000-U+2A6DF`: CJK Unified Ideographs Extension B
- `U+2A700-U+2B73F`: CJK Unified Ideographs Extension C
- `U+2B740-U+2B81F`: CJK Unified Ideographs Extension D
- `U+2B820-U+2CEAF`: CJK Unified Ideographs Extension E
- `U+2CEB0-U+2EBEF`: CJK Unified Ideographs Extension F
- `U+30000-U+3134F`: CJK Unified Ideographs Extension G
- `U+31350–U+323AF`: CJK Unified Ideographs Extension H

**12 CJK Unified Ideographs in the CJK Compatibility Ideographs block:**

- `U+FA0E`: 﨎
- `U+FA0F`: 﨏
- `U+FA11`: 﨑
- `U+FA13`: 﨓
- `U+FA14`: 﨔
- `U+FA1F`: 﨟
- `U+FA21`: 﨡
- `U+FA23`: 﨣
- `U+FA24`: 﨤
- `U+FA27`: 﨧
- `U+FA28`: 﨨
- `U+FA29`: 﨩

**2 characters in the CJK Symbols and Punctuation block that are often regarded as Chinese characters:**

- `U+3006`: 〆
- `U+3007`: 〇

**Variation Selectors:**

- `U+FE00-U+FE0F`: Variation Selectors
- `U+E0100-U+E01EF`: Variation Selectors Supplement

# Wrong Solutions

1. Solutions containing `\p{sc=Han}` (means the Han script in Unicode) is wrong because it selects more than Chinese characters
1. Solutions containing `\p{Ideo}` (means the Ideograph property in Unicode) is wrong because it selects more than Chinese characters
1. Solutions containing `\p{Variation_Selector}` is wrong because it also selects Mongolian variation selectors

# References

1. [Unicode Scripts](https://www.unicode.org/Public/UCD/latest/ucd/Scripts.txt)
1. [Unicode PropList](https://www.unicode.org/Public/UCD/latest/ucd/PropList.txt)
1. [Unicode codepoint properties in the Python `regex` library](https://github.com/mrabarnett/mrab-regex#unicode-codepoint-properties-including-scripts-and-blocks)
1. [Unicode property escapes in JavaScript (ECMA)](https://tc39.es/proposal-regexp-unicode-property-escapes/)
1. [Unicode property escapes in JavaScript (MDN)](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Regular_Expressions/Unicode_Property_Escapes)
1. [Surrogate Pair Calculator](http://russellcottrell.com/greek/utilities/SurrogatePairCalculator.htm)
