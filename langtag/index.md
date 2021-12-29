---
title: HTML 设置 `lang` 属性的意义
lang: zh-CN
keywords:
- HTML
- lang 属性
- 网页设计
- 字体排印学
- 本地化
- 中文本地化
math: |-
  <style>
  #TOC li { list-style-type: decimal; }
  header > h1 { counter-reset: my-awesome-counter; }
  body > h1 { counter-increment: my-awesome-counter; }
  body > h1::before { content: counter(my-awesome-counter) ". "; }
  textarea { font-family: inherit; }
  figure > img { border: 1px solid; }
  </style>
  <link rel="stylesheet" href="sans.css"/>
---

> 同时发布于知乎：[网页头部的声明应该是用 `lang="zh"` 还是 `lang="zh-cn"`？ - 三日月 綾香的回答](https://www.zhihu.com/question/20797118/answer/1045722170)
>
> 2020&#8239;年&#8239;7&#8239;月&#8239;16&#8239;日更新：「而中文、日文、韩文应当使用两侧对齐」改为「而中文、日文、韩文可以使用两侧对齐」，修改部分语句。

# 如果不设置 `lang` 属性…

```html
<p>天</p>
<p lang="zh-CN">天</p>
```

结果为：

:::{lang=ja .sans}
<p>天</p>
<p lang="zh-CN">天</p>
:::

![图中第一行的「天」字两横上长下短，第二行上短下长](1.png)

这是因为本页面外层设置了 `lang="ja"`，因此如果不设置 `lang` 属性，就会继承外层的设置，使浏览器默认使用日文字体。日文字体的「天」字上长下短，与中文习惯不符，影响用户的阅读体验。

有人会说，我当然不会把页面外层设置为日文。但是，如果页面没有设置 `lang` 属性，就会使用浏览器或操作系统设置的语言。用户的系统使用何种语言，是网页开发者无法控制的。

又有人会说，「天」字只有两横的长短区别，差别并不大，有必要关注吗？其实，这是很有必要的。

首先，虽然对于「天」字来说，日文字体与中文字体的差别并不大，但还有许多字差别较大，例如：

![图片取自知乎问题「为何有时电子设备上『门』会显示为『⿻冂丨』？」中[亜恵恵阿由的回答](https://www.zhihu.com/question/46637444/answer/102311689)<br/>示例字体为：上＝ヒラギノ角ゴ（Hiragino Kaku Gothic ProN），下＝ヒラギノ角ゴ 簡体中文（Hiragino Sans GB/冬青黑体）](1_2.jpg)

其次，即使网页使用日文字体，如果所有汉字都使用日文字体显示，达到风格上的统一，在一定程度上尚可接受。但是，许多日文字体缺少中国大陆的简体字，这些字会 fallback 到能显示大陆简体字的日文字体或大陆字体，从而出现字体混杂的问题。例如：

![图中为[阮一峰的网络日志](https://www.ruanyifeng.com/blog/2020/02/weekly-issue-95.html)在日文系统中出现使用日文字体与字体混杂的问题的截图<br/>图中上方的汉字一部分使用 Yu Mincho 字体，一部分使用 Source Han Sans 字体，出现字体混杂<br/>下方的汉字使用 Source Han Mono 字体，其中「复」字与中文差异明显](1_3.png)

这种问题只需要在网页 `html` 标签添加属性 `lang="zh-CN"` 即可解决。

有人会说，为什么会有人使用日文系统浏览中文网页呢？实际上，随着国际交流与合作日益密切，出于工作和学习的原因，不少中国人会使用日文系统，也有不少日本人会浏览中文网页。而且，上述字体问题不仅会在日语环境下出现，在其他外语环境下同样会出现。因此，考虑这一问题是很有必要的。

# `lang="zh-CN"`, `lang="zh-HK"` 与 `lang="zh-TW"` 的差异

```html
<p lang="zh-CN">骨</p>
<p lang="zh-HK">骨</p>
<p lang="zh-TW">骨</p>
```

结果为：

:::{lang=ja .sans}
<p lang="zh-CN">骨</p>
<p lang="zh-HK">骨</p>
<p lang="zh-TW">骨</p>
:::

![图中第一行的「骨」字上方朝左、下方作两横，第二行上方朝右、下方作两横，第三行上方朝右、下方作「点挑」](2.png)

这是因为中国大陆「骨」上方朝左，而香港、台湾朝右；大陆、香港「骨」下方作两横，而台湾作「点挑」。设置语言属性后，浏览器分别应用了三地的字体。

# `lang="zh-Hans"` 与 `lang="zh-Hant"` 的差异

```html
<p lang="zh-Hans">骨</p>
<p lang="zh-Hant">骨</p>
```

结果为：

:::{lang=ja .sans}
<p lang="zh-Hans">骨</p>
<p lang="zh-Hant">骨</p>
:::

![图中第一行的「骨」字上方朝左、下方作两横，第二行上方朝右、下方作「点挑」](3.png)

这是因为 `zh-Hans` 默认使用大陆字形，`zh-Hant` 默认使用台湾字形。

# `lang="zh-HK"` 与 `lang="zh-Hant-HK"` 有什么区别？

一般情况下没有区别，因为香港是使用繁体中文的地区，所以 `lang="zh-HK"` 就隐含了 `lang="zh-Hant-HK"`，二者的行为应该是等同的。

但是，在目前最新版的 Mozilla Firefox 中二者行为不同：

```html
<p lang="zh-HK"><q>你好</q></p>
<p lang="zh-Hant-HK"><q>你好</q></p>
```

结果为：

:::{lang=ja .sans}
<p lang="zh-HK"><q>你好</q></p>
<p lang="zh-Hant-HK"><q>你好</q></p>
:::

![图中第一行的「你好」使用的引号为“”，第二行为「」](5.png)

这可能只是一个 bug。

# `lang` 属性在西文中的差异

```html
<style>.upper { text-transform: uppercase; }</style>
<p class="upper" lang="en-US">shipping</p>
<p class="upper" lang="tr">shipping</p>
```

结果为：

:::{lang=ja .sans}
<style>.upper { text-transform: uppercase; }</style>
<p class="upper" lang="en-US">shipping</p>
<p class="upper" lang="tr">shipping</p>
:::

![图中第一行的 I 上方不带点，第二行的 İ 上方带点](6.png)

这是因为土耳其文有带点与不带点两种 i 字母，带点的小写 i 对应的是带点的大写 İ。

# `lang="en-GB"` 与 `lang="en-US"` 的差异

```html
<textarea lang="en-GB">center centre</textarea>
<textarea lang="en-US">center centre</textarea>
```

结果为：

:::{lang=ja .sans}
<textarea lang="en-GB">center centre</textarea>
<textarea lang="en-US">center centre</textarea>
:::

![图中第一个文本框显示 c-e-n-t-e-r 为拼写错误，第二个显示 c-e-n-t-r-e 为拼写错误](7.png)

这是因为在 Mozilla Firefox 中，拼写检查时会区分英国英语与美国英语。

# 是不是只要使用了相应地区的汉字字体，就没必要再使用 `lang` 属性指定地区了？

有人可能认为，既然 `lang` 属性会影响浏览器所使用的汉字字体，从而影响字形，那么只要用了相应地区的汉字字体，字形自然也就确定了，所以也就不必再指定 `lang` 属性了。

这种想法是不正确的。因为现代字体具有 OpenType 的 locl 特性，会根据 `lang` 属性改变字形。

```html
<style>.font-k { font-family: 'Source Han Sans SC', sans-serif; }</style>
<p class="font-k">天</p>
<p class="font-k" lang="zh-CN">天</p>
```

结果为：

:::{lang=ja .sans}
<style>.font-k { font-family: 'Source Han Sans SC', sans-serif; }</style>
<p class="font-k">天</p>
<p class="font-k" lang="zh-CN">天</p>
:::

![图中第一行的「天」字上长下短，第二行上短下长](8.png)

这是因为外层设置了 `lang="ja"`，因此如果不设置 `lang` 属性，就会继承外层的设置，使浏览器默认使用日文字形。虽然 Source Han Sans SC 默认为中国大陆字形，但是由于 OpenType 的 locl 特性，也会自动变为日文字形。

有人会说，我当然不会把页面外层设置为日文。但是，需要再次重申，如果页面没有设置 `lang` 属性，就会使用浏览器或操作系统设置的语言。用户的系统使用何种语言，是网页开发者无法控制的。

还有一个更常见的现象是引号问题。

```html
<div lang="en">
  <p>“你好”</p>
  <p lang="zh-CN">“你好”</p>
</div>
```

结果为：

:::{lang=ja .sans}
<div lang="en">
  <p>“你好”</p>
  <p lang="zh-CN">“你好”</p>
</div>
:::

![图中第一行的引号为半角，第二行为全角](8_2.png)

这是因为 Unicode 中并不区分全角与半角引号，具体的显示效果会由于 `lang` 属性的不同而不同。

# 中国大陆的简体中文网页应该设置 `lang="zh"` 还是 `lang="zh-CN"`？

从效果上看，二者并没有区别，都会使用大陆字形显示汉字。因此，设置哪个都是没有问题的。

但是，如果网页是简繁混排的，即同一网页中还会出现 `lang="zh-HK"` 或 `lang="zh-TW"`，则为了代码的可读性与可维护性，应该使用 `lang="zh-CN"`。

例如，使用楷体排版的多语言网页可以这样设置 CSS：

```css
:lang(zh), :lang(ja), :lang(ko) { text-align: justify; }
:lang(zh-CN) { font-family: KaiTi, cursive; }
:lang(zh-TW) { font-family: DFKai-SB, cursive; }
:lang(zh-HK) { font-family: DFPHKStdKai-B5, cursive; }
```

西文文本不宜使用两侧对齐，否则会造成[川流](https://zh.wikipedia.org/wiki/%E5%B7%9D%E6%B5%81_(%E5%AD%97%E4%BD%93%E6%8E%92%E5%8D%B0%E5%AD%A6))现象，而中文、日文、韩文可以使用两侧对齐。这时，使用 `lang="zh"`，可以一次性选择所有中文变体，即所有以 `zh` 起始的 `lang` 属性。

如果将 `lang="zh-CN"` 改为 `lang="zh"`，则上述 CSS 代码中的 `lang="zh-CN"` 也必须改为 `lang="zh"`。在维护过程中，有可能因为维护人员的疏忽，规则之间被调换了顺序，写作：

```css
:lang(zh), :lang(ja), :lang(ko) { text-align: justify; }
:lang(zh-TW) { font-family: DFKai-SB, cursive; }
:lang(zh-HK) { font-family: DFPHKStdKai-B5, cursive; }
:lang(zh) { font-family: KaiTi, cursive; }
```

这样就产生了 bug，因为这会导致 `:lang(zh-TW)` 与 `:lang(zh-HK)` 两条规则都被 `:lang(zh)` 覆盖。

# 应该使用 `lang="zh-Hant"` 类属性，还是应该使用 `lang="zh-TW"` 类属性？

如前文所述，这在显示效果上是没有差别的，因为 `zh-Hant` 默认使用台湾字形。但是，二者却有语义上的差别。

如果网页内容是一篇用现代文体写成的、使用繁体字的、讲述古代文化的文章，其地域性并不强，并不能直接看出是台湾、香港，或是其他使用繁体字的地区的文章，因此应该使用 `lang="zh-Hant"`。同理，如果这样的文章使用简体字写成，其地域性并不强，并不能直接看出是来自中国大陆，还是其他使用简体字的地区，因此应该使用 `lang="zh-Hans"`。

而如果网页内容是一篇与现代生活紧密相关的文章，则通常需要指明地区。这是因为不同地区的用字、用词与写作习惯均存在差异。例如，在用字上，中国大陆、香港「着」和「著」音义均不同，而台湾则一律用「著」；在用词上，中国大陆称「摩托车」，香港称「电单车」，而台湾称「机车」。这时就应该使用 `lang="zh-CN"`, `lang="zh-HK"`, `lang="zh-TW"` 等属性。

# 如何使 `lang="zh-Hant"` 使用香港或韩国字形？

`zh-Hant` 默认使用台湾字形。但是有人会说，自己的页面就是需要设置 `lang="zh-Hant"`，但是又[不想使用台湾字形](http://founder.acgvlyric.org/iu/doku.php/%E8%AA%AA%E6%96%87:%E8%87%BA%E6%A8%99%E4%B9%8B%E5%AE%B3)。这时可以使用 CSS 的 [`font-language-override`](https://developer.mozilla.org/en-US/docs/Web/CSS/font-language-override){lang=en-x-code} 属性。

```html
<style>
.glyph-hk:lang(zh-Hant) { font-language-override: "ZHH"; }
.glyph-kr:lang(zh-Hant) { font-language-override: "KOR"; }
</style>
<p lang="zh-Hant">霄</p>
<p class="glyph-hk" lang="zh-Hant">霄</p>
<p class="glyph-kr" lang="zh-Hant">霄</p>
```

结果为：

:::{lang=ja .sans}
<style>
.glyph-hk:lang(zh-Hant) { font-language-override: "ZHH"; }
.glyph-kr:lang(zh-Hant) { font-language-override: "KOR"; }
</style>
<p lang="zh-Hant">霄</p>
<p class="glyph-hk" lang="zh-Hant">霄</p>
<p class="glyph-kr" lang="zh-Hant">霄</p>
:::

![图中第一行的「霄」字上方不作横、「小」字向内、下方作「点挑」<br/>第二行上方不作横、「小」字向内、下方作两横<br/>第三行上方作横、「小」字向外、下方作两横](4.png)

`font-language-override` 属性的常用取值如下：

- 大陆字形：`ZHS`
- 台湾字形：`ZHT`
- 香港字形：`ZHH`
- 韩国字形：`KOR`
- 日本字形：`JAN`

# 为什么一般情况下不应使用以 `cdo`, `cjy`, `cmn`, `cnp`, `cpx`, `csp`, `czh`, `czo`, `gan`, `hak`, `hsn`, `lzh`, `mnp`, `nan`, `wuu`, `yue` 等起始的属性？

这有两个原因。

首先是出于兼容性的考量，因为只有一部分较新的浏览器支持这些属性。

例如，吴语维基百科设置了 `lang="wuu"`。对于日文系统，在目前最新版的 Edge 浏览器中，页面会出现字体混杂的问题。

![图中为吴语维基百科的[《徐家汇》页面](https://wuu.wikipedia.org/wiki/%E5%BE%90%E5%AE%B6%E6%B1%87)在日文系统中出现字体混杂的问题的截图<br/>图中的汉字一部分使用 Meiryo 字体，一部分使用宋体](10.png)

这是因为 Edge 不能识别 `wuu` 这类属性。由于系统语言为日语，因此 Edge 优先使用日文字体显示。Edge 默认的日文字体是 Meiryo 字体，该字体缺少中国大陆的简体字，因此 fallback 到宋体，从而出现字体混杂的问题。

当然，由于吴语维基百科使用吴语，与通常所指的汉语并不等同，因此确实需要使用 `wuu` 这一属性；而一般的汉语页面使用以 `zh` 起始的属性即可，不需要使用 `cmn`。这就涉及到第二个原因。

根据中国开发者的习惯，在编写中文页面时，通常会认为页面记录的是「汉语」，而不是「官话」。这是因为在中国人通常的观念中，与「英语」、「日语」等相对的是「汉语」，而「官话」则与「粤语」、「吴语」等相对。如果页面使用以 `zh` 起始的属性，更能反映这样的语义。而如果编写的页面中同时出现了普通话与粤语，这时对普通话内容使用以 `cmn` 起始的属性，对粤语内容使用以 `yue` 起始的属性，是较好的选择。

# 相关链接

- [CJKV Information Processing](https://book.douban.com/subject/3404546/){lang=en-US}
- [Language subtag lookup app](https://r12a.github.io/app-subtags/){lang=en-US}
- [ISO 639-1 列表](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)
- [ISO 639-3 列表](https://en.wikipedia.org/wiki/Wikipedia:WikiProject_Languages/List_of_ISO_639-3_language_codes_(2019))
- [ISO 15924 列表](https://www.unicode.org/iso15924/iso15924-codes.html)
- [OpenType Features in CSS](https://sparanoid.com/lab/opentype-features/){lang=en-US}
- [繁简中文转换概说](https://zhuanlan.zhihu.com/p/104314323)
- [思源黑体](https://github.com/adobe-fonts/source-han-sans)
