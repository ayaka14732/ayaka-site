---
title: 体验 OpenType 的 locl/smpl/trad/ss01-ss20 特性
lang: zh-CN
math: |-
  <style>
  .sample {
    border: 1px solid;
    font-family: unset;
    font-language-override: unset;
    text-align: center;
  }
  </style>
---

> 注：本文中的示例需要安装相应的字体才能显示。

# 简介

现在的字体有一类特性，可以在不改变字符的情况下改变显示出来的样子。

在浏览器中，可以通过设置 CSS 简单地实现。

# locl 特性

locl 特性可以根据地区的不同自动改变字形。这一特性是默认启用的。

比如下面的 HTML：

```html
<p lang="ko-KR" style="font-family: 'Source Han Serif K', serif;">羽</p>
<p lang="zh-CN" style="font-family: 'Source Han Serif K', serif;">羽</p>
```

:::{.sample}
<p lang="ko-KR" style="font-family: 'Source Han Serif K', serif;">羽</p>
<p lang="zh-CN" style="font-family: 'Source Han Serif K', serif;">羽</p>
:::

这里指定了韩国字形的字体 Source Han Serif K，但由于这一字体支持 locl 特性，当在 HTML 中指定 `lang="zh-CN"` 时，会自动使用大陆字形。

# smpl 特性与 trad 特性

这两种特性分别是将简体字形转换为繁体字形，以及将繁体字形转换为简体字形；有些日文字体也可以通过 trad 特性将新字形转换为旧字形。

例如下面的 HTML：

```html
<p style="font-family: 'Yu Mincho', serif;">国家</p>
<p style="font-family: 'Yu Mincho', serif; font-feature-settings: 'trad';">国家</p>
```

:::{.sample}
<p style="font-family: 'Yu Mincho', serif;">国家</p>
<p style="font-family: 'Yu Mincho', serif; font-feature-settings: 'trad';">国家</p>
:::

当然，这一字体的转换规则并不完善：

```html
<p style="font-family: 'Yu Mincho', serif;">花弁</p>
<p style="font-family: 'Yu Mincho', serif; font-feature-settings: 'trad';">花弁</p>
```

:::{.sample}
<p style="font-family: 'Yu Mincho', serif;">花弁</p>
<p style="font-family: 'Yu Mincho', serif; font-feature-settings: 'trad';">花弁</p>
:::

但是，更精确的转换在技术上是可行的。（注：见我的新文章《[正确实现简转繁字体](https://ayaka.shn.hk/s2tfont/)》）

虽然 trad 特性在日文字体中已经得到支持，但是我还没有找到支持 smpl 特性的字体。当然，等我有空也可以自己开发一个。

# ss01-ss20 特性

虽然我还没有找到支持 smpl 特性的字体，但有一个字体也可以达到把繁体字转为简体字的效果，而且它甚至可以转换为二简字等七种写法！这就是 [Evil Sung](https://github.com/ButTaiwan/evilsung)。

ss01-ss20 特性可以定义字符的 stylistic alternatives。OpenType 预留了 20 种 stylistic alternatives，这个字体一口气用了 7 种，定义了汉字的各种写法，分别是 1935 第一批 KMT、1964 简化字总表、1969 新加坡简体、1974 新加坡简体、1977 二简第一表、1977 二简第二表和 1986 简体字总表。

比如下面的 HTML：

```html
<p style="font-family: 'Evil Sung', serif;">寻梦？撑一支长篙。</p>
<p style="font-family: 'Evil Sung', serif; font-feature-settings: 'ss01';">寻梦？撑一支长篙。</p>
<p style="font-family: 'Evil Sung', serif; font-feature-settings: 'ss02';">寻梦？撑一支长篙。</p>
<p style="font-family: 'Evil Sung', serif; font-feature-settings: 'ss03';">寻梦？撑一支长篙。</p>
<p style="font-family: 'Evil Sung', serif; font-feature-settings: 'ss04';">寻梦？撑一支长篙。</p>
<p style="font-family: 'Evil Sung', serif; font-feature-settings: 'ss05';">寻梦？撑一支长篙。</p>
<p style="font-family: 'Evil Sung', serif; font-feature-settings: 'ss06';">寻梦？撑一支长篙。</p>
<p style="font-family: 'Evil Sung', serif; font-feature-settings: 'ss07';">寻梦？撑一支长篙。</p>
```

:::{.sample}
<p style="font-family: 'Evil Sung', serif;">寻梦？撑一支长篙。</p>
<p style="font-family: 'Evil Sung', serif; font-feature-settings: 'ss01';">寻梦？撑一支长篙。</p>
<p style="font-family: 'Evil Sung', serif; font-feature-settings: 'ss02';">寻梦？撑一支长篙。</p>
<p style="font-family: 'Evil Sung', serif; font-feature-settings: 'ss03';">寻梦？撑一支长篙。</p>
<p style="font-family: 'Evil Sung', serif; font-feature-settings: 'ss04';">寻梦？撑一支长篙。</p>
<p style="font-family: 'Evil Sung', serif; font-feature-settings: 'ss05';">寻梦？撑一支长篙。</p>
<p style="font-family: 'Evil Sung', serif; font-feature-settings: 'ss06';">寻梦？撑一支长篙。</p>
<p style="font-family: 'Evil Sung', serif; font-feature-settings: 'ss07';">寻梦？撑一支长篙。</p>
:::

要在 Google Chrome 中使用，首先安装 Custom CSS 这一拓展，然后找一个充满繁体字的页面。

添加 Custom CSS 即可：

```css
* {
  font-family: "Evil Sung", serif;
  font-feature-settings: "ss06";
}
```

类似地，在 Mozilla Firefox 中也可以通过 Stylus 这一拓展解决。

（作于 2020&#8239;年&#8239;2&#8239;月&#8239;24&#8239;日，更新于 2020&#8239;年&#8239;8&#8239;月&#8239;14&#8239;日）
