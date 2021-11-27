---
title: 体验 OpenType 的 locl/smpl/trad/ss01-ss20 特性
lang: zh-CN
keywords:
- OpenType 字体
- 字体特性
- CSS
- 简繁转换
math: |-
  <style>
  .sample {
    font-family: unset;
    font-language-override: unset;
    text-align: center;
  }
  .test1 { font-family: 'Source Han Serif K', serif; }
  .test2 { font-family: 'Yu Mincho', serif; }
  .trad { font-feature-settings: 'trad'; }
  .test3 { font-family: 'Evil Sung', serif; }
  .ss01 { font-feature-settings: 'ss01'; }
  .ss02 { font-feature-settings: 'ss02'; }
  .ss03 { font-feature-settings: 'ss03'; }
  .ss04 { font-feature-settings: 'ss04'; }
  .ss05 { font-feature-settings: 'ss05'; }
  .ss06 { font-feature-settings: 'ss06'; }
  .ss07 { font-feature-settings: 'ss07'; }
  </style>
---

> 注：本文中的示例需要安装相应的字体才可以显示。

# 简介

OpenType 字体有一类字体特性，可以在不改变字符码位的情况下改变显示出的字形。

在浏览器中，可以通过 CSS 简便地调用这类 OpenType 字体特性。

# locl 特性

locl 特性可以根据地区的不同自动改变字形。这一特性是默认启用的。

比如下面的 HTML：

```html
<style>.test1 { font-family: 'Source Han Serif K', serif; }</style>
<p lang="ko-KR" class="test1">羽</p>
<p lang="zh-CN" class="test1">羽</p>
```

显示效果为：

:::{.sample}
<p lang="ko-KR" class="test1">羽</p>
<p lang="zh-CN" class="test1">羽</p>
:::

这里指定了韩国字形的字体 Source Han Serif K，但由于这一字体支持 locl 特性，当在 HTML 中指定 `lang="zh-CN"` 时，会自动使用大陆字形。

# smpl 特性与 trad 特性

这两种特性分别是将简体字形转换为繁体字形，以及将繁体字形转换为简体字形；有些日文字体也可以通过 trad 特性将新字形转换为旧字形。

例如下面的 HTML：

```html
<style>
  .test2 { font-family: 'Yu Mincho', serif; }
  .trad { font-feature-settings: 'trad'; }
</style>
<p class="test2">国家</p>
<p class="test2 trad">国家</p>
```

显示效果为：

:::{.sample}
<p class="test2">国家</p>
<p class="test2 trad">国家</p>
:::

当然，这一字体的转换规则并不完善：

```html
<p class="test2">花弁</p>
<p class="test2 trad">花弁</p>
```

显示效果为：

:::{.sample}
<p class="test2">花弁</p>
<p class="test2 trad">花弁</p>
:::

但是，更精确的转换在技术上是可行的，见我的新文章《[正确实现简转繁字体](https://ayaka.shn.hk/s2tfont/)》。

虽然 trad 特性在日文字体中已经得到支持，但目前似乎还没有人制作支持 smpl 特性的字体。

# ss01-ss20 特性

虽然我还没有见过支持 smpl 特性的字体，但有一个字体也可以达到把繁体字转为简体字的效果，而且甚至可以转换为二简字等七种写法，这就是 [Evil Sung](https://github.com/ButTaiwan/evilsung) 字体。

ss01-ss20 特性可以定义字符的 stylistic alternatives。OpenType 预留了 20 种 stylistic alternatives，这个字体一下就用了 7 种，定义了汉字的各种写法，分别是 1935 第一批 KMT、1964 简化字总表、1969 新加坡简体、1974 新加坡简体、1977 二简第一表、1977 二简第二表和 1986 简体字总表。

比如下面的 HTML：

```html
<style>
  .test3 { font-family: 'Evil Sung', serif; }
  .ss01 { font-feature-settings: 'ss01'; }
  .ss02 { font-feature-settings: 'ss02'; }
  .ss03 { font-feature-settings: 'ss03'; }
  .ss04 { font-feature-settings: 'ss04'; }
  .ss05 { font-feature-settings: 'ss05'; }
  .ss06 { font-feature-settings: 'ss06'; }
  .ss07 { font-feature-settings: 'ss07'; }
</style>
<p class="test3">晚來天欲雪，能飲一杯無</p>
<p class="test3 ss01">晚來天欲雪，能飲一杯無</p>
<p class="test3 ss02">晚來天欲雪，能飲一杯無</p>
<p class="test3 ss03">晚來天欲雪，能飲一杯無</p>
<p class="test3 ss04">晚來天欲雪，能飲一杯無</p>
<p class="test3 ss05">晚來天欲雪，能飲一杯無</p>
<p class="test3 ss06">晚來天欲雪，能飲一杯無</p>
<p class="test3 ss07">晚來天欲雪，能飲一杯無</p>
```

显示效果为：

:::{.sample}
<p class="test3">晚來天欲雪，能飲一杯無</p>
<p class="test3 ss01">晚來天欲雪，能飲一杯無</p>
<p class="test3 ss02">晚來天欲雪，能飲一杯無</p>
<p class="test3 ss03">晚來天欲雪，能飲一杯無</p>
<p class="test3 ss04">晚來天欲雪，能飲一杯無</p>
<p class="test3 ss05">晚來天欲雪，能飲一杯無</p>
<p class="test3 ss06">晚來天欲雪，能飲一杯無</p>
<p class="test3 ss07">晚來天欲雪，能飲一杯無</p>
:::

要在 Chrome 或 Firefox 中使用 1977 二简第二表，可以安装 Stylus 拓展，然后打开一个繁体字页面，添加 Custom CSS：

```css
* { font-family: "Evil Sung", serif; font-feature-settings: "ss06"; }
```

（作于 2020&#8239;年&#8239;2&#8239;月&#8239;24&#8239;日，更新于 2020&#8239;年&#8239;8&#8239;月&#8239;14&#8239;日、2021&#8239;年&#8239;8&#8239;月&#8239;18&#8239;日与 2021&#8239;年&#8239;11&#8239;月&#8239;27&#8239;日）
