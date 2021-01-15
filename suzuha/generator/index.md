---
title: 鈴羽系統乘法表生成器
lang: zh-Hant
keywords:
- 铃羽系统
- 汉语
- 数制拓展
- 《切韵》音系
- 计算机科学
math: |-
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/pure/1.0.1/pure-min.css"/>
  <style>
  body { max-width: initial; }
  table { overflow: initial; white-space: nowrap; }
  table rt { font-size: 80%; margin: 0 2px; }
  </style>
  <script src="index.js"></script>
---

<form class="pure-form" onsubmit="handleChange(this.radix.value); return false"><p><label>進制：<input name="radix" type="number" min="2" max="64" value="10" required /></label><input class="pure-button" type="submit" value="應用"/></p></form><table></table>
