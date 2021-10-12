---
title: 在 FreeBSD 上使用 Fcitx5
lang: zh-HK
keywords:
- FreeBSD
- Fcitx5
- rime 輸入法
math: |-
  <meta name="description" content="FreeBSD 上安裝 Fcitx5，並使用 rime 輸入法輸入中文的方法"/>
---

安裝 Fcitx5 輸入法程式：

```sh
pkg install fcitx5 fcitx5-configtool fcitx5-gtk fcitx5-qt zh-fcitx5-rime zh-rime-essay
```

安裝時會提示執行以下命令：

```sh
cp /usr/local/share/applications/org.fcitx.Fcitx5.desktop ~/.config/autostart/
```

新建 `~/.xprofile`：

```sh
export GTK_IM_MODULE=fcitx
export QT_IM_MODULE=fcitx
export XMODIFIERS=@im=fcitx
```

重啓系統後，在輸入法圖標上點按右鍵，開啓 Fcitx5 配置，然後添加 rime，即可輸入中文。

注：要使用廣東話輸入法，請參考 [rime-cantonese](https://github.com/rime/rime-cantonese) 的 [FreeBSD 安裝教程](https://github.com/rime/rime-cantonese/wiki/FreeBSD-%E5%AE%89%E8%A3%9D%E6%95%99%E7%A8%8B)。

（2021&#8239;年&#8239;10&#8239;月&#8239;11&#8239;日）
