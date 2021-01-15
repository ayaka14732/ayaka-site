---
title: ssh 连接 AI Studio 服务器
keywords:
- AI Studio
- 服务器
- ssh
- 深度学习
- GPU
lang: zh-CN
---

> 另见《[ssh 连接 Google Colab 服务器](/colab/)》

# 简介

AI Studio 提供了免费的服务器资源，可以使用 AI Studio 的 CPU，也可以使用 Tesla V100 的 GPU 运行深度学习程序。然而，AI Studio 只提供了网页的操作界面，不能像平时操作服务器一样在真实的终端中输入命令。为此，本文提供一种借助中转服务器，使用 ssh 连接 AI Studio 服务器的方法。

AI Studio 的服务器实际上是一个 Docker 容器。用户操作时，用户名为 aistudio，没有 root 权限。服务器在 22 端口运行了 ssh 程序，但没有公网 IP 地址。

AI Studio 提供了 100 GB 的实时持久化存储，路径为 `~/work`。在配置环境时，可以将下载的文件放入这个目录内，下次启动时直接从这个目录加载，不需要重新下载。

# 实现思路

为了通过 ssh 连接 AI Studio 服务器，需要一个有公网 IP 的中转服务器。通过端口转发，将 AI Studio 服务器的 ssh 端口转发到中转服务器上，然后连接中转服务器，达到连接 AI Studio 服务器的目的。

# 实现步骤

## 将中转服务器的公钥加入 AI Studio 服务器

在 AI Studio 单元格中运行以下命令：

```python
!echo '<PUBLIC_KEY>' >> ~/.ssh/authorized_keys
```

其中，`<PUBLIC_KEY>` 要替换为中转服务器的 ssh 公钥。

## 配置 AI Studio 服务器

在 AI Studio 单元格中运行以下命令：

```python
%%sh
# Config ssh
if [ ! -d ~/work/ssh ] ; then
mkdir ~/work/ssh
ssh-keygen -t rsa -N "" -f ~/work/ssh/id_rsa
fi
cp ~/work/ssh/id_rsa ~/.ssh/id_rsa

# Improve bash environment
git clone -q --depth=1 https://github.com.cnpmjs.org/ohmybash/oh-my-bash.git ~/.oh-my-bash
cp ~/.oh-my-bash/templates/bashrc.osh-template ~/.bashrc
sed -i 's/OSH_THEME="font"/OSH_THEME="bobby-python"/' ~/.bashrc

# Install neofetch
git clone -q --depth=1 https://github.com.cnpmjs.org/dylanaraps/neofetch.git ~/.local/share/neofetch

# Config PATH
echo 'export PATH=/opt/conda/bin:$HOME/.local/share/neofetch:$PATH' >> ~/.bashrc
```

## 将 AI Studio 服务器的公钥加入中转服务器

在 AI Studio 单元格中运行以下命令：

```python
!cat ~/work/ssh/id_rsa.pub
```

可以查看上一步生成的 AI Studio 服务器的公钥。

然后，在中转服务器上运行：

```sh
echo '<PUBLIC_KEY>' >> ~/.ssh/authorized_keys
```

其中，`<PUBLIC_KEY>` 要替换为 AI Studio 服务器的公钥。

这个步骤只用执行一次，以后再启动 AI Studio 运行环境时不用执行。

## 将 AI Studio 服务器的 ssh 端口转发到中转服务器

在 AI Studio 的单元格中运行以下命令：

```python
%%sh --bg
ssh -fNT -R 7423:127.0.0.1:22 user@example.com
```

其中，`user@example.com` 要替换为中转服务器登录用户名和 IP 地址。

## 连接 AI Studio 服务器

首先在本地登录中转服务器：

```sh
ssh user@example.com
```

其中，`user@example.com` 要替换为中转服务器登录用户名和 IP 地址。

然后在中转服务器上登录 AI Studio 服务器：

```sh
ssh -p7423 aistudio@127.0.0.1
```

当在 CPU 和 GPU 运行环境之间切换时，AI Studio 服务器的 host key 会发生变化，导致无法再次在中转服务器上登录 AI Studio 服务器。此时可以在中转服务器上运行以下命令，删除原来储存的 host key：

```sh
ssh-keygen -R "[127.0.0.1]:7423"
```

然后重新登录 AI Studio 服务器。

# 注意事项

如果不是第一次启动运行环境，首先要等待网页上的提示 syncing 和 Unzipping files 消失后，再运行上述代码，否则会导致文件冲突。

# 备考

## 查看系统信息

```sh
neofetch
```

## 查看 GPU 使用情况

```sh
watch -n 1 nvidia-smi
```

## 保持 ssh 连接

为了防止 ssh 连接一段时间后就中断，可以在 ssh 选项中加入：

```ini
ServerAliveInterval=60
```

## 备份大文件

可以使用腾讯云对象存储 COS。在 AI Studio 服务器与腾讯云对象存储 COS 之间传输文件的速度非常快，通常在 20 MB/s 以上。

（2020&#8239;年&#8239;9&#8239;月&#8239;17&#8239;日）
