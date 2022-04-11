---
title: ssh 连接 Google Colab 服务器
keywords:
- Google Colab
- 服务器
- ssh
- 深度学习
- GPU
lang: zh-CN
---

# 简介

Google Colab 提供了免费的服务器资源，可以使用 Google Colab 的 CPU，也可以使用 GPU 运行深度学习程序。然而，Google Colab 只提供了网页的操作界面，不能像平时操作服务器一样在真实的终端中输入命令。为此，本文提供一种借助中转服务器，使用 ssh 连接 Google Colab 服务器的方法。

# 实现思路

为了通过 ssh 连接 Google Colab 服务器，需要一个有公网 IP 的中转服务器。通过端口转发，将 Google Colab 服务器的 ssh 端口转发到中转服务器上，然后连接中转服务器，达到连接 Google Colab 服务器的目的。

# 实现步骤

## 将中转服务器的公钥加入 Google Colab 服务器

在 Google Colab 单元格中运行以下命令：

```python
%%sh
mkdir -p ~/.ssh
echo '<PUBLIC_KEY>' >> ~/.ssh/authorized_keys
```

其中，`<PUBLIC_KEY>` 要替换为中转服务器的 ssh 公钥。

## 配置 Google Colab 服务器

在 Google Colab 单元格中运行以下命令：

```python
%%sh
# Update system and install necessary packages
apt update > /dev/null
yes | unminimize > /dev/null
apt install -qq -o=Dpkg::Use-Pty=0 openssh-server pwgen net-tools psmisc pciutils > /dev/null

# Config ssh
ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa

# Config sshd
echo ListenAddress 127.0.0.1 >> /etc/ssh/sshd_config
mkdir -p /var/run/sshd
/usr/sbin/sshd
```

另外，我还根据自己的使用习惯进行以下配置：

```python
%%sh
# Install packages
apt install -qq -o=Dpkg::Use-Pty=0 zsh neofetch python-virtualenv nano python3.8 python3.8-dev python3.8-distutils python3.8-venv > /dev/null

# Set default shell to zsh
chsh -s `which zsh`

# Install oh-my-zsh
yes | sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# Customize zsh theme
wget -q -P /root/.oh-my-zsh/custom/themes https://gist.githubusercontent.com/ayaka14732/d15e5a583de03a3b310019f869cc2302/raw/118b7b46ffcf560bf85a35460d7df8343abc62c1/af-magic-%25E7%25B6%25BE.zsh-theme
sed -r -i 's/robbyrussell/af-magic-綾/' ~/.zshrc

# Install pip for Python 3.8
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.8 get-pip.py
rm -f get-pip.py

# Set timezone
echo 'export TZ=":Asia/Hong_Kong"' >> ~/.zshrc

# Set alias
echo 'alias python=python3.8
alias pip="python3.8 -m pip"' >> ~/.zshrc
```

## 将 Google Colab 服务器的公钥加入中转服务器

在 Google Colab 单元格中运行以下命令：

```python
!cat ~/.ssh/id_rsa.pub
```

可以查看上一步生成的 Google Colab 服务器的公钥。

然后，在中转服务器上运行：

```sh
echo '<PUBLIC_KEY>' >> ~/.ssh/authorized_keys
```

其中，`<PUBLIC_KEY>` 要替换为 Google Colab 服务器的公钥。

## 将 Google Colab 服务器的 ssh 端口转发到中转服务器

在 Google Colab 的单元格中运行以下命令：

```python
%%sh --bg
ssh -fNT -R 7423:127.0.0.1:22 user@example.com
```

其中，`user@example.com` 要替换为中转服务器登录用户名和 IP 地址。

## 连接 Google Colab 服务器

首先在本地登录中转服务器：

```sh
ssh user@example.com
```

其中，`user@example.com` 要替换为中转服务器登录用户名和 IP 地址。

然后在中转服务器上登录 Google Colab 服务器：

```sh
ssh -p7423 root@127.0.0.1
```

当重新启动 Google Colab 运行环境时，Google Colab 服务器的 host key 会发生变化，导致无法再次在中转服务器上登录 Google Colab 服务器。此时可以在中转服务器上运行以下命令，删除原来储存的 host key：

```sh
ssh-keygen -R "[127.0.0.1]:7423"
```

然后重新登录 Google Colab 服务器。

# 备考

## 查看 GPU 型号

在 Google Colab 的单元格中运行以下命令：

```python
import torch
print(torch.cuda.get_device_name())
```

如果 GPU 型号不理想，可以尝试重新启动 Google Colab 运行环境。

## 查看系统信息

```sh
neofetch
```

## 查看 GPU 使用情况

```sh
LD_LIBRARY_PATH=/usr/lib64-nvidia watch -n 1 nvidia-smi
```

不能直接使用 `export LD_LIBRARY_PATH=/usr/lib64-nvidia`，这样会导致某些程序不能使用 GPU。

## 保持 ssh 连接

为了防止 ssh 连接一段时间后就中断，可以在 ssh 选项中加入：

```ini
ServerAliveInterval=60
```

## 挂载 Google Drive

在启动 8 小时左右，服务器资源会被回收。为了防止数据丢失，可以定期将中间结果保存到 Google Drive 中。将 Google Drive 挂载到 Google Colab 的方法是在 Google Colab 单元格中运行：

```python
from google.colab import drive
drive.mount('/content/gdrive')
```

然后按提示操作。

## 防止没有操作网页导致资源被回收

如果一段时间没有操作网页，服务器资源将被回收。为此，可以在网页上运行一段输出日志的代码，告诉 Google Colab 这个服务器还在用：

```python
import time
import datetime

with open('out.log', 'w') as f:
    for i in range(10000):
        print(datetime.datetime.now(), 'Doing hard work here i =', i)
        print(datetime.datetime.now(), 'Doing hard work here i =', i, file=f)
        time.sleep(1000)
```

（2020&#8239;年&#8239;9&#8239;月&#8239;17&#8239;日）
