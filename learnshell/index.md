---
title: 记一次学习 Shell 的经历
lang: zh-CN
---

今天又一次看名叫「[歴史さんぽ]{lang=ja}」的日语生肉的日本文学史，还是不怎么看得懂，不过评论区的交谈引起了我的注意：有一位大神找到了视频中提到的配套的讲义资料，发现资料的网址是 `http://static.manavee.com/files/reference/foo.pdf` 的格式，其中 `foo` 是一个正整数。该视频分为数课，因此讲义也是多个 PDF 文件，而要想找齐所有讲义，就要从数字 1 开始不断尝试，看对应的 PDF 文件，是否是该课程的讲义。

经过尝试容易发现，该网站上目前共有 1415 个文件。而经过评论区中三位大神接力，在千余次测试后，已经将文件基本找全。他们对知识的渴望与锲而不舍的精神使我非常佩服，但也觉得十分心疼 >\_<，我作为一名程序员，应该用编程的工具，为解决这个问题做出自己的贡献。

基本思路如下：

1. `i` 从 1 至 1415
1. 对于每个 `i`，从上述网址获取 PDF 文件
1. 提取 PDF 中的文本内容
1. 检测 PDF 中是否含有「[歴史さんぽ]{lang=ja}」字符串，若是，则下载该文件

下面要开始着手实现了。对于第 2 步，需要使用 `wget`，但是 Windows 上没有 `wget`。不过好在我曾在自己的一个小项目中用过这个命令，只需要下载一个 `wget.exe` 文件即可解决。

对于第 3 步，我的第一反应是使用 `strings`，奈何字符串在 PDF 中有特殊的存储格式，于是我另觅他法，找到了 stackoverflow 上的[一个回答](https://stackoverflow.com/a/26405241)，使用 `gswin64c` 解决。

接下来，我需要使用 Shell 解决这个问题，由于未曾学习过 Shell 的原因，诸如 if 语句啦、标准输入流作为文件啦、如何使用 `grep` 命令啦，基本上都是自己边查边用。经过一番学习后，我收获颇丰，将如下代码投入使用：

``` sh
for i in {1..1415}
do
  str="http://static.manavee.com/files/reference/$i.pdf"
  if [ `wget -qO- $str | gswin64c -sDEVICE=txtwrite -o- - | grep -c "歴史さんぽ"` -eq '0' ]
  then
    echo "$i: Not found"
  else
    echo "$i: Found"
    wget -q $str
  fi
done
```

然而，代码运行起来却不甚理想：某些 PDF 文件似乎是难啃的硬骨头，`gswin64c` 一处理就停不下来。起初我手动终止进程，然后修改代码跳过这些文件，但后来被跳过的文件多了，也是一件麻烦事。我觉得这违背编程的初衷了，一定要想个方法解决掉这个问题。

于是，改进后的思路如下：

**`sub.sh`**

1. 对于某个数字，从上述网址获取 PDF 文件
1. 提取 PDF 中的文本内容
1. 检测 PDF 中是否含有「[歴史さんぽ]{lang=ja}」字符串，若是，则下载该文件

**`ext.sh`**

1. `i` 从 1 至 1415
1. 对于每个 `i`，计时 20 秒，执行 `sub.sh`
1. 当上一步结束时，检测结束的原因：到底是事情已经做完了自动结束的，还是 20 秒数到了强制结束的？若是强制结束的，把 `i` 记录到 `log.txt` 中

然后，运行 `ext.sh` 即可。

而且，令人高兴的是，因为前面已经处理了一部分，所以这里并不需要再从 1 开始。

为了实现上述思路，我经过查找资料，知道 `timeout` 命令可以实现所需的「程序超时退出」功能。其中，如果程序超时，则退出状态数为 124。在 Shell 中，退出状态数用 `$?` 变量表示。另外，向 `log.txt` 中追加内容，是使用 `>>`。

最终的程序如下：

`sub.sh`

``` sh
str="http://static.manavee.com/files/reference/$1.pdf"
if [ `wget -qO- $str | gswin64c -sDEVICE=txtwrite -o- - | grep -c "歴史さんぽ"` -eq '0' ]
then
  echo "$1: Not found"
else
  echo "$1: Found"
  wget -q $str
fi
```

`ext.sh`

``` sh
for i in {1..1415}
do
  timeout 20s ./sub.sh $i
  if [ $? -eq 124 ]
  then
    echo $i >> log.txt
  fi
done
```

执行 `ext.sh`，程序就如愿以偿地运行起来了！

通过此次写程序，我解决了「寻找日本文学史讲义」这一大海捞针的问题，而且捕获了三位大神没有找到的一只漏网之鱼，这令我十分高兴。然而，更为重要的是，我学习了 Shell 的基本使用方法，掌握了很多与 Shell 有关的知识。对于这些知识，我将永远铭记在心，为我今后成为一名更加优秀的程序员埋下伏笔！

（2018&#8239;年&#8239;8&#8239;月&#8239;29&#8239;日）

[文件下载](日本文学史.zip)
