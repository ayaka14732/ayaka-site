---
title: 腾讯云 coscmd 通配符格式
lang: zh-CN
math: |-
  <meta name="description" content="腾讯云 coscmd 匹配文件的方法和想象中不太一样"/>
---

使用腾讯云 coscmd 将文件上传到腾讯云 COS 时，通常是使用 `--ignore` 参数指定要忽略的文件。例如，要忽略所有 `*.pkl` 和 `*.npz` 文件，可以使用：

```sh
coscmd upload -rs --delete -f . / --ignore '*.pkl,*.npz'
```

但是，`--ignore` 参数的用法比较特别，例如指定 `**/test` 不能成功忽略所有 `test` 目录，指定 `index.js` 也不能成功忽略当前目录下的 `index.js` 文件，因此我决定查找资料解决这个问题。

经过查看[源代码](https://github.com/tencentyun/coscmd/blob/d307ac3/coscmd/cos_sync.py#L21-L24)可以发现，coscmd 是使用 Python 标准库的 [`fnmatch.fnmatch`](https://docs.python.org/3/library/fnmatch.html#fnmatch.fnmatch) 函数判断文件名是否匹配的。`fnmatch` 函数不会对 `/` 作单独处理，而是把它当作文件名的一部分，按照匹配文件名的方式进行匹配。换句话说，`/` 和其他字符（例如 `a`, `b`, `1`, `2`, `正`）没有什么区别。例如：

```python
from fnmatch import fnmatch

print(fnmatch('index.js', 'index.js'))  # True

print(fnmatch('./index.js', 'index.js'))  # False
print(fnmatch('.正index.js', 'index.js'))  # False
print(fnmatch('./index.js', './index.js'))  # True

print(fnmatch('./test-1/src/index.js', 'test-1'))  # False
print(fnmatch('./test-1/src/index.js', '*/test-1/*'))  # True

print(fnmatch('./test-1/src/index.js', 'index.js'))  # False
print(fnmatch('./test-1/src/index.js', '*index.js'))  # True (but not good)
print(fnmatch('./test-1/src/index.js', '*/index.js'))  # True (good)
print(fnmatch('./test-1/src/abc-index.js', '*index.js'))  # True

print(fnmatch('./test-2/src/.gitignore', '.gitignore'))  # False
print(fnmatch('./test-2/src/.gitignore', './**/.gitignore'))  # True (but not good)
print(fnmatch('./test-2/src/.gitignore', './*/.gitignore'))  # True (good)
```

由此可以总结出 coscmd 通配符的常见用法如下：

- 忽略当前目录下的 `index.js` 文件：`./index.js`
- 忽略所有 `index.js` 文件：`*/index.js`
- 忽略当前目录下的 `*.csv` 文件：`./*.csv`
- 忽略所有 `*.csv` 文件：`*.csv`
- 忽略当前目录下的 `test` 子目录：`./test/*`
- 忽略所有 `test` 子目录：`*/test/*`
- 忽略所有 dot file：`*/.*`

（2022&#8239;年&#8239;3&#8239;月&#8239;1&#8239;日）
