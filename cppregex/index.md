---
title: C++ 正则表达式速查手册
lang: zh-CN
---

# 概述

C++ 中正则表达式通常使用 `R"(...)"` 的形式，相当于 Python 中的 `r"..."`，保证字符串内部的 \\ 等字符不被转义。

但是，`R"(...)"` 中的 `()` 是语法的一部分，一定不要当成字符串里面的内容。例如，`R"(1234)"` 表示字符串 `"1234"`，而不是 `"(1234)"`。

创建正则表达式使用如下形式：

```cpp
const std::regex re { R"(\S+)" };
```

如果要指定忽略大小写， 则使用如下形式：

```cpp
const std::regex re { R"(\S+)", std::regex::icase };
```

# 查找

## 判断是否完全匹配

使用 [`std::regex_match`](https://en.cppreference.com/w/cpp/regex/regex_match) 函数，第一个参数为字符串，第二个参数为正则表达式，返回值为 `bool`。

**示例：验证电话号码**

假设电话号码为 xxx-xxxxx 的形式

```cpp
import <iostream>;
import <regex>;

int main() {
    const std::regex re { R"(\d{3}-\d{5})" };

    const std::string str1 = "123-12345";
    const bool matched1 = std::regex_match(str1, re);
    std::cout << (matched1 ? "Yes" : "No") << std::endl;

    const std::string str2 = "123-12345a";
    const bool matched2 = std::regex_match(str2, re);
    std::cout << (matched2 ? "Yes" : "No") << std::endl;
}
```

输出：

```
Yes
No
```

**示例：密码格式验证**

在设置密码时，要求至少包含一个小写字母、一个大写字母、至少一个标点字符，并且至少有 6 个字符长。

同上例，第一个参数为待验证的密码字符串，第二个参数为正则表达式 `(?=.*[a-z])(?=.*[A-Z])(?=.*[[:punct:]]).{6,}`。

## 判断是否局部匹配

使用 [`std::regex_search`](https://en.cppreference.com/w/cpp/regex/regex_search) 函数，第一个参数为字符串，第二个参数为正则表达式，返回值为 `bool`。

它与 [`std::regex_match`](https://en.cppreference.com/w/cpp/regex/regex_match) 的区别是，`std::regex_search` 只要局部匹配就返回 `true`，而 `std::regex_match` 必须完全匹配整个字符串。

**示例：判断每行是否均在 20 字符以内**

```cpp
import <iostream>;
import <regex>;

int main() {
    const std::regex re { R"(^.{21,}$)" };
    const std::string str =
R"(0123456789012345
012345678901234567
0123456789012345678901
0123456)";
    const bool matched = std::regex_search(str, re);
    std::cout << (!matched ? "Yes" : "No") << std::endl;
}
```

输出：

```
No
```

## 统计匹配个数

使用 `<iterator>` 头文件的 `std::distance` 函数。

**示例：统计词数**

```cpp
import <iostream>;
import <regex>;
import <iterator>;

int main() {
    const std::regex re { R"(\S+)" };
    const std::string str = "Lorem ipsum dolor sit amet, consectetur";
    const std::sregex_iterator begin { str.cbegin(), str.cend(), re }, end;
    const auto count = std::distance(begin, end);
    std::cout << count << std::endl;
}
```

输出：

```
6
```

# 提取

## 完全匹配并提取分组

使用 [`std::regex_match`](https://en.cppreference.com/w/cpp/regex/regex_match) 函数，第一个参数为字符串，第二个参数为匹配结果，第三个参数为正则表达式。返回值为 `bool`，若匹配成功，则返回 `true`，且匹配结果存入第二个参数中。

**示例：匹配字母加数字**

```cpp
import <iostream>;
import <regex>;

int main() {
    const std::regex re { R"(([A-Za-z]+)([0-9]+))" };
    const std::string str = "ABC9876543";
    std::smatch match;
    if (std::regex_match(str, match, re)) {
        const std::ssub_match match_1 = match[1];
        const std::ssub_match match_2 = match[2];
        std::cout << match_1.str() << ' ' << match_2.str() << std::endl;
    }
}
```

输出：

```
ABC 9876543
```

## 局部匹配并提取分组

使用 [`std::regex_search`](https://en.cppreference.com/w/cpp/regex/regex_search) 函数，第一个参数为字符串，第二个参数为匹配结果，第三个参数为正则表达式。返回值为 `bool`，若匹配成功，则返回 `true`，且匹配结果存入第二个参数中。

它与 [`std::regex_match`](https://en.cppreference.com/w/cpp/regex/regex_match) 的区别是，`std::regex_search` 只要局部匹配就返回 `true`，而 `std::regex_match` 必须完全匹配整个字符串。

**示例：匹配长度为 5 的单词的前两个字母**

```cpp
import <iostream>;
import <regex>;

int main() {
    const std::regex re { R"(\b([a-z]{2})[a-z]{3}\b)" };
    const std::string str = "thread look welcome high merit cup";
    std::smatch match;
    if (std::regex_search(str, match, re)) {
        const std::ssub_match match_1 = match[1];
        std::cout << match_1.str() << std::endl;
    }
}
```

输出：

```
me
```

## 获取所有匹配

使用 [`std::sregex_iterator`](https://en.cppreference.com/w/cpp/regex/regex_iterator)，然后对迭代器使用 `it->str()`。

**示例：逐行输出所有单词**

```cpp
import <iostream>;
import <regex>;

int main() {
    const std::regex re { R"(\S+)" };
    const std::string str = "Lorem ipsum dolor sit amet, consectetur";
    for (std::sregex_iterator it { str.cbegin(), str.cend(), re }, end; it != end; ++it)
        std::cout << it->str() << std::endl;
}
```

输出：

```
Lorem
ipsum
dolor
sit
amet,
consectetur
```

## 从所有匹配中提取分组

使用 [`std::sregex_iterator`](https://en.cppreference.com/w/cpp/regex/regex_iterator)，然后对迭代器使用 `(*it)[1]`。

**示例：匹配十六进制数的数字部分**

```cpp
import <iostream>;
import <regex>;

int main() {
    const std::regex re { R"(0x([0-9a-f]+))" };
    const std::string str = "  0x1d782 0x86fa1  0x54b67 ";
    for (std::sregex_iterator it { str.cbegin(), str.cend(), re }, end; it != end; ++it)
        std::cout << (*it)[1] << std::endl;
}
```

输出：

```
1d782
86fa1
54b67
```

# 分割

## 分割字符串

**示例：根据空白字符分割单词（类似 Python 中 [`re.split`](https://docs.python.org/3/library/re.html#re.split)）**

```cpp
import <iostream>;
import <list>;
import <regex>;

std::list<std::string> split(const std::string& str, const std::regex& delim) {
    const std::sregex_token_iterator
        begin { str.cbegin(), str.cend(), delim, -1 }, end;
    return { begin, end };
}

int main() {
    const std::string str = "  Lorem   ipsum \n dolor sit  \t amet,   consectetur  ";
    const std::regex delim { R"(\s+)" };
    const auto tokens = split(str, delim);
    for (const std::string& token : tokens)
        std::cout << token << std::endl;
}
```

输出：

```

Lorem
ipsum
dolor
sit
amet,
consectetur
```

**示例：字符串包括字母和数字，根据字母和数字的边界分割字符串**

`"123abc345defghi6789"`

`"((\d+|[a-zA-Z]+))"`

转变为「获取所有匹配」类问题。

# 替换

## 替换字符串

使用 [`std::regex_replace`](https://en.cppreference.com/w/cpp/regex/regex_replace)。

**示例：将所有 `<h1>` 替换为 `<h2>`，`</h1>` 替换为 `</h2>`**

``` cpp
import <iostream>;
import <regex>;

int main() {
    const std::regex re { R"(<(\/?)h1>)" };
    const std::string str = "<h1>Apples</h1><p>I like apples.</p>\n"
                            "<h1>Bananas</h1><p>I also like bananas.</p>";
    const std::string res = std::regex_replace(str, re, "<$1h2>");
    std::cout << res << std::endl;
}
```

输出：

```
<h2>Apples</h2><p>I like apples.</p>
<h2>Bananas</h2><p>I also like bananas.</p>
```

（作于 2019&#8239;年&#8239;3&#8239;月&#8239;17&#8239;日，发布于 2020&#8239;年&#8239;3&#8239;月&#8239;17&#8239;日，补充于 2020&#8239;年&#8239;12&#8239;月&#8239;9&#8239;日）
