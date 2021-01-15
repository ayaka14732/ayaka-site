---
title: CCF 201403-3 命令行选项
lang: zh-CN
---

# 试题

**问题描述**

请你写一个命令行分析程序，用以分析给定的命令行里包含哪些选项。每个命令行由若干个字符串组成，它们之间恰好由一个空格分隔。这些字符串中的第一个为该命令行工具的名字，由小写字母组成，你的程序不用对它进行处理。在工具名字之后可能会包含若干选项，然后可能会包含一 些不是选项的参数。

选项有两类：带参数的选项和不带参数的选项。一个合法的无参数选项的形式是一个减号后面跟单个小写字母，如 "-a" 或 "-b"。而带参数选项则由两个由空格分隔的字符串构成，前者的格式要求与无参数选项相同，后者则是该选项的参数，是由小写字母，数字和减号组成的非空字符串。

该命令行工具的作者提供给你一个格式字符串以指定他的命令行工具需要接受哪些选项。这个字符串由若干小写字母和冒号组成，其中的每个小写字母表示一个该程序接受的选项。如果该小写字母后面紧跟了一个冒号，它就表示一个带参数的选项，否则则为不带参数的选项。例如，`ab:m:` 表示该程序接受三种选项，即 "-a"（不带参数），"-b"（带参数）， 以及 "-m"（带参数）。

命令行工具的作者准备了若干条命令行用以测试你的程序。对于每个命令行，你的工具应当一直向后分析。当你的工具遇到某个字符串既不是合法的选项，又不是某个合法选项的参数时，分析就停止。命令行剩余的未分析部分不构成该命令的选项，因此你的程序应当忽略它们。

**输入格式**

输入的第一行是一个格式字符串，它至少包含一个字符，且长度不超过 52。格式字符串只包含小写字母和冒号，保证每个小写字母至多出现一次，不会有两个相邻的冒号，也不会以冒号开头。

输入的第二行是一个正整数 $N \left( 1 \leq N \leq 20 \right)$，表示你需要处理的命令行的个数。

接下来有 $N$ 行，每行是一个待处理的命令行，它包括不超过 256 个字符。该命令行一定是若干个由单个空格分隔的字符串构成，每个字符串里只包含小写字母，数字和减号。

**输出格式**

输出有 $N$ 行。其中第 $i$ 行以 "Case i:" 开始，然后应当有恰好一个空格，然后应当按照字母升序输出该命令行中用到的所有选项的名称，对于带参数的选项，在输出它的名称之后还要输出它的参数。如果一个选项在命令行中出现了多次，只输出一次。如果一个带参数的选项在命令行中出现了多次，只输出最后一次出现时所带的参数。

**样例输入**

```
albw:x
4
ls -a -l -a documents -b
ls
ls -w 10 -x -w 15
ls -a -b -c -d -e -l
```

**样例输出**

```
Case 1: -a -l
Case 2:
Case 3: -w 15 -x
Case 4: -a -b
```

# 分析

由于题目要求「按照字母升序输出该命令行中用到的所有选项的名称」，所以选择 `std::map` 存储所有合法的命令行选项。

使用自定义数据结构保存命令行选项的属性。当输入特定的命令行选项（即 `-` 后的字母，以 `char` 类型存储）时，通过 `std::map` 映射到对应的属性。这种自定义数据结构以 `Data` 类型表示。因此，`std::map` 的类型是 `std::map<char, Data>`。

`Data` 类型需要提供的信息包括：该命令行选项是否带参数，用户是否输入了这个命令行选项，以及如果带参数且用户输入了这个选项，还需提供用户输入的参数。

分析这个类型可知，它是两种情况的并：「该命令行选项带参数」和「该命令行选项不带参数」。第一种情况又是两种小情况的并：「用户没有输入这个命令行选项」和「用户输入了这个命令行选项」；第二种情况也是两种小情况的并：「用户没有输入这个命令行选项」和「用户输入了这个命令行选项，它的值是 `std::string`」。

为了构造这样的类型，计算这个类型中所含元素的个数。设 `std::string` 为 $a$，则根据以上分析，`Data` 为 $(1 + 1) + (1 + a) = 3 + a$。于是，可以将其定义如下：

```cpp
class Data {
public:
    enum class Chirality { Left, Right };
    enum class State { Nothing, Just };
private:
    const Chirality chirality;
    State state;
    union Storage {
        struct Empty { } empty;
        std::string arg;
        Storage() : empty { } { }
        ~Storage() { }
    } storage;
public:
    Data(Chirality c) : chirality { c }, state { State::Nothing }, storage { } { }
    Data(const Data& d) : chirality { d.chirality }, state { d.state } {
        if (d.chirality == Chirality::Right && d.state == State::Just)
            ::new (&storage.arg) std::string { d.storage.arg };
    }
    ~Data() {
        if (chirality == Chirality::Right && state == State::Just)
            storage.arg.~basic_string();
    }
    Chirality getChirality() const { return chirality; }
    State getState() const { return state; }
    void setStateToJust() { state = State::Just; }
    std::string getArg() const { return storage.arg; }
    void setArg(const std::string& str) { ::new (&storage.arg) std::string { str }; }
};
```

在上述定义中，`Data` 可以看作由 `chirality` 和 `state` 的笛卡尔积和联合体 `storage` 组成。计算这个类型中所含元素的个数可知，`chirality` 中元素个数为 2，`state` 中元素个数也为 2。由于 `storage` 中也包括两种情况，需要额外的 2 个元素提供选择信息，所以消耗了 2 个元素。但它可以看作是 `empty` 和 `std::string` 的并，`empty` 只有唯一的可能性，因此元素个数为 1，再设 `std::string` 为 $a$，则根据以上分析，`Data` 为 $(2 \times 2) - 2 + (1 + a) = 3 + a$，与上述定义相同，因此这样的定义是合理的。

为 `std::map<char, Data>` 定义别名 `Pattern`：

```cpp
using Pattern = std::map<char, Data>;
```

当读入第一行时，程序解析第一行的内容并将其存入一个 `Pattern m` 中，作为解析用户输入的模板，称为 `parseRule` 函数，它可以定义如下：

$parseRule \left( m , list \right) = \begin{cases}
m& list = [] \\
parseRule \left( insert \left( n , \mathrm{inr} \, Nothing , m \right) , xs \right)& list = n : \mathsf{':'} : xs \\
parseRule \left( insert \left( n , \mathrm{inl} \, Nothing , m \right) , xs \right)& list = n : xs
\end{cases}$

于是，可以采用以下方式实现 `parseRule` 函数：

```cpp
Pattern parseRule(Pattern m, const char *p) {
    if (!p[0]) {
        return m;
    } else if (p[1] == ':') {
        m.emplace(p[0], Data::Chirality::Right);
        return parseRule(m, p + 2);
    } else {
        m.emplace(p[0], Data::Chirality::Left);
        return parseRule(m, p + 1);
    }
}
```

接下来读入个数 `n`，设置循环变量 `i` 从 1 到 `n` 循环。对于每次循环，先读入用户的一行输入，再使用 `split` 函数，按空格把输入分割成 token。由于「这些字符串中的第一个为该命令行工具的名字，由小写字母组成，你的程序不用对它进行处理」，所以丢弃第一个 token，将剩余的 token 和前面已经获得的 `Pattern m` 一起放入 `solve` 函数中处理。经过 `solve` 函数处理后，原来 `Pattern m` 中「用户没有输入这个命令行选项」的信息，会被「用户输入了这个命令行选项」所取代。

由于 C++ 语言没有内置的 `split` 函数，需要自己实现如下：

```cpp
std::forward_list<std::string> split(const std::string& str, const std::regex& delim = std::regex { R"(\s+)" }) {
    const std::sregex_token_iterator begin { str.cbegin(), str.cend(), delim, -1 }, end;
    return { begin, end };
}
```

`solve` 函数可以定义如下：

$solve \left( m , list \right) = \begin{cases}
\begin{cases}
solve \left( insert \left( n , \mathrm{inr} \, Just \, x , m ) \right) , xs \right)& lookup \left( n , m \right) = Just \left( \mathrm{inr} \, \_ \right) \\
solve \left( insert \left( n , \mathrm{inl} \, Just \, \top , m ) \right) , x : xs \right)& lookup \left( n , m \right) = Just \left( \mathrm{inl} \, \_ \right) \\
m& \text{otherwise}
\end{cases}& list = [\mathsf{'-'}, n] : x : xs \\
\begin{cases}
insert \left( n , \mathrm{inl} \, Just \, \top , m ) \right)& lookup \left( n , m \right) = Just \left( \mathrm{inl} \, \_ \right) \\
m& \text{otherwise}
\end{cases}& list = [[\mathsf{'-'}, n]] \\
m& \text{otherwise}
\end{cases}$

由此，可以将 `solve` 函数实现如下：

```cpp
Pattern solve(Pattern m, std::forward_list<std::string> xs) {
    if (xs.cbegin() != xs.cend() && std::next(xs.cbegin()) != xs.cend() && xs.front().length() == 2 && xs.front()[0] == '-') {
        const auto pos = m.find(xs.front()[1]);
        if (pos != m.end() && pos->second.getChirality() == Data::Chirality::Right) {
            pos->second.setStateToJust();
            pos->second.setArg(*std::next(xs.cbegin()));
            xs.pop_front();
            xs.pop_front();
            return solve(m, xs);
        } else if (pos != m.end() && pos->second.getChirality() == Data::Chirality::Left) {
            pos->second.setStateToJust();
            xs.pop_front();
            return solve(m, xs);
        }
    } else if (xs.cbegin() != xs.cend() && xs.front().length() == 2 && xs.front()[0] == '-') {
        const auto pos = m.find(xs.front()[1]);
        if (pos != m.end() && pos->second.getChirality() == Data::Chirality::Left)
            pos->second.setStateToJust();
    }
    return m;
}
```

经过 `solve` 函数处理后，得到了新的 `Pattern`，为此再定义一个 `getResult` 函数，将 `Pattern` 转换为题目规定的输出格式。在函数中，只需检测 `state` 的值是否为 `Just` 即可：

```cpp
std::string getResult(Pattern m) {
    std::ostringstream oss;
    for (const auto& it : m)
        if (it.second.getChirality() == Data::Chirality::Left && it.second.getState() == Data::State::Just)
            oss << ' ' << '-' << it.first;
        else if (it.second.getChirality() == Data::Chirality::Right && it.second.getState() == Data::State::Just)
            oss << ' ' << '-' << it.first << ' ' << it.second.getArg();
    return oss.str();
}
```

解题所需的函数已经定义好，`main` 函数可以表示如下：

```cpp
int main() {
    std::string line;
    std::getline(std::cin, line);
    const Pattern m = parseRule({ }, line.c_str());
    int n;
    std::getline(std::cin, line);
    std::istringstream iss { line };
    iss >> n;
    for (int i = 1; i <= n; i++) {
        std::getline(std::cin, line);
        std::forward_list<std::string> tokens { split(line) };
        tokens.pop_front();
        std::cout << "Case " << i << ':' << getResult(solve(m, tokens)) << std::endl;
    }
}
```

# 结果

```cpp
include <iostream>
include <sstream>
include <forward_list>
include <map>
include <regex>
include <new>

class Data {
public:
    enum class Chirality { Left, Right };
    enum class State { Nothing, Just };
private:
    const Chirality chirality;
    State state;
    union Storage {
        struct Empty { } empty;
        std::string arg;
        Storage() : empty { } { }
        ~Storage() { }
    } storage;
public:
    Data(Chirality c) : chirality { c }, state { State::Nothing }, storage { } { }
    Data(const Data& d) : chirality { d.chirality }, state { d.state } {
        if (d.chirality == Chirality::Right && d.state == State::Just)
            ::new (&storage.arg) std::string { d.storage.arg };
    }
    ~Data() {
        if (chirality == Chirality::Right && state == State::Just)
            storage.arg.~basic_string();
    }
    Chirality getChirality() const { return chirality; }
    State getState() const { return state; }
    void setStateToJust() { state = State::Just; }
    std::string getArg() const { return storage.arg; }
    void setArg(const std::string& str) { ::new (&storage.arg) std::string { str }; }
};

using Pattern = std::map<char, Data>;

Pattern parseRule(Pattern m, const char *p) {
    if (!p[0]) {
        return m;
    } else if (p[1] == ':') {
        m.emplace(p[0], Data::Chirality::Right);
        return parseRule(m, p + 2);
    } else {
        m.emplace(p[0], Data::Chirality::Left);
        return parseRule(m, p + 1);
    }
}

Pattern solve(Pattern m, std::forward_list<std::string> xs) {
    if (xs.cbegin() != xs.cend() && std::next(xs.cbegin()) != xs.cend() && xs.front().length() == 2 && xs.front()[0] == '-') {
        const auto pos = m.find(xs.front()[1]);
        if (pos != m.end() && pos->second.getChirality() == Data::Chirality::Right) {
            pos->second.setStateToJust();
            pos->second.setArg(*std::next(xs.cbegin()));
            xs.pop_front();
            xs.pop_front();
            return solve(m, xs);
        } else if (pos != m.end() && pos->second.getChirality() == Data::Chirality::Left) {
            pos->second.setStateToJust();
            xs.pop_front();
            return solve(m, xs);
        }
    } else if (xs.cbegin() != xs.cend() && xs.front().length() == 2 && xs.front()[0] == '-') {
        const auto pos = m.find(xs.front()[1]);
        if (pos != m.end() && pos->second.getChirality() == Data::Chirality::Left)
            pos->second.setStateToJust();
    }
    return m;
}

std::string getResult(Pattern m) {
    std::ostringstream oss;
    for (const auto& it : m)
        if (it.second.getChirality() == Data::Chirality::Left && it.second.getState() == Data::State::Just)
            oss << ' ' << '-' << it.first;
        else if (it.second.getChirality() == Data::Chirality::Right && it.second.getState() == Data::State::Just)
            oss << ' ' << '-' << it.first << ' ' << it.second.getArg();
    return oss.str();
}

std::forward_list<std::string> split(const std::string& str, const std::regex& delim = std::regex { R"(\s+)" }) {
    const std::sregex_token_iterator begin { str.cbegin(), str.cend(), delim, -1 }, end;
    return { begin, end };
}

int main() {
    std::string line;
    std::getline(std::cin, line);
    const Pattern m = parseRule({ }, line.c_str());
    int n;
    std::getline(std::cin, line);
    std::istringstream iss { line };
    iss >> n;
    for (int i = 1; i <= n; i++) {
        std::getline(std::cin, line);
        std::forward_list<std::string> tokens { split(line) };
        tokens.pop_front();
        std::cout << "Case " << i << ':' << getResult(solve(m, tokens)) << std::endl;
    }
}
```

评测结果：正确<br/>得分：100<br/>时间使用：15ms<br/>空间使用：544.0KB

:::{style="display: none;"}
```haskell
import Control.Applicative (empty)
import Data.Foldable (fold, for_)
import Data.Map.Strict (Map, (!?))
import qualified Data.Map.Strict as M (insert, toAscList)

type Pattern = Map Char (Either (Maybe ()) (Maybe String))

parseRule :: Pattern -> String -> Pattern
parseRule m [] = m
parseRule m (n : ':' : xs) = parseRule (M.insert n (Right Nothing) m) xs
parseRule m (n : xs) = parseRule (M.insert n (Left Nothing) m) xs

solve :: Pattern -> [String] -> Pattern
solve m (['-', n] : xa@(x : xs)) =
  case m !? n of
    Just (Right _) -> solve (M.insert n (Right (Just x)) m) xs
    Just (Left _) -> solve (M.insert n (Left (Just ())) m) xa
    Nothing -> m
solve m [['-', n]] =
  case m !? n of
    Just (Left _) -> M.insert n (Left (Just ())) m
    _ -> m
solve m _ = m

getResult :: Pattern -> String
getResult m = fold $ do
  (k, a) <- M.toAscList m
  case a of
    Right (Just s) -> pure (' ' : '-' : k : ' ' : s)
    Left (Just _) -> pure (' ' : '-' : k : [])
    _ -> empty

main :: IO ()
main = do
  m <- parseRule mempty <$> getLine
  n <- read <$> getLine :: IO Int
  for_ [1..n] $ \i -> do
    command <- getLine
    let tokens = tail $ words command
    putStrLn $ "Case " <> show i <> (':' : getResult (solve m tokens))
```
:::

（2019&#8239;年&#8239;3&#8239;月&#8239;10&#8239;日）
