#!/bin/sh
cd $1
pandoc --fail-if-warnings --metadata-file=`git rev-parse --show-toplevel`/pandoc.yaml --listings -s index.md -o index.tex
sed -r -i 's/\\usepackage\{(polyglossia|bidi)\}//g' index.tex
sed -r -i 's/\\set(main|other)language\[.*\]\{.*\}//g' index.tex
sed -i 's/\([[:digit:]]\+\)\\,年\\,\([[:digit:]]\+\)\\,月\\,\([[:digit:]]\+\)\\,日/\1 年 \2 月 \3 日/' index.tex
xelatex -interaction=errorstopmode -halt-on-error --shell-escape index.tex
xelatex -interaction=errorstopmode -halt-on-error --shell-escape index.tex
