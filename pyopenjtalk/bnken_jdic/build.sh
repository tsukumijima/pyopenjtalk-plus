#!/bin/bash

# シェルスクリプトの実行時にエラーが発生したらスクリプトを終了する
set -Eeuo pipefail

# このスクリプトがあるディレクトリに移動
cd "$(dirname "$0")"

# 既存の辞書を削除
## 既存の辞書が存在しなくてもエラーにしない
rm -f *.dic

# 辞書をビルド (automake が必要)
aclocal
autoconf
automake --add-missing
./configure
make
