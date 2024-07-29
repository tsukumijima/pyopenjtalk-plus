# pyopenjtalk-plus

[![PyPI](https://img.shields.io/pypi/v/pyopenjtalk-plus.svg)](https://pypi.python.org/pypi/pyopenjtalk-plus)
[![Python package](https://github.com/tsukumijima/pyopenjtalk-plus/actions/workflows/ci.yml/badge.svg)](https://github.com/tsukumijima/pyopenjtalk-plus/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)

pyopenjtalk-plus は、各フォークでの改善を一つのコードベースにまとめ、さらなる改善を加えることを目的とした、[pyopenjtalk](https://github.com/r9y9/pyopenjtalk) の派生ライブラリです。

## Changes in this fork

- パッケージ名を `pyopenjtalk-plus` に変更
  - ライブラリ名は `pyopenjtalk` から変更されておらず、[pyopenjtalk](https://github.com/r9y9/pyopenjtalk) 本家同様に `import pyopenjtalk` でインポートできる
  - [pyopenjtalk](https://github.com/r9y9/pyopenjtalk) 本家のドロップイン代替として利用できる
- 明示的に Python 3.11 / 3.12 をサポート対象に追加
  - CI 対象の Python バージョンも 3.11 / 3.12 メインに変更した
- Windows・Mac (x64 / arm64)・Linux すべての事前ビルド済み wheels を PyPI に公開
  - pyopenjtalk は hts_engine_API・OpenJTalk・Cython に依存しており、ビルド環境の構築難易度が比較的高い
    - 特に Windows においては MSVC のインストールが必要となる
  - 事前ビルド済みの wheels を PyPI に公開することで、簡単にインストールできるようにした
- 依存関係の numpy を 1.x 系に固定
  - numpy 2.x では互換性のない変更が多数行われており、もとよりレガシーな設計である現行の pyopenjtalk(-plus) では動作しないと考えられるため
- [litagin02/pyopenjtalk](https://github.com/litagin02/pyopenjtalk) での変更を取り込み、`pyopenjtalk.unset_user_dict()` 関数を追加
  - VOICEVOX で利用されている [VOICEVOX/pyopenjtalk](https://github.com/VOICEVOX/pyopenjtalk) には、VOICEVOX ENGINE で利用するためのユーザー辞書機能が独自に追加されている
  - その後 pyopenjtalk v0.3.5 で同等のユーザー辞書機能が実装された
    - VOICEVOX/pyopenjtalk の `set_user_dict()` 関数が `update_global_jtalk_with_user_dict()` 関数になるなど、同等の機能ながら関数名は変更されている
    - …が、どういう訳か VOICEVOX/pyopenjtalk には存在した「設定したユーザー辞書をリセットする」関数が実装されていない
  - このため litagin02/pyopenjtalk では VOICEVOX/pyopenjtalk から `pyopenjtalk.unset_user_dict()` 関数が移植されており、pyopenjtalk-plus でもこの実装を継承した
- [VOICEVOX/pyopenjtalk](https://github.com/VOICEVOX/pyopenjtalk) での変更を取り込み
  - [VOICEVOX/open_jtalk](https://github.com/VOICEVOX/open_jtalk) での改変点を前提とした変更が多数含まれる
- submodule の OpenJTalk を [tsukumijima/open_jtalk](https://github.com/tsukumijima/open_jtalk) に変更
  - このフォークでは、この pyopenjtalk-plus 向けに [VOICEVOX/open_jtalk](https://github.com/VOICEVOX/open_jtalk) から取り込んだ多数の有用な変更が加えられている
- submodule の hts_engine_API を [syoyo/hts_engine_API](https://github.com/syoyo/hts_engine_API) に変更
  - このフォークでは、https://github.com/r9y9/hts_engine_API/issues/9 に挙げられている問題が修正されている
- ライブラリの開発環境構築・ビルド・コード整形・テストを `taskipy` によるタスクランナーで管理
- 利用予定のない Travis CI 向けファイルを削除

## Installation (TODO)

```bash
pip install pyopenjtalk-plus
```

## Development

開発環境は macOS / Linux 、Python は 3.11 が前提です。

```bash
# submodule ごとリポジトリを clone
git clone --recursive https://github.com/tsukumijima/pyopenjtalk-plus.git

# ライブラリ自身とその依存関係を .venv/ 以下の仮想環境にインストールし、開発環境を構築
pip install taskipy
task install

# ライブラリの wheel と sdist をビルドし、dist/ に出力
task build

# コード整形
task lint
task format

# テストの実行
task test
```

下記ならびに [docs/](docs/) 以下のドキュメントは、[pyopenjtalk](https://github.com/r9y9/pyopenjtalk) 本家のドキュメントを改変なしでそのまま引き継いでいます。  
これらのドキュメントの内容が pyopenjtalk-plus にも通用するかは保証されません。

-------

# pyopenjtalk

[![PyPI](https://img.shields.io/pypi/v/pyopenjtalk.svg)](https://pypi.python.org/pypi/pyopenjtalk)
[![Python package](https://github.com/r9y9/pyopenjtalk/actions/workflows/ci.yaml/badge.svg)](https://github.com/r9y9/pyopenjtalk/actions/workflows/ci.yaml)
[![Build Status](https://app.travis-ci.com/r9y9/pyopenjtalk.svg?branch=master)](https://app.travis-ci.com/r9y9/pyopenjtalk)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![DOI](https://zenodo.org/badge/143748865.svg)](https://zenodo.org/badge/latestdoi/143748865)

A python wrapper for [OpenJTalk](http://open-jtalk.sp.nitech.ac.jp/).

The package consists of two core components:

- Text processing frontend based on OpenJTalk
- Speech synthesis backend using HTSEngine

## Notice

- The package is built with the [modified version of OpenJTalk](https://github.com/r9y9/open_jtalk). The modified version provides the same functionality with some improvements (e.g., cmake support) but is technically different from the one from HTS working group.
- The package also uses the [modified version of hts_engine_API](https://github.com/r9y9/hts_engine_API). The same applies as above.

Before using the pyopenjtalk package, please have a look at the LICENSE for the two software.

## Build requirements

The python package relies on cython to make python bindings for open_jtalk and hts_engine_API. You must need the following tools to build and install pyopenjtalk:

- C/C++ compilers (to build C/C++ extentions)
- cmake
- cython

## Supported platforms

- Linux
- Mac OSX
- Windows (MSVC) (see [this PR](https://github.com/r9y9/pyopenjtalk/pull/13))

## Installation

```
pip install pyopenjtalk
```

## Development

To build the package locally, you will need to make sure to clone open_jtalk and hts_engine_API.

```
git submodule update --recursive --init
```

and then run

```
pip install -e .
```

## Quick demo

Please check the notebook version [here (nbviewer)](https://nbviewer.jupyter.org/github/r9y9/pyopenjtalk/blob/master/docs/notebooks/Demo.ipynb).

### TTS

```py
In [1]: import pyopenjtalk

In [2]: from scipy.io import wavfile

In [3]: x, sr = pyopenjtalk.tts("おめでとうございます")

In [4]: wavfile.write("test.wav", sr, x.astype(np.int16))
```

### Run text processing frontend only

```py
In [1]: import pyopenjtalk

In [2]: pyopenjtalk.extract_fullcontext("こんにちは")
Out[2]:
['xx^xx-sil+k=o/A:xx+xx+xx/B:xx-xx_xx/C:xx_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:xx_xx#xx_xx@xx_xx|xx_xx/G:5_5%0_xx_xx/H:xx_xx/I:xx-xx@xx+xx&xx-xx|xx+xx/J:1_5/K:1+1-5',
'xx^sil-k+o=N/A:-4+1+5/B:xx-xx_xx/C:09_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:5_5#0_xx@1_1|1_5/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-5@1+1&1-1|1+5/J:xx_xx/K:1+1-5',
'sil^k-o+N=n/A:-4+1+5/B:xx-xx_xx/C:09_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:5_5#0_xx@1_1|1_5/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-5@1+1&1-1|1+5/J:xx_xx/K:1+1-5',
'k^o-N+n=i/A:-3+2+4/B:xx-xx_xx/C:09_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:5_5#0_xx@1_1|1_5/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-5@1+1&1-1|1+5/J:xx_xx/K:1+1-5',
'o^N-n+i=ch/A:-2+3+3/B:xx-xx_xx/C:09_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:5_5#0_xx@1_1|1_5/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-5@1+1&1-1|1+5/J:xx_xx/K:1+1-5',
'N^n-i+ch=i/A:-2+3+3/B:xx-xx_xx/C:09_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:5_5#0_xx@1_1|1_5/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-5@1+1&1-1|1+5/J:xx_xx/K:1+1-5',
'n^i-ch+i=w/A:-1+4+2/B:xx-xx_xx/C:09_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:5_5#0_xx@1_1|1_5/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-5@1+1&1-1|1+5/J:xx_xx/K:1+1-5',
'i^ch-i+w=a/A:-1+4+2/B:xx-xx_xx/C:09_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:5_5#0_xx@1_1|1_5/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-5@1+1&1-1|1+5/J:xx_xx/K:1+1-5',
'ch^i-w+a=sil/A:0+5+1/B:xx-xx_xx/C:09_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:5_5#0_xx@1_1|1_5/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-5@1+1&1-1|1+5/J:xx_xx/K:1+1-5',
'i^w-a+sil=xx/A:0+5+1/B:xx-xx_xx/C:09_xx+xx/D:xx+xx_xx/E:xx_xx!xx_xx-xx/F:5_5#0_xx@1_1|1_5/G:xx_xx%xx_xx_xx/H:xx_xx/I:1-5@1+1&1-1|1+5/J:xx_xx/K:1+1-5',
'w^a-sil+xx=xx/A:xx+xx+xx/B:xx-xx_xx/C:xx_xx+xx/D:xx+xx_xx/E:5_5!0_xx-xx/F:xx_xx#xx_xx@xx_xx|xx_xx/G:xx_xx%xx_xx_xx/H:1_5/I:xx-xx@xx+xx&xx-xx|xx+xx/J:xx_xx/K:1+1-5']
```

Please check `lab_format.pdf` in [HTS-demo_NIT-ATR503-M001.tar.bz2](http://hts.sp.nitech.ac.jp/archives/2.3/HTS-demo_NIT-ATR503-M001.tar.bz2) for more details about full-context labels.


### Grapheme-to-phoeneme (G2P)

```py
In [1]: import pyopenjtalk

In [2]: pyopenjtalk.g2p("こんにちは")
Out[2]: 'k o N n i ch i w a'

In [3]: pyopenjtalk.g2p("こんにちは", kana=True)
Out[3]: 'コンニチワ'
```

### Create/Apply user dictionary

1. Create a CSV file (e.g. `user.csv`) and write custom words like below:

```csv
ＧＮＵ,,,1,名詞,一般,*,*,*,*,ＧＮＵ,グヌー,グヌー,2/3,*
```

2. Call `mecab_dict_index` to compile the CSV file.

```python
In [1]: import pyopenjtalk

In [2]: pyopenjtalk.mecab_dict_index("user.csv", "user.dic")
reading user.csv ... 1
emitting double-array: 100% |###########################################|

done!
```

3. Call `update_global_jtalk_with_user_dict` to apply the user dictionary.

```python
In [3]: pyopenjtalk.g2p("GNU")
Out[3]: 'j i i e n u y u u'

In [4]: pyopenjtalk.update_global_jtalk_with_user_dict("user.dic")

In [5]: pyopenjtalk.g2p("GNU")
Out[5]: 'g u n u u'
```

### About `run_marine` option

After v0.3.0, the `run_marine` option has been available for estimating the Japanese accent with the DNN-based method (see [marine](https://github.com/6gsn/marine)). If you want to use the feature, please install pyopenjtalk as below;

```shell
pip install pyopenjtalk[marine]
```

And then, you can use the option as the following examples;

```python
In [1]: import pyopenjtalk

In [2]: x, sr = pyopenjtalk.tts("おめでとうございます", run_marine=True) # for TTS

In [3]: label = pyopenjtalk.extract_fullcontext("こんにちは", run_marine=True) # for text processing frontend only
```


## LICENSE

- pyopenjtalk: MIT license ([LICENSE.md](LICENSE.md))
- Open JTalk: Modified BSD license ([COPYING](https://github.com/r9y9/open_jtalk/blob/1.10/src/COPYING))
- htsvoice in this repository: Please check [pyopenjtalk/htsvoice/README.md](pyopenjtalk/htsvoice/README.md).
- marine: Apache 2.0 license ([LICENSE](https://github.com/6gsn/marine/blob/main/LICENSE))

## Acknowledgements

HTS Working Group for their dedicated efforts to develop and maintain Open JTalk.
