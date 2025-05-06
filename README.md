# pyopenjtalk-plus

[![PyPI](https://img.shields.io/pypi/v/pyopenjtalk-plus.svg)](https://pypi.python.org/pypi/pyopenjtalk-plus)
[![Python package](https://github.com/tsukumijima/pyopenjtalk-plus/actions/workflows/ci.yml/badge.svg)](https://github.com/tsukumijima/pyopenjtalk-plus/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)

pyopenjtalk-plus は、各フォークでの改善を一つのコードベースにまとめ、さらなる改善を加えることを目的とした、[pyopenjtalk](https://github.com/r9y9/pyopenjtalk) の派生ライブラリです。

## Changes in this fork

- **パッケージ名を `pyopenjtalk-plus` に変更**
  - ライブラリ名は `pyopenjtalk` から変更されておらず、[pyopenjtalk](https://github.com/r9y9/pyopenjtalk) 本家同様に `import pyopenjtalk` でインポートできる
  - [pyopenjtalk](https://github.com/r9y9/pyopenjtalk) 本家のドロップイン代替として利用できる
- **明示的に Python 3.11 / 3.12 / 3.13 をサポート対象に追加**
  - CI 対象の Python バージョンも 3.11 以降メインに変更した
- **Windows・macOS (x64 / arm64)・Linux すべての事前ビルド済み wheels を PyPI に公開**
  - pyopenjtalk は hts_engine_API・OpenJTalk・Cython に依存しており、ビルド環境の構築難易度が比較的高い
    - 特に Windows においては MSVC のインストールが必要となる
  - 事前ビルド済みの wheels を PyPI に公開することで、ビルド環境のない PC でも簡単にインストール可能にすることを意図している
- **Python 側と Cython 側の両方に型ヒント (Type Hints) を追加**
  - Cython モジュールの型ヒントは [sabonerune/pyopenjtalk (enh/add-stub-files ブランチ)](https://github.com/sabonerune/pyopenjtalk/tree/enh/add-stub-files) での変更を一部改変の上で取り込んだもの
- **Cython を 3.0 系に更新**
  - https://github.com/cython/cython/issues/5982 の通り、Python 3.13 では一部の非推奨 C API が削除されている
  - Cython 0.x 系では Python 3.13 以降のビルドに失敗するため、Cython 3.0 系に更新した
- **numpy 2.x 系に対応**
  - numpy 2.x 系では互換性のない変更が多数行われているが、[公式ドキュメント](https://numpy.org/doc/stable/dev/depending_on_numpy.html#numpy-2-0-specific-advice) によると「numpy 2.x 系でビルドした wheel であれば numpy 1.x 系でも動作する」らしい
    - pyopenjtalk-plus では、numpy 2.x 系でビルドした wheel を公開することで対応した
  - ただし、marine (marine-plus) のコードは numpy 2.x 系との互換性がないため (OverflowError が発生する) 、marine と併用する際は引き続き numpy 1.x 系が必要となる
- **`pyopenjtalk.run_frontend()` 関数に CLI インターフェイスを追加**
  - コマンドライン引数としてテキストを受け取り、そのテキストを処理した結果を標準出力に出力する
  - 実行例: `python -m pyopenjtalk "あらゆる現実を、すべて自分の方へねじ曲げたのだ。"`
- **OpenJTalk 向けシステム辞書を、pyopenjtalk では初回実行時に自動ダウンロードされる [open_jtalk_dic_utf_8-1.11.tar.gz](https://github.com/r9y9/open_jtalk/releases/download/v1.11.1/open_jtalk_dic_utf_8-1.11.tar.gz) から、[独自にカスタマイズした pyopenjtalk-plus 向け辞書](pyopenjtalk/dictionary/) (wheel に同梱) に変更**
  - この辞書は [n5-suzuki/pyopenjtalk](https://github.com/n5-suzuki/pyopenjtalk) にかつて含まれていた mecab-naist-jdic のカスタム版をベースに、[jpreprocess/naist-jdic](https://github.com/jpreprocess/naist-jdic) での改良点を取り込み、さらに独自に改良したもの
  - [jpreprocess/naist-jdic](https://github.com/jpreprocess/naist-jdic) では mecab-naist-jdic に jpreprocess 向けの改良が施されており、辞書データのミスの修正など有用な変更が多かったことから、修正内容をほぼすべて適用した
  - pyopenjtalk 本家で実装されていた `_lazy_init()` 関数内での辞書ダウンロード処理は pyopenjtalk-plus での辞書同梱に伴い削除している
    - 辞書データがなければ pyopenjtalk は動作しないため (つまり辞書をダウンロードしない選択肢はなく必須) 、毎回追加でダウンロードするよりも wheel に直接含めた方が安定性の面でよりベターだと考えた
    - pyopenjtalk-plus の辞書データは 100MB 以上あるが (wheel 自体は圧縮が効いて 25MB 程度) 、せいぜい数十 MB のサイズ節約よりもアクセント・読み推定精度の向上を優先した
  - カスタム辞書は pyproject.toml のあるディレクトリで `task build-dictionary` を実行するとビルドできる
    - 管理の簡便化のため、ビルド済みの辞書データ (*.bin / *.dic) はこの Git リポジトリに含めている 
- **`pyopenjtalk.run_frontend()` や `pyopenjtalk.g2p()` でも `run_marine=True` を指定し [marine](https://github.com/6gsn/marine) によるアクセント推定を行えるようにした**
  - 以前から `pyopenjtalk.extract_fullcontext()` では marine による AI アクセント推定が可能だったが、`pyopenjtalk.run_frontend()` や `pyopenjtalk.g2p()` にも実装した
    - 「デフォルトの学習済みモデルは JSUT コーパスのみから学習されており、論文に記載されている性能とは異なる」(≒ marine 開発元の LINE 社内では独自の音声コーパスを用いてより高性能な学習済みモデルを作成・運用している) 旨が marine の README に記載されている
    - このためデフォルトの学習済みモデルを使う限り、現状 OpenJTalk のアクセント推定の方が精度が高い傾向にある
  - pyopenjtalk-plus では PyTorch への依存が発生することからデフォルトの依存関係には含めていないが、別途 marine / marine-plus をインストールすれば利用可能
  - marine 本家は Windows や Python 3.12 以降に非対応な上、非推奨警告が多数出力される問題があるため、これらの問題に対処した [marine-plus](https://github.com/tsukumijima/marine-plus) の利用を推奨します
    - `pip install marine-plus` で marine 本家の代わりに marine-plus をインストールできる
- **[litagin02/pyopenjtalk](https://github.com/litagin02/pyopenjtalk) での変更を取り込み、`pyopenjtalk.unset_user_dict()` 関数を追加**
  - VOICEVOX で利用されている [VOICEVOX/pyopenjtalk](https://github.com/VOICEVOX/pyopenjtalk) には、VOICEVOX ENGINE で利用するためのユーザー辞書機能が独自に実装されている
  - その後、pyopenjtalk v0.3.4 で VOICEVOX/pyopenjtalk と同等のユーザー辞書機能が実装された
    - VOICEVOX/pyopenjtalk の `set_user_dict()` 関数が `update_global_jtalk_with_user_dict()` 関数になるなど、同等の機能ながら関数名は変更されている
    - …が、どういう訳か VOICEVOX/pyopenjtalk には存在した「設定したユーザー辞書をリセットする」関数が実装されていない
  - このため litagin02/pyopenjtalk では VOICEVOX/pyopenjtalk から `pyopenjtalk.unset_user_dict()` 関数が移植されており、pyopenjtalk-plus でもこの実装を継承した
  - このほか、クロスプラットフォームで wheel をビルドするための GitHub Actions ワークフローもこのフォークから取り込んだもの
- **[VOICEVOX/pyopenjtalk](https://github.com/VOICEVOX/pyopenjtalk) での変更を取り込み、多数の改良点を反映**
  - [OpenJTalk の VOICEVOX 向けフォーク (VOICEVOX/open_jtalk)](https://github.com/VOICEVOX/open_jtalk) での変更内容を前提とした変更が多数含まれる
  - 取り込んだ変更点 (一部):
    - text2mecab() 関数を安全に改良し、エラー発生時に適切な RuntimeError を送出する
    - ARM 版 Windows でビルド可能にする
    - Windows で辞書の保存先パスに日本語を含むマルチバイト文字が含まれるとエラーが発生する問題を修正
    - 各環境でのビルドに関連する諸問題を修正
    - (OpenJTalk 側のみ) OpenJTalk 本体だけでユーザー辞書を読み込める `Mecab_load_with_userdic()` 関数を追加
    - (OpenJTalk 側のみ) 辞書のコンパイルに利用される `mecab-dict-index` モジュールにログ出力を抑制する `--quiet` オプションを追加
    - (OpenJTalk 側のみ) `mecab-dict-index` モジュールの `main()` 関数 (元は CLI コマンド用) をコメントアウト
      - OpenJTalk は MeCab のソースコードがベース、その MeCab 自体も非常にレガシーなソフトウェアで、お世辞にも綺麗なコードではない
      - このためか pyopenjtalk の辞書コンパイル機能は「CLI コマンド `mecab-dict-index` の argv と argc に相当する値を、ライブラリ側から OpenJTalk の `mecab_dict_index()` 関数 (`mecab-dict-index` コマンドのエントリーポイント) の引数として注入する」という非常にトリッキーかつ強引な手法で実装されている
      - どのみち pyopenjtalk 向け OpenJTalk では `mecab-dict-index` コマンドをビルドする必要がない
- **[n5-suzuki/pyopenjtalk](https://github.com/n5-suzuki/pyopenjtalk) での変更を取り込み、多数の改良点を反映**
  - 複数の読み方をする漢字の読みに対し [SudachiPy](https://github.com/WorksApplications/SudachiPy) で形態素解析を行い、得られた結果を使い OpenJTalk から返された `list[NJDFeature]` 内の値を補正する実装がユニーク
  - 「何」を「なん」と読むか「なに」と読むかを判定するための [scikit-learn で実装された機械学習モデルによるロジック](pyopenjtalk/yomi_model/nani_predict.py) に関して、学習済みモデルを ONNX に変換し、scikit-learn 0.24.2 への依存なしに動かせるよう改良した
    - 当該モデルは scikit-learn 0.24.2 でしか動作しないが、3年以上前にリリースされた極めて古いバージョンにつき Python 3.11 以降では動作せず、依存関係の問題もありインストール自体が困難になってきている
    - 学習用コードは含まれていなかったため推測するしかないが、モデルのバイナリに含まれる文字列から、RandomForestClassifier を用いた比較的単純な機械学習モデルだと推測される
    - [ONNX 変換ツール](pyopenjtalk/yomi_model/convert_onnx.py) を自作した上で ONNX に変換し、[推論コード](pyopenjtalk/yomi_model/nani_predict.py) も ONNXRuntime を用いて推論するよう変更した
      - この変更により依存関係に ONNXRuntime が追加されるが、すでに機械学習関連の他ライブラリの依存関係に含まれていることも多く、実用上問題ないと判断した
- **[korguchi/pyopenjtalk](https://github.com/korguchi/pyopenjtalk) での変更を取り込み、多数の改良点を反映**
  - このフォークで利用されている [korguchi/open_jtalk](https://github.com/korguchi/open_jtalk) では、「クァ」「グヮ」「デェ」「フュ」「シィ」などの比較的珍しい音素のサポートが追加されている
  - ほかにも「！」（感嘆符）を「記号/一般」として正しく推定するための改良など、概ね副作用なしに精度向上が見込めることから、有用性を鑑みほぼそのままマージした
  - この関係で、pyopenjtalk 本家とは一部音素での挙動が異なる
- **[sabonerune/pyopenjtalk](https://github.com/sabonerune/pyopenjtalk) での変更を取り込み、スレッドセーフ化と一部 Cython コードの nogil 化を達成**
  - スレッドセーフ化により、複数スレッドから安全に pyopenjtalk を呼び出せるようになった
  - 一部 Cython コードの nogil 化により、若干のパフォーマンス向上も見込める
  - https://github.com/r9y9/pyopenjtalk/pull/87 と https://github.com/r9y9/pyopenjtalk/pull/88 の内容を一部改変の上で取り込んだ
- **submodule の OpenJTalk を [tsukumijima/open_jtalk](https://github.com/tsukumijima/open_jtalk) に変更**
  - このフォークでは、pyopenjtalk-plus 向けに下記のフォーク版 OpenJTalk での改善内容を取り込んでいる
    - [VOICEVOX/open_jtalk](https://github.com/VOICEVOX/open_jtalk)
    - [a-ejiri/open_jtalk](https://github.com/a-ejiri/open_jtalk)
    - [sophiefy/open_jtalk](https://github.com/sophiefy/open_jtalk)
    - [korguchi/open_jtalk](https://github.com/korguchi/open_jtalk)
- **submodule の hts_engine_API を [syoyo/hts_engine_API](https://github.com/syoyo/hts_engine_API) に変更**
  - このフォークでは、https://github.com/r9y9/hts_engine_API/issues/9 に挙げられている問題が修正されている
- **ライブラリの開発環境構築・ビルド・コード整形・テストを `taskipy` によるタスクランナーでの管理に変更**
- **利用予定のない Travis CI 向けファイルを削除**
- **不要な依存関係の削除、依存バージョンの整理**
- **その他精度の改善・コードのクリーンアップ、非推奨警告の解消など**

## Installation

下記コマンドを実行して、ライブラリをインストールできます。

```bash
pip install pyopenjtalk-plus
```

## Development

開発環境は macOS / Linux 、Python バージョンは 3.11 が前提です。

```bash
# submodule ごとリポジトリを clone
git clone --recursive https://github.com/tsukumijima/pyopenjtalk-plus.git
cd pyopenjtalk-plus

# ライブラリ自身とその依存関係を .venv/ 以下の仮想環境にインストールし、開発環境を構築
pip install taskipy
task install

# コード整形
task lint
task format

# テストの実行
task test

# pyopenjtalk/dictionary/ 以下にある MeCab / OpenJTalk 辞書をビルド
## ビルド成果物は同ディレクトリに *.bin / *.dic として出力される
## ビルド後の辞書データは数百 MB あるバイナリファイルだが、取り回しやすいよう敢えて Git 管理下に含めている
task build-dictionary

# ライブラリの wheel と sdist をビルドし、dist/ に出力
task build

# ビルド成果物をクリーンアップ
task clean
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
