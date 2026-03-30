# pyopenjtalk-plus 開発ガイド

## プロジェクト概要

OpenJTalk の Python バインディング。Cython で C ライブラリをラップし、日本語テキストから音素・フルコンテキストラベルを生成する。
r9y9/pyopenjtalk のフォークであり、アクセント推定の改善・踊り字対応・形態素-音素マッピング API 等を独自に追加している。
形態素-音素マッピング API は Haqumei (Rust 再実装: https://github.com/stellanomia/haqumei) からインターフェイスや一部ロジックを改良した上で移植した。

## ビルド・テスト

```bash
# リンター（型チェック含む）・フォーマッター（変更後は必須）
uv run task lint
uv run task format

# Cython ビルド (pyx 変更後は必須)
uv run task build-ext

# ビルド済み成果物をクリーンアップ
uv run task clean

# デフォルト辞書のビルド
uv run task build-dictionary

# テスト実行
uv run task test

# 特定テストのみ実行
uv run pytest tests/test_openjtalk.py -k "test_g2p_mapping"
```

**重要: グローバルの python/pytest ではなく必ず `uv run python` を使うこと。**

## アーキテクチャ

### テキスト処理パイプライン

```
テキスト
  → text2mecab (テキスト正規化: 半角→全角変換等。半角スペースは全角スペースに変換される)
  → MeCab (形態素解析)
  → mecab2njd (MeCab feature → NJD ノード変換)
  → NJD 処理 (njd_set_pronunciation → njd_set_digit → njd_set_accent_phrase
              → njd_set_accent_type → njd_set_unvoiced_vowel → njd_set_long_vowel)
  → Python 側後処理 (apply_postprocessing: 踊り字展開・アクセント修正・漢字読み修正等)
  → njd2jpcommon (NJD → JPCommon ノード変換)
  → JPCommon_make_label (フルコンテキストラベル生成)
```

### ファイル構成

- `openjtalk.pyx` — Cython 実装。C ライブラリとの低レベルインターフェース
- `openjtalk.pyi` — 型スタブ。**pyx と Docstring を完全一致させること**
- `__init__.py` — Python 公開 API。後処理やアライメントロジックを含む
- `types.py` — TypedDict 定義 (NJDFeature, MecabMorph, SurfacePhonemeMapping)
- `utils.py` — 後処理関数 (踊り字展開、アクセント修正等)
- `htsengine.pyx` — HTS Engine のバインディング。2026 年現在ではもっぱらテキスト処理ライブラリとして使われているため、積極的にメンテナンスされていない
- `lib/open_jtalk/` — Open JTalk C ライブラリ (submodule)

## コードから読み取りにくい重要なコンテキスト

### 記号,空白 フィルタの経緯

`run_mecab()` は MeCab の結果から "記号,空白" を含む feature を除外してから NJD に渡す。
Open JTalk の C 実装にはこのフィルタは存在せず、pyopenjtalk-plus 独自の処理。

理由: `text2mecab` が半角スペースを全角スペースに変換し、MeCab がそれを "記号,空白" としてトークン化する。
このトークンが NJD を通ると `pron=、` に変換され、JPCommon が `pau` としてラベルに挿入してしまう。
フィルタにより、スペースは TTS 出力に影響を与えず黙って無視される。

`run_mecab_detailed()` はアライメント用に全トークンを返す必要があるため、フィルタしない。
代わりに `is_ignored=True` フラグで "記号,空白" を識別し、アライメント時に `sp` として出力する。

### NJD 処理がトークンの数と surface を変える

NJD の各処理ステップは MeCab の元の形態素と 1:1 対応しない変換を行う。
これが `make_phoneme_mapping` のアライメントロジックを複雑にしている根本原因。

- `njd_set_digit`: 数字の漢字変換。"１２３" → "百","二","十","三" のように桁の漢字を**挿入**する (NJD ノード増加)。
  "１０" → "十" のように trailing 0 を**吸収**する (NJD ノード減少)。"７" → "七" のように surface を**変更**する (数は変わらない)。
- `njd_set_long_vowel`: 長音 'ー' を先行 Word のモーラとして吸収する (JPCommon Word 減少)。
  これは Cython 側の `make_phoneme_mapping` で処理済みのため、Python 側では考慮不要。
- `process_odori_features` (Python 側後処理): 踊り字展開。"学生","々","活" → "学生","生活" のように
  MeCab morph の粒度と NJD feature の粒度がずれる (morph が余る)。

### Lattice ノード走査とメモリライフタイム

`_run_mecab_detailed()` は `Mecab_analysis()` 後に MeCab の lattice ノードを直接走査して
未知語フラグ (`node.stat == MECAB_UNK_NODE`) やコスト情報を取得する。

元の Open JTalk の `Mecab_analysis()` は解析後に `lattice->clear()` を呼んでいたが、
これだと lattice ノードが解放されて走査できない。
そのため C 側を修正し、`lattice->clear()` は `Mecab_refresh()` に移動してある (`lib/open_jtalk/src/mecab/src/mecab.cpp`)。
`Mecab_refresh()` は Python 側の `try/finally` で確実に呼ばれる。

`Mecab_get_feature()` が返す feature 文字列は `strdup()` でコピー済みなので lattice とは独立。
一方、lattice ノードの `surface` と `feature` ポインタは lattice 内部メモリを指すため、
`Mecab_refresh()` 後はアクセスできない。

### JPCommon の Word-Mora-Phoneme 階層

Cython 側の `make_phoneme_mapping()` は `JPCommon_make_label()` を呼ばずに、
`JPCommonLabel_push_word()` を個別に呼び出して Word-Mora-Phoneme 階層を構築する。
`JPCommon_make_label()` は HTS ラベル文字列を生成する重い処理であり、音素マッピングには不要。

各 `push_word` 呼び出しの前後で `label.word_tail` の変化を観察し、
新しい Word が生成されたかを追跡する (ptr_to_idx マッピング)。
ポーズ形態素 ("、"/"？"/"！") や長音吸収された 'ー' では Word が生成されないため、
ptr_to_idx に含まれず音素が空のままになる。

### _run_njd_from_mecab の二重変換パターン

`_run_njd_from_mecab()` は NJD → Python dict → NJD → Python dict という二重変換を行う。
一見冗長だが、`apply_original_rule_before_chaining()` が Python dict を直接操作して
アクセント結合規則を変更するため (サ変動詞・接頭語・動詞連続等)、この構造が必要。
NJD C 構造体を直接操作するのは危険なため、安全な Python 側で処理している。

### OpenJTalk 用辞書の品詞体系

OpenJTalk は naist-jdic の品詞体系に依存している。
一般的な MeCab 用辞書 (ipadic, unidic 等) を使うと品詞 ID や feature フォーマットが異なり、
NJD 処理が誤動作またはクラッシュする。ユーザー辞書作成時も naist-jdic 互換のフォーマットが必須。

### スレッド安全性

`OpenJTalk` クラスの公開メソッドは `@_lock_manager()` デコレータで排他制御されている。
ロックは非リエントラントな `threading.Lock()` で、同一インスタンスへの同時アクセスを防ぐ。
`run_frontend()` は `run_frontend_detailed()` に委譲するためロックを取らない
（二重ロックを避けるため）。

グローバルインスタンスは `_global_jtalk()` コンテキストマネージャ経由でアクセスされる。

## make_phoneme_mapping() のアライメントロジック

`__init__.py` の `make_phoneme_mapping(njd_features, morphs=)` は最もセンシティブな処理。
base_mapping (Cython 側の NJD ベース音素マッピング) と morphs (MeCab 形態素) を突合する。

### NJD 処理による morph/base のずれ

| パターン | 原因 | morph vs base | 対処 |
|---------|------|---------------|------|
| 踊り字展開 | process_odori_features | morph > base | morph を 2 つ消費 |
| 数字展開 | njd_set_digit (桁の漢字挿入) | morph < base | morph を消費しない |
| 数字縮約 | njd_set_digit (trailing 0 吸収) | morph > base | 連続 digit morph を追加消費 |
| surface 変化 | njd_set_digit (７→七) | morph = base | morph を 1 つ消費 |
| 長音吸収 | njd_set_long_vowel | Cython 側で処理済み | base_mapping でマージ済み |

### 判定ロジック

1. **踊り字判定**: morph に踊り字文字 (々ゝゞヽヾ) が含まれるか
2. **バランスチェック**: 残りの非 ignored morph 数 ≤ 残りの base 数なら NJD 挿入
3. **数字縮約**: advance 後に連続する全角数字 morph をバランスが取れるまで追加消費

## Cython 開発の注意点

- `.pxd` で `cdef extern from` 内のフィールドを省略しても、C コンパイラが正しいオフセットを計算する
- `cdef` 宣言は関数スコープの先頭に置く必要がある
- C リソースは必ず `try/finally` で `*_refresh()` を呼んで解放する
- `with nogil:` ブロック内では Python オブジェクトにアクセスできない
- MeCab の `surface` は null 終端ではないので `[:node.length]` でスライスする
- MeCab の `feature` は null 終端なので `<bytes>` キャストで読める

## コーディング規約

- 文字列リテラルはダブルクォートで統一（ruff format は pyx に効かないので手動で統一する）
- pyx と pyi の Docstring は完全一致させるべき
- `__init__.py` 内で他の関数を参照する場合は `pyopenjtalk.` prefix を付け、`OpenJTalk` クラスのメソッドと明確に区別すべき
- 辞書関連の Docstring では「MeCab ユーザー辞書」ではなく「OpenJTalk 用のユーザー辞書」と書く
  （naist-jdic 互換の品詞体系が必要なため）
- 同一の意味を持つ引数 (jtalk, text, njd_features 等) は全関数で Docstring の記述を統一する
