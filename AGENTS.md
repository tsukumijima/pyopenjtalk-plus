# pyopenjtalk-plus 開発ガイド

## プロジェクト概要

OpenJTalk の Python バインディング。Cython で C ライブラリをラップし、日本語テキストから音素・フルコンテキストラベルを生成する。  
r9y9/pyopenjtalk のフォークであり、アクセント推定の改善・踊り字対応・形態素-音素マッピング API 等を独自に追加している。  
形態素-音素マッピング API に関しては、Haqumei (Rust 再実装: https://github.com/o24s/haqumei) からインターフェイスや一部ロジックを改良した上で移植したあと、独自に多数のアライメントバグを修正している。

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
- `_mapping.py` — 形態素-音素マッピングと、NJD・MeCab 形態素のアライメントの実装。Haqumei の open_jtalk/mapping.rs に対応する。グローバルインスタンスの借り出しは `__init__.py` 側の公開ラッパーが担う
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

MeCab の `Lattice` 型・概念を指す本文と Docstring では、固有名として先頭を大文字の `Lattice` と表記する。変数名や C/C++ の識別子は実装どおり小文字を維持する。

`_run_mecab_detailed()` は `Mecab_analysis()` 後に MeCab の Lattice ノードを直接走査して未知語フラグ (`node.stat == MECAB_UNK_NODE`) やコスト情報を取得する。

元の Open JTalk の `Mecab_analysis()` は解析後に `lattice->clear()` を呼んでいたが、これだと Lattice ノードが解放されて走査できない。そのため C 側を修正し、`lattice->clear()` は `Mecab_refresh()` に移動してある (`lib/open_jtalk/src/mecab/src/mecab.cpp`)。`Mecab_refresh()` は Python 側の `try/finally` で確実に呼ばれる。

`Mecab_get_feature()` が返す feature 文字列は `strdup()` でコピー済みなので Lattice とは独立。一方、Lattice ノードの `surface` と `feature` ポインタは Lattice 内部メモリを指すため、`Mecab_refresh()` 後はアクセスできない。

### JPCommon の Word-Mora-Phoneme 階層

Cython 側の `make_phoneme_mapping()` は `JPCommon_make_label()` を呼ばずに、`JPCommonLabel_push_word()` を個別に呼び出して Word-Mora-Phoneme 階層を構築する。  
`JPCommon_make_label()` は HTS ラベル文字列を生成する重い処理であり、音素マッピングには不要。

各 `push_word` 呼び出しの前後で `label.word_tail` の変化を観察し、新しい Word が生成されたかを追跡する (ptr_to_idx マッピング)。  
ポーズ形態素 ("、"/"？"/"！") や長音吸収された 'ー' では Word が生成されないため、ptr_to_idx に含まれず音素が空のままになる。

### _run_njd_from_mecab の二重変換パターン

`_run_njd_from_mecab()` は NJD → Python dict → NJD → Python dict という二重変換を行う。  
一見冗長だが、`apply_original_rule_before_chaining()` が Python dict を直接操作してアクセント結合規則を変更するため (サ変動詞・接頭語・動詞連続等)、この構造が必要。  
NJD C 構造体を直接操作するのは危険なため、安全な Python 側で処理している。

### nani_predict の決定論的ガードの経緯

「何」の読み補正 (`predict_nani_reading()`) は、後続形態素が を/が/に/も/より/する の場合に常に「ナニ」と確定する決定論的ガード (`is_high_confidence_nani_context()`) を、ONNX モデルの判定より先に適用する。    
このガードはモデル本体の修正より先に入った経緯があり、冗長な二重処理として削除してはいけない。

背景: nani_model は scikit-learn の OneHotEncoder + RandomForestClassifier を skl2onnx で ONNX 化したもので、入力は直後形態素の6素性 (pos・pos_group・pron・ctype・cform) のみ。    
旧 skl2onnx の出力はクラス 1 の確率だけを葉に格納し、ONNX Runtime 1.25 以前は残りを 1 - score で暗黙補完していたが、この補完が ONNX Runtime 1.26 で削除された。    
同じ ONNX ファイルでもランタイムのバージョンで判定が変わる状態になり、誤判定が「何を」「何が」のような本来ナニと確定できる文脈で実データ上集中した。

対応は 2026/07/17 に2段階で行われた。まず決定論的ガードを追加して誤判定の集中した文脈をモデルから剥がし (`37c1483`)、続いて ONNX モデルを明示的な二クラス確率へ移行して (`de20553`、`scripts/migrate_nani_model_onnx.py`)、`predict()` は明示確率の argmax で判定する形へ修正した (回帰テストは `tests/test_nani_predict.py`)。

モデル修正後もガードを残置しているのは、モデルの誤判定が集中した文脈を確定的に処理でき、モデルの仕事を「何か」「何で」や文末のような真に曖昧な残余へ限定できるため。    
ナニ/ナンは後続形態素の音韻・文法でほぼ決まる局所的な現象であり、この「ルールで自明な多数を剥がし、モデルは曖昧な残余だけを見る」構造が高い精度の主要因として機能している。

### OpenJTalk 用辞書の品詞体系

pyopenjtalk-plus では、lib/open_jtalk/src/mecab-naist-jdic/ 以下の辞書ではなく、pyopenjtalk/dictionary/ 以下にある、独自に改良を重ねたデフォルト辞書を、日々誤りを見つけては修正しながらデフォルト辞書として使用している。  
lib/open_jtalk/src/mecab-naist-jdic/ はメンテナンスしておらず、OpenJTalk 1.11 時代からほとんど修正されていないため注意。

**デフォルト辞書を更新した際は、必ず `uv run task scripts/sort_dictionary_csv.py` で CSV をソートしたあと、`uv run task build-dictionary` で `sys.dic` ファイルの再ビルドが必要。**  
**`sys.dic` は Git 管理がギリギリなくらい巨大なバイナリブロブのため、ユーザーから明示的な指示がない限り絶対にステージ・コミットしてはならない。** `build-dictionary` を実行した事実や、コミットメッセージへの記載を理由に含めてはならない。  
ビルド済み `sys.dic` の Git 反映は、CSV の変更が FIX したリリース直前のタイミングだけに限る。

OpenJTalk は naist-jdic の品詞体系に依存している。  
一般的な MeCab 用辞書 (ipadic, unidic 等) を使うと品詞 ID や feature フォーマットが異なり、NJD 処理が誤動作またはクラッシュする。  
ユーザー辞書作成時も naist-jdic 互換のフォーマットが必須。

**`pyopenjtalk/dictionary/naist-jdic.csv` は行数が多く diff 管理が困難なため、原則として直接編集しない。**  
v0.4.1-post8 以降の naist-jdic.csv 向け修正は、すべて `scripts/modify_dictionary.py` 内の定数に集約する。  
`naist-jdic.csv` を手で直した場合は、必ず同内容を `modify_dictionary.py` 側へコード化してからコミットすること。手動 CSV コミットだけだと、次回スクリプト実行時に巻き戻る。

`scripts/modify_dictionary.py` の修正は、新規エントリ・読み・コスト・フィールド修正など既存の変更種別へ統合する。凍結済みの削除台帳は末尾に保ち、後日の修正を追加しない。

## 公開 API の設計方針

pyopenjtalk-plus は **精度改善を最優先** する fork である。r9y9/pyopenjtalk や OpenJTalk 本体との **出力等価性は保証しない**。本家と同じ挙動が必要な用途では pyopenjtalk 本家を使う。

公開 API に載せるのは、このリポジトリの利用者が実際に使う処理だけである。Haqumei にある、研究用である、だけでは採用理由にならない。

### ノブを設ける基準

公開 API に bool 引数 (ノブ) を追加するのは、**メリットとデメリットが両方あり、用途によって ON/OFF を選びたい処理** に限る。  
大体のケースで精度が上がり、採用しない理由がない改善は **既定 ON** とし、個別フラグは設けない。後処理をまとめて止めたい場合は `use_vanilla=True` を使う。

`use_vanilla=True` のとき無効化される主な処理 (NJD 数詞補正・異体字正規化・未知漢字読み・文脈読み・外来語仮名復元・アクセント句分割・踊り字展開等) は、個別トグルとして再公開しない。

`use_vanilla` の意味論は「**既定 ON の暗黙の自動後処理を一括 OFF にする。引数で明示的にオプトインした機能 (run_marine・iu_pronunciation・発音復元・use_tsqyomi) は vanilla より常に優先する**」である。  
明示指定まで無効化すると、指定した引数が黙って無視される罠になるため、この優先順位を崩してはならない。

### 公開フロントエンドで残すノブ (v0.4.1-post9 以降)

- `use_vanilla`: pyopenjtalk-plus 独自後処理の一括 OFF
- `use_tsqyomi` / `use_sudachi_kanji_yomi` / `predict_nani`: 読み選択経路の切替 (tsqyomi 使用時は Sudachi と nani モデルは自動 OFF)
- `run_marine`: アクセント推定 (重い・任意)
- `iu_pronunciation`: 「言う」系の発音方式 (方式によっては TTS に不向き)
- `use_read_as_pron` / `revert_long_vowels` / `revert_yotsugana`: 発音復元 (TTS 用途と相性が分かれる)
- `normalize_mode`: Unicode 正規化
- `is_non_pause_symbol`: 記号のポーズ判定の拡張

低レベル API (`OpenJTalk.run_frontend` 等) の `restore_unknown_katakana` / `modify_numeral_reading` は、分割実行向けの明示指定として公開フロントエンドとは別レイヤーで維持する。

`g2p_prosody` / `g2p_mapping_prosody` は Haqumei 互換の韻律 API として維持する。  
`format` の3形式 (`Default` = tdmelodic 風のピッチ変化記号、`Prefix` = `L_a` 形式、`Numeric` = `a:0` 形式) は、実装が出力整形の数行の分岐に閉じており、削って Haqumei 追従の摩擦を作るより残す方が単純なため、3形式とも維持する。

### 静的データの配置

辞書外の大規模読み表・異体字表は `pyopenjtalk/data/` のようなサブパッケージに置かず、`_known_symbols.py` と同型の **`pyopenjtalk/_foo.py` モジュール** に Python 定数として埋め込む (例: `_itaiji_map.py`, `_unihan_readings_map.py`)。  
ライセンス表記は生成 `.py` 先頭のコメントに簡潔に埋め込み、別ファイル `LICENSE` / `README.md` は置かない。Unihan 表は `scripts/generate_unihan_readings_map.py` で `_unihan_readings_map.py` を再生成する。

## Haqumei からの取り込み

[Haqumei](https://github.com/o24s/haqumei) の変更のうち、このリポジトリの利用者が実際に使う読み精度・辞書・アライメント・後処理は入れる。使わない機能は入れない。Haqumei にあることだけを理由に採用してはならない。

Kanalizer 連携、batch API、低水準ラティス API の公開は使わないので入れない。

異音分離も入れない。これは撥音「ン」や促音「ッ」を、後続の子音に応じて `Nm` / `Ng` / `clp` / `clt` のような別記号へ細分する処理で、同じ音素の調音上の変異をラベルに出す研究用 API である。Haqumei では `AllophoneOptions`、`use_allophones`、`split_n_allophones`、`resolve_allophones` として実装されている。本リポジトリの TTS / G2P は OpenJTalk の標準音素 (`N`, `cl`) のまま使う。公開引数にも載せない。歴史改変でも初手から載せない。

登坂車線は TTS 向けに長音を落とした「トハンシャセン」を維持する。Haqumei は「トーハンシャセン」である。この差は取り込みより前から `modify_dictionary.py` にある。著作権の発音「チョサッケン」は OpenJTalk 由来の既存値で、Haqumei の製品辞書だけが読みに合わせて「チョサクケン」へ変えている。plus が上書きした項目ではない。早急の cost 2000 と揃いの cost 5754 も pyopenjtalk-plus 側の既存値を残す。これらは `modify_dictionary.py` へ移さず、naist-jdic 側のエントリのままにする。

方向を表す「方」は、Sudachi が返す「ホウ」という読みを維持しつつ、OpenJTalk の発音は「ホー」とする。異体字の通用字体化と Sudachi による送り仮名語の読み補完は Haqumei にないが、こちらでは入れる (`_itaiji_map.py` / `normalize_itaiji` / 未知漢字読みの Sudachi 経路)。LOCAL_EXACT、tsqyomi、marine は pyopenjtalk-plus 固有なので、Haqumei へ逆方向には移植しない。

未知漢字は `use_vanilla=False` の既定で、Unihan の 1 文字読みと Sudachi の語単位読み (送り仮名語) を常に適用する。明示指定だけ有効にする中間状態は履歴に残さない。

ユーザー辞書の読み保護は、マーキングを入力テキスト上の文字位置 (`char_span`) で行い、`apply_postprocessing` での復元は形態素の添字のままにする。復元より前の後処理は NJD 形態素数を変えない (その場での書き換えだけ) ことが前提である。形態素を増減する後処理を足すときは、復元も文字位置へ切り替える。

読み保護が守るのは read / pron / mora_size と、ユーザー辞書に登録されたアクセント核 (acc) である。  
acc は marine とアクセント補正がすべて終わった後、形態素数が変わる踊り字展開より前に登録値へ復元し、個々の後処理側には保護の分岐を実装しない。  
アクセント結合 (chain_flag) は NJD が文脈に応じて決めるものなので、保護の対象にしない。

文脈読み規則 (`modify_context_reading`) へ規則や語彙を足すときは、まず辞書の生起コスト調整だけで解決できないかを実測する。  
文頭の「時が経つ」が接尾辞のジと誤読される問題は、後処理の規則ではなく「時 (名詞一般・トキ)」の生起コストを 8731 から 7500 へ下げるだけで解決した (7900 以下なら文頭でトキが選ばれ、「開催時」「15時」「梅雨時」の接尾・複合用法は連接コストの差で崩れない)。  
語彙集合へ足してよいのは、前後の特定の語によって読みが変わり、生起コストでは表現できない場合だけである。  
Haqumei より条件を意図的に絞った規則 (「寺」をジと読ませる前接語を実証済みの「霊山」だけに限る、「章」をショウと読ませる前接語を「記念」だけに限る等) には、絞った理由をコメントで書く。

naist-jdic 向けの修正は `scripts/modify_dictionary.py` を通す。読み候補の実測確認は audit スクリプトで行う。`heteronyms.csv` は手動編集で、`modify_dictionary.py` の対象外である。

tsqyomi 使用時に後処理から外すのは Sudachi と「何」モデルだけである。文脈で決まる読み補正まで止めてはならない。

Haqumei 由来の変更は、コミット本文の末尾に `ref: https://github.com/o24s/haqumei/commit/<fullhash>` を付けてよい。入れなかった差分や pyopenjtalk-plus 独自の拡張は、参照だけに頼らず通常の日本語で理由を書く。セッション内の語彙をコミット本文へ入れてはならない。

## __init__.py の関数配置

`pyopenjtalk/_mapping.py` の `make_phoneme_mapping()` を正例とする。

- **同一ファイルから 1 回しか呼ばれない** private ヘルパーは、**呼び出し元関数の直下**に関数内関数としてネストする
- 関数内関数は**呼び出しコールツリー順**に並べる（呼ばれる側を先、呼ぶ側を後）
- **2 箇所以上から呼ぶ**場合のみモジュールレベル `_foo` に置く（例: `_normalize_unknown_itaiji` は `run_frontend` と `run_frontend_detailed` の 2 呼び出し → モジュールレベル維持）
- **複数ファイルから import される**処理は `utils.py` 等の既存配置規約に従う（`utils/` 以外に関数だけのモジュールは作らない）
- 既存のモジュールレベル `_foo` を安易に増やさない。バックポート時の新規追加から適用し、触ったコミットのリライト時に既存違反も直す

`_get_njd_feature_char_spans` は `mark_user_dictionary_reading_protection` 内へネスト済み。  
`g2p_mapping` / `make_phoneme_mapping` 内の `_build_caller_text_spans_by_mecab_character` 等も正例。

### heteronyms.csv と naist-jdic.csv の役割分担

`heteronyms.csv` は tsqyomi 向けの同形異音語候補辞書であり、件数が少ないため**手動で編集する**。`modify_dictionary.py` の変更対象には含めない。

同形異音語関連のエントリは**すべて `heteronyms.csv` へ移す**。移行後は **`naist-jdic.csv` から当該読みの行をすべて削除する**（死にエントリとして残さない）。naist-jdic.csv は MeCab 既定ラティス用であり、同形異音語の候補一覧を載せる場所ではない。読み候補の維持・劣後管理は heteronyms.csv 側の責務とする。

`heteronyms.csv` 側では、移行後に劣後・撤回したい読みも**行ごと消さない**。生起コスト **10000** を基本の死にエントリとして残し、同 surface 内にそれ以上のコストを持つ競合候補がある場合は、それより高い値に上げて tsqyomi 候補から実質選ばれないようにする。  
2026/08/10 以降、かつて「死にエントリだから消すべきだ」と判断され行削除されてしまった候補を復元した。**今後は死にエントリであっても heteronyms.csv から削除しない（他の読み方も存在するが、ごく稀なためあえて主流な読みに完全に固定していることをコード上明確にするため）。**

### スレッド安全性

`OpenJTalk` クラスの公開メソッドは `@_lock_manager()` デコレータで排他制御されている。  
ロックは非リエントラントな `threading.Lock()` で、同一インスタンスへの同時アクセスを防ぐ。
`run_frontend()` と `run_frontend_detailed()` はそれぞれ独立して `@_lock_manager()` を持ち、互いに委譲しない。

かつては `run_frontend()` が `run_frontend_detailed()` に委譲する構造だったが (`0b23fce`)、後者だけで使う MeCab 形態素詳細の構築コストを `run_frontend()` の呼び出し元にまで負わせるため、Haqumei バックポート (`22d8cb0`) で独立した軽量経路として再実装された。
両者の重複を委譲へ戻す提案をする場合は、この経緯と `22d8cb0` の速度改善意図を先に確認すること。

グローバルインスタンスは `_global_jtalk()` コンテキストマネージャ経由でアクセスされる。`run_frontend()` / `run_frontend_detailed()` は `_resolve_jtalk()` でフロントエンド全体の借り出しを保持し、`update_global_jtalk_with_user_dict()` / `unset_user_dict()` の `replace()` は進行中の処理を待ってから同じマネージャー内のインスタンスを交換する。

多数のスレッドから同時に `g2p()` などを呼ぶ高並列用途では、グローバルインスタンス内のロックによって MeCab と NJD の処理が直列化される。その場合はスレッドごとに `OpenJTalk()` を明示生成し、`jtalk=` 引数で渡す。各インスタンスは独立した MeCab 辞書ハンドルを持つ。

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
- アクセント推定ライブラリの公式表記は、文頭や見出しを含めて小文字の `marine` で統一する
- pyx と pyi の Docstring は完全一致させるべき
- `__init__.py` 内の関数は直接参照する。関数のグローバル名前空間はモジュール辞書と同一なので、テストから公開 API を差し替える目的で自己参照エイリアスや `pyopenjtalk.` prefix を追加しない
- 辞書関連の Docstring では「MeCab ユーザー辞書」ではなく「OpenJTalk 用のユーザー辞書」と書く（naist-jdic 互換の品詞体系が必要なため）
- 同一の意味を持つ引数 (jtalk, text, njd_features 等) は全関数で Docstring の記述を統一する

### Docstring の句点 (Google Style)

- **概要** (Args より前): 文として句点 `。` を付ける。`test_*` 関数の Docstring は引数・戻り値・例外の節を設けず、概要文の末尾を句点 `。` で閉じる
- **Args / Returns / Raises / NOTE の各エントリ**: 1行の短い説明で足りるときは**行末に句点を付けない** (例: `text (str): Unicode 日本語テキスト`)
- **1エントリが複文になるとき** (続き行がある、または1行内で True/False 等を並べる): 文と文の**中間**には `。` を付ける。**そのエントリの最終行**だけ行末句点なし (例: `デフォルト: False`)
- フロントエンド系オプションの Args 文言は `g2p()` を正とし、`extract_fullcontext` / `run_frontend` 等と揃える
- 一括置換スクリプトは使わず、エントリ単位で目視確認する (pyx と pyi の Docstring は完全一致)

### コメントと Docstring の記述規範（文脈の解説・語彙圧縮の禁止）

このリポジトリは OpenJTalk C ライブラリの制約、MeCab の Lattice 挙動、独自の辞書コスト調整、tsqyomi 連携など、**非常に複雑でセンシティブな文脈**を多数抱えている。  
そのため、コードのコメントやテストコードの Docstring は、後から読む人間だけでなく将来のエージェントにとっても意図が即座に正しく伝わるよう、解像度高く自然な日本語で記述する。

1. **不自然な語彙圧縮・雑な言い回しの禁止**:
   - `「ヒトツキ」読みを維持する`、`キュウ読み`、`ダマ読み`、`フソク読み`、`アキラ読み`、`ブン読み` のように、カタカナ表記に助詞を介さず「読み」を直結させた破綻した日本語（不自然な語彙圧縮）は絶対に書かない。
   - `既定の「ヒトツキ」という読みを維持する`、`連濁して「ダマ」と読まれることを確認する`、`「アキラ」という読みのまま維持されることを確認する` のように、自然で論理的な日本語にする。
   - 後続のどの処理（アクセント句結合やポーズ判定等）でなぜその処理が必要なのか、設計上の意図・理由を明確に書く。

2. **テストコードにおける解説・Docstring の具体性**:
   - テストの Docstring やインラインコメントを、抽象的・官僚的な1行要約（例: `時間量の四分は、局地的な地名読みより優先する。`、`「に」を含む死に関連語の後で助詞の読みが重複しない。`）で済ませない。文脈が欠落し、何を守るためのテストなのかが分からなくなる。
   - 「**どのような入力に対して、どのような背景・理由から、どうなることを期待するテストなのか**」を具体的に解説する。
     - **悪い例**: `漢語・外来語に続く接尾辞の球は、生産的なキュウ読みを選ぶ。`
     - **良い例**: `漢語や外来語に接尾辞「球」が続く複合語（「ボール球」「樹脂球」など）では、「キュー」と発音されることを確認する。`
     - **悪い例**: `時間量の四分は、局地的な地名読みより優先する。`
     - **良い例**: `時間量の「四分」が、奈良県橿原市の局地的な地名（「シブ」）に誤爆せず、「ヨンプン」と読まれることを確認する。`

3. **読み・発音表記のカギ括弧囲みの徹底**:
   - コメントや Docstring 内で言及される単語の読みや発音（カナ表記・音素表記・記号等）は、地の文と混同しないよう必ずカギ括弧（「」）で囲む（例: `「ヒトツキ」`、`「ホウ」`、`「ホー」`、`「ヅ」/「ヂ」`、`「pau」`）。

4. **あえて厚く書かれた泥臭い経緯・文脈の保護（過剰な要約・脱臭の禁止）**:
   - `pyopenjtalk/tsqyomi/inference.py`（デフォルト辞書における正規化不備の経緯）や `AGENTS.md` の「コードから読み取りにくい重要なコンテキスト」のように、**あえて詳細に泥臭い経緯や試行錯誤・過去の障害対応が書かれているコメントを、勝手に「長いから」「整理する」といって要約・削除（脱臭）してはならない**。
   - 削るべきなのは「バイブコーディング由来の無責任なコメント」や「語彙圧縮による日本語の破綻」であり、設計意図や歴史的経緯の解像度を下げてはならない。

