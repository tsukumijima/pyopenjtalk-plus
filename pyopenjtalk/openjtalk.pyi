# flake8: noqa

from collections.abc import Sequence
from typing import Any, Iterable

from .types import MeCabMorph, MeCabNBestPath, NJDFeature
from .tsqyomi.types import ReadingAnalysis

class OpenJTalk:
    def __init__(
        self,
        dn_mecab: bytes = b"/usr/local/dic",
        userdic: bytes = b"",
        userdic_reading_protection: Sequence[bool] | None = None,
    ) -> None:
        """
        OpenJTalk のテキスト処理フロントエンドの Cython 実装。
        通常は pyopenjtalk モジュール経由で使用するが、低レベル API として直接インスタンス化も可能。

        Args:
            dn_mecab (bytes): MeCab システム辞書のディレクトリパス
            userdic (bytes): OpenJTalk 用ユーザー辞書 (.dic) のパス。空バイト列の場合は無視される。複数指定時はカンマ区切り。デフォルト: 空
            userdic_reading_protection (Sequence[bool] | None): 各ユーザー辞書の読み候補を tsqyomi による MeCab feature 差し替えから保護するか
                None の場合は全辞書を未保護として扱う。デフォルト: None

        Raises:
            ValueError: `userdic_reading_protection` の要素数が辞書数と一致しない場合
            TypeError: 保護フラグに bool 以外が含まれる場合
            RuntimeError: MeCab 初期化に失敗した場合

        NOTE:
            公開メソッドは `@_lock_manager()` で直列化される。`Mecab` / `NJD` / `JPCommon` はインスタンス内で共有される。
            `Mecab_refresh()` は Python 側の `try/finally` から呼び出し、lattice ノード走査後に MeCab 内部状態を解放する。
        """
        pass

    def normalize_for_mecab(self, text: str | bytes | bytearray) -> str:
        """
        OpenJTalk の MeCab 入力と同じ規則で本文を正規化する。

        Args:
            text (str | bytes | bytearray): 入力テキスト (str の場合は UTF-8 にエンコードされる)

        Returns:
            str: 正規化されたテキスト
        """
        pass

    def run_mecab(self, text: str | bytes | bytearray) -> list[str]:
        """
        MeCab で形態素解析を実行する。"記号,空白" は除外される。
        全トークン (未知語フラグ・コスト情報含む) が必要な場合は代わりに run_mecab_detailed() を使うこと。

        Args:
            text (str | bytes | bytearray): 入力テキスト (str の場合は UTF-8 にエンコードされる)

        Returns:
            list[str]: MeCab の feature 文字列のリスト ("記号,空白" を除く)
        """
        pass

    def run_mecab_detailed(
        self, text: str | bytes | bytearray
    ) -> tuple[list[str], list[MeCabMorph]]:
        """
        MeCab を1回だけ実行し、run_mecab() 互換の features と詳細 morphs を返す。
        詳細 morphs には "記号,空白" も含まれ、未知語フラグ・コスト情報を保持する。

        Args:
            text (str | bytes | bytearray): 入力テキスト (str の場合は UTF-8 にエンコードされる)

        Returns:
            tuple[list[str], list[MeCabMorph]]: (フィルタ済み features, 全 morphs)
                features は run_mecab() と同等 ("記号,空白" を除く)
                morphs は lattice 走査で構築した詳細形態素列 ("記号,空白" も含む)

        NOTE:
            `Mecab_analysis()` 後に lattice ノードを走査し、未知語フラグ・コスト・文字位置を取得する。
            未知語に連結された連続記号は、既知記号辞書を使って1文字ずつ morph へ分割する。
        """
        pass

    def run_mecab_nbest_features(
        self, text: str | bytes | bytearray, max_paths: int = 5
    ) -> list[MeCabNBestPath]:
        """
        MeCab の n-best 候補を features / morphs / path_cost 付きで返す。
        features は run_njd_from_mecab() に渡せる形式で、morphs は run_mecab_detailed()[1] と同じ詳細形式を持つ。

        Args:
            text (str | bytes | bytearray): 入力テキスト (str の場合は UTF-8 にエンコードされる)
            max_paths (int): 取得する最大候補数 (MeCab の上限に合わせて 1-512 を受け付ける)

        Returns:
            list[MeCabNBestPath]: MeCab n-best 候補パスのリスト
        """
        pass

    def analyze_mecab_candidates(
        self,
        text: str | bytes | bytearray,
        target_spans: Sequence[tuple[int, int]],
    ) -> ReadingAnalysis:
        """
        MeCab の補正前最良経路と全候補ノードをコピーして返す。
        戻り値は MeCab の生ポインタを含まず、呼び出し完了後にモデル推論へ安全に渡せる。

        Args:
            text (str | bytes | bytearray): 入力テキスト (str の場合は UTF-8 にエンコードされる)
            target_spans (Sequence[tuple[int, int]]): 候補経路を列挙する正規化本文上の半開区間

        Returns:
            ReadingAnalysis: 正規化本文、最良経路、候補グラフのコピー

        NOTE:
            MeCab を NBEST モードで解析し、候補ノード間の接続辺を取得する。コスト変更や最良経路の再計算は行わない。
            戻り値は lattice ノードの Python コピーのみで、呼び出し完了後は `Mecab_refresh()` で C 側 lattice を解放する。
            tsqyomi はこの戻り値をロック外でモデル推論へ渡せる。
        """
        pass

    def run_njd_from_mecab(self, mecab_features: list[str]) -> list[NJDFeature]:
        """
        MeCab の feature 文字列のリストから NJD 処理を実行する。
        run_mecab() の戻り値をそのまま渡す想定。
        数字正規化・アクセント句設定・長音処理などの NJD ルールが適用される。

        Args:
            mecab_features (list[str]): MeCab の feature 文字列のリスト

        Returns:
            list[NJDFeature]: NJDNode 用 features
        """
        pass

    def run_frontend(self, text: str | bytes | bytearray) -> list[NJDFeature]:
        """
        OpenJTalk のテキスト処理フロントエンドを実行する。
        MeCab 形態素詳細を構築せず、NJD features のみを返す軽量経路。

        Args:
            text (str | bytes | bytearray): 入力テキスト (str の場合は UTF-8 にエンコードされる)

        Returns:
            list[NJDFeature]: NJDNode 用 features
        """
        pass

    def run_frontend_detailed(
        self, text: str | bytes | bytearray
    ) -> tuple[list[NJDFeature], list[MeCabMorph]]:
        """
        OpenJTalk のテキスト処理フロントエンドを MeCab 形態素詳細付きで実行する。
        MeCab 解析を 1 回だけ実行し、NJD features と MeCab morphs を同時に返す。

        Args:
            text (str | bytes | bytearray): 入力テキスト (str の場合は UTF-8 にエンコードされる)

        Returns:
            tuple[list[NJDFeature], list[MeCabMorph]]: (NJD features, MeCab morphs)
                NJD features は run_frontend() と、MeCab morphs は run_mecab_detailed() と同一の結果
        """
        pass

    def extract_phonemes(self, features: Iterable[NJDFeature]) -> list[str]:
        """
        NJD features からフラットな音素列を直接抽出する。
        HTS フルコンテキストラベル文字列は生成せず、JPCommonLabel の音素連結リストをそのまま走査する。

        Args:
            features (Iterable[NJDFeature]): NJDNode 用 features (run_frontend() の戻り値)

        Returns:
            list[str]: フラットな音素列

        NOTE:
            `try/finally` で `JPCommon_refresh()` と `NJD_refresh()` を呼び、インスタンス共有バッファを解放する。
        """
        pass

    def make_label(self, features: Iterable[NJDFeature]) -> list[str]:
        """
        HTS 音声合成用のフルコンテキストラベルを返す。

        Args:
            features (Iterable[NJDFeature]): NJDNode 用 features (run_frontend() の戻り値)

        Returns:
            list[str]: フルコンテキストラベル文字列のリスト

        NOTE:
            `try/finally` で `JPCommon_refresh()` と `NJD_refresh()` を呼び、ラベル文字列と中間バッファを解放する。
        """
        pass

    def make_phoneme_mapping(self, features: Iterable[NJDFeature]) -> list[dict[str, Any]]:
        """
        NJD features から各形態素に対応する音素列のマッピングを生成する。
        JPCommon の Word-Mora-Phoneme 階層を構築し、各 feature に音素を割り当てる。
        NJD の pron が短ポーズを表す記号へ ["pau"] を割り当て、括弧類は空の音素列で保持する。
        長音吸収マージにより、戻り値の長さが入力と異なる場合がある。

        Args:
            features (Iterable[NJDFeature]): NJDNode 用 features (run_frontend() の戻り値)

        Returns:
            list[dict[str, Any]]: NJDFeature の全フィールド + phonemes を含む辞書のリスト。
                MeCab の未知語情報や features が必要な場合は pyopenjtalk.make_phoneme_mapping() を使用すること

        Raises:
            RuntimeError: JPCommonLabel の内部アロケーション失敗時

        NOTE:
            `JPCommon_make_label()` は呼ばず、`JPCommonLabel_push_word()` で Word-Mora-Phoneme 階層だけ構築する。
            ポーズ形態素 ("、"/"？"/"！") や長音吸収された 'ー' では Word が生成されず、対応 feature の音素は空のままになる。
            長音吸収で隣接 feature がマージされ、戻り値の要素数が入力より少なくなる場合がある。
        """
        pass

    def g2p(
        self, text: str | bytes | bytearray, kana: bool = False, join: bool = True
    ) -> list[str] | str:
        """
        文字から音素への変換 (G2P) 。

        Args:
            text (str | bytes | bytearray): 入力テキスト (str の場合は UTF-8 にエンコードされる)
            kana (bool): True の場合、カタカナで発音を返す。False の場合は音素形式。デフォルト: False
            join (bool): True の場合、音素またはカタカナを単一の文字列に連結する。デフォルト: True

        Returns:
            str | list[str]: kana と join の組み合わせにより、str または list[str] を返す
        """
        pass

def mecab_dict_index(dn_mecab: bytes, path: bytes, out_path: bytes) -> int:
    """
    OpenJTalk 用のユーザー辞書を CSV からビルドする。低レベル API 。
    通常は pyopenjtalk.mecab_dict_index() を使用すること。
    CSV は naist-jdic 互換の品詞体系で記述する必要がある。

    Args:
        dn_mecab (bytes): MeCab システム辞書のディレクトリパス
        path (bytes): ユーザー CSV ファイルのパス
        out_path (bytes): 出力辞書ファイルのパス

    Returns:
        int: mecab-dict-index の戻り値 (0: 成功, 非 0: 失敗)
    """
    ...

def build_mecab_dictionary(dn_mecab: bytes) -> int:
    """
    OpenJTalk 用のシステム辞書を再ビルドする。低レベル API 。
    通常は pyopenjtalk.build_mecab_dictionary() を使用すること。

    Args:
        dn_mecab (bytes): MeCab システム辞書のディレクトリパス

    Returns:
        int: mecab-dict-index の戻り値 (0: 成功, 非 0: 失敗)
    """
    ...

def apply_original_rule_before_chaining(njd_features: list[NJDFeature]) -> list[NJDFeature]:
    """
    NJD features に chaining 前の独自ルールを適用する。内部用。
    サ変接続・接頭語・動詞連続・連用形・助動詞などのアクセント結合規則を適用する。

    Args:
        njd_features (list[NJDFeature]): NJDNode 用 features 。インプレースで更新される

    Returns:
        list[NJDFeature]: 更新後の njd_features（同一オブジェクト）
    """
    ...
