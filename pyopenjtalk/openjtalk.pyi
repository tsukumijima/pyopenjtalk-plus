# flake8: noqa

from typing import Any, Iterable

from .types import MeCabMorph, NJDFeature

class OpenJTalk:
    def __init__(self, dn_mecab: bytes = b"/usr/local/dic", userdic: bytes = b"") -> None:
        """
        OpenJTalk のテキスト処理フロントエンドの Cython 実装。
        通常は pyopenjtalk モジュール経由で使用するが、低レベル API として直接インスタンス化も可能。

        Args:
            dn_mecab (bytes): MeCab システム辞書のディレクトリパス。
            userdic (bytes): MeCab ユーザー辞書のパス。空バイト列の場合は無視される。デフォルトは空。
        """
        pass

    def run_mecab(self, text: str | bytes | bytearray) -> list[str]:
        """
        MeCab で形態素解析を実行する。"記号,空白" は除外される。
        全トークン (未知語フラグ・コスト情報含む) が必要な場合は代わりに run_mecab_detailed() を使うこと。

        Args:
            text (str | bytes | bytearray): 入力テキスト。str の場合は UTF-8 にエンコードされる。

        Returns:
            list[str]: MeCab の feature 文字列のリスト ("記号,空白" を除く) 。
        """
        pass

    def run_mecab_detailed(self, text: str | bytes | bytearray) -> list[MeCabMorph]:
        """
        MeCab の形態素解析結果を未知語フラグ・コスト情報付きで返す。
        通常の run_mecab() と異なり、"記号,空白" もフィルタせずに全トークンを返す。

        Args:
            text (str | bytes | bytearray): 入力テキスト。str の場合は UTF-8 にエンコードされる。

        Returns:
            list[MeCabMorph]: MeCab の形態素解析結果のリスト。
        """
        pass

    def run_njd_from_mecab(self, mecab_features: list[str]) -> list[NJDFeature]:
        """
        MeCab の feature 文字列のリストから NJD 処理を実行する。
        run_mecab() の戻り値をそのまま渡す想定。
        数字正規化・アクセント句設定・長音処理などの NJD ルールが適用される。

        Args:
            mecab_features (list[str]): MeCab の feature 文字列のリスト。

        Returns:
            list[NJDFeature]: NJDNode 用 features 。
        """
        pass

    def run_frontend(self, text: str | bytes | bytearray) -> list[NJDFeature]:
        """
        OpenJTalk のテキスト処理フロントエンドを実行する。
        run_frontend_detailed() に委譲し、NJD features のみを返す。

        Args:
            text (str | bytes | bytearray): 入力テキスト。str の場合は UTF-8 にエンコードされる。

        Returns:
            list[NJDFeature]: NJDNode 用 features 。
        """
        pass

    def run_frontend_detailed(
        self, text: str | bytes | bytearray
    ) -> tuple[list[NJDFeature], list[MeCabMorph]]:
        """
        OpenJTalk のテキスト処理フロントエンドを MeCab 形態素詳細付きで実行する。
        MeCab 解析を 1 回だけ実行し、NJD features と MeCab morphs を同時に返す。

        Args:
            text (str | bytes | bytearray): 入力テキスト。str の場合は UTF-8 にエンコードされる。

        Returns:
            tuple[list[NJDFeature], list[MeCabMorph]]: (NJD features, MeCab morphs)
                NJD features は run_frontend() と、MeCab morphs は run_mecab_detailed() と同一の結果。
        """
        pass

    def make_label(self, features: Iterable[NJDFeature]) -> list[str]:
        """
        HTS 音声合成用のフルコンテキストラベルを返す。

        Args:
            features (Iterable[NJDFeature]): NJDNode 用 features (run_frontend() の戻り値) 。

        Returns:
            list[str]: フルコンテキストラベル文字列のリスト。
        """
        pass

    def make_phoneme_mapping(self, features: Iterable[NJDFeature]) -> list[dict[str, Any]]:
        """
        NJD features から各形態素に対応する音素列のマッピングを生成する。
        JPCommon の Word-Mora-Phoneme 階層を構築し、各 feature に音素を割り当てる。
        ポーズ記号 ("、"/"？"/"！") は常に ['pau'] として保持する。
        長音吸収マージにより、戻り値の長さが入力と異なる場合がある。

        Args:
            features (Iterable[NJDFeature]): NJDNode 用 features (run_frontend() の戻り値) 。

        Returns:
            list[dict[str, Any]]: NJDFeature の全フィールド + phonemes を含む辞書のリスト。
                MeCab の未知語情報や features が必要な場合は pyopenjtalk.make_phoneme_mapping() を使用すること。

        Raises:
            RuntimeError: JPCommonLabel の内部アロケーション失敗時。
        """
        pass

    def g2p(
        self, text: str | bytes | bytearray, kana: bool = False, join: bool = True
    ) -> list[str] | str:
        """
        文字から音素への変換 (G2P) 。

        Args:
            text (str | bytes | bytearray): 入力テキスト。str の場合は UTF-8 にエンコードされる。
            kana (bool): True の場合、カタカナで発音を返す。False の場合は音素形式。デフォルトは False 。
            join (bool): True の場合、音素またはカタカナを単一の文字列に連結する。デフォルトは True 。

        Returns:
            str | list[str]: kana と join の組み合わせにより、str または list[str] を返す。
        """
        pass

def mecab_dict_index(dn_mecab: bytes, path: bytes, out_path: bytes) -> int:
    """
    OpenJTalk 用のユーザー辞書を CSV からビルドする。低レベル API 。
    通常は pyopenjtalk.mecab_dict_index() を使用すること。
    CSV は naist-jdic 互換の品詞体系で記述する必要がある。

    Args:
        dn_mecab (bytes): MeCab システム辞書のディレクトリパス。
        path (bytes): ユーザー CSV ファイルのパス。
        out_path (bytes): 出力辞書ファイルのパス。

    Returns:
        int: mecab-dict-index の戻り値 (0: 成功, 非 0: 失敗) 。
    """
    ...

def build_mecab_dictionary(dn_mecab: bytes) -> int:
    """
    OpenJTalk 用のシステム辞書を再ビルドする。低レベル API 。
    通常は pyopenjtalk.build_mecab_dictionary() を使用すること。

    Args:
        dn_mecab (bytes): MeCab システム辞書のディレクトリパス。

    Returns:
        int: mecab-dict-index の戻り値 (0: 成功, 非 0: 失敗) 。
    """
    ...
