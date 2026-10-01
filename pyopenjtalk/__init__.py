"""pyopenjtalk-plus: OpenJTalk の Python バインディングとテキスト処理フロントエンドの公開 API。"""

from __future__ import annotations

import atexit
import os
from collections.abc import Callable, Generator, Sequence
from contextlib import ExitStack, contextmanager
from importlib.resources import as_file, files
from os.path import exists
from pathlib import Path
from threading import Condition, Lock
from typing import Any, Generic, TypeVar, cast

import numpy as np
import numpy.typing as npt


try:
    from .version import __version__  # noqa
except ImportError:
    raise ImportError("BUG: version.py doesn't exist. Please file a bug report.")

from . import _mapping, _prosody
from ._mapping import (
    default_is_non_pause_symbol,
    mark_user_dictionary_reading_protection,
)
from .htsengine import HTSEngine
from .openjtalk import OpenJTalk
from .openjtalk import build_mecab_dictionary as _build_mecab_dictionary
from .openjtalk import mecab_dict_index as _mecab_dict_index
from .types import (
    IuPronunciation,
    MeCabMorph,
    MeCabNBestPath,
    NJDFeature,
    NormalizeMode,
    ProsodyFormat,
    SurfacePhonemeMapping,
    SurfaceProsodyMapping,
    UserDictionaryEntry,
)
from .types import (
    JPCommonMappingEntry as JPCommonMappingEntry,
)
from .utils import (
    merge_njd_marine_features,
    modify_acc_after_chaining,
    modify_context_reading,
    modify_kanji_yomi,
    modify_old_province_yomi,
    normalize_iu,
    normalize_text,
    normalize_unknown_itaiji,
    predict_nani_reading,
    process_odori_features,
    read_unknown_kanji,
    restore_loanword_kana,
    retreat_acc_nuc,
    revert_pron_to_read,
    split_prefix_accent_phrase,
    suppress_unnatural_auxiliary_u_long_vowel,
)


_file_manager = ExitStack()
atexit.register(_file_manager.close)

_pyopenjtalk_ref = files(__name__)
_dic_dir_name = "dictionary"

# Dictionary directory
# defaults to the directory containing the dictionaries built into the package
OPEN_JTALK_DICT_DIR = os.environ.get(
    "OPEN_JTALK_DICT_DIR",
    str(_file_manager.enter_context(as_file(_pyopenjtalk_ref / _dic_dir_name))),
).encode("utf-8")

# Default mei_normal.voice for HMM-based TTS
DEFAULT_HTS_VOICE = str(
    _file_manager.enter_context(as_file(_pyopenjtalk_ref / "htsvoice/mei_normal.htsvoice"))
).encode("utf-8")

# 複数の読みを持つ漢字のリスト
MULTI_READ_KANJI_LIST = [
    '風','何','観','方','出','時','上','下','君','手','嫌','表',
    '対','色','人','前','後','角','金','頭','筆','水','間','棚',
    # 以下、Wikipedia「同形異音語」からミスりそうな漢字を抜粋 (ただしこれらは NN 使わない限り完璧な判定は無理な気がする…)
    # Sudachi の方が不正確な '汚','通','臭','辛' は除外した
    # ref: https://ja.wikipedia.org/wiki/%E5%90%8C%E5%BD%A2%E7%95%B0%E9%9F%B3%E8%AA%9E
    '床','入','来','塗','怒','包','被','開','弾','捻','潜','支','抱','行','降','種','訳','糞',
    # 以下、Wikipedia「同形異音語」記事内「読み方が3つ以上ある同形異音語」より
    '空','性','体','等','生','止','堪','捩',
    # 以下、独自に追加
    '家','縁','労','中','高','低','気','要','退','面','色','主','術','直','片','緒','小','大','値',
    # 他にも日付（月・火・水・木・金・土・日）も入るが、当面は入れない (金を除く)
]  # fmt: skip
_MULTI_READ_KANJI_SET_EXCLUDING_NANI = frozenset(
    kanji for kanji in MULTI_READ_KANJI_LIST if kanji != "何"
)


_T = TypeVar("_T")


def _lazy_init() -> None:
    """
    互換性維持のための no-op 初期化フック。

    pyopenjtalk-plus では辞書のダウンロード処理を削除しているが、
    VOICEVOX 等が `_lazy_init()` を直接呼び出すため残置している。
    """

    pass


class _ReplaceableInstanceManager(Generic[_T]):
    """
    利用中の処理を完了させてから交換できる共有インスタンスを管理する。
    """

    def __init__(self, instance_factory: Callable[[], _T]) -> None:
        """
        インスタンスを遅延生成するマネージャーを初期化する。

        Args:
            instance_factory (Callable[[], _T]): 初回の借り出し時に呼ぶファクトリ
        """

        self._instance: _T | None = None
        self._instance_factory = instance_factory
        self._mutex = Lock()
        self._condition = Condition(self._mutex)
        self._active_leases = 0
        self._is_replacing = False

    @contextmanager
    def __call__(self) -> Generator[_T, None, None]:
        """
        共有インスタンスを借り出す。

        Yields:
            _T: 遅延生成または固定のシングルトン
        """

        with self._condition:
            # 交換開始後の呼び出しは、旧インスタンスを取得せず交換完了まで待機
            while self._is_replacing is True:
                self._condition.wait()
            instance = self._instance
            if instance is None:
                instance = self._instance_factory()
                self._instance = instance
            self._active_leases += 1
        try:
            yield instance
        finally:
            with self._condition:
                self._active_leases -= 1
                # 交換処理が待つのは最後の借り出しが返却される瞬間だけ
                if self._active_leases == 0:
                    self._condition.notify_all()

    def replace(self, instance: _T) -> None:
        """
        進行中の借り出しを待って共有インスタンスを交換する。

        Args:
            instance (_T): 次の借り出しから返すインスタンス

        NOTE:
            非リエントラント。借り出し中 (`with _global_jtalk()` 内) から呼んではいけない。
            `run_frontend()` の後処理中に `update_global_jtalk_with_user_dict()` や
            `unset_user_dict()` を呼ぶと、借り出し完了待ちでデッドロックする
        """

        with self._condition:
            # 複数の交換要求も順番に処理し、交換中は新規借り出しを止める
            while self._is_replacing is True:
                self._condition.wait()
            self._is_replacing = True
            try:
                while self._active_leases > 0:
                    self._condition.wait()
                self._instance = instance
            finally:
                self._is_replacing = False
                self._condition.notify_all()


class _ExclusiveInstanceManager(Generic[_T]):
    """一連の操作が完了するまで排他的に貸し出すインスタンスを管理する。"""

    def __init__(self, instance_factory: Callable[[], _T]) -> None:
        """
        インスタンスを遅延生成するマネージャーを初期化する。

        Args:
            instance_factory (Callable[[], _T]): 初回の借り出し時に呼ぶファクトリ
        """

        self._instance: _T | None = None
        self._instance_factory = instance_factory
        self._mutex = Lock()

    @contextmanager
    def __call__(self) -> Generator[_T, None, None]:
        """
        コンテキスト内の操作全体を排他してインスタンスを貸し出す。

        Yields:
            _T: 遅延生成されたシングルトン
        """

        # HTSEngine の設定変更と合成を一体として扱うため、返却までロックを保持
        with self._mutex:
            if self._instance is None:
                self._instance = self._instance_factory()
            yield self._instance


# Global instance of OpenJTalk
_global_jtalk: _ReplaceableInstanceManager[OpenJTalk] = _ReplaceableInstanceManager(
    lambda: OpenJTalk(dn_mecab=OPEN_JTALK_DICT_DIR),
)
# 連続する update / unset が直前のマネージャーを待たずに差し替えるのを防ぐ
_global_jtalk_swap_lock = Lock()

# Global instance of HTSEngine
# mei_normal.voice is used as default
_global_htsengine: _ExclusiveInstanceManager[HTSEngine] = _ExclusiveInstanceManager(
    lambda: HTSEngine(DEFAULT_HTS_VOICE),
)
# Global instance of marine
_global_marine = None


@contextmanager
def _resolve_jtalk(jtalk: OpenJTalk | None) -> Generator[OpenJTalk, None, None]:
    """
    呼び出し元指定またはグローバルの `OpenJTalk` インスタンスを返す。

    Args:
        jtalk (OpenJTalk | None): 明示インスタンス。None なら `_global_jtalk()` を使う

    Yields:
        OpenJTalk: 処理中の借り出しが記録された OpenJTalk インスタンス

    NOTE:
        呼び出し元がインスタンスを渡した場合は、グローバルインスタンスの交換待機へ影響しない
    """

    # 明示インスタンスはグローバルの辞書交換と独立しているため、そのまま使用
    if jtalk is not None:
        yield jtalk
        return
    # グローバル利用中は辞書交換が同じインスタンスを途中で無効化しないよう借り出しを記録
    with _global_jtalk() as instance:
        yield instance


def g2p(
    text: str,
    kana: bool = False,
    join: bool = True,
    *,
    run_marine: bool = False,
    use_vanilla: bool = False,
    use_tsqyomi: bool = False,
    use_sudachi_kanji_yomi: bool = True,
    predict_nani: bool = True,
    normalize_mode: NormalizeMode = "None",
    iu_pronunciation: IuPronunciation | None = None,
    use_read_as_pron: bool = False,
    revert_long_vowels: bool = False,
    revert_yotsugana: bool = False,
    jtalk: OpenJTalk | None = None,
) -> list[str] | str:
    """
    文字から音素への変換処理 (G2P) 。pyopenjtalk.run_frontend() のラッパー。

    Args:
        text (str): Unicode 日本語テキスト
        kana (bool): True の場合、カタカナで発音を返す。False の場合は音素形式 (デフォルト: False)
        join (bool): True の場合、音素またはカタカナを単一の文字列に連結する (デフォルト: True)
        run_marine (bool): marine を用いたアクセント推定を行うか (デフォルト: False)
            有効にするには `pip install pyopenjtalk-plus[marine]` で marine をインストールする必要がある
        use_vanilla (bool): True の場合、pyopenjtalk-plus 独自の後処理を省略し、
            OpenJTalk の素の NJDFeature をそのまま後段に流す
            ただし発音復元オプション (use_read_as_pron 等) は use_vanilla とは独立して適用される (デフォルト: False)
        use_tsqyomi (bool): True の場合、ロード済みの tsqyomi で文脈に合う読み候補を選ぶ
            Sudachi と「何」モデルによる読み変更を省き、tsqyomi の選択を維持する (デフォルト: False)
        use_sudachi_kanji_yomi (bool): True の場合、Sudachi による同形異音語の読み補正を行う
            use_tsqyomi が True の場合は tsqyomi を優先し、常に無効化される (デフォルト: True)
        predict_nani (bool): True の場合、ONNX モデルで単独形態素として出現した「何」の読みを推定する
            use_tsqyomi が True の場合は tsqyomi を優先し、常に無効化される (デフォルト: True)
        normalize_mode (NormalizeMode): 入力テキストに適用する Unicode 正規化方式
            `"NFC"` は結合文字を正規化し、`"NFKC"` は半角カナなどの互換文字も正規化する (デフォルト: `"None"`)
        iu_pronunciation (IuPronunciation | None): 「言う」や「という」などの定型表現に含まれる「イウ」を、どう発音するかの方式
            "Iu"、"Yuu"、"KanjiIu"、"KanjiYuu"、"YuuBase"、"KanjiYuuBase" のいずれかを指定する。None の場合は辞書の発音のままにし、指定した場合は use_vanilla の設定に関係なく適用される (デフォルト: None)
        use_read_as_pron (bool): True の場合、全ての発音を強制的に読みに置き換える
            助詞「は」も「ハ」になるため、TTS 用途には適さない (デフォルト: False)
            このオプションが True の場合、revert_long_vowels / revert_yotsugana の指定に関係なく
            全ての pron が read で上書きされる
        revert_long_vowels (bool): True の場合、辞書が自動的に長音化した発音を元に復元する
            pron に「ー」が含まれ、かつ orig に「ー」が含まれていない場合のみ復元する
            助詞 (は→ワ, へ→エ) の発音は「ー」を含まないため影響を受けず維持される
            (例: 「効果」コーカ → コウカ / 「人生」ジンセー → ジンセイ) (デフォルト: False)
        revert_yotsugana (bool): True の場合、四つ仮名 (ヅ・ヂ) の発音統合を元に復元する
            read に「ヅ」「ヂ」が含まれている場合、pron を read で上書きする
            (例: 「気づかず」キズカズ → キヅカズ / 「鼻血」ハナジ → ハナヂ) (デフォルト: False)
        jtalk (OpenJTalk | None): 使用する OpenJTalk インスタンス。None ならグローバルインスタンスを使う

    Returns:
        Union[list[str], str]: G2P 結果を返す。join が True の場合は str 、False の場合は list[str] を返す
    """
    njd_features = run_frontend(
        text,
        run_marine=run_marine,
        use_vanilla=use_vanilla,
        use_tsqyomi=use_tsqyomi,
        use_sudachi_kanji_yomi=use_sudachi_kanji_yomi,
        predict_nani=predict_nani,
        normalize_mode=normalize_mode,
        iu_pronunciation=iu_pronunciation,
        use_read_as_pron=use_read_as_pron,
        revert_long_vowels=revert_long_vowels,
        revert_yotsugana=revert_yotsugana,
        jtalk=jtalk,
    )

    if kana is False:
        # run_frontend() の借り出しは返却済みなので、音素抽出の間だけ再度借り出す
        with _resolve_jtalk(jtalk) as resolved_jtalk:
            prons = resolved_jtalk.extract_phonemes(njd_features)
        if join:
            prons = " ".join(prons)
        return prons

    # kana
    prons = []
    for n in njd_features:
        if n["pos"] == "記号":
            p = n["string"]
        else:
            p = n["pron"]
        # remove special chars
        for c in "’":
            p = p.replace(c, "")
        prons.append(p)
    if join:
        prons = "".join(prons)
    return prons


def g2p_prosody(
    text: str,
    *,
    format: ProsodyFormat = "Default",
    is_non_pause_symbol: Callable[[str], bool] = default_is_non_pause_symbol,
    run_marine: bool = False,
    use_vanilla: bool = False,
    use_tsqyomi: bool = False,
    use_sudachi_kanji_yomi: bool = True,
    predict_nani: bool = True,
    normalize_mode: NormalizeMode = "None",
    iu_pronunciation: IuPronunciation | None = None,
    use_read_as_pron: bool = False,
    revert_long_vowels: bool = False,
    revert_yotsugana: bool = False,
    jtalk: OpenJTalk | None = None,
) -> list[str]:
    """
    テキストを、高低の記号とアクセント句の区切り、句読点の記号を含む音素列に変換する。
    Haqumei の g2p_prosody と同じ記号で出力する。

    Args:
        text (str): Unicode 日本語テキスト
        format (ProsodyFormat): ピッチの表記方式 (`Default` は高低が変わる位置に `[` `]`、`Prefix` は `H_` / `L_`、`Numeric` は `:1` / `:0` を使う) (デフォルト: `"Default"`)
        is_non_pause_symbol (Callable[[str], bool]): True を返した記号は音素なしで保持し、False を返した短ポーズ記号には `pau` を割り当てる。
            既定では括弧・引用符だけを音素なしで保持する
        run_marine (bool): marine を用いたアクセント推定を行うか (デフォルト: False)
            有効にするには `pip install pyopenjtalk-plus[marine]` で marine をインストールする必要がある
        use_vanilla (bool): True の場合、pyopenjtalk-plus 独自の後処理を省略し、
            OpenJTalk の素の NJDFeature をそのまま後段に流す
            ただし発音復元オプション (use_read_as_pron 等) は use_vanilla とは独立して適用される (デフォルト: False)
        use_tsqyomi (bool): True の場合、ロード済みの tsqyomi で文脈に合う読み候補を選ぶ
            Sudachi と「何」モデルによる読み変更を省き、tsqyomi の選択を維持する (デフォルト: False)
        use_sudachi_kanji_yomi (bool): True の場合、Sudachi による同形異音語の読み補正を行う
            use_tsqyomi が True の場合は tsqyomi を優先し、常に無効化される (デフォルト: True)
        predict_nani (bool): True の場合、ONNX モデルで単独形態素として出現した「何」の読みを推定する
            use_tsqyomi が True の場合は tsqyomi を優先し、常に無効化される (デフォルト: True)
        normalize_mode (NormalizeMode): 入力テキストに適用する Unicode 正規化方式
            `"NFC"` は結合文字を正規化し、`"NFKC"` は半角カナなどの互換文字も正規化する (デフォルト: `"None"`)
        iu_pronunciation (IuPronunciation | None): 「言う」や「という」などの定型表現に含まれる「イウ」を、どう発音するかの方式
            "Iu"、"Yuu"、"KanjiIu"、"KanjiYuu"、"YuuBase"、"KanjiYuuBase" のいずれかを指定する。None の場合は辞書の発音のままにし、指定した場合は use_vanilla の設定に関係なく適用される (デフォルト: None)
        use_read_as_pron (bool): True の場合、全ての発音を強制的に読みに置き換える
            助詞「は」も「ハ」になるため、TTS 用途には適さない (デフォルト: False)
            このオプションが True の場合、revert_long_vowels / revert_yotsugana の指定に関係なく
            全ての pron が read で上書きされる
        revert_long_vowels (bool): True の場合、辞書が自動的に長音化した発音を元に復元する
            pron に「ー」が含まれ、かつ orig に「ー」が含まれていない場合のみ復元する
            助詞 (は→ワ, へ→エ) の発音は「ー」を含まないため影響を受けず維持される
            (例: 「効果」コーカ → コウカ / 「人生」ジンセー → ジンセイ) (デフォルト: False)
        revert_yotsugana (bool): True の場合、四つ仮名 (ヅ・ヂ) の発音統合を元に復元する
            read に「ヅ」「ヂ」が含まれている場合、pron を read で上書きする
            (例: 「気づかず」キズカズ → キヅカズ / 「鼻血」ハナジ → ハナヂ) (デフォルト: False)
        jtalk (OpenJTalk | None): 使用する OpenJTalk インスタンス。None ならグローバルインスタンスを使う

    Returns:
        list[str]: 先頭の ^ と末尾の $、ピッチの記号、アクセント句の区切り、句読点の記号を含む音素列

    Raises:
        ValueError: 呼び出し元入力と MeCab 正規化本文の対応付けに失敗した場合
        RuntimeError: 音素マッピングとフルコンテキストラベルの対応付けに失敗した場合
    """

    prosody_mapping = g2p_mapping_prosody(
        text,
        is_non_pause_symbol=is_non_pause_symbol,
        run_marine=run_marine,
        use_vanilla=use_vanilla,
        use_tsqyomi=use_tsqyomi,
        use_sudachi_kanji_yomi=use_sudachi_kanji_yomi,
        predict_nani=predict_nani,
        normalize_mode=normalize_mode,
        iu_pronunciation=iu_pronunciation,
        use_read_as_pron=use_read_as_pron,
        revert_long_vowels=revert_long_vowels,
        revert_yotsugana=revert_yotsugana,
        jtalk=jtalk,
    )
    return _prosody.format_prosody_phonemes(prosody_mapping, format)


def g2p_mapping(
    text: str,
    *,
    is_non_pause_symbol: Callable[[str], bool] = default_is_non_pause_symbol,
    run_marine: bool = False,
    use_vanilla: bool = False,
    use_tsqyomi: bool = False,
    use_sudachi_kanji_yomi: bool = True,
    predict_nani: bool = True,
    normalize_mode: NormalizeMode = "None",
    iu_pronunciation: IuPronunciation | None = None,
    use_read_as_pron: bool = False,
    revert_long_vowels: bool = False,
    revert_yotsugana: bool = False,
    jtalk: OpenJTalk | None = None,
) -> list[SurfacePhonemeMapping]:
    """
    テキストから形態素-音素マッピングを一括で取得する便利ラッパー。
    内部で pyopenjtalk.run_frontend_detailed() と pyopenjtalk.make_phoneme_mapping() を呼び出し、
    MeCab 未知語フラグ・無視トークン情報付きの音素マッピングを返す。

    Args:
        text (str): Unicode 日本語テキスト
        is_non_pause_symbol (Callable[[str], bool]): True を返した記号は音素なしで保持し、False を返した短ポーズ記号には `pau` を割り当てる。
            既定では括弧・引用符だけを音素なしで保持する
        run_marine (bool): marine を用いたアクセント推定を行うか (デフォルト: False)
            有効にするには `pip install pyopenjtalk-plus[marine]` で marine をインストールする必要がある
        use_vanilla (bool): True の場合、pyopenjtalk-plus 独自の後処理を省略し、
            OpenJTalk の素の NJDFeature をそのまま後段に流す
            ただし発音復元オプション (use_read_as_pron 等) は use_vanilla とは独立して適用される (デフォルト: False)
        use_tsqyomi (bool): True の場合、ロード済みの tsqyomi で文脈に合う読み候補を選ぶ
            Sudachi と「何」モデルによる読み変更を省き、tsqyomi の選択を維持する (デフォルト: False)
        use_sudachi_kanji_yomi (bool): True の場合、Sudachi による同形異音語の読み補正を行う
            use_tsqyomi が True の場合は tsqyomi を優先し、常に無効化される (デフォルト: True)
        predict_nani (bool): True の場合、ONNX モデルで単独形態素として出現した「何」の読みを推定する
            use_tsqyomi が True の場合は tsqyomi を優先し、常に無効化される (デフォルト: True)
        normalize_mode (NormalizeMode): 入力テキストに適用する Unicode 正規化方式
            `"NFC"` は結合文字を正規化し、`"NFKC"` は半角カナなどの互換文字も正規化する (デフォルト: `"None"`)
        iu_pronunciation (IuPronunciation | None): 「言う」や「という」などの定型表現に含まれる「イウ」を、どう発音するかの方式
            "Iu"、"Yuu"、"KanjiIu"、"KanjiYuu"、"YuuBase"、"KanjiYuuBase" のいずれかを指定する。None の場合は辞書の発音のままにし、指定した場合は use_vanilla の設定に関係なく適用される (デフォルト: None)
        use_read_as_pron (bool): True の場合、全ての発音を強制的に読みに置き換える
            助詞「は」も「ハ」になるため、TTS 用途には適さない (デフォルト: False)
            このオプションが True の場合、revert_long_vowels / revert_yotsugana の指定に関係なく
            全ての pron が read で上書きされる
        revert_long_vowels (bool): True の場合、辞書が自動的に長音化した発音を元に復元する
            pron に「ー」が含まれ、かつ orig に「ー」が含まれていない場合のみ復元する
            助詞 (は→ワ, へ→エ) の発音は「ー」を含まないため影響を受けず維持される
            (例: 「効果」コーカ → コウカ / 「人生」ジンセー → ジンセイ) (デフォルト: False)
        revert_yotsugana (bool): True の場合、四つ仮名 (ヅ・ヂ) の発音統合を元に復元する
            read に「ヅ」「ヂ」が含まれている場合、pron を read で上書きする
            (例: 「気づかず」キズカズ → キヅカズ / 「鼻血」ハナジ → ハナヂ) (デフォルト: False)
        jtalk (OpenJTalk | None): 使用する OpenJTalk インスタンス。None ならグローバルインスタンスを使う

    Returns:
        list[SurfacePhonemeMapping]: 各形態素に対応する音素列のマッピング (未知語・無視トークン情報付き)

    Raises:
        ValueError: 呼び出し元入力と MeCab 正規化本文の対応付けに失敗した or `char_span` が入力全体を1度ずつ覆わない場合
    """

    njd_features, morphs = run_frontend_detailed(
        text,
        run_marine=run_marine,
        use_vanilla=use_vanilla,
        use_tsqyomi=use_tsqyomi,
        use_sudachi_kanji_yomi=use_sudachi_kanji_yomi,
        predict_nani=predict_nani,
        normalize_mode=normalize_mode,
        iu_pronunciation=iu_pronunciation,
        use_read_as_pron=use_read_as_pron,
        revert_long_vowels=revert_long_vowels,
        revert_yotsugana=revert_yotsugana,
        jtalk=jtalk,
    )
    mapping = make_phoneme_mapping(
        njd_features,
        morphs=morphs,
        jtalk=jtalk,
        caller_text=text,
        normalize_mode=normalize_mode,
        is_non_pause_symbol=is_non_pause_symbol,
    )

    # 値を返す前に char_span の座標が壊れていないかを確かめ、壊れていたら明示的にエラーにする
    _mapping.check_caller_char_spans(mapping, text, "g2p_mapping")
    return mapping


def g2p_mapping_prosody(
    text: str,
    *,
    is_non_pause_symbol: Callable[[str], bool] = default_is_non_pause_symbol,
    run_marine: bool = False,
    use_vanilla: bool = False,
    use_tsqyomi: bool = False,
    use_sudachi_kanji_yomi: bool = True,
    predict_nani: bool = True,
    normalize_mode: NormalizeMode = "None",
    iu_pronunciation: IuPronunciation | None = None,
    use_read_as_pron: bool = False,
    revert_long_vowels: bool = False,
    revert_yotsugana: bool = False,
    jtalk: OpenJTalk | None = None,
) -> list[SurfaceProsodyMapping]:
    """
    g2p_mapping() と同じ形態素-音素マッピングに、各音素の高低とアクセント句の区切り、句読点の種類を重ねて返す。
    高低と区切りは、同じ NJD features から作ったフルコンテキストラベルから読み取る。

    Args:
        text (str): Unicode 日本語テキスト
        is_non_pause_symbol (Callable[[str], bool]): True を返した記号は音素なしで保持し、False を返した短ポーズ記号には `pau` を割り当てる。
            既定では括弧・引用符だけを音素なしで保持する
        run_marine (bool): marine を用いたアクセント推定を行うか (デフォルト: False)
            有効にするには `pip install pyopenjtalk-plus[marine]` で marine をインストールする必要がある
        use_vanilla (bool): True の場合、pyopenjtalk-plus 独自の後処理を省略し、
            OpenJTalk の素の NJDFeature をそのまま後段に流す
            ただし発音復元オプション (use_read_as_pron 等) は use_vanilla とは独立して適用される (デフォルト: False)
        use_tsqyomi (bool): True の場合、ロード済みの tsqyomi で文脈に合う読み候補を選ぶ
            Sudachi と「何」モデルによる読み変更を省き、tsqyomi の選択を維持する (デフォルト: False)
        use_sudachi_kanji_yomi (bool): True の場合、Sudachi による同形異音語の読み補正を行う
            use_tsqyomi が True の場合は tsqyomi を優先し、常に無効化される (デフォルト: True)
        predict_nani (bool): True の場合、ONNX モデルで単独形態素として出現した「何」の読みを推定する
            use_tsqyomi が True の場合は tsqyomi を優先し、常に無効化される (デフォルト: True)
        normalize_mode (NormalizeMode): 入力テキストに適用する Unicode 正規化方式
            `"NFC"` は結合文字を正規化し、`"NFKC"` は半角カナなどの互換文字も正規化する (デフォルト: `"None"`)
        iu_pronunciation (IuPronunciation | None): 「言う」や「という」などの定型表現に含まれる「イウ」を、どう発音するかの方式
            "Iu"、"Yuu"、"KanjiIu"、"KanjiYuu"、"YuuBase"、"KanjiYuuBase" のいずれかを指定する。None の場合は辞書の発音のままにし、指定した場合は use_vanilla の設定に関係なく適用される (デフォルト: None)
        use_read_as_pron (bool): True の場合、全ての発音を強制的に読みに置き換える
            助詞「は」も「ハ」になるため、TTS 用途には適さない (デフォルト: False)
            このオプションが True の場合、revert_long_vowels / revert_yotsugana の指定に関係なく
            全ての pron が read で上書きされる
        revert_long_vowels (bool): True の場合、辞書が自動的に長音化した発音を元に復元する
            pron に「ー」が含まれ、かつ orig に「ー」が含まれていない場合のみ復元する
            助詞 (は→ワ, へ→エ) の発音は「ー」を含まないため影響を受けず維持される
            (例: 「効果」コーカ → コウカ / 「人生」ジンセー → ジンセイ) (デフォルト: False)
        revert_yotsugana (bool): True の場合、四つ仮名 (ヅ・ヂ) の発音統合を元に復元する
            read に「ヅ」「ヂ」が含まれている場合、pron を read で上書きする
            (例: 「気づかず」キズカズ → キヅカズ / 「鼻血」ハナジ → ハナヂ) (デフォルト: False)
        jtalk (OpenJTalk | None): 使用する OpenJTalk インスタンス。None ならグローバルインスタンスを使う

    Returns:
        list[SurfaceProsodyMapping]: 高低と韻律の区切りを持つ形態素-音素マッピング

    Raises:
        ValueError: 呼び出し元入力と MeCab 正規化本文の対応付けに失敗した or `char_span` が入力全体を1度ずつ覆わない場合
        RuntimeError: 音素マッピングとフルコンテキストラベルの対応付けに失敗した場合
    """

    # 解析・マッピング・ラベルの生成を同じインスタンスの借り出しの中で続けて行い、途中で辞書が入れ替わって対応がずれないようにする
    with _resolve_jtalk(jtalk) as resolved_jtalk:
        njd_features, morphs = run_frontend_detailed(
            text,
            run_marine=run_marine,
            use_vanilla=use_vanilla,
            use_tsqyomi=use_tsqyomi,
            use_sudachi_kanji_yomi=use_sudachi_kanji_yomi,
            predict_nani=predict_nani,
            normalize_mode=normalize_mode,
            iu_pronunciation=iu_pronunciation,
            use_read_as_pron=use_read_as_pron,
            revert_long_vowels=revert_long_vowels,
            revert_yotsugana=revert_yotsugana,
            jtalk=resolved_jtalk,
        )
        mapping = make_phoneme_mapping(
            njd_features,
            morphs=morphs,
            jtalk=resolved_jtalk,
            caller_text=text,
            normalize_mode=normalize_mode,
            is_non_pause_symbol=is_non_pause_symbol,
        )
        labels = resolved_jtalk.make_label(njd_features)

    # 高低を重ねる前に、g2p_mapping() と同じく char_span の座標が壊れていないかを確かめる
    _mapping.check_caller_char_spans(mapping, text, "g2p_mapping_prosody")
    return _prosody.make_prosody_mapping(mapping, labels)


def load_marine_model(model_dir: str | None = None, dict_dir: str | None = None) -> None:
    """
    marine の Predictor をグローバルに1回だけ初期化する。

    Args:
        model_dir (str | None): marine モデルディレクトリ。None なら marine の既定
        dict_dir (str | None): marine 後処理語彙ディレクトリ。None なら marine の既定

    Raises:
        ImportError: `pyopenjtalk-plus[marine]` 未インストールの場合
    """

    global _global_marine
    if _global_marine is None:
        try:
            from marine.predict import Predictor  # type: ignore[reportMissingImports]
        except ImportError:
            raise ImportError("Please install marine by `pip install pyopenjtalk-plus[marine]`")
        _global_marine = Predictor(model_dir=model_dir, postprocess_vocab_dir=dict_dir)


def estimate_accent(njd_features: list[NJDFeature]) -> list[NJDFeature]:
    """
    marine を用いたアクセント推定処理。

    Args:
        njd_features (list[NJDFeature]): NJDNode 用 features (pyopenjtalk.run_frontend() の戻り値)

    Returns:
        list[NJDFeature]: marine による推定結果付きの NJDNode 用 features
    """
    global _global_marine
    if _global_marine is None:
        load_marine_model()
        assert _global_marine is not None

    from marine.utils.openjtalk_util import convert_njd_feature_to_marine_feature  # type: ignore[reportMissingImports] # noqa: I001

    # marine は NJDFeature を独自の TypedDict として宣言していて、読み保護の印 (is_reading_protected) の分だけこちらの型とフィールドが食い違う
    # list は要素の型について不変なので中身が同じ構造でも型検査を通らず、境界で Any として渡している
    marine_feature = convert_njd_feature_to_marine_feature(cast(Any, njd_features))
    marine_results = cast(
        dict[str, Any],
        _global_marine.predict([marine_feature], require_open_jtalk_format=True),
    )
    njd_features = merge_njd_marine_features(njd_features, marine_results)
    return njd_features


def modify_filler_accent(njd: list[NJDFeature]) -> list[NJDFeature]:
    """
    フィラー直後の名詞が誤って前アクセント句へ連結されるのを防ぐ。

    Args:
        njd (list[NJDFeature]): NJDNode 用 features

    Returns:
        list[NJDFeature]: フィラーのアクセント核補正と、直後名詞の `chain_flag` 修正を反映した features
    """

    modified_njd = []
    is_after_filler = False
    for features in njd:
        if features["pos"] == "フィラー":
            if features["acc"] > features["mora_size"]:
                features["acc"] = 0
            is_after_filler = True

        elif is_after_filler:
            if features["pos"] == "名詞":
                features["chain_flag"] = 0
            is_after_filler = False
        modified_njd.append(features)

    return modified_njd


def preserve_noun_accent(
    input_njd: list[NJDFeature], predicted_njd: list[NJDFeature]
) -> list[NJDFeature]:
    """
    marine の推定のあとも、読みが1つしかない名詞のアクセント核は、OpenJTalk の入力側の値のままにする。
    ただし、読みの補正でアクセント句が短くなり、入力側の核が句の外に出てしまう場合は、marine の推定した核を使う。

    Args:
        input_njd (list[NJDFeature]): marine 適用前の NJD features
        predicted_njd (list[NJDFeature]): marine 適用後の NJD features

    Returns:
        list[NJDFeature]: 句の長さを超えない対象の名詞の `acc` を入力側の値で上書きした predicted_njd 相当の list
    """

    predicted_accents = [feature["acc"] for feature in predicted_njd]
    return_njd = []
    for f_input, f_pred in zip(input_njd, predicted_njd):
        if f_pred["pos"] == "名詞" and f_pred["string"] not in MULTI_READ_KANJI_LIST:
            f_pred["acc"] = f_input["acc"]
        return_njd.append(f_pred)

    # 読みの補正でアクセント句が短くなり、入力側の核が句の外に出た場合は、marine の推定した核に戻す
    phrase_head_index = 0
    phrase_mora_size = 0
    for feature_index, feature in enumerate(return_njd):
        if feature_index > 0 and feature["chain_flag"] in (0, -1):
            if return_njd[phrase_head_index]["acc"] > phrase_mora_size:
                return_njd[phrase_head_index]["acc"] = predicted_accents[phrase_head_index]
            phrase_head_index = feature_index
            phrase_mora_size = 0
        phrase_mora_size += feature["mora_size"]
    if len(return_njd) > 0 and return_njd[phrase_head_index]["acc"] > phrase_mora_size:
        return_njd[phrase_head_index]["acc"] = predicted_accents[phrase_head_index]

    return return_njd


def extract_fullcontext(
    text: str,
    *,
    run_marine: bool = False,
    use_vanilla: bool = False,
    use_tsqyomi: bool = False,
    use_sudachi_kanji_yomi: bool = True,
    predict_nani: bool = True,
    normalize_mode: NormalizeMode = "None",
    iu_pronunciation: IuPronunciation | None = None,
    use_read_as_pron: bool = False,
    revert_long_vowels: bool = False,
    revert_yotsugana: bool = False,
    jtalk: OpenJTalk | None = None,
) -> list[str]:
    """
    テキストからフルコンテキストラベルを抽出する。

    Args:
        text (str): Unicode 日本語テキスト
        run_marine (bool): marine を用いたアクセント推定を行うか (デフォルト: False)
            有効にするには `pip install pyopenjtalk-plus[marine]` で marine をインストールする必要がある
        use_vanilla (bool): True の場合、pyopenjtalk-plus 独自の後処理を省略し、
            OpenJTalk の素の NJDFeature をそのまま後段に流す
            ただし発音復元オプション (use_read_as_pron 等) は use_vanilla とは独立して適用される (デフォルト: False)
        use_tsqyomi (bool): True の場合、ロード済みの tsqyomi で文脈に合う読み候補を選ぶ
            Sudachi と「何」モデルによる読み変更を省き、tsqyomi の選択を維持する (デフォルト: False)
        use_sudachi_kanji_yomi (bool): True の場合、Sudachi による同形異音語の読み補正を行う
            use_tsqyomi が True の場合は tsqyomi を優先し、常に無効化される (デフォルト: True)
        predict_nani (bool): True の場合、ONNX モデルで単独形態素として出現した「何」の読みを推定する
            use_tsqyomi が True の場合は tsqyomi を優先し、常に無効化される (デフォルト: True)
        normalize_mode (NormalizeMode): 入力テキストに適用する Unicode 正規化方式
            `"NFC"` は結合文字を正規化し、`"NFKC"` は半角カナなどの互換文字も正規化する (デフォルト: `"None"`)
        iu_pronunciation (IuPronunciation | None): 「言う」や「という」などの定型表現に含まれる「イウ」を、どう発音するかの方式
            "Iu"、"Yuu"、"KanjiIu"、"KanjiYuu"、"YuuBase"、"KanjiYuuBase" のいずれかを指定する。None の場合は辞書の発音のままにし、指定した場合は use_vanilla の設定に関係なく適用される (デフォルト: None)
        use_read_as_pron (bool): True の場合、全ての発音を強制的に読みに置き換える
            助詞「は」も「ハ」になるため、TTS 用途には適さない (デフォルト: False)
            このオプションが True の場合、revert_long_vowels / revert_yotsugana の指定に関係なく
            全ての pron が read で上書きされる
        revert_long_vowels (bool): True の場合、辞書が自動的に長音化した発音を元に復元する
            pron に「ー」が含まれ、かつ orig に「ー」が含まれていない場合のみ復元する
            助詞 (は→ワ, へ→エ) の発音は「ー」を含まないため影響を受けず維持される
            (例: 「効果」コーカ → コウカ / 「人生」ジンセー → ジンセイ) (デフォルト: False)
        revert_yotsugana (bool): True の場合、四つ仮名 (ヅ・ヂ) の発音統合を元に復元する
            read に「ヅ」「ヂ」が含まれている場合、pron を read で上書きする
            (例: 「気づかず」キズカズ → キヅカズ / 「鼻血」ハナジ → ハナヂ) (デフォルト: False)
        jtalk (OpenJTalk | None): 使用する OpenJTalk インスタンス。None ならグローバルインスタンスを使う

    Returns:
        list[str]: フルコンテキストラベルのリスト
    """
    njd_features = run_frontend(
        text,
        run_marine=run_marine,
        use_vanilla=use_vanilla,
        use_tsqyomi=use_tsqyomi,
        use_sudachi_kanji_yomi=use_sudachi_kanji_yomi,
        predict_nani=predict_nani,
        normalize_mode=normalize_mode,
        iu_pronunciation=iu_pronunciation,
        use_read_as_pron=use_read_as_pron,
        revert_long_vowels=revert_long_vowels,
        revert_yotsugana=revert_yotsugana,
        jtalk=jtalk,
    )
    return make_label(njd_features, jtalk=jtalk)


def synthesize(
    labels: list[str] | tuple[Any, list[str]],
    speed: float = 1.0,
    half_tone: float = 0.0,
) -> tuple[npt.NDArray[np.float64], int]:
    """
    OpenJTalk の音声合成バックエンドを実行する。

    Args:
        labels (list): フルコンテキストラベル
        speed (float): 話速 (デフォルト: 1.0)
        half_tone (float): 追加の半音 (デフォルト: 0)

    Returns:
        np.ndarray: 音声波形 (dtype: np.float64)
        int: サンプリング周波数 (デフォルト: 48000)
    """
    if isinstance(labels, tuple) and len(labels) == 2:
        labels = labels[1]

    with _global_htsengine() as htsengine:
        sr = htsengine.get_sampling_frequency()
        htsengine.set_speed(speed)
        htsengine.add_half_tone(half_tone)
        return htsengine.synthesize(labels), sr


def tts(
    text: str,
    speed: float = 1.0,
    half_tone: float = 0.0,
    *,
    run_marine: bool = False,
    use_vanilla: bool = False,
    use_tsqyomi: bool = False,
    use_sudachi_kanji_yomi: bool = True,
    predict_nani: bool = True,
    normalize_mode: NormalizeMode = "None",
    iu_pronunciation: IuPronunciation | None = None,
    use_read_as_pron: bool = False,
    revert_long_vowels: bool = False,
    revert_yotsugana: bool = False,
    jtalk: OpenJTalk | None = None,
) -> tuple[npt.NDArray[np.float64], int]:
    """
    テキストから音声を合成する。

    Args:
        text (str): Unicode 日本語テキスト
        speed (float): 話速 (デフォルト: 1.0)
        half_tone (float): 追加の半音 (デフォルト: 0)
        run_marine (bool): marine を用いたアクセント推定を行うか (デフォルト: False)
            有効にするには `pip install pyopenjtalk-plus[marine]` で marine をインストールする必要がある
        use_vanilla (bool): True の場合、pyopenjtalk-plus 独自の後処理を省略し、
            OpenJTalk の素の NJDFeature をそのまま後段に流す
            ただし発音復元オプション (use_read_as_pron 等) は use_vanilla とは独立して適用される (デフォルト: False)
        use_tsqyomi (bool): True の場合、ロード済みの tsqyomi で文脈に合う読み候補を選ぶ
            Sudachi と「何」モデルによる読み変更を省き、tsqyomi の選択を維持する (デフォルト: False)
        use_sudachi_kanji_yomi (bool): True の場合、Sudachi による同形異音語の読み補正を行う
            use_tsqyomi が True の場合は tsqyomi を優先し、常に無効化される (デフォルト: True)
        predict_nani (bool): True の場合、ONNX モデルで単独形態素として出現した「何」の読みを推定する
            use_tsqyomi が True の場合は tsqyomi を優先し、常に無効化される (デフォルト: True)
        normalize_mode (NormalizeMode): 入力テキストに適用する Unicode 正規化方式
            `"NFC"` は結合文字を正規化し、`"NFKC"` は半角カナなどの互換文字も正規化する (デフォルト: `"None"`)
        iu_pronunciation (IuPronunciation | None): 「言う」や「という」などの定型表現に含まれる「イウ」を、どう発音するかの方式
            "Iu"、"Yuu"、"KanjiIu"、"KanjiYuu"、"YuuBase"、"KanjiYuuBase" のいずれかを指定する。None の場合は辞書の発音のままにし、指定した場合は use_vanilla の設定に関係なく適用される (デフォルト: None)
        use_read_as_pron (bool): True の場合、全ての発音を強制的に読みに置き換える
            助詞「は」も「ハ」になるため、TTS 用途には適さない (デフォルト: False)
            このオプションが True の場合、revert_long_vowels / revert_yotsugana の指定に関係なく
            全ての pron が read で上書きされる
        revert_long_vowels (bool): True の場合、辞書が自動的に長音化した発音を元に復元する
            pron に「ー」が含まれ、かつ orig に「ー」が含まれていない場合のみ復元する
            助詞 (は→ワ, へ→エ) の発音は「ー」を含まないため影響を受けず維持される
            (例: 「効果」コーカ → コウカ / 「人生」ジンセー → ジンセイ) (デフォルト: False)
        revert_yotsugana (bool): True の場合、四つ仮名 (ヅ・ヂ) の発音統合を元に復元する
            read に「ヅ」「ヂ」が含まれている場合、pron を read で上書きする
            (例: 「気づかず」キズカズ → キヅカズ / 「鼻血」ハナジ → ハナヂ) (デフォルト: False)
        jtalk (OpenJTalk | None): 使用する OpenJTalk インスタンス。None ならグローバルインスタンスを使う

    Returns:
        np.ndarray: 音声波形 (dtype: np.float64)
        int: サンプリング周波数 (デフォルト: 48000)
    """
    return synthesize(
        extract_fullcontext(
            text,
            run_marine=run_marine,
            use_vanilla=use_vanilla,
            use_tsqyomi=use_tsqyomi,
            use_sudachi_kanji_yomi=use_sudachi_kanji_yomi,
            predict_nani=predict_nani,
            normalize_mode=normalize_mode,
            iu_pronunciation=iu_pronunciation,
            use_read_as_pron=use_read_as_pron,
            revert_long_vowels=revert_long_vowels,
            revert_yotsugana=revert_yotsugana,
            jtalk=jtalk,
        ),
        speed,
        half_tone,
    )


def apply_postprocessing(
    text: str,
    njd_features: list[NJDFeature],
    *,
    run_marine: bool = False,
    use_vanilla: bool = False,
    use_sudachi_kanji_yomi: bool = True,
    predict_nani: bool = True,
    normalize_mode: NormalizeMode = "None",
    iu_pronunciation: IuPronunciation | None = None,
    use_read_as_pron: bool = False,
    revert_long_vowels: bool = False,
    revert_yotsugana: bool = False,
    jtalk: OpenJTalk | None = None,
) -> list[NJDFeature]:
    """
    加工されていない生の NJD features に後処理を適用する。
    run_frontend() / run_frontend_detailed() の通常経路・tsqyomi 適用経路の双方で呼び出される。

    Args:
        text (str): Unicode 日本語テキスト
        njd_features (list[NJDFeature]): NJDNode 用 features (pyopenjtalk.run_frontend() の戻り値)
        run_marine (bool): marine を用いたアクセント推定を行うか (デフォルト: False)
            有効にするには `pip install pyopenjtalk-plus[marine]` で marine をインストールする必要がある
        use_vanilla (bool): True の場合、pyopenjtalk-plus 独自の後処理を省略し、
            OpenJTalk の素の NJDFeature をそのまま後段に流す
            ただし発音復元オプション (use_read_as_pron 等) は use_vanilla とは独立して適用される (デフォルト: False)
        use_sudachi_kanji_yomi (bool): True の場合、Sudachi による同形異音語の読み補正を行う (デフォルト: True)
        predict_nani (bool): True の場合、ONNX モデルで単独形態素として出現した「何」の読みを推定する (デフォルト: True)
        normalize_mode (NormalizeMode): 入力テキストに適用する Unicode 正規化方式
            `"NFC"` は結合文字を正規化し、`"NFKC"` は半角カナなどの互換文字も正規化する (デフォルト: `"None"`)
        iu_pronunciation (IuPronunciation | None): 「言う」や「という」などの定型表現に含まれる「イウ」を、どう発音するかの方式
            "Iu"、"Yuu"、"KanjiIu"、"KanjiYuu"、"YuuBase"、"KanjiYuuBase" のいずれかを指定する。None の場合は辞書の発音のままにし、指定した場合は use_vanilla の設定に関係なく適用される (デフォルト: None)
        use_read_as_pron (bool): True の場合、全ての発音を強制的に読みに置き換える
            助詞「は」も「ハ」になるため、TTS 用途には適さない (デフォルト: False)
            このオプションが True の場合、revert_long_vowels / revert_yotsugana の指定に関係なく
            全ての pron が read で上書きされる
        revert_long_vowels (bool): True の場合、辞書が自動的に長音化した発音を元に復元する
            pron に「ー」が含まれ、かつ orig に「ー」が含まれていない場合のみ復元する
            助詞 (は→ワ, へ→エ) の発音は「ー」を含まないため影響を受けず維持される
            (例: 「効果」コーカ → コウカ / 「人生」ジンセー → ジンセイ) (デフォルト: False)
        revert_yotsugana (bool): True の場合、四つ仮名 (ヅ・ヂ) の発音統合を元に復元する
            read に「ヅ」「ヂ」が含まれている場合、pron を read で上書きする
            (例: 「気づかず」キズカズ → キヅカズ / 「鼻血」ハナジ → ハナヂ) (デフォルト: False)
        jtalk (OpenJTalk | None): 使用する OpenJTalk インスタンス。None ならグローバルインスタンスを使う

    NOTE:
        発音復元オプション (use_read_as_pron, revert_long_vowels, revert_yotsugana) は
        use_vanilla の設定に関係なく、明示的に指定された場合のみ独立して適用される

    Returns:
        list[NJDFeature]: 後処理後の NJDNode 用 features
    """
    text = normalize_text(text, normalize_mode)
    # 読み保護を指定したユーザー辞書の形態素は、後処理の前の読み・発音・モーラ数・アクセント核を控えておく
    protected_readings = [
        (index, feature["read"], feature["pron"], feature["mora_size"], feature["acc"])
        for index, feature in enumerate(njd_features)
        if feature.get("is_reading_protected", False) is True
    ]
    if use_vanilla is False:
        # フィラーのアクセントは読み変更より先に補正する既存の処理順序を維持する
        njd_features = modify_filler_accent(njd_features)
        if predict_nani is True:
            njd_features = predict_nani_reading(njd_features)
        if use_sudachi_kanji_yomi is True:
            njd_features = modify_kanji_yomi(
                text,
                njd_features,
                _MULTI_READ_KANJI_SET_EXCLUDING_NANI,
            )
        njd_features = suppress_unnatural_auxiliary_u_long_vowel(njd_features)
        njd_features = modify_context_reading(njd_features)
        njd_features = modify_old_province_yomi(njd_features)
        njd_features = restore_loanword_kana(njd_features)
        njd_features = read_unknown_kanji(njd_features, text)

        # 読みを書き換える補正が増えても個別に例外を書かずに済むよう、保護した形態素の読みはここでまとめて辞書の値に戻す
        ## 形態素の位置で戻すので、ここまでの後処理は形態素の数を変えてはならない (変える処理を足すなら文字位置で戻す形に変える)
        for index, read, pronunciation, mora_size, _acc in protected_readings:
            feature = njd_features[index]
            feature["read"] = read
            feature["pron"] = pronunciation
            feature["mora_size"] = mora_size

    # marine には読みとモーラ数を確定した形態素を渡し、補正後の発音に合ったアクセントを推定させる
    ## use_vanilla=True でも、明示的に指定された marine は適用する
    if run_marine:
        pred_njd_features = estimate_accent(njd_features)
        njd_features = preserve_noun_accent(njd_features, pred_njd_features)

    if use_vanilla is False:
        # 読みを確定したあとで接頭辞の後ろのアクセント句を分け、分けたあとの句でアクセントの補正を計算する
        njd_features = split_prefix_accent_phrase(njd_features)
        njd_features = retreat_acc_nuc(njd_features)
        njd_features = modify_acc_after_chaining(njd_features)

    # ユーザー辞書に登録したアクセント核は、marine とアクセントの補正のあとで登録した値に戻して最優先で守る
    ## アクセント句のつながり (chain_flag) は NJD が前後の文脈から決めるものなので守らず、形態素の核の位置だけを登録した値に保つ
    ## use_vanilla=True でも明示的に指定した marine は動くので、この戻しは use_vanilla に関係なく行う
    ## 形態素の数が変わる踊り字の展開より前に戻し、控えたときの位置との対応を保つ
    for index, _read, _pronunciation, _mora_size, acc in protected_readings:
        njd_features[index]["acc"] = acc

    if use_vanilla is False:
        with _resolve_jtalk(jtalk) as resolved_jtalk:
            njd_features = process_odori_features(njd_features, jtalk=resolved_jtalk)
    # 発音復元は use_vanilla の設定に関係なく、明示的に指定された場合のみ独立して適用する
    if use_read_as_pron is True or revert_long_vowels is True or revert_yotsugana is True:
        njd_features = revert_pron_to_read(
            njd_features,
            use_read_as_pron=use_read_as_pron,
            revert_long_vowels=revert_long_vowels,
            revert_yotsugana=revert_yotsugana,
        )
    # 「言う」の発音の方式も、明示的に指定された場合だけ適用する (発音の復元より後に置き、どの入口から呼んでも同じ結果を返す)
    if iu_pronunciation is not None:
        njd_features = normalize_iu(njd_features, iu_pronunciation)
    return njd_features


def run_frontend(
    text: str,
    *,
    run_marine: bool = False,
    use_vanilla: bool = False,
    use_tsqyomi: bool = False,
    use_sudachi_kanji_yomi: bool = True,
    predict_nani: bool = True,
    normalize_mode: NormalizeMode = "None",
    iu_pronunciation: IuPronunciation | None = None,
    use_read_as_pron: bool = False,
    revert_long_vowels: bool = False,
    revert_yotsugana: bool = False,
    jtalk: OpenJTalk | None = None,
) -> list[NJDFeature]:
    """
    OpenJTalk のテキスト処理フロントエンドを実行する。
    pyopenjtalk.run_frontend_detailed() のラッパー。NJD features のみを返す。

    Args:
        text (str): Unicode 日本語テキスト
        run_marine (bool): marine を用いたアクセント推定を行うか (デフォルト: False)
            有効にするには `pip install pyopenjtalk-plus[marine]` で marine をインストールする必要がある
        use_vanilla (bool): True の場合、pyopenjtalk-plus 独自の後処理を省略し、
            OpenJTalk の素の NJDFeature をそのまま後段に流す
            ただし発音復元オプション (use_read_as_pron 等) は use_vanilla とは独立して適用される (デフォルト: False)
        use_tsqyomi (bool): True の場合、ロード済みの tsqyomi で文脈に合う読み候補を選ぶ
            Sudachi と「何」モデルによる読み変更を省き、tsqyomi の選択を維持する (デフォルト: False)
        use_sudachi_kanji_yomi (bool): True の場合、Sudachi による同形異音語の読み補正を行う
            use_tsqyomi が True の場合は tsqyomi を優先し、常に無効化される (デフォルト: True)
        predict_nani (bool): True の場合、ONNX モデルで単独形態素として出現した「何」の読みを推定する
            use_tsqyomi が True の場合は tsqyomi を優先し、常に無効化される (デフォルト: True)
        normalize_mode (NormalizeMode): 入力テキストに適用する Unicode 正規化方式
            `"NFC"` は結合文字を正規化し、`"NFKC"` は半角カナなどの互換文字も正規化する (デフォルト: `"None"`)
        iu_pronunciation (IuPronunciation | None): 「言う」や「という」などの定型表現に含まれる「イウ」を、どう発音するかの方式
            "Iu"、"Yuu"、"KanjiIu"、"KanjiYuu"、"YuuBase"、"KanjiYuuBase" のいずれかを指定する。None の場合は辞書の発音のままにし、指定した場合は use_vanilla の設定に関係なく適用される (デフォルト: None)
        use_read_as_pron (bool): True の場合、全ての発音を強制的に読みに置き換える
            助詞「は」も「ハ」になるため、TTS 用途には適さない (デフォルト: False)
            このオプションが True の場合、revert_long_vowels / revert_yotsugana の指定に関係なく
            全ての pron が read で上書きされる
        revert_long_vowels (bool): True の場合、辞書が自動的に長音化した発音を元に復元する
            pron に「ー」が含まれ、かつ orig に「ー」が含まれていない場合のみ復元する
            助詞 (は→ワ, へ→エ) の発音は「ー」を含まないため影響を受けず維持される
            (例: 「効果」コーカ → コウカ / 「人生」ジンセー → ジンセイ) (デフォルト: False)
        revert_yotsugana (bool): True の場合、四つ仮名 (ヅ・ヂ) の発音統合を元に復元する
            read に「ヅ」「ヂ」が含まれている場合、pron を read で上書きする
            (例: 「気づかず」キズカズ → キヅカズ / 「鼻血」ハナジ → ハナヂ) (デフォルト: False)
        jtalk (OpenJTalk | None): 使用する OpenJTalk インスタンス。None ならグローバルインスタンスを使う

    Returns:
        list[NJDFeature]: NJDNode 用 features
    """
    text = normalize_text(text, normalize_mode)
    with _resolve_jtalk(jtalk) as inference_jtalk:
        if use_vanilla is False:
            # 辞書で読めない異体字だけを通用字へ置き換え、辞書で読める旧字体に固有の読みは残す
            text = normalize_unknown_itaiji(text, inference_jtalk)
        # 読み保護の印はどの辞書の語かが分かる詳細形態素から付けるので、読み保護を指定したユーザー辞書があるときだけ詳細形態素を作る
        # 指定がなければ、run_frontend_detailed() と違って詳細形態素を作らず、普段どおり軽い処理のまま動く
        reading_protection = inference_jtalk.userdic_reading_protection
        needs_morphs = any(reading_protection)
        if use_tsqyomi is True:
            njd_features, morphs = _run_frontend_with_tsqyomi(
                text,
                jtalk=inference_jtalk,
                include_morphs=needs_morphs,
                restore_unknown_katakana=use_vanilla is False,
                modify_numeral_reading=use_vanilla is False,
            )
        elif needs_morphs is True:
            njd_features, morphs = inference_jtalk.run_frontend_detailed(
                text,
                restore_unknown_katakana=use_vanilla is False,
                modify_numeral_reading=use_vanilla is False,
            )
        else:
            njd_features = inference_jtalk.run_frontend(
                text,
                restore_unknown_katakana=use_vanilla is False,
                modify_numeral_reading=use_vanilla is False,
            )
            morphs = []
        mark_user_dictionary_reading_protection(njd_features, morphs, reading_protection)

        # 読みとアクセントの後処理は apply_postprocessing() でまとめて行う
        ## tsqyomi を使うときは、tsqyomi が選んだ読みと競合する Sudachi の読み補正と「何」の読み推定だけを、下の引数で無効化して渡す
        ## 前後の語だけで読みが1つに決まる補正 (modify_context_reading() など) は tsqyomi と競合しないので、apply_postprocessing() の中で tsqyomi を使うときも適用する
        njd_features = apply_postprocessing(
            text,
            njd_features,
            run_marine=run_marine,
            use_vanilla=use_vanilla,
            use_sudachi_kanji_yomi=use_sudachi_kanji_yomi if use_tsqyomi is False else False,
            predict_nani=predict_nani if use_tsqyomi is False else False,
            normalize_mode="None",  # 既に normalize_text() で正規化されているため、再度正規化しない
            iu_pronunciation=iu_pronunciation,
            use_read_as_pron=use_read_as_pron,
            revert_long_vowels=revert_long_vowels,
            revert_yotsugana=revert_yotsugana,
            jtalk=inference_jtalk,
        )
    return njd_features


def run_frontend_detailed(
    text: str,
    *,
    run_marine: bool = False,
    use_vanilla: bool = False,
    use_tsqyomi: bool = False,
    use_sudachi_kanji_yomi: bool = True,
    predict_nani: bool = True,
    normalize_mode: NormalizeMode = "None",
    iu_pronunciation: IuPronunciation | None = None,
    use_read_as_pron: bool = False,
    revert_long_vowels: bool = False,
    revert_yotsugana: bool = False,
    jtalk: OpenJTalk | None = None,
) -> tuple[list[NJDFeature], list[MeCabMorph]]:
    """
    OpenJTalk のテキスト処理フロントエンドを MeCab 形態素詳細付きで実行する。
    MeCab で形態素解析を 1 回だけ実行し、NJD features と MeCab morphs を同時に返す。
    pyopenjtalk.run_frontend() と異なり、MeCab の未知語フラグ・コスト情報付きの morphs も取得できる。

    Args:
        text (str): Unicode 日本語テキスト
        run_marine (bool): marine を用いたアクセント推定を行うか (デフォルト: False)
            有効にするには `pip install pyopenjtalk-plus[marine]` で marine をインストールする必要がある
        use_vanilla (bool): True の場合、pyopenjtalk-plus 独自の後処理を省略し、
            OpenJTalk の素の NJDFeature をそのまま後段に流す
            ただし発音復元オプション (use_read_as_pron 等) は use_vanilla とは独立して適用される (デフォルト: False)
        use_tsqyomi (bool): True の場合、ロード済みの tsqyomi で文脈に合う読み候補を選ぶ
            Sudachi と「何」モデルによる読み変更を省き、tsqyomi の選択を維持する (デフォルト: False)
        use_sudachi_kanji_yomi (bool): True の場合、Sudachi による同形異音語の読み補正を行う
            use_tsqyomi が True の場合は tsqyomi を優先し、常に無効化される (デフォルト: True)
        predict_nani (bool): True の場合、ONNX モデルで単独形態素として出現した「何」の読みを推定する
            use_tsqyomi が True の場合は tsqyomi を優先し、常に無効化される (デフォルト: True)
        normalize_mode (NormalizeMode): 入力テキストに適用する Unicode 正規化方式
            `"NFC"` は結合文字を正規化し、`"NFKC"` は半角カナなどの互換文字も正規化する (デフォルト: `"None"`)
        iu_pronunciation (IuPronunciation | None): 「言う」や「という」などの定型表現に含まれる「イウ」を、どう発音するかの方式
            "Iu"、"Yuu"、"KanjiIu"、"KanjiYuu"、"YuuBase"、"KanjiYuuBase" のいずれかを指定する。None の場合は辞書の発音のままにし、指定した場合は use_vanilla の設定に関係なく適用される (デフォルト: None)
        use_read_as_pron (bool): True の場合、全ての発音を強制的に読みに置き換える
            助詞「は」も「ハ」になるため、TTS 用途には適さない (デフォルト: False)
            このオプションが True の場合、revert_long_vowels / revert_yotsugana の指定に関係なく
            全ての pron が read で上書きされる
        revert_long_vowels (bool): True の場合、辞書が自動的に長音化した発音を元に復元する
            pron に「ー」が含まれ、かつ orig に「ー」が含まれていない場合のみ復元する
            助詞 (は→ワ, へ→エ) の発音は「ー」を含まないため影響を受けず維持される
            (例: 「効果」コーカ → コウカ / 「人生」ジンセー → ジンセイ) (デフォルト: False)
        revert_yotsugana (bool): True の場合、四つ仮名 (ヅ・ヂ) の発音統合を元に復元する
            read に「ヅ」「ヂ」が含まれている場合、pron を read で上書きする
            (例: 「気づかず」キズカズ → キヅカズ / 「鼻血」ハナジ → ハナヂ) (デフォルト: False)
        jtalk (OpenJTalk | None): 使用する OpenJTalk インスタンス。None ならグローバルインスタンスを使う

    Returns:
        tuple[list[NJDFeature], list[MeCabMorph]]: (NJD features, MeCab morphs)
            - NJD features: pyopenjtalk.run_frontend() と同一の結果が得られる
            - MeCab morphs: pyopenjtalk.run_mecab_detailed()[1] と同一の結果が得られる
    """
    text = normalize_text(text, normalize_mode)
    with _resolve_jtalk(jtalk) as inference_jtalk:
        if use_vanilla is False:
            # 辞書で読めない異体字だけを通用字へ置き換え、辞書で読める旧字体に固有の読みは残す
            text = normalize_unknown_itaiji(text, inference_jtalk)
        if use_tsqyomi is True:
            njd_features, morphs = _run_frontend_with_tsqyomi(
                text,
                jtalk=inference_jtalk,
                include_morphs=True,
                restore_unknown_katakana=use_vanilla is False,
                modify_numeral_reading=use_vanilla is False,
            )
        else:
            njd_features, morphs = inference_jtalk.run_frontend_detailed(
                text,
                restore_unknown_katakana=use_vanilla is False,
                modify_numeral_reading=use_vanilla is False,
            )
        mark_user_dictionary_reading_protection(
            njd_features,
            morphs,
            inference_jtalk.userdic_reading_protection,
        )
        njd_features = apply_postprocessing(
            text,
            njd_features,
            run_marine=run_marine,
            use_vanilla=use_vanilla,
            use_sudachi_kanji_yomi=use_sudachi_kanji_yomi if use_tsqyomi is False else False,
            predict_nani=predict_nani if use_tsqyomi is False else False,
            normalize_mode="None",  # 既に normalize_text() で正規化されているため、再度正規化しない
            iu_pronunciation=iu_pronunciation,
            use_read_as_pron=use_read_as_pron,
            revert_long_vowels=revert_long_vowels,
            revert_yotsugana=revert_yotsugana,
            jtalk=inference_jtalk,
        )
    return njd_features, morphs


def _run_frontend_with_tsqyomi(
    text: str,
    *,
    jtalk: OpenJTalk,
    include_morphs: bool = True,
    restore_unknown_katakana: bool = True,
    modify_numeral_reading: bool = True,
) -> tuple[list[NJDFeature], list[MeCabMorph]]:
    """
    tsqyomi で MeCab feature を選び、NJD 処理後の features と morphs を返す。
    Python 側後処理は apply_postprocessing() に委譲する。

    Args:
        text (str): 正規化済みの Unicode 日本語テキスト
        jtalk (OpenJTalk): 候補解析と NJD 処理に使う OpenJTalk インスタンス
        include_morphs (bool): 詳細形態素列を返す場合は True
        restore_unknown_katakana (bool): True の場合、未知カタカナ語の品詞とアクセントを MeCab の結果から復元する
        modify_numeral_reading (bool): True の場合、分数の分母の「分」を「ブン」、2つ以上続く「〇」を「マル」と読む

    Returns:
        tuple[list[NJDFeature], list[MeCabMorph]]: NJD features と差し替え後の形態素列
    """

    # tsqyomi を使う場合のみインポートする
    from .tsqyomi.inference import select_mecab_features_with_tsqyomi

    mecab_features, morphs = select_mecab_features_with_tsqyomi(
        text,
        jtalk,
        include_morphs=include_morphs,
    )
    njd_features = jtalk.run_njd_from_mecab(
        mecab_features,
        restore_unknown_katakana=restore_unknown_katakana,
        modify_numeral_reading=modify_numeral_reading,
    )
    return njd_features, morphs


def make_label(njd_features: list[NJDFeature], jtalk: OpenJTalk | None = None) -> list[str]:
    """
    HTS 音声合成用のフルコンテキストラベルを返す。

    Args:
        njd_features (list[NJDFeature]): NJDNode 用 features (pyopenjtalk.run_frontend() の戻り値)
        jtalk (OpenJTalk | None): 使用する OpenJTalk インスタンス。None ならグローバルインスタンスを使う

    Returns:
        list[str]: フルコンテキストラベル文字列のリスト
    """
    with _resolve_jtalk(jtalk) as resolved_jtalk:
        return resolved_jtalk.make_label(njd_features)


def make_phoneme_mapping(
    njd_features: list[NJDFeature],
    morphs: list[MeCabMorph] | None = None,
    jtalk: OpenJTalk | None = None,
    *,
    caller_text: str | None = None,
    normalize_mode: NormalizeMode = "None",
    is_non_pause_symbol: Callable[[str], bool] = default_is_non_pause_symbol,
) -> list[SurfacePhonemeMapping]:
    """
    NJD features から各形態素に対応する音素列のマッピングを返す。
    Cython 側の OpenJTalk.make_phoneme_mapping() で基本マッピングを取得し、
    morphs が渡された場合は MeCab morphs とアライメントして is_unknown / is_ignored を付与する。

    morphs を省略した場合は is_unknown=False 、is_ignored は音素列の空判定から推定される。
    このとき `char_span` は NJD 後処理後の surface を連結した座標系であり、
    `g2p_mapping(text=...)` が返す呼び出し元入力文上の半開区間とは一致しない。
    morphs を渡す場合、踊り字展開や数字正規化により NJD と MeCab の粒度がずれることがあるが、
    アライメントロジックが自動的に補正する。音素列自体は常に正しい値が得られる。
    句読点などの記号は surface として保持し、is_non_pause_symbol() が True を返す記号の phonemes は空にする。
    False を返す記号には、JPCommon が短ポーズを生成しなかった場合も phonemes に `pau` を割り当てるので、音素を連結した結果が extract_phonemes() と一致しないことがある。
    morphs 付きで対応する morph を特定できないエントリの `char_span` は、未特定を表す `(0, 0)` になる。

    Args:
        njd_features (list[NJDFeature]): NJDNode 用 features (pyopenjtalk.run_frontend() の戻り値)
        morphs (list[MeCabMorph] | None): MeCab の形態素解析結果 (pyopenjtalk.run_frontend_detailed() の戻り値)
            None の場合は is_unknown / is_ignored の推定精度が下がる
        jtalk (OpenJTalk | None): 使用する OpenJTalk インスタンス。None ならグローバルインスタンスを使う
        caller_text (str | None): `char_span` の座標系に使う正規化前の入力文
            None の場合は MeCab 正規化本文上の座標を使う
        normalize_mode (NormalizeMode): caller_text に適用した Unicode 正規化方式 (デフォルト: `"None"`)
        is_non_pause_symbol (Callable[[str], bool]): True を返した記号は音素なしで保持し、False を返した短ポーズ記号には `pau` を割り当てる。
            既定では括弧・引用符だけを音素なしで保持する

    Returns:
        list[SurfacePhonemeMapping]: 各形態素に対応する音素列のマッピング

    Raises:
        ValueError: caller_text と MeCab 正規化本文の対応付けに失敗した場合。
            `g2p_mapping()` 経由でも `_build_caller_text_spans_by_mecab_character()` からそのまま伝播する
    """

    # 借り出しの間は辞書交換が待機するため、Cython 呼び出しと座標変換が同じ辞書で完結する
    with _resolve_jtalk(jtalk) as inference_jtalk:
        return _mapping.make_phoneme_mapping(
            njd_features,
            morphs,
            inference_jtalk,
            caller_text=caller_text,
            normalize_mode=normalize_mode,
            is_non_pause_symbol=is_non_pause_symbol,
        )


def mecab_dict_index(path: str, out_path: str, dn_mecab: str | None = None) -> None:
    """
    OpenJTalk 用のユーザー辞書を CSV からビルドする。
    CSV は naist-jdic 互換の品詞体系で記述する必要がある。

    Args:
        path (str): OpenJTalk 用のユーザー辞書 CSV (naist-jdic 互換) のパス
        out_path (str): OpenJTalk 用のユーザー辞書ファイル (.dic) の出力先パス
        dn_mecab (str | None): OpenJTalk/naist-jdic 互換の MeCab システム辞書のパス
    """
    if not exists(path):
        raise FileNotFoundError(f"No such file or directory: {path}")
    if dn_mecab is None:
        dn_mecab = OPEN_JTALK_DICT_DIR.decode("utf-8")
    if not exists(dn_mecab):
        raise FileNotFoundError(f"No such file or directory: {dn_mecab}")
    out_path_parent = Path(out_path).resolve().parent
    if out_path_parent.exists() is False:
        raise FileNotFoundError(f"No such directory: {out_path_parent}")
    r = _mecab_dict_index(dn_mecab.encode("utf-8"), path.encode("utf-8"), out_path.encode("utf-8"))

    # NOTE: mecab load returns 1 if success, but mecab_dict_index return the opposite
    # yeah it's confusing...
    if r != 0:
        raise RuntimeError("Failed to create user dictionary")


def update_global_jtalk_with_user_dict(
    paths: str | list[str] | list[UserDictionaryEntry],
) -> None:
    """
    グローバル OpenJTalk インスタンスにユーザー辞書を適用する。
    注意: この関数を実行すると、pyopenjtalk モジュールのグローバル状態が変更される。

    Args:
        paths (str | list[str] | list[UserDictionaryEntry]): ユーザー辞書ファイル (.dic) と読み保護の指定
            UserDictionaryEntry の is_reading_protected が True の場合、そのユーザー辞書が与えた読みを tsqyomi を含む後段の読み補正から保護する

    Raises:
        ValueError: 空のリスト、UserDictionaryEntry のキー、またはリスト内のパスが不正な場合
        TypeError: リストの要素型が不正か、文字列と UserDictionaryEntry が混在する場合
        FileNotFoundError: 指定したユーザー辞書ファイルが存在しない場合
        RuntimeError: OpenJTalk またはユーザー辞書の初期化に失敗した場合
    """

    if isinstance(paths, str):
        dic_paths = paths.split(",")
        reading_protection = [False] * len(dic_paths)
    else:
        raw_paths = cast(Sequence[object], paths)
        if len(raw_paths) == 0:
            raise ValueError("paths must contain at least one user dictionary")
        # 未対応の要素型と、対応済みの2形式を混在させた入力を別のエラーとして報告する
        if any(isinstance(entry, (str, dict)) is False for entry in raw_paths):
            raise TypeError("paths must contain only strings or UserDictionaryEntry values")
        is_string_list = all(isinstance(entry, str) for entry in raw_paths)
        is_entry_list = all(isinstance(entry, dict) for entry in raw_paths)
        if is_string_list is True:
            dic_paths = cast(list[str], paths)
            reading_protection = [False] * len(dic_paths)
        elif is_entry_list is True:
            dictionary_entries = cast(list[UserDictionaryEntry], paths)
            dic_paths = []
            reading_protection = []
            for entry in dictionary_entries:
                if set(entry) != {"dic_path", "is_reading_protected"}:
                    raise ValueError(
                        "UserDictionaryEntry must contain dic_path and is_reading_protected"
                    )
                # TypedDict の注釈だけでは実行時入力を制限できないため、辞書を開く前に型も検査する
                if type(entry["dic_path"]) is not str or entry["dic_path"] == "":
                    raise TypeError("UserDictionaryEntry.dic_path must be a non-empty string")
                if type(entry["is_reading_protected"]) is not bool:
                    raise TypeError("UserDictionaryEntry.is_reading_protected must be bool")
                dic_paths.append(entry["dic_path"])
                reading_protection.append(entry["is_reading_protected"])
        else:
            raise TypeError("paths must not mix strings and UserDictionaryEntry values")

        # リストの1要素を C 側で複数辞書と解釈すると、読み保護フラグとの対応が崩れる
        if any("," in dic_path for dic_path in dic_paths):
            raise ValueError("user dictionary paths in a list must not contain commas")

    # 連結前の各要素を検査し、空要素と存在しないパスを元の表記で報告する
    for dic_path in dic_paths:
        if dic_path.strip() == "":
            raise ValueError("user dictionary path must not be empty")
    for dic_path in dic_paths:
        if not exists(dic_path):
            raise FileNotFoundError(f"No such file or directory: {dic_path}")
    paths_str = ",".join(dic_paths)

    with _global_jtalk_swap_lock:
        # 新しい辞書の初期化中は旧インスタンスを引き続き利用可能にする
        new_jtalk = OpenJTalk(
            dn_mecab=OPEN_JTALK_DICT_DIR,
            userdic=paths_str.encode("utf-8"),
            userdic_reading_protection=reading_protection,
        )
        _global_jtalk.replace(new_jtalk)


def unset_user_dict() -> None:
    """
    ユーザー辞書の適用を解除する。
    注意: この関数を実行すると、pyopenjtalk モジュールのグローバル状態が変更される。
    """
    with _global_jtalk_swap_lock:
        # デフォルト辞書の初期化中は旧インスタンスを引き続き利用可能にする
        new_jtalk = OpenJTalk(dn_mecab=OPEN_JTALK_DICT_DIR)
        _global_jtalk.replace(new_jtalk)


def run_mecab(text: str, jtalk: OpenJTalk | None = None) -> list[str]:
    """
    MeCab で形態素解析を実行する。"記号,空白" は除外される。
    全トークン（未知語フラグ・コスト情報含む）が必要な場合は代わりに pyopenjtalk.run_mecab_detailed() を使うこと。

    Args:
        text (str): Unicode 日本語テキスト
        jtalk (OpenJTalk | None): 使用する OpenJTalk インスタンス。None ならグローバルインスタンスを使う

    Returns:
        list[str]: MeCab の feature 文字列のリスト ("記号,空白" を除く)
    """
    with _resolve_jtalk(jtalk) as jtalk:
        return jtalk.run_mecab(text)


def run_mecab_detailed(
    text: str,
    jtalk: OpenJTalk | None = None,
) -> tuple[list[str], list[MeCabMorph]]:
    """
    MeCab を1回だけ実行し、run_mecab() 互換の features と詳細 morphs を返す。
    詳細 morphs には記号,空白 も含まれ、各トークンの is_unknown フラグにより辞書登録の有無を判定できる。

    Args:
        text (str): Unicode 日本語テキスト
        jtalk (OpenJTalk | None): 使用する OpenJTalk インスタンス。None ならグローバルインスタンスを使う

    Returns:
        tuple[list[str], list[MeCabMorph]]: (フィルタ済み features, 全 morphs)
            features は pyopenjtalk.run_mecab() と同等 (記号,空白 を除く)
            morphs は未知語フラグ・コスト情報付きの全トークン (記号,空白 も含む)
    """

    with _resolve_jtalk(jtalk) as jtalk:
        return jtalk.run_mecab_detailed(text)


def run_mecab_nbest_features(
    text: str,
    max_paths: int = 5,
    *,
    jtalk: OpenJTalk | None = None,
) -> list[MeCabNBestPath]:
    """
    MeCab の n-best 候補を features / morphs / path_cost 付きで返す。
    features は pyopenjtalk.run_njd_from_mecab() に渡せる形式で、
    OpenJTalk の後処理を維持したまま候補パスごとの読み・発音を比較できる。
    ただし n-best の morphs は、pyopenjtalk.run_mecab_detailed() の記号単位分割を適用しない。

    Args:
        text (str): Unicode 日本語テキスト
        max_paths (int): 取得する最大候補数 (MeCab の上限に合わせて 1-512 を受け付ける)
        jtalk (OpenJTalk | None): 使用する OpenJTalk インスタンス (None ならグローバルインスタンスを使う)

    Returns:
        list[MeCabNBestPath]: MeCab n-best 候補パスのリスト
    """

    with _resolve_jtalk(jtalk) as jtalk:
        return jtalk.run_mecab_nbest_features(text, max_paths)


def run_njd_from_mecab(
    mecab_features: list[str],
    jtalk: OpenJTalk | None = None,
    *,
    restore_unknown_katakana: bool = False,
    modify_numeral_reading: bool = True,
) -> list[NJDFeature]:
    """
    MeCab の feature 文字列のリストから NJD 処理を実行する。
    pyopenjtalk.run_mecab() の戻り値をそのまま渡す想定。数字正規化・アクセント句設定・長音処理などの NJD ルールが適用される。

    Args:
        mecab_features (list[str]): MeCab の feature 文字列のリスト
        jtalk (OpenJTalk | None): 使用する OpenJTalk インスタンス。None ならグローバルインスタンスを使う
        restore_unknown_katakana (bool): True の場合、未知カタカナ語の品詞とアクセントを MeCab の結果から復元する (デフォルト: False)
        modify_numeral_reading (bool): True の場合、分数の分母の「分」を「ブン」、2つ以上続く「〇」を「マル」と読む (デフォルト: True)

    Returns:
        list[NJDFeature]: NJDNode 用 features
    """
    with _resolve_jtalk(jtalk) as jtalk:
        return jtalk.run_njd_from_mecab(
            mecab_features,
            restore_unknown_katakana,
            modify_numeral_reading,
        )


def build_mecab_dictionary(dn_mecab: str | None = None) -> None:
    """
    MeCab システム辞書を再ビルドする。

    Args:
        dn_mecab (str | None): MeCab システム辞書のディレクトリパス (None の場合はグローバル辞書ディレクトリを使う、デフォルト: None)
    """
    if dn_mecab is None:
        dn_mecab = OPEN_JTALK_DICT_DIR.decode("utf-8")

    # remove *.dic / *.bin files
    dict_path = Path(dn_mecab)
    for file in dict_path.glob("*.dic"):
        file.unlink()
    for file in dict_path.glob("*.bin"):
        file.unlink()

    # Build mecab dictionary
    r = _build_mecab_dictionary(dn_mecab.encode("utf-8"))

    # NOTE: mecab load returns 1 if success, but mecab_dict_index return the opposite
    # yeah it's confusing...
    if r != 0:
        raise RuntimeError("Failed to build dictionary")
