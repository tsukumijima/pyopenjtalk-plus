from __future__ import annotations

import atexit
import os
from collections.abc import Callable, Generator, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager
from importlib.resources import as_file, files
from os.path import exists
from pathlib import Path
from threading import Lock
from typing import Any, Literal, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt


try:
    from .version import __version__  # noqa
except ImportError:
    raise ImportError("BUG: version.py doesn't exist. Please file a bug report.")

from .htsengine import HTSEngine
from .openjtalk import OpenJTalk
from .openjtalk import build_mecab_dictionary as _build_mecab_dictionary
from .openjtalk import mecab_dict_index as _mecab_dict_index
from .types import (
    MeCabMorph,
    MeCabNBestPath,
    NJDFeature,
    SurfacePhonemeMapping,
    UserDictionaryEntry,
)
from .utils import (
    merge_njd_marine_features,
    modify_acc_after_chaining,
    modify_kanji_yomi,
    normalize_text,
    predict_nani_reading,
    process_odori_features,
    retreat_acc_nuc,
    revert_pron_to_read,
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

# 踊り字展開 (process_odori_features()) で morph/NJD のずれを検出するための文字集合
_ODORI_CHARS = frozenset("々ゝゞヽヾ")
# 数字正規化後の NJD ノードと MeCab morph を局所的に対応させるための文字集合
_DIGIT_MORPH_SURFACES = frozenset("０１２３４５６７８９0123456789")
_KANJI_NUMBER_SURFACES = frozenset("一二三四五六七八九十百千万億兆〇零")

_T = TypeVar("_T")


def _lazy_init() -> None:
    # pyopenjtalk-plus では辞書のダウンロード処理は削除されているが、
    # _lazy_init() を直接呼び出している VOICEVOX などへの互換性のために残置している
    pass


def _global_instance_manager(
    instance_factory: Union[Callable[[], _T], None] = None,
    instance: Union[_T, None] = None,
) -> Callable[[], AbstractContextManager[_T]]:
    assert instance_factory is not None or instance is not None
    _instance = instance
    mutex = Lock()

    @contextmanager
    def manager() -> Generator[_T, None, None]:
        nonlocal _instance
        with mutex:
            if _instance is None:
                _instance = instance_factory()  # type: ignore
            yield _instance

    return manager


# Global instance of OpenJTalk
_global_jtalk = _global_instance_manager(lambda: OpenJTalk(dn_mecab=OPEN_JTALK_DICT_DIR))
# 連続する update / unset が古い manager ではなく直前の manager を待たずに差し替えるのを防ぐ
_global_jtalk_swap_lock = Lock()


@contextmanager
def _resolve_jtalk(jtalk: Union[OpenJTalk, None]) -> Generator[OpenJTalk, None, None]:
    # 呼び出し元がインスタンスを渡した場合はグローバル mutex を取らず、そのまま使う
    ## _global_jtalk の mutex は非リエントラントなので、解決済みインスタンスを下位へ渡して再取得を避ける
    if jtalk is not None:
        yield jtalk
    else:
        with _global_jtalk() as instance:
            yield instance


# Global instance of HTSEngine
# mei_normal.voice is used as default
_global_htsengine = _global_instance_manager(lambda: HTSEngine(DEFAULT_HTS_VOICE))
# Global instance of Marine
_global_marine = None


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
    normalize_mode: Literal["None", "NFC", "NFKC"] = "None",
    use_read_as_pron: bool = False,
    revert_long_vowels: bool = False,
    revert_yotsugana: bool = False,
    jtalk: Union[OpenJTalk, None] = None,
) -> Union[list[str], str]:
    """
    文字から音素への変換処理 (G2P) 。pyopenjtalk.run_frontend() のラッパー。

    Args:
        text (str): Unicode 日本語テキスト
        kana (bool): True の場合、カタカナで発音を返す。False の場合は音素形式。デフォルト: False
        join (bool): True の場合、音素またはカタカナを単一の文字列に連結する。デフォルト: True
        run_marine (bool): marine を用いたアクセント推定を行うか。デフォルト: False
            有効にするには `pip install pyopenjtalk-plus[marine]` で marine をインストールする必要がある
        use_vanilla (bool): True の場合、pyopenjtalk-plus 独自の後処理を省略し、
            OpenJTalk の素の NJDFeature をそのまま後段に流す。
            ただし発音復元オプション (use_read_as_pron 等) は use_vanilla とは独立して適用される
            デフォルト: False
        use_tsqyomi (bool): True の場合、ロード済みの tsqyomi で文脈に合う読み候補を選ぶ。
            Sudachi と「何」モデルによる読み変更を省き、tsqyomi の選択を維持する
            デフォルト: False
        use_sudachi_kanji_yomi (bool): True の場合、Sudachi による同形異音語の読み補正を行う。
            use_tsqyomi が True の場合は tsqyomi を優先し、常に無効化される
            デフォルト: True
        predict_nani (bool): True の場合、ONNX モデルで単独形態素として出現した「何」の読みを推定する。
            use_tsqyomi が True の場合は tsqyomi を優先し、常に無効化される
            デフォルト: True
        normalize_mode (Literal["None", "NFC", "NFKC"]): 入力テキストに適用する Unicode 正規化方式。
            `"NFC"` は結合文字を正規化し、`"NFKC"` は半角カナなどの互換文字も正規化する
            デフォルト: `"None"`
        use_read_as_pron (bool): True の場合、全ての発音を強制的に読みに置き換える。
            助詞「は」も「ハ」になるため、TTS 用途には適さない。デフォルト: False
            このオプションが True の場合、revert_long_vowels / revert_yotsugana の指定に関係なく
            全ての pron が read で上書きされる
        revert_long_vowels (bool): True の場合、辞書が自動的に長音化した発音を元に復元する。
            pron に「ー」が含まれ、かつ orig に「ー」が含まれていない場合のみ復元する。
            助詞 (は→ワ, へ→エ) の発音は「ー」を含まないため影響を受けず維持される。
            (例: 「効果」コーカ → コウカ / 「人生」ジンセー → ジンセイ)
            デフォルト: False
        revert_yotsugana (bool): True の場合、四つ仮名 (ヅ・ヂ) の発音統合を元に復元する。
            read に「ヅ」「ヂ」が含まれている場合、pron を read で上書きする。
            (例: 「気づかず」キズカズ → キヅカズ / 「鼻血」ハナジ → ハナヂ)
            デフォルト: False
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
        use_read_as_pron=use_read_as_pron,
        revert_long_vowels=revert_long_vowels,
        revert_yotsugana=revert_yotsugana,
        jtalk=jtalk,
    )

    if not kana:
        # run_frontend() 側でグローバル mutex は解放済みなので、音素抽出だけ短く取り直す
        with _resolve_jtalk(jtalk) as jtalk:
            prons = jtalk.extract_phonemes(njd_features)
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


def g2p_mapping(
    text: str,
    *,
    run_marine: bool = False,
    use_vanilla: bool = False,
    use_tsqyomi: bool = False,
    use_sudachi_kanji_yomi: bool = True,
    predict_nani: bool = True,
    normalize_mode: Literal["None", "NFC", "NFKC"] = "None",
    use_read_as_pron: bool = False,
    revert_long_vowels: bool = False,
    revert_yotsugana: bool = False,
    jtalk: Union[OpenJTalk, None] = None,
) -> list[SurfacePhonemeMapping]:
    """
    テキストから形態素-音素マッピングを一括で取得する便利ラッパー。
    内部で pyopenjtalk.run_frontend_detailed() と pyopenjtalk.make_phoneme_mapping() を呼び出し、
    MeCab 未知語フラグ・無視トークン情報付きの音素マッピングを返す。

    Args:
        text (str): Unicode 日本語テキスト
        run_marine (bool): marine を用いたアクセント推定を行うか。デフォルト: False
            有効にするには `pip install pyopenjtalk-plus[marine]` で marine をインストールする必要がある
        use_vanilla (bool): True の場合、pyopenjtalk-plus 独自の後処理を省略し、
            OpenJTalk の素の NJDFeature をそのまま後段に流す。
            ただし発音復元オプション (use_read_as_pron 等) は use_vanilla とは独立して適用される
            デフォルト: False
        use_tsqyomi (bool): True の場合、ロード済みの tsqyomi で文脈に合う読み候補を選ぶ。
            Sudachi と「何」モデルによる読み変更を省き、tsqyomi の選択を維持する
            デフォルト: False
        use_sudachi_kanji_yomi (bool): True の場合、Sudachi による同形異音語の読み補正を行う。
            use_tsqyomi が True の場合は tsqyomi を優先し、常に無効化される
            デフォルト: True
        predict_nani (bool): True の場合、ONNX モデルで単独形態素として出現した「何」の読みを推定する。
            use_tsqyomi が True の場合は tsqyomi を優先し、常に無効化される
            デフォルト: True
        normalize_mode (Literal["None", "NFC", "NFKC"]): 入力テキストに適用する Unicode 正規化方式。
            `"NFC"` は結合文字を正規化し、`"NFKC"` は半角カナなどの互換文字も正規化する
            デフォルト: `"None"`
        use_read_as_pron (bool): True の場合、全ての発音を強制的に読みに置き換える。
            助詞「は」も「ハ」になるため、TTS 用途には適さない。デフォルト: False
            このオプションが True の場合、revert_long_vowels / revert_yotsugana の指定に関係なく
            全ての pron が read で上書きされる
        revert_long_vowels (bool): True の場合、辞書が自動的に長音化した発音を元に復元する。
            pron に「ー」が含まれ、かつ orig に「ー」が含まれていない場合のみ復元する。
            助詞 (は→ワ, へ→エ) の発音は「ー」を含まないため影響を受けず維持される。
            (例: 「効果」コーカ → コウカ / 「人生」ジンセー → ジンセイ)
            デフォルト: False
        revert_yotsugana (bool): True の場合、四つ仮名 (ヅ・ヂ) の発音統合を元に復元する。
            read に「ヅ」「ヂ」が含まれている場合、pron を read で上書きする。
            (例: 「気づかず」キズカズ → キヅカズ / 「鼻血」ハナジ → ハナヂ)
            デフォルト: False
        jtalk (OpenJTalk | None): 使用する OpenJTalk インスタンス。None ならグローバルインスタンスを使う

    Returns:
        list[SurfacePhonemeMapping]: 各形態素に対応する音素列のマッピング (未知語・無視トークン情報付き)
    """

    njd_features, morphs = run_frontend_detailed(
        text,
        run_marine=run_marine,
        use_vanilla=use_vanilla,
        use_tsqyomi=use_tsqyomi,
        use_sudachi_kanji_yomi=use_sudachi_kanji_yomi,
        predict_nani=predict_nani,
        normalize_mode=normalize_mode,
        use_read_as_pron=use_read_as_pron,
        revert_long_vowels=revert_long_vowels,
        revert_yotsugana=revert_yotsugana,
        jtalk=jtalk,
    )
    return make_phoneme_mapping(njd_features, morphs=morphs, jtalk=jtalk)


def load_marine_model(model_dir: Union[str, None] = None, dict_dir: Union[str, None] = None):
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

    marine_feature = convert_njd_feature_to_marine_feature(njd_features)
    marine_results = cast(
        dict[str, Any],
        _global_marine.predict([marine_feature], require_open_jtalk_format=True),
    )
    njd_features = merge_njd_marine_features(njd_features, marine_results)
    return njd_features


def modify_filler_accent(njd: list[NJDFeature]) -> list[NJDFeature]:
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
    return_njd = []
    for f_input, f_pred in zip(input_njd, predicted_njd):
        if f_pred["pos"] == "名詞" and f_pred["string"] not in MULTI_READ_KANJI_LIST:
            f_pred["acc"] = f_input["acc"]
        return_njd.append(f_pred)

    return return_njd


def extract_fullcontext(
    text: str,
    *,
    run_marine: bool = False,
    use_vanilla: bool = False,
    use_tsqyomi: bool = False,
    use_sudachi_kanji_yomi: bool = True,
    predict_nani: bool = True,
    normalize_mode: Literal["None", "NFC", "NFKC"] = "None",
    use_read_as_pron: bool = False,
    revert_long_vowels: bool = False,
    revert_yotsugana: bool = False,
    jtalk: Union[OpenJTalk, None] = None,
) -> list[str]:
    """
    テキストからフルコンテキストラベルを抽出する。

    Args:
        text (str): Unicode 日本語テキスト
        run_marine (bool): marine を用いたアクセント推定を行うか。デフォルト: False
            有効にするには `pip install pyopenjtalk-plus[marine]` で marine をインストールする必要がある
        use_vanilla (bool): True の場合、pyopenjtalk-plus 独自の後処理を省略し、
            OpenJTalk の素の NJDFeature をそのまま後段に流す。
            ただし発音復元オプション (use_read_as_pron 等) は use_vanilla とは独立して適用される
            デフォルト: False
        use_tsqyomi (bool): True の場合、ロード済みの tsqyomi で文脈に合う読み候補を選ぶ。
            Sudachi と「何」モデルによる読み変更を省き、tsqyomi の選択を維持する
            デフォルト: False
        use_sudachi_kanji_yomi (bool): True の場合、Sudachi による同形異音語の読み補正を行う。
            use_tsqyomi が True の場合は tsqyomi を優先し、常に無効化される
            デフォルト: True
        predict_nani (bool): True の場合、ONNX モデルで単独形態素として出現した「何」の読みを推定する。
            use_tsqyomi が True の場合は tsqyomi を優先し、常に無効化される
            デフォルト: True
        normalize_mode (Literal["None", "NFC", "NFKC"]): 入力テキストに適用する Unicode 正規化方式。
            `"NFC"` は結合文字を正規化し、`"NFKC"` は半角カナなどの互換文字も正規化する
            デフォルト: `"None"`
        use_read_as_pron (bool): True の場合、全ての発音を強制的に読みに置き換える。
            助詞「は」も「ハ」になるため、TTS 用途には適さない。デフォルト: False
            このオプションが True の場合、revert_long_vowels / revert_yotsugana の指定に関係なく
            全ての pron が read で上書きされる
        revert_long_vowels (bool): True の場合、辞書が自動的に長音化した発音を元に復元する。
            pron に「ー」が含まれ、かつ orig に「ー」が含まれていない場合のみ復元する。
            助詞 (は→ワ, へ→エ) の発音は「ー」を含まないため影響を受けず維持される。
            (例: 「効果」コーカ → コウカ / 「人生」ジンセー → ジンセイ)
            デフォルト: False
        revert_yotsugana (bool): True の場合、四つ仮名 (ヅ・ヂ) の発音統合を元に復元する。
            read に「ヅ」「ヂ」が含まれている場合、pron を read で上書きする。
            (例: 「気づかず」キズカズ → キヅカズ / 「鼻血」ハナジ → ハナヂ)
            デフォルト: False
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
        use_read_as_pron=use_read_as_pron,
        revert_long_vowels=revert_long_vowels,
        revert_yotsugana=revert_yotsugana,
        jtalk=jtalk,
    )
    return make_label(njd_features, jtalk=jtalk)


def synthesize(
    labels: Union[list[str], tuple[Any, list[str]]],
    speed: float = 1.0,
    half_tone: float = 0.0,
) -> tuple[npt.NDArray[np.float64], int]:
    """
    OpenJTalk の音声合成バックエンドを実行する。

    Args:
        labels (list): フルコンテキストラベル
        speed (float): 話速 (デフォルト 1.0)
        half_tone (float): 追加の半音 (デフォルト 0)

    Returns:
        np.ndarray: 音声波形 (dtype: np.float64)
        int: サンプリング周波数 (デフォルト: 48000)
    """
    if isinstance(labels, tuple) and len(labels) == 2:
        labels = labels[1]

    global _global_htsengine
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
    normalize_mode: Literal["None", "NFC", "NFKC"] = "None",
    use_read_as_pron: bool = False,
    revert_long_vowels: bool = False,
    revert_yotsugana: bool = False,
    jtalk: Union[OpenJTalk, None] = None,
) -> tuple[npt.NDArray[np.float64], int]:
    """
    テキストから音声を合成する。

    Args:
        text (str): Unicode 日本語テキスト
        speed (float): 話速 (デフォルト 1.0)
        half_tone (float): 追加の半音 (デフォルト 0)
        run_marine (bool): marine を用いたアクセント推定を行うか。デフォルト: False
            有効にするには `pip install pyopenjtalk-plus[marine]` で marine をインストールする必要がある
        use_vanilla (bool): True の場合、pyopenjtalk-plus 独自の後処理を省略し、
            OpenJTalk の素の NJDFeature をそのまま後段に流す。
            ただし発音復元オプション (use_read_as_pron 等) は use_vanilla とは独立して適用される
            デフォルト: False
        use_tsqyomi (bool): True の場合、ロード済みの tsqyomi で文脈に合う読み候補を選ぶ。
            Sudachi と「何」モデルによる読み変更を省き、tsqyomi の選択を維持する
            デフォルト: False
        use_sudachi_kanji_yomi (bool): True の場合、Sudachi による同形異音語の読み補正を行う。
            use_tsqyomi が True の場合は tsqyomi を優先し、常に無効化される
            デフォルト: True
        predict_nani (bool): True の場合、ONNX モデルで単独形態素として出現した「何」の読みを推定する。
            use_tsqyomi が True の場合は tsqyomi を優先し、常に無効化される
            デフォルト: True
        normalize_mode (Literal["None", "NFC", "NFKC"]): 入力テキストに適用する Unicode 正規化方式。
            `"NFC"` は結合文字を正規化し、`"NFKC"` は半角カナなどの互換文字も正規化する
            デフォルト: `"None"`
        use_read_as_pron (bool): True の場合、全ての発音を強制的に読みに置き換える。
            助詞「は」も「ハ」になるため、TTS 用途には適さない。デフォルト: False
            このオプションが True の場合、revert_long_vowels / revert_yotsugana の指定に関係なく
            全ての pron が read で上書きされる
        revert_long_vowels (bool): True の場合、辞書が自動的に長音化した発音を元に復元する。
            pron に「ー」が含まれ、かつ orig に「ー」が含まれていない場合のみ復元する。
            助詞 (は→ワ, へ→エ) の発音は「ー」を含まないため影響を受けず維持される。
            (例: 「効果」コーカ → コウカ / 「人生」ジンセー → ジンセイ)
            デフォルト: False
        revert_yotsugana (bool): True の場合、四つ仮名 (ヅ・ヂ) の発音統合を元に復元する。
            read に「ヅ」「ヂ」が含まれている場合、pron を read で上書きする。
            (例: 「気づかず」キズカズ → キヅカズ / 「鼻血」ハナジ → ハナヂ)
            デフォルト: False
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
    normalize_mode: Literal["None", "NFC", "NFKC"] = "None",
    use_read_as_pron: bool = False,
    revert_long_vowels: bool = False,
    revert_yotsugana: bool = False,
    jtalk: Union[OpenJTalk, None] = None,
) -> list[NJDFeature]:
    """
    加工されていない生の NJD features に後処理を適用する。
    run_frontend() / run_frontend_detailed() の通常経路・tsqyomi 適用経路の双方で呼び出される。

    Args:
        text (str): Unicode 日本語テキスト
        njd_features (list[NJDFeature]): NJDNode 用 features (pyopenjtalk.run_frontend() の戻り値)
        run_marine (bool): marine を用いたアクセント推定を行うか。デフォルト: False
            有効にするには `pip install pyopenjtalk-plus[marine]` で marine をインストールする必要がある
        use_vanilla (bool): True の場合、pyopenjtalk-plus 独自の後処理を省略し、
            OpenJTalk の素の NJDFeature をそのまま後段に流す。
            ただし発音復元オプション (use_read_as_pron 等) は use_vanilla とは独立して適用される
            デフォルト: False
        use_sudachi_kanji_yomi (bool): True の場合、Sudachi による同形異音語の読み補正を行う
            デフォルト: True
        predict_nani (bool): True の場合、ONNX モデルで単独形態素として出現した「何」の読みを推定する
            デフォルト: True
        normalize_mode (Literal["None", "NFC", "NFKC"]): 入力テキストに適用する Unicode 正規化方式。
            `"NFC"` は結合文字を正規化し、`"NFKC"` は半角カナなどの互換文字も正規化する
            デフォルト: `"None"`
        use_read_as_pron (bool): True の場合、全ての発音を強制的に読みに置き換える。
            助詞「は」も「ハ」になるため、TTS 用途には適さない。デフォルト: False
            このオプションが True の場合、revert_long_vowels / revert_yotsugana の指定に関係なく
            全ての pron が read で上書きされる
        revert_long_vowels (bool): True の場合、辞書が自動的に長音化した発音を元に復元する。
            pron に「ー」が含まれ、かつ orig に「ー」が含まれていない場合のみ復元する。
            助詞 (は→ワ, へ→エ) の発音は「ー」を含まないため影響を受けず維持される。
            (例: 「効果」コーカ → コウカ / 「人生」ジンセー → ジンセイ)
            デフォルト: False
        revert_yotsugana (bool): True の場合、四つ仮名 (ヅ・ヂ) の発音統合を元に復元する。
            read に「ヅ」「ヂ」が含まれている場合、pron を read で上書きする。
            (例: 「気づかず」キズカズ → キヅカズ / 「鼻血」ハナジ → ハナヂ)
            デフォルト: False
        jtalk (OpenJTalk | None): 使用する OpenJTalk インスタンス。None ならグローバルインスタンスを使う

    NOTE:
        発音復元オプション (use_read_as_pron, revert_long_vowels, revert_yotsugana) は
        use_vanilla の設定に関係なく、明示的に指定された場合のみ独立して適用される

    Returns:
        list[NJDFeature]: 後処理後の NJDNode 用 features
    """
    text = normalize_text(text, normalize_mode)
    if run_marine:
        pred_njd_features = estimate_accent(njd_features)
        njd_features = preserve_noun_accent(njd_features, pred_njd_features)
    if use_vanilla is False:
        # filler アクセントは読み変更より先に補正する既存の処理順序を維持する
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
        njd_features = retreat_acc_nuc(njd_features)
        njd_features = modify_acc_after_chaining(njd_features)
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
    return njd_features


def run_frontend(
    text: str,
    *,
    run_marine: bool = False,
    use_vanilla: bool = False,
    use_tsqyomi: bool = False,
    use_sudachi_kanji_yomi: bool = True,
    predict_nani: bool = True,
    normalize_mode: Literal["None", "NFC", "NFKC"] = "None",
    use_read_as_pron: bool = False,
    revert_long_vowels: bool = False,
    revert_yotsugana: bool = False,
    jtalk: Union[OpenJTalk, None] = None,
) -> list[NJDFeature]:
    """
    OpenJTalk のテキスト処理フロントエンドを実行する。
    pyopenjtalk.run_frontend_detailed() のラッパー。NJD features のみを返す。

    Args:
        text (str): Unicode 日本語テキスト
        run_marine (bool): marine を用いたアクセント推定を行うか。デフォルト: False
            有効にするには `pip install pyopenjtalk-plus[marine]` で marine をインストールする必要がある
        use_vanilla (bool): True の場合、pyopenjtalk-plus 独自の後処理を省略し、
            OpenJTalk の素の NJDFeature をそのまま後段に流す。
            ただし発音復元オプション (use_read_as_pron 等) は use_vanilla とは独立して適用される
            デフォルト: False
        use_tsqyomi (bool): True の場合、ロード済みの tsqyomi で文脈に合う読み候補を選ぶ。
            Sudachi と「何」モデルによる読み変更を省き、tsqyomi の選択を維持する
            デフォルト: False
        use_sudachi_kanji_yomi (bool): True の場合、Sudachi による同形異音語の読み補正を行う。
            use_tsqyomi が True の場合は tsqyomi を優先し、常に無効化される
            デフォルト: True
        predict_nani (bool): True の場合、ONNX モデルで単独形態素として出現した「何」の読みを推定する。
            use_tsqyomi が True の場合は tsqyomi を優先し、常に無効化される
            デフォルト: True
        normalize_mode (Literal["None", "NFC", "NFKC"]): 入力テキストに適用する Unicode 正規化方式。
            `"NFC"` は結合文字を正規化し、`"NFKC"` は半角カナなどの互換文字も正規化する
            デフォルト: `"None"`
        use_read_as_pron (bool): True の場合、全ての発音を強制的に読みに置き換える。
            助詞「は」も「ハ」になるため、TTS 用途には適さない。デフォルト: False
            このオプションが True の場合、revert_long_vowels / revert_yotsugana の指定に関係なく
            全ての pron が read で上書きされる
        revert_long_vowels (bool): True の場合、辞書が自動的に長音化した発音を元に復元する。
            pron に「ー」が含まれ、かつ orig に「ー」が含まれていない場合のみ復元する。
            助詞 (は→ワ, へ→エ) の発音は「ー」を含まないため影響を受けず維持される。
            (例: 「効果」コーカ → コウカ / 「人生」ジンセー → ジンセイ)
            デフォルト: False
        revert_yotsugana (bool): True の場合、四つ仮名 (ヅ・ヂ) の発音統合を元に復元する。
            read に「ヅ」「ヂ」が含まれている場合、pron を read で上書きする。
            (例: 「気づかず」キズカズ → キヅカズ / 「鼻血」ハナジ → ハナヂ)
            デフォルト: False
        jtalk (OpenJTalk | None): 使用する OpenJTalk インスタンス。None ならグローバルインスタンスを使う

    Returns:
        list[NJDFeature]: NJDNode 用 features
    """
    text = normalize_text(text, normalize_mode)
    if use_tsqyomi is True:
        # `_global_jtalk()` の mutex は参照取得だけに使い、tsqyomi 推論中に他スレッドがグローバルロック待ちにならないようにする
        with _resolve_jtalk(jtalk) as resolved_jtalk:
            inference_jtalk = resolved_jtalk

        njd_features, _ = _run_frontend_with_tsqyomi(
            text,
            jtalk=inference_jtalk,
            include_morphs=False,
        )
    else:
        with _resolve_jtalk(jtalk) as resolved_jtalk:
            inference_jtalk = resolved_jtalk
            njd_features = inference_jtalk.run_frontend(text)

    # tsqyomi 使用時は読み候補確定済みなので Sudachi と nani_predict モデルを適用しない
    njd_features = apply_postprocessing(
        text,
        njd_features,
        run_marine=run_marine,
        use_vanilla=use_vanilla,
        use_sudachi_kanji_yomi=use_sudachi_kanji_yomi if use_tsqyomi is False else False,
        predict_nani=predict_nani if use_tsqyomi is False else False,
        normalize_mode="None",  # 既に normalize_text() で正規化されているため、再度正規化しない
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
    normalize_mode: Literal["None", "NFC", "NFKC"] = "None",
    use_read_as_pron: bool = False,
    revert_long_vowels: bool = False,
    revert_yotsugana: bool = False,
    jtalk: Union[OpenJTalk, None] = None,
) -> tuple[list[NJDFeature], list[MeCabMorph]]:
    """
    OpenJTalk のテキスト処理フロントエンドを MeCab 形態素詳細付きで実行する。
    MeCab で形態素解析を 1 回だけ実行し、NJD features と MeCab morphs を同時に返す。
    pyopenjtalk.run_frontend() と異なり、MeCab の未知語フラグ・コスト情報付きの morphs も取得できる。

    Args:
        text (str): Unicode 日本語テキスト
        run_marine (bool): marine を用いたアクセント推定を行うか。デフォルト: False
            有効にするには `pip install pyopenjtalk-plus[marine]` で marine をインストールする必要がある
        use_vanilla (bool): True の場合、pyopenjtalk-plus 独自の後処理を省略し、
            OpenJTalk の素の NJDFeature をそのまま後段に流す。
            ただし発音復元オプション (use_read_as_pron 等) は use_vanilla とは独立して適用される
            デフォルト: False
        use_tsqyomi (bool): True の場合、ロード済みの tsqyomi で文脈に合う読み候補を選ぶ。
            Sudachi と「何」モデルによる読み変更を省き、tsqyomi の選択を維持する
            デフォルト: False
        use_sudachi_kanji_yomi (bool): True の場合、Sudachi による同形異音語の読み補正を行う。
            use_tsqyomi が True の場合は tsqyomi を優先し、常に無効化される
            デフォルト: True
        predict_nani (bool): True の場合、ONNX モデルで単独形態素として出現した「何」の読みを推定する。
            use_tsqyomi が True の場合は tsqyomi を優先し、常に無効化される
            デフォルト: True
        normalize_mode (Literal["None", "NFC", "NFKC"]): 入力テキストに適用する Unicode 正規化方式。
            `"NFC"` は結合文字を正規化し、`"NFKC"` は半角カナなどの互換文字も正規化する
            デフォルト: `"None"`
        use_read_as_pron (bool): True の場合、全ての発音を強制的に読みに置き換える。
            助詞「は」も「ハ」になるため、TTS 用途には適さない。デフォルト: False
            このオプションが True の場合、revert_long_vowels / revert_yotsugana の指定に関係なく
            全ての pron が read で上書きされる
        revert_long_vowels (bool): True の場合、辞書が自動的に長音化した発音を元に復元する。
            pron に「ー」が含まれ、かつ orig に「ー」が含まれていない場合のみ復元する。
            助詞 (は→ワ, へ→エ) の発音は「ー」を含まないため影響を受けず維持される。
            (例: 「効果」コーカ → コウカ / 「人生」ジンセー → ジンセイ)
            デフォルト: False
        revert_yotsugana (bool): True の場合、四つ仮名 (ヅ・ヂ) の発音統合を元に復元する。
            read に「ヅ」「ヂ」が含まれている場合、pron を read で上書きする。
            (例: 「気づかず」キズカズ → キヅカズ / 「鼻血」ハナジ → ハナヂ)
            デフォルト: False
        jtalk (OpenJTalk | None): 使用する OpenJTalk インスタンス。None ならグローバルインスタンスを使う

    Returns:
        tuple[list[NJDFeature], list[MeCabMorph]]: (NJD features, MeCab morphs)
            - NJD features: pyopenjtalk.run_frontend() と同一の結果が得られる
            - MeCab morphs: pyopenjtalk.run_mecab_detailed() と同一の結果が得られる
    """
    text = normalize_text(text, normalize_mode)
    if use_tsqyomi is True:
        # 候補解析と NJD の C 処理だけが OpenJTalk のロックを取り、モデル推論はコピー済みデータで行う
        with _resolve_jtalk(jtalk) as resolved_jtalk:
            inference_jtalk = resolved_jtalk

        njd_features, morphs = _run_frontend_with_tsqyomi(
            text,
            jtalk=inference_jtalk,
            include_morphs=True,
        )
    else:
        with _resolve_jtalk(jtalk) as resolved_jtalk:
            inference_jtalk = resolved_jtalk
            njd_features, morphs = inference_jtalk.run_frontend_detailed(text)
    njd_features = apply_postprocessing(
        text,
        njd_features,
        run_marine=run_marine,
        use_vanilla=use_vanilla,
        use_sudachi_kanji_yomi=use_sudachi_kanji_yomi if use_tsqyomi is False else False,
        predict_nani=predict_nani if use_tsqyomi is False else False,
        normalize_mode="None",  # 既に normalize_text() で正規化されているため、再度正規化しない
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
) -> tuple[list[NJDFeature], list[MeCabMorph]]:
    """
    tsqyomi で MeCab feature を選び、NJD 処理後の features と morphs を返す。
    Python 側後処理は apply_postprocessing() に委譲する。

    Args:
        text (str): 正規化済みの Unicode 日本語テキスト
        jtalk (OpenJTalk): 候補解析と NJD 処理に使う OpenJTalk インスタンス
        include_morphs (bool): 詳細形態素列を返す場合は True

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
    njd_features = jtalk.run_njd_from_mecab(mecab_features)
    return njd_features, morphs


def make_label(njd_features: list[NJDFeature], jtalk: Union[OpenJTalk, None] = None) -> list[str]:
    """
    HTS 音声合成用のフルコンテキストラベルを返す。

    Args:
        njd_features (list[NJDFeature]): NJDNode 用 features (pyopenjtalk.run_frontend() の戻り値)
        jtalk (OpenJTalk | None): 使用する OpenJTalk インスタンス。None ならグローバルインスタンスを使う

    Returns:
        list[str]: フルコンテキストラベル文字列のリスト
    """
    with _resolve_jtalk(jtalk) as jtalk:
        return jtalk.make_label(njd_features)


def make_phoneme_mapping(
    njd_features: list[NJDFeature],
    morphs: Union[list[MeCabMorph], None] = None,
    jtalk: Union[OpenJTalk, None] = None,
) -> list[SurfacePhonemeMapping]:
    """
    NJD features から各形態素に対応する音素列のマッピングを返す。
    Cython 側の OpenJTalk.make_phoneme_mapping() で基本マッピングを取得し、
    morphs が渡された場合は MeCab morphs とアライメントして is_unknown / is_ignored を付与する。

    morphs を省略した場合は is_unknown=False 、is_ignored は音素列の空判定から推定される。
    morphs を渡す場合、踊り字展開や数字正規化により NJD と MeCab の粒度がずれることがあるが、
    アライメントロジックが自動的に補正する。音素列自体は常に正しい値が得られる。
    pause-like な記号は surface として保持されるが、
    JPCommon が実際に短ポーズを生成しない場合は phonemes は空のまま返る。

    Args:
        njd_features (list[NJDFeature]): NJDNode 用 features (pyopenjtalk.run_frontend() の戻り値)
        morphs (list[MeCabMorph] | None): MeCab の形態素解析結果 (pyopenjtalk.run_frontend_detailed() の戻り値)
            None の場合は is_unknown / is_ignored の推定精度が下がる
        jtalk (OpenJTalk | None): 使用する OpenJTalk インスタンス。None ならグローバルインスタンスを使う

    Returns:
        list[SurfacePhonemeMapping]: 各形態素に対応する音素列のマッピング
    """

    def _base_to_detail(
        base: dict[str, Any],
        phonemes: list[str],
        features: Union[list[str], None] = None,
        is_unknown: bool = False,
        is_ignored: bool = False,
    ) -> SurfacePhonemeMapping:
        """base_mapping のエントリから SurfacePhonemeMapping を構築する。"""

        return {
            "surface": base["surface"],
            "phonemes": phonemes,
            "features": features if features is not None else [],
            "pos": base["pos"],
            "pos_group1": base["pos_group1"],
            "pos_group2": base["pos_group2"],
            "pos_group3": base["pos_group3"],
            "ctype": base["ctype"],
            "cform": base["cform"],
            "orig": base["orig"],
            "read": base["read"],
            "pron": base["pron"],
            "accent_nucleus": base["accent_nucleus"],
            "mora_count": base["mora_count"],
            "chain_rule": base["chain_rule"],
            "chain_flag": base["chain_flag"],
            "is_unknown": is_unknown,
            "is_ignored": is_ignored,
        }

    def _sp_entry(surface: str, is_unknown: bool = False) -> SurfacePhonemeMapping:
        """is_ignored な morph 用の sp エントリを構築する。"""

        return {
            "surface": surface,
            "phonemes": ["sp"],
            "features": [],
            "pos": "記号",
            "pos_group1": "空白",
            "pos_group2": "*",
            "pos_group3": "*",
            "ctype": "*",
            "cform": "*",
            "orig": surface,
            "read": surface,
            "pron": surface,
            "accent_nucleus": 0,
            "mora_count": 0,
            "chain_rule": "*",
            "chain_flag": -1,
            "is_unknown": is_unknown,
            "is_ignored": True,
        }

    # Cython レベルで基本マッピングと長音吸収マージを取得
    with _resolve_jtalk(jtalk) as jtalk:
        base_mapping = jtalk.make_phoneme_mapping(njd_features)

    # morphs が渡されていない場合: NJDFeature ベースで is_unknown を推定
    # njd_set_pronunciation が mora_size=0 のノードの読みを補完し、pos を "フィラー" に上書きする
    # この判定は辞書に元からフィラーとして登録された既知語（ゔぁ等）でも True になるため、
    # MeCab の is_unknown より範囲が広い（偽陽性がある）ため、正確な判定には morphs が必要
    if morphs is None:
        return [
            _base_to_detail(
                entry,
                entry["phonemes"],
                is_unknown=(entry["pos"] == "フィラー" and entry["chain_rule"] == "*"),
                is_ignored=len(entry["phonemes"]) == 0,
            )
            for entry in base_mapping
        ]

    # base_mapping と morphs のアライメント: is_unknown / is_ignored を付与する

    # 全 morphs が ignored の場合は全て sp として返す
    has_valid_morph = any(morph["is_ignored"] is False for morph in morphs)
    if has_valid_morph is False:
        return [_sp_entry(morph["surface"], is_unknown=morph["is_unknown"]) for morph in morphs]

    result: list[SurfacePhonemeMapping] = []
    morph_idx = 0
    for base_idx, base_entry in enumerate(base_mapping):
        current_surface = base_entry["surface"]
        current_phonemes = base_entry["phonemes"]

        # is_ignored な morph を先に sp として出力
        while morph_idx < len(morphs):
            morph = morphs[morph_idx]
            if morph["is_ignored"] is True:
                result.append(_sp_entry(morph["surface"], is_unknown=morph["is_unknown"]))
                morph_idx += 1
            else:
                break

        if morph_idx >= len(morphs):
            # morphs が尽きた: 後処理で feature 数が変動しうるため出力を継続
            result.append(
                _base_to_detail(base_entry, current_phonemes, is_ignored=len(current_phonemes) == 0)
            )
            continue

        morph = morphs[morph_idx]

        # 完全一致: morph と NJD feature の surface が一致
        if current_surface == morph["surface"]:
            phonemes = list(current_phonemes)

            # 未知語を NJD が読点扱いした場合も、区切り記号と誤認させず unk へ戻す
            if morph["is_unknown"] is True and (len(phonemes) == 0 or phonemes == ["pau"]):
                phonemes = ["unk"]

            # is_ignored は音素列が空かで判定 (MeCab の is_ignored とは異なるセマンティクス)
            result.append(
                _base_to_detail(
                    base_entry,
                    phonemes,
                    is_unknown=morph["is_unknown"],
                    is_ignored=len(current_phonemes) == 0,
                    features=morph["features"],
                )
            )
            morph_idx += 1

        # 先頭一致: NJD が複数の morph を結合したケース
        elif current_surface.startswith(morph["surface"]):
            # 記号だけの NJD ノードは、発音を増やさず詳細形態素の表層粒度へ戻す
            ## MeCab の通常出力が連続記号を1ノードへまとめても、lattice から復元した morphs は
            ## 1文字ずつ保持されるため、最初の形態素だけへ NJD のポーズ音素を割り当てる
            symbol_morphs: list[MeCabMorph] = []
            symbol_surface = ""
            symbol_morph_idx = morph_idx
            while symbol_morph_idx < len(morphs) and len(symbol_surface) < len(current_surface):
                symbol_morph = morphs[symbol_morph_idx]
                if (
                    symbol_morph["is_ignored"] is True
                    or len(symbol_morph["surface"]) != 1
                    or symbol_morph["surface"].isalnum() is True
                ):
                    break
                symbol_morphs.append(symbol_morph)
                symbol_surface += symbol_morph["surface"]
                symbol_morph_idx += 1

            is_restored_symbol_chunk = (
                len(symbol_morphs) > 1
                and symbol_surface == current_surface
                and (len(current_phonemes) == 0 or current_phonemes == ["pau"])
            )
            if is_restored_symbol_chunk is True:
                for symbol_idx, symbol_morph in enumerate(symbol_morphs):
                    symbol_mapping = _base_to_detail(
                        base_entry,
                        list(current_phonemes) if symbol_idx == 0 else [],
                        features=symbol_morph["features"],
                        is_unknown=symbol_morph["is_unknown"],
                        is_ignored=False,
                    )
                    symbol_mapping["surface"] = symbol_morph["surface"]
                    symbol_mapping["orig"] = symbol_morph["surface"]
                    result.append(symbol_mapping)
                morph_idx = symbol_morph_idx
                continue

            is_unknown_word = False
            matched_len = 0
            internal_ignored_entries: list[SurfacePhonemeMapping] = []

            while morph_idx < len(morphs):
                inner_morph = morphs[morph_idx]

                # 結合語の内部にある空白は、表層の構成要素を先に出してから直後へ戻す
                ## その場で result へ追加すると、まだ未出力の結合語より空白が前へ移動してしまう
                if inner_morph["is_ignored"] is True:
                    internal_ignored_entries.append(
                        _sp_entry(inner_morph["surface"], is_unknown=inner_morph["is_unknown"])
                    )
                    morph_idx += 1
                    continue

                remaining = current_surface[matched_len:]

                if remaining.startswith(inner_morph["surface"]):
                    # いずれかの構成トークンが未知語なら全体を未知語とみなす
                    is_unknown_word = is_unknown_word or inner_morph["is_unknown"]
                    matched_len += len(inner_morph["surface"])
                    morph_idx += 1

                    if matched_len == len(current_surface):
                        break
                else:
                    break

            phonemes = list(current_phonemes)

            # 結合語を構成する未知語が読点扱いされた場合も unk へ戻す
            if is_unknown_word is True and (len(phonemes) == 0 or phonemes == ["pau"]):
                phonemes = ["unk"]

            result.append(
                _base_to_detail(
                    base_entry,
                    phonemes,
                    is_unknown=is_unknown_word,
                    is_ignored=len(current_phonemes) == 0,
                )
            )
            result.extend(internal_ignored_entries)

        # 不一致: 数字正規化・踊り字展開等で surface が変化したケース
        # 以下の 3 パターンに応じて morph_idx の消費数を制御する:
        #   A) 踊り字展開 (morph 数 > NJD 数): morph を 2 つ消費 (踊り字 + 結合先)
        #   B) NJD 数字展開 (NJD 数 > morph 数): morph を消費しない
        #   C) その他の不一致 (surface 変化のみ): morph を 1 つ消費
        else:
            # 不一致ブランチでは morph と NJD の surface が異なるため、
            # morph の features をこのエントリに紐づけると嘘データになる (features は空リスト)
            result.append(
                _base_to_detail(
                    base_entry,
                    list(current_phonemes),
                    is_ignored=len(current_phonemes) == 0,
                )
            )

            current_morph_surface = morphs[morph_idx]["surface"]
            has_odori = any(c in _ODORI_CHARS for c in current_morph_surface)

            # A) 踊り字展開: 踊り字 morph + 結合先 morph を消費
            # 踊り字展開では、単独の踊り字 morph ('々' 等) と後続の漢字 morph が
            # 結合されて 1 つの NJD feature になる (例: morphs['々','活'] → NJD '生活')
            if has_odori is True:
                morph_idx += 1
                # 結合先 morph の判定: current_surface の末尾と次の morph の surface が一致
                # 結合先がないケース (例: '学生々' → NJD='生') では追加消費しない
                if morph_idx < len(morphs):
                    ahead = morphs[morph_idx]
                    if ahead["is_ignored"] is not True and current_surface.endswith(
                        ahead["surface"]
                    ):
                        morph_idx += 1

            else:
                # 数字以外の surface 変化は対応する morph を1つだけ消費する
                ## 数字ブロック以外の残数を判断材料にすると、後段のノード結合で数字の対応までずれる
                if current_morph_surface not in _DIGIT_MORPH_SURFACES:
                    morph_idx += 1
                else:
                    # 現在位置から連続する数字 morph と漢数字 mapping だけを数える
                    ## 数字展開と後段の英単語結合などが同時に起きても、互いの増減を相殺させない
                    digit_morph_count = 0
                    for remaining_morph in morphs[morph_idx:]:
                        if remaining_morph["is_ignored"] is True:
                            continue
                        if remaining_morph["surface"] not in _DIGIT_MORPH_SURFACES:
                            break
                        digit_morph_count += 1

                    digit_mapping_count = int(current_surface in _KANJI_NUMBER_SURFACES)
                    for remaining_base in base_mapping[base_idx + 1 :]:
                        if remaining_base["surface"] not in _KANJI_NUMBER_SURFACES:
                            break
                        digit_mapping_count += 1

                    # 現在の mapping 以降へ残す morph 数を引き、現在位置で必要な分だけ消費する
                    target_remaining_morphs = max(digit_mapping_count - 1, 0)
                    needed_non_ignored = max(
                        digit_morph_count - target_remaining_morphs,
                        0,
                    )
                    consumed_non_ignored = 0
                    consumed_ignored_entries: list[SurfacePhonemeMapping] = []
                    while morph_idx < len(morphs) and consumed_non_ignored < needed_non_ignored:
                        remaining_morph = morphs[morph_idx]
                        if remaining_morph["is_ignored"] is True:
                            # NJD は空白を除いた数字列を縮約するが、公開 mapping では元の空白を sp として保持する
                            consumed_ignored_entries.append(
                                _sp_entry(
                                    remaining_morph["surface"],
                                    is_unknown=remaining_morph["is_unknown"],
                                )
                            )
                        else:
                            if remaining_morph["surface"] not in _DIGIT_MORPH_SURFACES:
                                break
                            consumed_non_ignored += 1
                        morph_idx += 1
                    result.extend(consumed_ignored_entries)

    # morphs 末尾に残った is_ignored トークンを sp として回収
    while morph_idx < len(morphs):
        morph = morphs[morph_idx]
        if morph["is_ignored"] is True:
            result.append(_sp_entry(morph["surface"], is_unknown=morph["is_unknown"]))
        morph_idx += 1

    return result


def mecab_dict_index(path: str, out_path: str, dn_mecab: Union[str, None] = None) -> None:
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
    paths: Union[str, list[str], list[UserDictionaryEntry]],
) -> None:
    """
    グローバル OpenJTalk インスタンスにユーザー辞書を適用する。
    注意: この関数を実行すると、pyopenjtalk モジュールのグローバル状態が変更される。

    Args:
        paths (str | list[str] | list[UserDictionaryEntry]): ユーザー辞書ファイル (.dic) と読み保護の指定

    Raises:
        ValueError: 空のリスト、UserDictionaryEntry のキー、またはリスト内のパスが不正な場合
        TypeError: リストの要素型が不正か、文字列と UserDictionaryEntry が混在する場合
        FileNotFoundError: 指定したユーザー辞書ファイルが存在しない場合
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

    global _global_jtalk
    with _global_jtalk_swap_lock:
        with _global_jtalk():
            _global_jtalk = _global_instance_manager(
                instance=OpenJTalk(
                    dn_mecab=OPEN_JTALK_DICT_DIR,
                    userdic=paths_str.encode("utf-8"),
                    userdic_reading_protection=reading_protection,
                ),
            )


def unset_user_dict() -> None:
    """
    ユーザー辞書の適用を解除する。
    注意: この関数を実行すると、pyopenjtalk モジュールのグローバル状態が変更される。
    """
    global _global_jtalk
    with _global_jtalk_swap_lock:
        with _global_jtalk():
            _global_jtalk = _global_instance_manager(
                instance=OpenJTalk(dn_mecab=OPEN_JTALK_DICT_DIR),
            )


def run_mecab(text: str, jtalk: Union[OpenJTalk, None] = None) -> list[str]:
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
    jtalk: Union[OpenJTalk, None] = None,
) -> list[MeCabMorph]:
    """
    MeCab の形態素解析結果を未知語フラグ・コスト情報付きで返す。
    通常の pyopenjtalk.run_mecab() と異なり、記号,空白 もフィルタせずに全トークンを返す。
    各トークンの is_unknown フラグにより、辞書に登録されている単語かどうかを判定できる。

    Args:
        text (str): Unicode 日本語テキスト
        jtalk (OpenJTalk | None): 使用する OpenJTalk インスタンス。None ならグローバルインスタンスを使う

    Returns:
        list[MeCabMorph]: MeCab の形態素解析結果のリスト
    """

    with _resolve_jtalk(jtalk) as jtalk:
        return jtalk.run_mecab_detailed(text)


def run_mecab_nbest_features(
    text: str,
    max_paths: int = 5,
    *,
    jtalk: Union[OpenJTalk, None] = None,
) -> list[MeCabNBestPath]:
    """
    MeCab の n-best 候補を features / morphs / path_cost 付きで返す。
    features は pyopenjtalk.run_njd_from_mecab() に渡せる形式で、
    OpenJTalk の後処理を維持したまま候補パスごとの読み・発音を比較できる。

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
    mecab_features: list[str], jtalk: Union[OpenJTalk, None] = None
) -> list[NJDFeature]:
    """
    MeCab の feature 文字列のリストから NJD 処理を実行する。
    pyopenjtalk.run_mecab() の戻り値をそのまま渡す想定。数字正規化・アクセント句設定・長音処理などの NJD ルールが適用される。

    Args:
        mecab_features (list[str]): MeCab の feature 文字列のリスト
        jtalk (OpenJTalk | None): 使用する OpenJTalk インスタンス。None ならグローバルインスタンスを使う

    Returns:
        list[NJDFeature]: NJDNode 用 features
    """
    with _resolve_jtalk(jtalk) as jtalk:
        return jtalk.run_njd_from_mecab(mecab_features)


def build_mecab_dictionary(dn_mecab: Union[str, None] = None) -> None:
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
