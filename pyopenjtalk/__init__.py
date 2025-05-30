from __future__ import annotations

import atexit
import os
from collections.abc import Callable, Generator
from contextlib import ExitStack, contextmanager
from importlib.resources import as_file, files
from os.path import exists
from pathlib import Path
from threading import Lock
from typing import Any, TypeVar, Union

import numpy as np
import numpy.typing as npt
from typing_extensions import Literal


try:
    from .version import __version__  # noqa
except ImportError:
    raise ImportError("BUG: version.py doesn't exist. Please file a bug report.")

from .hougen import (
    convert_babytalk_style,
    convert_d2r_style,
    convert_hatsuonbin_style,
    convert_s2z_style,
    convert_tt2t_style,
    modify_kansai_accent,
    modify_kansai_hougen,
    modify_kyusyu_hougen,
)
from .htsengine import HTSEngine
from .openjtalk import OpenJTalk
from .openjtalk import build_mecab_dictionary as _build_mecab_dictionary
from .openjtalk import mecab_dict_index as _mecab_dict_index
from .types import NJDFeature
from .utils import (
    merge_njd_marine_features,
    modify_acc_after_chaining,
    modify_kanji_yomi,
    process_odori_features,
    retreat_acc_nuc,
)


_file_manager = ExitStack()
atexit.register(_file_manager.close)

_pyopenjtalk_ref = files(__name__)
_dic_dir_name = "dictionary"

_user_dic_dir = Path(_pyopenjtalk_ref) / "user_dictionary" 


def is_dic_file(file: Path) -> bool:
    supported_extensions = [".dic"]
    return file.suffix.lower() in supported_extensions


_dic_files = [str(file) for file in _user_dic_dir.rglob("*") if is_dic_file(file)]

_default_user_dict = ",".join(_dic_files)

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
    '家','縁','労','中','高','低','気','要','退','面','色','主','術','直','片','緒','小','大',
    # 他にも日付（月・火・水・木・金・土・日）も入るが、当面は入れない (金を除く)
]  # fmt: skip

_T = TypeVar("_T")


def _lazy_init() -> None:
    # pyopenjtalk-plus では辞書のダウンロード処理は削除されているが、
    # _lazy_init() を直接呼び出している VOICEVOX などへの互換性のために残置している
    pass


def _global_instance_manager(
    instance_factory: Union[Callable[[], _T], None] = None,
    instance: Union[_T, None] = None,
):
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
_global_jtalk = _global_instance_manager(
    lambda: OpenJTalk(dn_mecab=OPEN_JTALK_DICT_DIR, userdic=_default_user_dict.encode("utf-8"))
)
# Global instance of HTSEngine
# mei_normal.voice is used as default
_global_htsengine = _global_instance_manager(lambda: HTSEngine(DEFAULT_HTS_VOICE))
# Global instance of Marine
_global_marine = None


def g2p(
    text: str,
    kana: bool = False,
    join: bool = True,
    run_marine: bool = False,
    use_vanilla: bool = False,
    dialect: Literal["Kansai", "Kyusyu", "BabyTalk", "Hatsuonbin", "TTtoT", "StoZ", "DtoR"]
    | None = None,
    jtalk: Union[OpenJTalk, None] = None,
) -> Union[list[str], str]:
    """Grapheme-to-phoeneme (G2P) conversion

    This is just a convenient wrapper around `run_frontend`.

    Args:
        text (str): Unicode Japanese text.
        kana (bool): If True, returns the pronunciation in katakana, otherwise in phone.
          Default is False.
        join (bool): If True, concatenate phones or katakana's into a single string.
          Default is True.
        run_marine (bool): Whether to estimate accent using marine.
          Default is False. If you want to activate this option, you need to install marine
          by `pip install pyopenjtalk-plus[marine]`
        use_vanilla (bool): If True, returns the vanilla NJDFeature list.
          Default is False.
        jtalk (OpenJTalk, optional): OpenJTalk instance to use. If None, use global instance.
          Default is None.

    Returns:
        Union[List[str], str]: G2P result in 1) str if join is True 2) List[str] if join is False.
    """
    njd_features = run_frontend(
        text, run_marine=run_marine, use_vanilla=use_vanilla, dialect=dialect, jtalk=jtalk
    )

    if not kana:
        labels = make_label(njd_features, jtalk=jtalk)
        prons = list(map(lambda s: s.split("-")[1].split("+")[0], labels[1:-1]))
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


def load_marine_model(model_dir: Union[str, None] = None, dict_dir: Union[str, None] = None):
    global _global_marine
    if _global_marine is None:
        try:
            from marine.predict import Predictor
        except ImportError:
            raise ImportError("Please install marine by `pip install pyopenjtalk-plus[marine]`")
        _global_marine = Predictor(model_dir=model_dir, postprocess_vocab_dir=dict_dir)


def estimate_accent(njd_features: list[NJDFeature]) -> list[NJDFeature]:
    """Accent estimation using marine

    This function requires marine (https://github.com/6gsn/marine)

    Args:
        njd_result (List[NJDFeature]): features generated by OpenJTalk.

    Returns:
        List[NJDFeature]: features for NJDNode with estimation results by marine.
    """
    global _global_marine
    if _global_marine is None:
        load_marine_model()
        assert _global_marine is not None
    from marine.utils.openjtalk_util import convert_njd_feature_to_marine_feature

    marine_feature = convert_njd_feature_to_marine_feature(njd_features)
    marine_results = _global_marine.predict([marine_feature], require_open_jtalk_format=True)
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
    run_marine: bool = False,
    use_vanilla: bool = False,
    dialect: Literal["Kansai", "Kyusyu", "BabyTalk", "Hatsuonbin", "TTtoT", "StoZ", "DtoR"]
    | None = None,
    jtalk: Union[OpenJTalk, None] = None,
) -> list[str]:
    """Extract full-context labels from text

    Args:
        text (str): Input text
        run_marine (bool): Whether to estimate accent using marine.
          Default is False. If you want to activate this option, you need to install marine
          by `pip install pyopenjtalk-plus[marine]`
        use_vanilla (bool): If True, returns the vanilla NJDFeature list.
          Default is False.
        jtalk (OpenJTalk, optional): OpenJTalk instance to use. If None, use global instance.
          Default is None.

    Returns:
        List[str]: List of full-context labels
    """
    njd_features = run_frontend(
        text, run_marine=run_marine, use_vanilla=use_vanilla, dialect=dialect, jtalk=jtalk
    )
    return make_label(njd_features, jtalk=jtalk)


def synthesize(
    labels: Union[list[str], tuple[Any, list[str]]],
    speed: float = 1.0,
    half_tone: float = 0.0,
) -> tuple[npt.NDArray[np.float64], int]:
    """Run OpenJTalk's speech synthesis backend

    Args:
        labels (list): Full-context labels
        speed (float): speech speed rate. Default is 1.0.
        half_tone (float): additional half-tone. Default is 0.

    Returns:
        np.ndarray: speech waveform (dtype: np.float64)
        int: sampling frequency (defualt: 48000)
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
    run_marine: bool = False,
    use_vanilla: bool = False,
    jtalk: Union[OpenJTalk, None] = None,
) -> tuple[npt.NDArray[np.float64], int]:
    """Text-to-speech

    Args:
        text (str): Input text
        speed (float): speech speed rate. Default is 1.0.
        half_tone (float): additional half-tone. Default is 0.
        run_marine (bool): Whether to estimate accent using marine.
          Default is False. If you want activate this option, you need to install marine
          by `pip install pyopenjtalk-plus[marine]`
        use_vanilla (bool): If True, returns the vanilla NJDFeature list.
          Default is False.
        jtalk (OpenJTalk, optional): OpenJTalk instance to use. If None, use global instance.
          Default is None.

    Returns:
        np.ndarray: speech waveform (dtype: np.float64)
        int: sampling frequency (defualt: 48000)
    """
    return synthesize(
        extract_fullcontext(
            text,
            run_marine=run_marine,
            use_vanilla=use_vanilla,
            jtalk=jtalk,
        ),
        speed,
        half_tone,
    )


def run_frontend(
    text: str,
    run_marine: bool = False,
    use_vanilla: bool = False,
    dialect: Literal["Kansai", "Kyusyu", "BabyTalk", "Hatsuonbin", "TTtoT", "StoZ", "DtoR"]
    | None = None,
    jtalk: Union[OpenJTalk, None] = None,
) -> list[NJDFeature]:
    """Run OpenJTalk's text processing frontend

    Args:
        text (str): Unicode Japanese text.
        run_marine (bool): Whether to estimate accent using marine.
          Default is False. If you want to activate this option, you need to install marine
          by `pip install pyopenjtalk-plus[marine]`
        use_vanilla (bool): If True, returns the vanilla NJDFeature list.
          Default is False.
        jtalk (OpenJTalk, optional): OpenJTalk instance to use. If None, use global instance.
          Default is None.

    Returns:
        List[NJDFeature]: features for NJDNode.
    """
    kansai = False
    if dialect is not None:
        if "Kansai" in dialect:
            kansai = True

    if jtalk is not None:
        njd_features = jtalk.run_frontend(text, kansai=kansai)
    else:
        global _global_jtalk
        with _global_jtalk() as jtalk:
            njd_features = jtalk.run_frontend(text, kansai=kansai)
    if run_marine:
        pred_njd_features = estimate_accent(njd_features)
        njd_features = preserve_noun_accent(njd_features, pred_njd_features)
    if use_vanilla is False:
        njd_features = modify_filler_accent(njd_features)
        njd_features = modify_kanji_yomi(text, njd_features, MULTI_READ_KANJI_LIST)
        njd_features = retreat_acc_nuc(njd_features)
        njd_features = modify_acc_after_chaining(njd_features)
        njd_features = process_odori_features(njd_features, jtalk=jtalk)

    if dialect is not None:
        if "Kyusyu" in dialect:
            njd_features = modify_kyusyu_hougen(njd_features)
        if "Kansai" in dialect:
            njd_features = modify_kansai_hougen(njd_features)
            njd_features = modify_kansai_accent(njd_features)
        if "BabyTalk" in dialect:
            njd_features = convert_babytalk_style(njd_features)
        if "Hatsuonbin" in dialect:
            njd_features = convert_hatsuonbin_style(njd_features)
        if "TTtoT" in dialect:
            njd_features = convert_tt2t_style(njd_features)
        if "StoZ" in dialect:
            njd_features = convert_s2z_style(njd_features)
        if "DtoR" in dialect:
            njd_features = convert_d2r_style(njd_features)

    return njd_features


def make_label(njd_features: list[NJDFeature], jtalk: Union[OpenJTalk, None] = None) -> list[str]:
    """Make full-context label using features

    Args:
        njd_features (List[NJDFeature]): features for NJDNode.
        jtalk (OpenJTalk, optional): OpenJTalk instance to use. If None, use global instance.
          Default is None.

    Returns:
        List[str]: full-context labels.
    """
    if jtalk is not None:
        return jtalk.make_label(njd_features)
    global _global_jtalk
    with _global_jtalk() as jtalk:
        return jtalk.make_label(njd_features)


def mecab_dict_index(path: str, out_path: str, dn_mecab: Union[str, None] = None) -> None:
    """Create user dictionary

    Args:
        path (str): path to user csv
        out_path (str): path to output dictionary
        dn_mecab (optional. str): path to mecab dictionary
    """
    if not exists(path):
        raise FileNotFoundError(f"no such file or directory: {path}")
    if dn_mecab is None:
        dn_mecab = OPEN_JTALK_DICT_DIR.decode("utf-8")
    r = _mecab_dict_index(dn_mecab.encode("utf-8"), path.encode("utf-8"), out_path.encode("utf-8"))

    # NOTE: mecab load returns 1 if success, but mecab_dict_index return the opposite
    # yeah it's confusing...
    if r != 0:
        raise RuntimeError("Failed to create user dictionary")


def update_global_jtalk_with_user_dict(paths: Union[str, list[str]]) -> None:
    """Update global openjtalk instance with the user dictionary

    Note that this will change the global state of the openjtalk module.

    Args:
        paths (Union[str, List[str]]): path to user dictionary
            (can specify multiple user dictionaries in the list)
    """
    unset_user_dict()

    if isinstance(paths, str):
        paths_str = f"{_default_user_dict},{paths}"
        paths = paths.split(",")
    else:
        paths_str = ",".join(paths)
        paths_str = f"{_default_user_dict},{paths_str}"

    # 全てのユーザー辞書パスの存在を確認
    for p in paths:
        if not exists(p):
            raise FileNotFoundError(f"no such file or directory: {p}")

    global _global_jtalk
    with _global_jtalk():
        _global_jtalk = _global_instance_manager(
            instance=OpenJTalk(dn_mecab=OPEN_JTALK_DICT_DIR, userdic=paths_str.encode("utf-8")),
        )


def unset_user_dict() -> None:
    """Stop applying user dictionary"""
    global _global_jtalk
    with _global_jtalk():
        _global_jtalk = _global_instance_manager(
            instance=OpenJTalk(dn_mecab=OPEN_JTALK_DICT_DIR),
        )


def build_mecab_dictionary(dn_mecab: Union[str, None] = None) -> None:
    """Build mecab dictionary

    Args:
        dn_mecab (optional. str): path to mecab dictionary
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
