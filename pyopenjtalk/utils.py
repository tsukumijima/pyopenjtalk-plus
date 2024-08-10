from typing import Any, Dict, List

from sudachipy import dictionary, tokenizer

from .types import NJDFeature
from .yomi_model.nani_predict import predict


def merge_njd_marine_features(
    njd_features: List[NJDFeature], marine_results: Dict[str, Any]
) -> List[NJDFeature]:
    features = []

    marine_accs = marine_results["accent_status"]
    marine_chain_flags = marine_results["accent_phrase_boundary"]

    assert (
        len(njd_features) == len(marine_accs) == len(marine_chain_flags)
    ), "Invalid sequence sizes in njd_results, marine_results"

    for node_index, njd_feature in enumerate(njd_features):
        _feature = {}
        for feature_key in njd_feature.keys():
            if feature_key == "acc":
                _feature["acc"] = int(marine_accs[node_index])
            elif feature_key == "chain_flag":
                _feature[feature_key] = int(marine_chain_flags[node_index])
            else:
                _feature[feature_key] = njd_feature[feature_key]
        features.append(_feature)
    return features


def modify_kanji_yomi(
    text: str, pyopen_njd: List[NJDFeature], multi_read_kanji_list: List[str]
) -> List[NJDFeature]:
    sudachi_yomi = sudachi_analyze(text, multi_read_kanji_list)
    return_njd = []
    pre_dict = None

    for dict in reversed(pyopen_njd):
        if dict["orig"] in multi_read_kanji_list:
            try:
                correct_yomi = sudachi_yomi.pop()
            except IndexError:
                return pyopen_njd
            if correct_yomi[0] != dict["orig"]:
                return pyopen_njd
            elif dict["orig"] == "何":
                is_read_nan = predict([pre_dict])
                if is_read_nan == 1:
                    dict["pron"] = "ナン"
                    dict["read"] = "ナン"
                else:
                    dict["pron"] = "ナニ"
                    dict["read"] = "ナニ"
                return_njd.append(dict)

            else:
                if correct_yomi[0] == "方" and correct_yomi[1] == "ホウ":
                    correct_yomi[1] = "ホオ"
                dict["pron"] = correct_yomi[1]
                dict["read"] = correct_yomi[1]
                return_njd.append(dict)
        else:
            return_njd.append(dict)
        pre_dict = dict

    return_njd.reverse()
    return return_njd


def sudachi_analyze(text: str, multi_read_kanji_list: List[str]) -> List[List[str]]:
    """
    複数の読み方をする漢字の読みを sudachi で形態素解析した結果をリストで返す
    例: 風がこんな風に吹く → [('風', 'カゼ'), ('風', 'フウ')]

    Args:
        text (str): 読み対象となるテキスト
        multi_read_kanji_list (List[str]): 複数の読み方をする漢字のリスト(ex : 何、風、方)

    Returns:
        yomi_list (List[List[str]]): 漢字とその読み方のリスト
    """

    text = text.replace("ー", "")
    tokenizer_obj = dictionary.Dictionary().create()
    mode = tokenizer.Tokenizer.SplitMode.C
    m_list = tokenizer_obj.tokenize(text, mode)
    yomi_list = [
        [m.surface(), m.reading_form()] for m in m_list if m.surface() in multi_read_kanji_list
    ]
    return yomi_list


def retreat_acc_nuc(njd_features: List[NJDFeature]) -> List[NJDFeature]:
    """
    長母音、重母音、撥音がアクセント核に来た場合にひとつ前のモーラにアクセント核がズレるルールの実装

    Args:
        njd_features (List[NJDFeature]): run_frontend() の結果

    Returns:
        List[NJDFeature]: 修正後の njd_features
    """

    inappropriate_for_nuclear_chars = ["ー", "ッ", "ン"]
    delete_youon = str.maketrans("", "", "ャュョァィゥェォ")
    for _, njd in enumerate(njd_features):
        # アクセント境界直後の node (chain_flag 0 or -1) にアクセント核の位置の情報が入っている
        if njd["chain_flag"] in [0, -1]:
            head = njd
            acc = njd["acc"]
            phase_len = 0

        phase_len += njd["mora_size"]
        pron = njd["pron"].translate(delete_youon)
        if len(pron) == 0:
            pron = njd["pron"]

        if acc > 0:
            if acc <= njd["mora_size"]:
                try:
                    nuc_pron = pron[acc - 1]
                except IndexError:
                    nuc_pron = pron[0]
                if nuc_pron in inappropriate_for_nuclear_chars:
                    head["acc"] += -1
                acc = -1
            else:
                acc = acc - njd["mora_size"]

    return njd_features


def modify_acc_after_chaining(njd_features: List[NJDFeature]) -> List[NJDFeature]:
    """
    品詞「特殊・マス」は直前に接続する動詞にアクセント核がある場合、アクセント核を「ま」に移動させる法則がある
    書きます → か[きま]す, 参ります → ま[いりま]す
    書いております → [か]いております

    Args:
        njd_features (List[NJDFeature]): run_frontend() の結果

    Returns:
        List[NJDFeature]: 修正後の njd_features
    """

    for njd in njd_features:
        # アクセント境界直後の node (chain_flag 0 or -1) にアクセント核の位置の情報が入っている
        if njd["chain_flag"] in [0, -1]:
            is_after_nuc = False
            head = njd
            acc = njd["acc"]
            phase_len = 0
        # acc = 0 の場合は「特殊・マス」は存在しないと考えてよい
        if acc == 0:
            continue
        elif is_after_nuc:
            if njd["ctype"] == "特殊・マス":
                head["acc"] = phase_len + 1 if njd["cform"] != "未然形" else phase_len + 2
            elif njd["ctype"] == "特殊・ナイ":
                head["acc"] = phase_len
            elif njd["orig"] in ["れる", "られる", "すぎる", "せる", "させる"]:
                head["acc"] = phase_len + njd["acc"]
            else:
                is_after_nuc = False
                acc = 0
            phase_len += njd["mora_size"]

        else:
            phase_len += njd["mora_size"]
            if acc <= njd["mora_size"]:
                is_after_nuc = True
            else:
                acc = acc - njd["mora_size"]

    return njd_features
