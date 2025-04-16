from typing import Any, Union

from sudachipy import dictionary, tokenizer

from .openjtalk import OpenJTalk
from .types import NJDFeature
from .yomi_model.nani_predict import predict


def merge_njd_marine_features(
    njd_features: list[NJDFeature], marine_results: dict[str, Any]
) -> list[NJDFeature]:
    features = []

    marine_accs = marine_results["accent_status"]
    marine_chain_flags = marine_results["accent_phrase_boundary"]

    assert len(njd_features) == len(marine_accs) == len(marine_chain_flags), (
        "Invalid sequence sizes in njd_results, marine_results"
    )

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
    text: str, pyopen_njd: list[NJDFeature], multi_read_kanji_list: list[str]
) -> list[NJDFeature]:
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


def sudachi_analyze(text: str, multi_read_kanji_list: list[str]) -> list[list[str]]:
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


def retreat_acc_nuc(njd_features: list[NJDFeature]) -> list[NJDFeature]:
    """
    長母音、重母音、撥音がアクセント核に来た場合にひとつ前のモーラにアクセント核がズレるルールの実装

    Args:
        njd_features (List[NJDFeature]): run_frontend() の結果

    Returns:
        List[NJDFeature]: 修正後の njd_features
    """

    if not njd_features:
        return njd_features

    inappropriate_for_nuclear_chars = ["ー", "ッ", "ン"]
    delete_youon = str.maketrans("", "", "ャュョァィゥェォ")
    phase_len = 0
    acc = 0
    head = njd_features[0]

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


def modify_acc_after_chaining(njd_features: list[NJDFeature]) -> list[NJDFeature]:
    """
    品詞「特殊・マス」は直前に接続する動詞にアクセント核がある場合、アクセント核を「ま」に移動させる法則がある
    書きます → か[きま]す, 参ります → ま[いりま]す
    書いております → [か]いております

    Args:
        njd_features (List[NJDFeature]): run_frontend() の結果

    Returns:
        List[NJDFeature]: 修正後の njd_features
    """

    if not njd_features:
        return njd_features

    acc = 0
    is_after_nuc = False
    phase_len = 0
    head = njd_features[0]

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


def process_odori_features(
    njd_features: list[NJDFeature],
    jtalk: Union[OpenJTalk, None] = None,
) -> list[NJDFeature]:
    """踊り字（々）と一の字点（ゝ、ゞ、ヽ、ヾ）の読みを適切に処理する後処理関数

    OpenJTalk の挙動に合わせて、連続する踊り字を処理する
    踊り字の数に応じて読みを繰り返す：
    - 「叙々苑」→「ジョジョエン」
    - 「叙々々苑」→「ジョジョジョエン」
    - 「叙々々々苑」→「ジョジョジョジョエン」

    また、複数漢字や複数トークンの場合は、前の読みをそのまま使用する：
    - 「部分々々」→「ブブン」
    - 「其他々々」→「ソノホカ」
    - 「前進々々」→「ゼンシンゼンシン」

    さらに、単独の踊り字で直前のトークンが複数漢字の場合は、適宜直前と直後の漢字を使って再解析：
    - 「結婚式々場」→「ケッコンシキシキジョウ」
    - 「民主々義」→「ミンシュシュギ」
    - 「学生々活」→「ガクセイセイカツ」

    一の字点（ゝ、ゞ、ヽ、ヾ）は直前の文字を繰り返す：
    - 「こゝろ」→「こころ」
    - 「みすゞ」→「みすず」
    - 「づゝ」→「づつ」
    - 「ぶゞ漬け」→「ぶぶ漬け」

    Args:
        njd_features (List[NJDFeature]): OpenJTalk の形態素解析結果
        jtalk (Union[OpenJTalk, None], optional): OpenJTalk インスタンス。
            単独の踊り字の直前の漢字を再解析する場合に使用。デフォルトは None。

    Returns:
        List[NJDFeature]: 踊り字の読みを修正した形態素解析結果
    """

    def is_dancing(orig: str) -> bool:
        """文字列が踊り字のみで構成されているかを判定する

        Args:
            orig (str): 判定対象の文字列

        Returns:
            bool: 踊り字のみで構成されている場合は True
        """
        return set(orig) == {"々"}

    def is_odoriji(orig: str) -> bool:
        """文字列が一の字点のみで構成されているかを判定する

        Args:
            orig (str): 判定対象の文字列

        Returns:
            bool: 一の字点のみで構成されている場合は True
        """
        return set(orig) <= {"ゝ", "ゞ", "ヽ", "ヾ"}

    def count_odori(orig: str) -> int:
        """文字列に含まれる踊り字の数をカウントする

        Args:
            orig (str): カウント対象の文字列

        Returns:
            int: 踊り字の数
        """
        return orig.count("々")

    def is_kanji_token(token: NJDFeature) -> bool:
        """トークンが漢字を含むかを判定する

        Args:
            token (NJDFeature): 判定対象のトークン

        Returns:
            bool: 漢字を含む場合は True
        """
        # 品詞が記号の場合は False
        if token["pos"] == "記号":
            return False
        # 原形に漢字が含まれているかを判定
        return any(0x4E00 <= ord(c) <= 0x9FFF for c in token["orig"])

    def is_single_kanji_token(token: NJDFeature) -> bool:
        """トークンが1文字の漢字で構成されているかを判定する

        Args:
            token (NJDFeature): 判定対象のトークン

        Returns:
            bool: 1文字の漢字で構成されている場合は True
        """
        return (
            is_kanji_token(token)
            and len(token["orig"]) == 1
            and 0x4E00 <= ord(token["orig"][0]) <= 0x9FFF
        )

    def needs_reanalysis(
        odori_feature: NJDFeature,
        prev_feature: NJDFeature,
        next_feature: Union[NJDFeature, None] = None,
    ) -> tuple[bool, str, Union[str, None]]:
        """踊り字の直前の漢字を再解析する必要があるかを判定

        Args:
            odori_feature (NJDFeature): 踊り字のトークン
            prev_feature (NJDFeature): 直前のトークン
            next_feature (Union[NJDFeature, None], optional): 後続のトークン

        Returns:
            Tuple[bool, str, Union[str, None]]: (再解析が必要か, 再解析する漢字, 後続の漢字)
        """
        # 踊り字が単独（1文字）でない場合は再解析不要
        if count_odori(odori_feature["orig"]) != 1:
            return False, "", None

        # 直前のトークンが漢字を含まない場合は再解析不要
        if not is_kanji_token(prev_feature):
            return False, "", None

        # 直前のトークンが複数文字で構成されている場合
        if len(prev_feature["orig"]) > 1:
            # 直前のトークンの最後の漢字を抽出
            last_char = prev_feature["orig"][-1]
            if 0x4E00 <= ord(last_char) <= 0x9FFF:
                # 後続のトークンが1文字の漢字の場合は、その漢字も含めて再解析
                if next_feature is not None and is_single_kanji_token(next_feature):
                    return True, last_char, next_feature["orig"]
                # それ以外の場合は最後の漢字のみを再解析
                return True, last_char, None

        return False, "", None

    def reanalyze_kanji(kanji: str, jtalk: OpenJTalk) -> list[NJDFeature]:
        """漢字を再解析して読みを取得

        Args:
            kanji (str): 解析対象の漢字
            jtalk (OpenJTalk): OpenJTalk インスタンス

        Returns:
            List[NJDFeature]: 解析結果
        """
        features = jtalk.run_frontend(kanji)
        return features

    def process_odoriji(
        odori_feature: NJDFeature,
        prev_feature: NJDFeature,
    ) -> NJDFeature:
        """一の字点の読みを処理する

        Args:
            odori_feature (NJDFeature): 一の字点のトークン
            prev_feature (NJDFeature): 直前のトークン

        Returns:
            NJDFeature: 読みを修正したトークン
        """
        # 直前のトークンの読みを取得
        # 読みとモーラサイズを1文字ずつに分解
        prev_read_chars = []
        prev_pron_chars = []
        prev_mora_sizes = []

        # カタカナを1文字ずつに分解
        i = 0
        while i < len(prev_feature["read"]):
            char = prev_feature["read"][i]
            # 小書き文字の処理
            if i + 1 < len(prev_feature["read"]) and prev_feature["read"][i + 1] in {"ャ", "ュ", "ョ", "ァ", "ィ", "ゥ", "ェ", "ォ"}:  # fmt: skip
                prev_read_chars.append(char + prev_feature["read"][i + 1])
                i += 2
            else:
                prev_read_chars.append(char)
                i += 1

        i = 0
        while i < len(prev_feature["pron"]):
            char = prev_feature["pron"][i]
            # 小書き文字の処理
            if i + 1 < len(prev_feature["pron"]) and prev_feature["pron"][i + 1] in {"ャ", "ュ", "ョ", "ァ", "ィ", "ゥ", "ェ", "ォ"}:  # fmt: skip
                prev_pron_chars.append(char + prev_feature["pron"][i + 1])
                i += 2
            else:
                prev_pron_chars.append(char)
                i += 1

        # モーラサイズを文字数に応じて分配
        mora_per_char = prev_feature["mora_size"] / len(prev_read_chars)
        prev_mora_sizes = [mora_per_char] * len(prev_read_chars)

        # 最後の文字の読みを取得
        prev_read = prev_read_chars[-1]
        prev_pron = prev_pron_chars[-1]
        prev_mora_size = prev_mora_sizes[-1]

        # 濁点化のマッピング
        dakuten_map = {
            "カ": "ガ", "キ": "ギ", "ク": "グ", "ケ": "ゲ", "コ": "ゴ",
            "サ": "ザ", "シ": "ジ", "ス": "ズ", "セ": "ゼ", "ソ": "ゾ",
            "タ": "ダ", "チ": "ヂ", "ツ": "ヅ", "テ": "デ", "ト": "ド",
            "ハ": "バ", "ヒ": "ビ", "フ": "ブ", "ヘ": "ベ", "ホ": "ボ",
            "か": "が", "き": "ぎ", "く": "ぐ", "け": "げ", "こ": "ご",
            "さ": "ざ", "し": "じ", "す": "ず", "せ": "ぜ", "そ": "ぞ",
            "た": "だ", "ち": "ぢ", "つ": "づ", "て": "で", "と": "ど",
            "は": "ば", "ひ": "び", "ふ": "ぶ", "へ": "べ", "ほ": "ぼ",
        }  # fmt: skip

        # 濁点の逆引きマッピング
        dakuten_reverse_map = {v: k for k, v in dakuten_map.items()}

        # 一の字点の種類を判定
        odori_char = odori_feature["orig"]
        if odori_char in {"ゝ", "ヽ"}:
            # 濁点なしの場合は直前の読みの濁点なしバージョンを使用
            odori_feature["read"] = dakuten_reverse_map.get(prev_read, prev_read)
            odori_feature["pron"] = dakuten_reverse_map.get(prev_pron, prev_pron)
            odori_feature["mora_size"] = int(prev_mora_size)
        elif odori_char in {"ゞ", "ヾ"}:
            # 濁点ありの場合は直前の読みを濁点化
            # 読みを濁点化
            odori_feature["read"] = dakuten_map.get(prev_read, prev_read)
            odori_feature["pron"] = dakuten_map.get(prev_pron, prev_pron)
            odori_feature["mora_size"] = int(prev_mora_size)

        # 記号扱いにすると後の処理で誤作動するケースがありそうな気がするので、適当に一般名詞としておく
        if odori_feature["pos"] == "記号":
            odori_feature["pos"] = "名詞"
            odori_feature["pos_group1"] = "一般"
            odori_feature["pos_group2"] = "*"
            odori_feature["pos_group3"] = "*"
            odori_feature["ctype"] = "*"
            odori_feature["cform"] = "*"

        return odori_feature

    i = 0
    while i < len(njd_features):
        if is_dancing(njd_features[i]["orig"]):
            # 単独の踊り字で再解析が必要な場合
            if i > 0 and jtalk is not None:
                next_feature = njd_features[i + 1] if i + 1 < len(njd_features) else None
                needs_reanalysis_flag, target_kanji, next_kanji = needs_reanalysis(
                    njd_features[i], njd_features[i - 1], next_feature
                )
                if needs_reanalysis_flag:
                    # 後続の漢字も含めて再解析する場合
                    if next_kanji is not None:
                        analyzed = reanalyze_kanji(target_kanji + next_kanji, jtalk)
                        # 再解析結果を踊り字トークンに反映し、後続の漢字トークンを削除
                        njd_features[i : i + 2] = analyzed
                        i += len(analyzed)
                        continue
                    else:
                        # 最後の漢字のみを再解析
                        analyzed = reanalyze_kanji(target_kanji, jtalk)
                        # 再解析結果を踊り字トークンに反映
                        njd_features[i] = analyzed[0]
                        # 記号扱いにすると後の処理で誤作動するケースがありそうな気がするので、適当に一般名詞としておく
                        njd_features[i]["pos"] = "名詞"
                        njd_features[i]["pos_group1"] = "一般"
                        njd_features[i]["pos_group2"] = "*"
                        njd_features[i]["pos_group3"] = "*"
                        njd_features[i]["ctype"] = "*"
                        njd_features[i]["cform"] = "*"
                        i += 1
                        continue

            # 連続する踊り字トークンを特定
            start = i
            end = i
            total_odori = 0
            while end < len(njd_features) and is_dancing(njd_features[end]["orig"]):
                total_odori += count_odori(njd_features[end]["orig"])
                end += 1

            # 直前の漢字トークンを抽出
            normal_list = []
            j = start - 1
            collected_chars = 0
            while j >= 0:
                if is_kanji_token(njd_features[j]):
                    normal_list.append(njd_features[j])
                    collected_chars += len(njd_features[j]["orig"])
                    # 踊り字が2文字以上の場合は2文字分、1文字の場合は1文字分まで収集
                    if collected_chars >= (2 if total_odori >= 2 else 1):
                        break
                j -= 1
            normal_list.reverse()  # 元の順序に戻す

            # 前に適切な漢字がない場合はスキップ
            if not normal_list:
                i = end
                continue

            # 置換用の読みを決定
            # 単一漢字の場合は踊り字の数に応じて繰り返し、
            # 複数漢字の場合はそのまま使用
            is_single_kanji = len(normal_list) == 1 and len(normal_list[0]["orig"]) == 1
            if is_single_kanji:
                # 単一漢字の場合
                base_read = normal_list[0]["read"]
                base_pron = normal_list[0]["pron"]
                base_mora_size = normal_list[0]["mora_size"]
            else:
                # 複数漢字の場合
                base_read = "".join(item["read"] for item in normal_list)
                base_pron = "".join(item["pron"] for item in normal_list)
                base_mora_size = sum(item["mora_size"] for item in normal_list)

            # 連続する踊り字トークンを処理
            processed_odori = 0
            for j in range(start, end):
                current_odori = count_odori(njd_features[j]["orig"])
                if is_single_kanji:
                    # 単一漢字の場合は踊り字の数に応じて繰り返す
                    njd_features[j]["read"] = base_read * current_odori
                    njd_features[j]["pron"] = base_pron * current_odori
                    njd_features[j]["mora_size"] = base_mora_size * current_odori
                else:
                    # 複数漢字の場合はそのまま使用
                    njd_features[j]["read"] = base_read
                    njd_features[j]["pron"] = base_pron
                    njd_features[j]["mora_size"] = base_mora_size

                processed_odori += current_odori

                # 記号扱いにすると後の処理で誤作動するケースがありそうな気がするので、適当に一般名詞としておく
                if njd_features[j]["pos"] == "記号":
                    njd_features[j]["pos"] = "名詞"
                    njd_features[j]["pos_group1"] = "一般"
                    njd_features[j]["pos_group2"] = "*"
                    njd_features[j]["pos_group3"] = "*"
                    njd_features[j]["ctype"] = "*"
                    njd_features[j]["cform"] = "*"

            i = end
        elif is_odoriji(njd_features[i]["orig"]):
            # 一の字点の処理
            if i > 0:
                njd_features[i] = process_odoriji(njd_features[i], njd_features[i - 1])
            i += 1
        else:
            i += 1

    return njd_features
