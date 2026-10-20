"""
正規化前の顔文字と腕・持ち物の形を、形態素解析後の単語境界と照合する。
"""

import unicodedata

from ._mapping import build_caller_text_spans_by_mecab_character, njd_feature_char_spans
from .openjtalk import OpenJTalk
from .types import MeCabMorph, NJDFeature, NormalizeMode


_EYES = "・･ﾟ゜°•●○◕╹>＞<＜≧≦∩＾^;；'＇‘’｀´`￣-˘ಠ◉б≖⌒ゝゞ〃ﾉノ/／＼\\˙◔◠◡☉⊙꒪ↀ⩌òóŏÒÓìíˊˋ˃˂❛◍º＝=╯╰◞◟☋☌☍✹☆★へつ↑↓◜◝՞Φ¯‾ᴖ⁰Ծ✖✘l*+ーಥ눈╮╭ಲ൭ˇᵕᴗ"
_MOUTHS = "ω∀▽∇Ддεз﹏ヮ◡ㅅ益皿_＿‿ーｰ-～〜~︿ᵕᴗᗜ⊖△◇○〇口ρڡڼਊة㉦⌓³₃⊱ᯅ⤚。oｏOＯㅂ‸︶⌣⁀Δ∧ᆺ▿﹃꒳Θワx0.ȏᗨへエළ╯ᴥｪェꞈˬ"
_ARMS = "ヾヽゞゝﾉノつっ⊂⊃∩╰╯╭╮┌┐└┘┗┛┏┓」｢｣⌒✌☝☛☚٩۶งว੭ʅʃᕦᕤ＼／ιυщوΣ∑mopqdbcvεσρぅ彡卍ฅ∪լ∠_「ｼﾞ"
_ORNAMENTS = " 　*＊｡。.,，､、:：♡♥❤✧✿❀๑〃＠@ｰー̀́͜ิﾞ゛゙ˇ▔▰灬˶⸝︶⌣|＃#∗❁〄☞☜̵ุ̣̥̑⁾ั∞⋈✗ㆀ†ूु∽͟͞~"
_PAIRS = {
    "(": ")",
    "（": "）",
    "⎛": "⎞",
    "꒰": "꒱",
    "₍": "₎",
    "ʕ": "ʔ",
    "∪": "∪",
    "⎝": "⎠",
    "|": ")",
    "｜": "）",
}
_STRONG_MOUTHS = "ω∀▽∇Ддεз﹏ヮᴥｪェᵕ꒳"
_BARS = "-－−ーｰ―─━═╌┄"


def exclude_kaomoji_from_numeral_processing(
    raw_text: str,
    text: str,
    mecab_features: list[str],
    morphs: list[MeCabMorph],
    normalize_mode: NormalizeMode,
    *,
    jtalk: OpenJTalk,
) -> list[str]:
    """
    顔文字に含まれる数詞・助数詞・英字を、NJD の数詞処理へ渡す前に記号として扱う。
    「10 m(_ _)m」の腕の「m」をメートルの助数詞として結合させず、ユーザー辞書の語や本文と重なる顔文字候補は元の解析を保つ。

    Args:
        raw_text (str): Unicode 正規化前の入力文
        text (str): フロントエンドが解析した Unicode 日本語テキスト
        mecab_features (list[str]): MeCab の feature 文字列のリスト
        morphs (list[MeCabMorph]): 同じ解析から得た MeCab 形態素列
        normalize_mode (NormalizeMode): 入力文に適用した Unicode 正規化方式
        jtalk (OpenJTalk): 本文の正規化に使うインスタンス

    Returns:
        list[str]: 顔文字の語を記号に置き換えた feature 列
    """

    _, accepted_morphs = silence_kaomoji(raw_text, text, [], morphs, normalize_mode, jtalk=jtalk)
    face_spans = [
        morph["char_span"] for morph in accepted_morphs if morph["features"][1] == "顔文字"
    ]
    result: list[str] = []
    normalized = jtalk.normalize_for_mecab(text)
    # 候補解析が空白を除いて連続記号をまとめても、照合した文字位置は元の本文へ戻す
    character_positions = [i for i, character in enumerate(normalized) if character != "　"]
    matching_text = "".join(normalized[i] for i in character_positions)
    position = 0
    for feature in mecab_features:
        fields = feature.split(",")
        surface = fields[0]
        # NJD 用の素性では連続記号がまとまるため、1文字ずつの詳細形態素とは文字範囲で照合する
        matching_surface = surface.replace("　", "")
        matching_start = matching_text.index(matching_surface, position)
        matching_end = matching_start + len(matching_surface)
        position = matching_end
        start = character_positions[matching_start]
        end = character_positions[matching_end - 1] + 1
        if any(a <= start and end <= b for a, b in face_spans) and (
            fields[2] in ("数", "アルファベット") or fields[3] == "助数詞"
        ):
            feature = f"{surface},記号,読点,顔文字,*,*,*,{surface},、,、,0/0,*"
        result.append(feature)
    return result


def silence_kaomoji(
    raw_text: str,
    text: str,
    features: list[NJDFeature],
    morphs: list[MeCabMorph],
    normalize_mode: NormalizeMode,
    *,
    jtalk: OpenJTalk,
) -> tuple[list[NJDFeature], list[MeCabMorph]]:
    """
    顔文字を1つの要素にまとめ、読みと発音を空にする。
    語の一部やユーザー辞書の語と重なる候補は解析結果を残し、語と重なる腕や持ち物だけを顔文字の範囲から外す。

    Args:
        raw_text (str): Unicode 正規化前の入力文
        text (str): フロントエンドが解析した Unicode 日本語テキスト
        features (list[NJDFeature]): NJD 処理後の形態素列
        morphs (list[MeCabMorph]): 同じ解析から得た MeCab 形態素列
        normalize_mode (NormalizeMode): 入力文に適用した Unicode 正規化方式
        jtalk (OpenJTalk): 本文の正規化に使うインスタンス

    Returns:
        tuple[list[NJDFeature], list[MeCabMorph]]: 顔文字をまとめた NJD 形態素列と MeCab 形態素列
    """

    parts = find_kaomoji_parts(raw_text)
    if len(parts) == 0:
        return features, morphs
    normalized = jtalk.normalize_for_mecab(text)
    source_spans = build_caller_text_spans_by_mecab_character(
        normalized, raw_text, normalize_mode, inference_jtalk=jtalk
    )
    features = _split_symbol_features(features, morphs)
    feature_spans = njd_feature_char_spans(features, morphs)
    accepted: list[tuple[int, int]] = []
    for core_start, core_end, start, end in parts:
        core_indices = [
            i for i, (a, b) in enumerate(source_spans) if a < core_end and core_start < b
        ]
        extent_indices = [i for i, (a, b) in enumerate(source_spans) if a < end and start < b]
        if not core_indices or not extent_indices:
            continue
        core_start, core_end = core_indices[0], core_indices[-1] + 1
        start, end = extent_indices[0], extent_indices[-1] + 1
        is_rejected = False
        for morph in morphs:
            a, b = morph["char_span"]
            if b <= start or a >= end:
                continue
            is_registered = 0 < morph["dictionary_index"] < 255
            intersects_core = a < core_end and core_start < b
            if intersects_core and (is_registered or a < start or b > end):
                is_rejected = True
                break
            is_lexical = (
                morph["is_unknown"] is False
                and morph["features"][0] != "記号"
                and len(morph["surface"]) > 1
                and not (
                    morph["surface"][0] in _ARMS and all(c in _BARS for c in morph["surface"][1:])
                )
            )
            if intersects_core is False and (is_registered or is_lexical or a < start or b > end):
                if b <= core_start:
                    start = max(start, b)
                elif a >= core_end:
                    end = min(end, a)
        # NJD が顔の輪郭と本文を1つにまとめた場合も、本文まで消さず元の解析を残す
        if is_rejected or any(
            a < end and start < b and (a < start or b > end) for a, b in feature_spans
        ):
            continue
        if accepted and start < accepted[-1][1]:
            accepted[-1] = (accepted[-1][0], max(end, accepted[-1][1]))
        else:
            accepted.append((start, end))
    result_features: list[NJDFeature] = []
    result_morphs: list[MeCabMorph] = []
    feature_index = morph_index = 0
    for start, end in accepted:
        while feature_index < len(features) and feature_spans[feature_index][1] <= start:
            result_features.append(features[feature_index])
            feature_index += 1
        while feature_index < len(features) and feature_spans[feature_index][0] < end:
            feature_index += 1
        while morph_index < len(morphs) and morphs[morph_index]["char_span"][1] <= start:
            result_morphs.append(morphs[morph_index])
            morph_index += 1
        template = morphs[morph_index].copy()
        while morph_index < len(morphs) and morphs[morph_index]["char_span"][0] < end:
            morph_index += 1
        surface = normalized[start:end]
        result_features.append(
            NJDFeature(
                string=surface,
                orig=surface,
                pos="その他",
                pos_group1="顔文字",
                pos_group2="*",
                pos_group3="*",
                ctype="*",
                cform="*",
                read="",
                pron="",
                acc=0,
                mora_size=0,
                chain_rule="*",
                chain_flag=1,
                is_reading_protected=False,
                is_reading_rewritten=True,
            )
        )
        template.update(
            surface=surface,
            char_span=(start, end),
            is_unknown=False,
            is_ignored=False,
            dictionary_index=255,
            is_tsqyomi_selected=False,
            is_tsqyomi_target=False,
            features=["その他", "顔文字", "*", "*", "*", "*", surface, "", "", "0/0", "*", "1"],
        )
        result_morphs.append(template)
    result_features.extend(features[feature_index:])
    result_morphs.extend(morphs[morph_index:])
    return result_features, result_morphs


def _split_symbol_features(
    features: list[NJDFeature], morphs: list[MeCabMorph]
) -> list[NJDFeature]:
    """
    複数の顔文字にまたがる記号の NJD ノードを、MeCab の文字範囲ごとに分ける。
    「＾＾）　（＾＾）」のように顔文字と空白が1ノードにまとまっても、それぞれの顔文字だけを無音にできるようにする。

    Args:
        features (list[NJDFeature]): NJD 処理後の形態素列
        morphs (list[MeCabMorph]): 同じ解析から得た MeCab 形態素列

    Returns:
        list[NJDFeature]: 記号の結合ノードを分割した形態素列
    """

    spans = njd_feature_char_spans(features, morphs)
    result: list[NJDFeature] = []
    morph_index = 0
    for feature, (start, end) in zip(features, spans, strict=True):
        # NJD ノードが覆う文字範囲と重なる MeCab 形態素を、入力順に集める
        while morph_index < len(morphs) and morphs[morph_index]["char_span"][1] <= start:
            morph_index += 1
        end_index = morph_index
        while end_index < len(morphs) and morphs[end_index]["char_span"][0] < end:
            end_index += 1
        parts = morphs[morph_index:end_index]
        # 記号の表層を形態素の連結で再現できる場合だけ分割し、語の解析結果は保つ
        if (
            feature["pos"] != "記号"
            or feature["pron"] != "、"
            or len(parts) < 2
            or "".join(morph["surface"] for morph in parts).replace("　", "")
            != feature["string"].replace("　", "")
        ):
            result.append(feature)
            continue
        for morph in parts:
            # 空白は元の MeCab 形態素列で保持し、NJD へは各記号のノードだけを渡す
            if morph["is_ignored"] is True:
                continue
            part = feature.copy()
            part["string"] = part["orig"] = morph["surface"]
            result.append(part)
    return result


def find_kaomoji_parts(text: str) -> list[tuple[int, int, int, int]]:
    """
    正規化前の文字位置で、顔の本体と腕・持ち物を含む範囲を返す。

    Args:
        text (str): Unicode 正規化前の入力文

    Returns:
        list[tuple[int, int, int, int]]: (本体の開始、本体の終端、全体の開始、全体の終端) の一覧
    """

    parts: list[tuple[int, int, int, int]] = []
    consumed = 0
    # 本文の引用符は、顔の外側にある対応の取れた組だけを残す
    quotes: set[int] = set()
    for index, original in enumerate(text):
        if index < consumed:
            continue
        end = _extra_face_end(text, index)
        if end is None and original in _PAIRS:
            depth = 0
            for at in range(index + 1, min(len(text), index + 50)):
                c = _fold(text[at])
                if c in "⎛꒰₍\n\r":
                    break
                if c == "(":
                    depth += 1
                if c in ")⎞꒱₎ʔ⎠" or (original == "∪" and c == "∪"):
                    if c == ")" and depth > 0:
                        depth -= 1
                    else:
                        if text[at] == _PAIRS[original] or (original == "⎝" and c == "⎞"):
                            end = at + 1
                        break
        if end is None:
            continue
        start, extent_end = index, end
        for at in range(index - 1, max(consumed - 1, index - 17), -1):
            c = text[at]
            if (
                not _arm(c)
                or c in "ｼﾞ"
                or (_letter(c) and ((at > 0 and _letter(text[at - 1])) or _letter(text[at + 1])))
            ):
                break
            start = at
        last = ""
        for at in range(end, min(len(text), end + 16)):
            c = text[at]
            if not (_arm(c) or (c in "ｼﾞ" and last in "ﾉｼ")) or (
                _letter(c)
                and (_letter(text[at - 1]) or (at + 1 < len(text) and _letter(text[at + 1])))
            ):
                break
            extent_end = at + 1
            last = c
        if _extra_face_end(text, index) != end and not _body(
            "".join(_fold(c) for c in text[index + 1 : end - 1]),
            start != index or extent_end != end or original in "|｜",
        ):
            continue
        # 濁点・結合文字・異体字セレクタも顔文字の範囲に含める
        while extent_end < len(text) and _decoration(text[extent_end]):
            extent_end += 1
        if text[extent_end - 1] == "\u200d" or (start > 0 and text[start - 1] == "\u200d"):
            continue
        start, extent_end = _extend_objects(text, index, end, start, extent_end)
        parts.append((index, end, start, extent_end))
        consumed = end
    stack: list[tuple[int, str]] = []
    for index, c in enumerate(text):
        if any(a <= index < b for a, b, _, _ in parts):
            continue
        if c in "「『":
            stack.append((index, c))
        elif stack and (stack[-1][1], c) in (("「", "」"), ("『", "』")):
            quotes.update((stack.pop()[0], index))
    return [
        (
            a,
            b,
            max([s, *[q + 1 for q in quotes if s <= q < a]]),
            min([e, *[q for q in quotes if b <= q < e]]),
        )
        for a, b, s, e in parts
    ]


def _extra_face_end(text: str, start: int) -> int | None:
    """
    手で囲んだ顔、つながった顔、輪郭のない定型の顔の終端を返す。

    Args:
        text (str): Unicode 正規化前の入力文
        start (int): 顔の開始文字位置

    Returns:
        int | None: 顔の終端 (定型の顔でなければ None)
    """

    # 本文の各文字位置から呼ばれるため、調べる先頭の形と長さを限定して長文のコピーを抑える
    if text[start] not in ">＞눈ᓀ/(（ʕ":
        return None
    tail = text[start : start + 100]
    for form in (">ω<", "＞ω＜", "눈_눈", "ᓀ‸ᓂ", "/ᐠ｡ꞈ｡ᐟ\\"):
        if tail.startswith(form):
            return start + len(form)
    for head in ("(:3[", "(¦3[", "('､3[", "('、3["):
        if tail.startswith(head):
            for index, c in enumerate(tail[len(head) : len(head) + 9]):
                if c == "]" and 2 <= index <= 8:
                    return start + len(head) + index + 1
                if c not in "_＿▓":
                    break
    if not tail.startswith(("(", "（", "ʕ")):
        return None
    folded = "".join(_fold(c) for c in tail[:100])
    # 「(ﾉ)ω(ヾ)」では、左右の手と中央の口を一緒に認識する
    if len(folded) > 5 and folded[1] in "ﾉノ/" and folded[2] == ")":
        opening = folded.find("(", 3, 20)
        if (
            opening >= 0
            and opening + 2 < len(folded)
            and _covering_pair(folded[1], folded[opening + 1])
            and folded[opening + 2] == ")"
        ):
            middle = folded[3:opening]
            if (len(middle) == 1 and middle in _STRONG_MOUTHS) or _body(middle, True):
                return start + opening + 3
    # 手つなぎや頬つつきは、接続する両端に顔がある場合だけ認める
    cursor = 1
    last_end: int | None = None
    left_end = next(
        (i for i in range(cursor, min(len(folded), cursor + 21)) if folded[i] in "()人"), None
    )
    if left_end is None or not _social_body(
        folded[cursor:left_end], False, folded[left_end] == "("
    ):
        return None
    delimiter = folded[left_end]
    cursor = left_end + 1
    for _ in range(7):
        missing_left = delimiter == ")"
        if delimiter == ")":
            if folded[cursor : cursor + 2] == "人(":
                cursor += 2
                missing_left = False
            elif cursor < len(folded) and folded[cursor] in "σっ":
                cursor += 1
                if cursor < len(folded) and folded[cursor] in '"”':
                    cursor += 1
                if cursor >= len(folded) or folded[cursor] != ")":
                    break
                cursor += 1
            elif folded[cursor : cursor + 1] == ")":
                cursor += 1
        right_end = next(
            (i for i in range(cursor, min(len(folded), cursor + 21)) if folded[i] in "()人"), None
        )
        if (
            right_end is None
            or folded[right_end] != ")"
            or not _social_body(folded[cursor:right_end], missing_left, False)
        ):
            break
        cursor = right_end + 1
        last_end = start + cursor
        delimiter = ")"
    return last_end


def _social_body(text: str, missing_left: bool, missing_right: bool) -> bool:
    """連結された顔文字の本体を、輪郭と目の省略を考慮して判定する。"""

    if not text or len(text) > 20:
        return False
    if any(c.isascii() and c.isalnum() for c in text):
        return not missing_left and not missing_right and _body(text, False)
    # 輪郭を省いた連結では口を確認し、「つー>ﾟ)))≫彡」の釣り竿を途中まで2つ目の顔にしない
    if (missing_left or missing_right) and not any(c in _STRONG_MOUTHS for c in text):
        return False
    if _body(text, True):
        return True
    clean = text.strip(" *｡")
    if not clean:
        return False
    return (
        missing_left
        and clean[0] in _STRONG_MOUTHS
        and all(_allowed(c) for c in clean[1:])
        and any(c in _EYES for c in clean[1:])
    ) or (
        missing_right
        and clean[-1] in _STRONG_MOUTHS
        and all(_allowed(c) for c in clean[:-1])
        and any(c in _EYES for c in clean[:-1])
    )


def _body(text: str, has_arms: bool) -> bool:
    """
    目と口の組み合わせから顔の本体が成立するかを調べる。
    英数字を含む形は、変数や数式を顔と取り違えないよう、確認できる目と口の組に限定する。

    Args:
        text (str): 判定用に全角記号を半角へ戻した顔の内部
        has_arms (bool): 顔の外に腕があるか

    Returns:
        bool: 顔として成立する場合は True
    """

    if not 2 <= len(text) <= 48:
        return False
    t = text.strip()
    clean = "".join(c for c in t if c not in " *｡´`｀〃ᐢ′‵๑")
    if len(clean) >= 2 and _covering_pair(clean[0], clean[-1]):
        if len(clean) == 2 and not clean[0].isascii():
            return True
        if len(clean) > 3 and not clean[0].isascii() and _body(clean[1:-1], False):
            return True
        if len(clean) == 3 and (
            clean[1] in _STRONG_MOUTHS or (not clean[0].isascii() and clean[1] in "-_ー")
        ):
            return True
    if all(c in "-ー= " for c in t):
        return False
    if "三" in t:
        return t.count("三") <= 2 and all(_body(part, has_arms) for part in t.split("三"))
    if (
        len(t) == 3
        and t[0] in "TtOo0xXQq3"
        and t[0].lower() == t[2].lower()
        and (
            t[1] == "_"
            or (t[0] in "TtOoXxQq" and t[1] in "ω﹏")
            or (t[0] in "TtQq" and t[1] == ".")
        )
    ):
        return True
    if t == "><" or (has_arms and t in ("・・", "･･", "・・。", "。。", "_ _")):
        return True
    eyes = "".join(c for c in t if c not in " *｡")
    if len(eyes) >= 4 and eyes[0] == eyes[-1] == "o":
        eyes = eyes[1:-1]
    if (
        len(eyes) == 2
        and eyes[0] == eyes[1]
        and (eyes[0] in "^˘ˇ" or (has_arms and eyes[0] in "･・"))
    ):
        return True
    if "ε:" in t and all(_arm(c) or c == " " for c in t.replace("ε:", "")):
        return True
    for head in (":3", ":D", ":0", "ε:", "'､3", "'、3"):
        if t.startswith(head) and all(_arm(c) or c == " " for c in t[len(head) :]):
            return True
    if "(" in t or ")" in t:
        return _body(t.replace("(", "").replace(")", ""), has_arms)
    if any(c.isascii() and c.isalnum() for c in text):
        is_hands = all(_allowed(c) or c in _MOUTHS or c in "opcqbdl" for c in t)
        is_o_mouth = (
            "o" in t and all(_allowed(c) or c == "o" for c in t) and any(c in "^＾" for c in t)
        )
        if not is_hands and not is_o_mouth:
            return False
        other_mouth = any(c != "o" and c in _MOUTHS for c in text)
        text = "".join(
            c for c in text if not (is_hands and (c in "pcqbdP" or (c == "o" and other_mouth)))
        )
    if any(c in "≡Ξ三" for c in text):
        chunks = text.replace("Ξ", "≡").replace("三", "≡").split("≡")
        return len(chunks) <= 3 and all(_body(part, has_arms) for part in chunks)
    one_eye = has_arms or any(c in "ヾヽゞゝﾉノつっ⊂⊃" for c in text)
    for index, c in enumerate(text):
        if c in _MOUTHS and all(_allowed(x) for x in text[:index] + text[index + 1 :]):
            before = any(x in _EYES for x in text[:index])
            after = any(x in _EYES for x in text[index + 1 :])
            if (before and after) or (one_eye and c in "ω∀▽∇Ддεзヮ益皿◡" and (before or after)):
                return True
    return False


def _covering_pair(a: str, b: str) -> bool:
    """口や目を覆う左右の手の組かを調べる。"""

    return (a, b) in (
        ("⊃", "⊂"),
        ("つ", "⊂"),
        ("っ", "c"),
        ("p", "q"),
        ("q", "p"),
        ("∩", "∩"),
        ("ฅ", "ฅ"),
    ) or (a in "ﾉノ/" and b in "ﾉノヾ\\")


def _fold(c: str) -> str:
    """全角英数字・記号を形の判定用に半角へ戻す。"""

    return chr(ord(c) - 0xFEE0) if "！" <= c <= "～" else {"−": "-", "　": " "}.get(c, c)


def _arm(c: str) -> bool:
    """腕に使える文字かを調べる。"""

    return c in _ARMS or _fold(c) in _ARMS or c in "/\\"


def _letter(c: str) -> bool:
    """腕と混同しやすい英数字やギリシャ文字かを調べる。"""

    c = _fold(c)
    return (c.isascii() and (c.isalnum() or c == "_")) or (
        "\u0370" <= c <= "\u052f" and c.isalpha()
    )


def _allowed(c: str) -> bool:
    """顔の装飾・目・腕に使える文字かを調べる。"""

    return c in _EYES or _arm(c) or c in _ORNAMENTS or _decoration(c) or c in "🎀💢💦\u200bლᐢˇP"


def _decoration(c: str) -> bool:
    """顔に続く結合文字や異体字セレクタかを調べる。"""

    return unicodedata.category(c).startswith("M") or c in "ﾞﾟ\u200d" or "\ufe00" <= c <= "\ufe0f"


def _extend_objects(
    text: str, core_start: int, core_end: int, start: int, end: int
) -> tuple[int, int]:
    """
    顔の手に接続した団子・飲み物・棒の先の図形を含めた範囲を返す。
    「○○○円」「☆5個」のように数や単位が続く図形は本文として残す。

    Args:
        text (str): Unicode 正規化前の入力文
        core_start (int): 顔の本体の開始
        core_end (int): 顔の本体の終端
        start (int): 腕を含む候補の開始
        end (int): 腕を含む候補の終端

    Returns:
        tuple[int, int]: 持ち物を含む開始と終端
    """

    for is_left in (False, True):
        boundary = start if is_left else end
        existing = text[start:core_start] if is_left else text[core_end:end]
        tail = text[:boundary][::-1] if is_left else text[boundary:]
        count = _object_length(tail, any(c in "つっ⊂⊃ﾉノヾヽoｏcｃ_＿/／＼\\" for c in existing))
        if count > 0:
            if is_left:
                start -= count
            else:
                end += count
    return start, end


def _object_length(text: str, has_hand: bool) -> int:
    """
    手の先の持ち物が成立する場合に、その文字数を返す。

    Args:
        text (str): 腕の外側から順に並べた文字列
        has_hand (bool): 顔の輪郭の外に手があるか

    Returns:
        int: 持ち物の文字数 (成立しない場合は0)
    """

    if has_hand is False and not text.lstrip(" 　").startswith("︻"):
        return 0
    index = 0
    while index < len(text) and index < 2 and text[index] in " 　":
        index += 1
    bars_start = index
    while index < len(text) and index - bars_start < 12 and text[index] in _BARS:
        index += 1
    bars = index - bars_start
    shape_start = index
    if index >= len(text):
        return 0
    if bars and text[index] in "○●◎〇◯￮":
        while index < len(text) and text[index] in "○●◎〇◯￮" and index - shape_start < 6:
            index += 1
        if index - shape_start < 2 or (index < len(text) and text[index] in "○●◎〇◯￮"):
            return 0
        if index < len(text) and text[index] in _BARS:
            index += 1
    elif bars and text[index] in "☆★✿❀":
        while index < len(text) and text[index] in "☆★✿❀" and index - shape_start < 8:
            index += 1
        if index < len(text) and text[index] in "☆★✿❀":
            return 0
        ornament_start = index
        while (
            index < len(text) and index - ornament_start < 8 and text[index] in "。｡.ﾟ゜*＊･・:：✧✨"
        ):
            index += 1
    elif bars and text[index] == ">":
        index += 1
        if index >= len(text) or text[index] not in "ﾟ゜":
            return 0
        index += 1
        gills_start = index
        while index < len(text) and index - gills_start < 8 and text[index] == ")":
            index += 1
        if (
            index == gills_start
            or index + 1 >= len(text)
            or text[index] not in "≫>"
            or text[index + 1] != "彡"
        ):
            return 0
        index += 2
        tail_end = last_wave = index
        while index < len(text) and index - tail_end < 8 and text[index] in "~ 　":
            if text[index] == "~":
                last_wave = index + 1
            index += 1
        index = last_wave
    elif bars and text.startswith("[二二]", index):
        index += 4
    elif bars and text.startswith("{}", index):
        index += 2
        count = 1
        while text.startswith("@{}", index) and count < 4:
            index += 3
            count += 1
        if count < 2:
            return 0
        if index < len(text) and text[index] in _BARS:
            index += 1
    elif not bars and text[index] == "︻":
        index += 1
        if index < len(text) and text[index] == "┻":
            index += 1
        if index >= len(text) or text[index] not in "┳デ":
            return 0
        index += 1
        barrel_start = index
        while index < len(text) and index - barrel_start < 8 and text[index] in "═━":
            index += 1
        if index == barrel_start or index >= len(text) or text[index] != "一":
            return 0
        index += 1
    elif (
        not bars
        and index + 1 < len(text)
        and (
            (text[index] in "￭■□" and text[index + 1] in "PDCＰＤＣ")
            or (text[index] in "PDCＰＤＣ" and text[index + 1] in "￭■□")
        )
    ):
        index += 2
        steam_start = index
        while index < len(text) and index - steam_start < 4 and text[index] in '~～〜”"':
            index += 1
    elif not bars and text[index] in "旦且🍡☕🍵":
        cup = text[index]
        index += 1
        if index < len(text) and text[index] in "\ufe0e\ufe0f":
            index += 1
        steam_start = index
        while index < len(text) and index - steam_start < 4 and text[index] in '~～〜”"':
            index += 1
        if cup in "旦且" and index == steam_start:
            return 0
    else:
        return 0
    if index < len(text) and text[index] in "\ufe0e\ufe0f":
        index += 1
    if index < len(text) and _decoration(text[index]):
        return 0
    if index < len(text) and "\U0001f3fb" <= text[index] <= "\U0001f3ff":
        return 0
    following = text[index:].lstrip(" 　")
    if following and (
        following[0].isnumeric()
        or (following[0].isascii() and following[0].isalnum())
        or following[0] in "円件個点番号年月日時分秒割％%式項口人台本桁"
    ):
        return 0
    return index
