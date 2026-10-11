"""顔文字の無読化と、公開 API 間の読み・音素・文字位置の対応を確かめる。"""

from collections.abc import Iterator
from pathlib import Path

import pytest

import pyopenjtalk
import pyopenjtalk.tsqyomi as tsqyomi
from pyopenjtalk.types import NormalizeMode
from pyopenjtalk.utils import normalize_text


_FACES = [
    "(´・ω・`)ノ",
    "(^_^)",
    "(-'ω')",
    "(-ノ_ノ)",
    "(T_T)",
    "(0_0)",
    "(⊃-⊂)",
    "(ﾉ)ω(ヾ)",
    "(*>ω<)ω<*)",
    "(^ω^)人(^ω^)",
    "(^o^)人(^o^)",
    "(T_T)人(T_T)",
    "(0_0)人(0_0)",
    "(^ω^人^ω^)",
    "(^ω^)σ)ω^)",
    "(^ω^)っ)ω^)",
    '(^ω^)σ")ω^)',
    "(^ω^)σ”)ω^)",
    "(ﾉｼ｀･ω･)",
    "(^ω^)ω^)ω^)",
    "(๑′ฅฅ‵๑)",
    "(´∩′•ω•∩)",
    "(^_^)つー○○○",
    "(^_^)つ☕️",
    "(^_^)つ︻┳═一",
    "(^_^)つー[二二]",
    "(^_^)つー{}@{}",
    "(^_^)つ■D",
    "(^_^)つー>ﾟ)))≫彡~~",
]


@pytest.mark.parametrize("mode", ["None", "NFC", "NFKC"])
@pytest.mark.parametrize("face", _FACES)
def test_face_is_one_silent_element(
    face: str, mode: NormalizeMode, capfd: pytest.CaptureFixture[str]
) -> None:
    """
    「(´・ω・`)ノ」などの顔文字は、正規化後の表層を残した1要素として返し、読みと音素を空にすることを確認する。
    顔の「ω」や腕の「ノ」を読み上げたり、顔の内部の記号から休止を作ったりせず、通常・詳細フロントエンドと全出力 API で同じ範囲を扱う。
    """

    features, morphs = pyopenjtalk.run_frontend_detailed(face, normalize_mode=mode)
    assert features == pyopenjtalk.run_frontend(face, normalize_mode=mode)
    assert len(features) == len(morphs) == 1
    assert features[0]["pos_group1"] == "顔文字"
    assert features[0]["read"] == features[0]["pron"] == ""
    assert features[0]["mora_size"] == features[0]["acc"] == 0
    assert pyopenjtalk.g2p(face, kana=True, normalize_mode=mode) == ""
    assert pyopenjtalk.g2p(face, normalize_mode=mode) == ""
    assert pyopenjtalk.extract_fullcontext(face, normalize_mode=mode) == []
    assert pyopenjtalk.g2p_prosody(face, normalize_mode=mode) == []
    mapping = pyopenjtalk.g2p_mapping(face, normalize_mode=mode)
    assert len(mapping) == 1
    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    assert mapping[0]["surface"] == jtalk.normalize_for_mecab(normalize_text(face, mode))
    assert mapping[0]["char_span"] == (0, len(face))
    assert mapping[0]["phonemes"] == []
    assert mapping[0]["read"] == mapping[0]["pron"] == ""
    assert mapping[0]["is_ignored"] is True
    assert mapping[0]["is_unknown"] is False
    assert capfd.readouterr().err == ""


@pytest.mark.parametrize(
    "text",
    [
        "コ(^_^)ーヒー",
        "コー(^_^)ーヒー",
        "前(^_^)ー後",
        "前(^_^)ーー後",
        "(^_^)こんにちは",
        "こんにちは(^_^)世界",
        "1(^_^)2人",
        "です(^_^)！？",
        "前(^_^)。後",
        "前(^_^) (T_T)後",
        "三(^_^)軒",
        "一冊(^_^)配った",
    ],
)
def test_surrounding_speech_agrees_across_apis(text: str) -> None:
    """
    「コ(^_^)ーヒー」のように顔の後ろに長音がある文でも、顔を除いた音素が G2P・マッピング・ラベルで一致することを確認する。
    顔の空の要素へ前の語の音素を割り当てず、元の入力における表層と文字位置を残す。
    """

    assert pyopenjtalk.run_frontend(text) == pyopenjtalk.run_frontend_detailed(text)[0]
    mapping = pyopenjtalk.g2p_mapping(text)
    phones = [phone for word in mapping for phone in word["phonemes"] if phone not in ("sp", "pau")]
    assert phones == [phone for phone in pyopenjtalk.g2p(text, join=False) if phone != "pau"]
    labels = pyopenjtalk.extract_fullcontext(text)
    assert phones == [
        phone
        for label in labels[1:-1]
        if (phone := label.split("-", 1)[1].split("+", 1)[0]) != "pau"
    ]
    end = 0
    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    for word in mapping:
        start, next_end = word["char_span"]
        assert start >= end
        end = next_end
        if word["pos_group1"] == "顔文字":
            assert word["surface"] == jtalk.normalize_for_mecab(text[start:next_end])
            assert word["phonemes"] == []
    if text == "コ(^_^)ーヒー":
        assert phones == ["k", "o", "o", "h", "i", "i"]


@pytest.mark.parametrize("faces", ["(^^) (^^)", "(^^)　(^^)", "(^^)  (^^)"])
def test_spaced_faces_keep_following_sentence(faces: str) -> None:
    """
    「おはよう(^^) (^^)今日は晴れです。」で、空白を挟んだ顔文字だけを無音にし、後続の「今日は晴れです」を読むことを確認する。
    MeCab が顔文字の記号と空白をまとめても、本文の形態素を顔文字の範囲へ割り当てず、各マッピングと韻律 API に本文を残す。
    """

    text = f"おはよう{faces}今日は晴れです。"
    assert pyopenjtalk.g2p(text, kana=True) == "オハヨーキョーワハレデス。"
    assert pyopenjtalk.g2p(text) == pyopenjtalk.g2p("おはよう今日は晴れです。")
    assert pyopenjtalk.run_frontend(text) == pyopenjtalk.run_frontend_detailed(text)[0]
    for mapping in (pyopenjtalk.g2p_mapping(text), pyopenjtalk.g2p_mapping_prosody(text)):
        assert "".join(text[slice(*word["char_span"])] for word in mapping) == text
        assert len([word for word in mapping if word["pos_group1"] == "顔文字"]) == 2
        assert any(word["surface"] == "今日" for word in mapping)
    assert pyopenjtalk.g2p_prosody(text)


@pytest.fixture(scope="module")
def tsqyomi_model() -> Iterator[None]:
    """顔文字の数詞への影響を実モデルの候補解析でも確認するため、CPU のモデルをロードする。"""

    pytest.importorskip("onnxruntime")
    is_model_already_loaded = tsqyomi.is_model_loaded()
    if is_model_already_loaded is False:
        tsqyomi.load_model(["CPUExecutionProvider"])
    yield
    if is_model_already_loaded is False and tsqyomi.is_model_loaded():
        tsqyomi.unload_model()


@pytest.mark.parametrize("use_tsqyomi", [False, True])
@pytest.mark.parametrize("face", ["m(_ _)m", "m(_ _ )m", "ｍ（＿　＿）ｍ"])
def test_face_arm_does_not_change_preceding_number_accent(
    face: str, use_tsqyomi: bool, tsqyomi_model: None
) -> None:
    """
    「答えは10 m(_ _)m。」では、顔文字の腕の「m」をメートルの助数詞にせず、「答えは10。」と同じ「ジュ＼ー」の核を保つことを確認する。
    通常解析と候補選択の両方で数詞処理の前に腕を除外し、無音化後の読み・ラベル・韻律に助数詞の影響を残さない。
    """

    text = f"答えは10 {face}。"
    plain = "答えは10。"
    for features in (
        pyopenjtalk.run_frontend(text, use_tsqyomi=use_tsqyomi),
        pyopenjtalk.run_frontend_detailed(text, use_tsqyomi=use_tsqyomi)[0],
    ):
        ten = next(feature for feature in features if feature["string"] == "十")
        assert (ten["pron"], ten["acc"], ten["mora_size"]) == ("ジュー", 1, 2)
    assert pyopenjtalk.g2p(text, use_tsqyomi=use_tsqyomi) == pyopenjtalk.g2p(
        plain, use_tsqyomi=use_tsqyomi
    )
    assert pyopenjtalk.extract_fullcontext(
        text, use_tsqyomi=use_tsqyomi
    ) == pyopenjtalk.extract_fullcontext(plain, use_tsqyomi=use_tsqyomi)
    assert [
        token for token in pyopenjtalk.g2p_prosody(text, use_tsqyomi=use_tsqyomi) if token != "sp"
    ] == pyopenjtalk.g2p_prosody(plain, use_tsqyomi=use_tsqyomi)
    assert pyopenjtalk.g2p_mapping_prosody(text, use_tsqyomi=use_tsqyomi)


@pytest.mark.parametrize("prefix", ["1 10日", "1 24日", "1,000ー"])
def test_number_before_face_keeps_day_and_following_sentence(prefix: str) -> None:
    """
    「1 24日(^^)今日は晴れです。」では、数詞と日付を読み、顔文字だけを無音にして後続の本文を残すことを確認する。
    数字を消費した文字位置がずれると「四日」が顔文字の範囲へ入り込むため、数字・日付・長音の後でも通常の読みを保持する。
    """

    text = prefix + "(^^)今日は晴れです。"
    assert pyopenjtalk.g2p(text) == pyopenjtalk.g2p(prefix + "今日は晴れです。")
    assert pyopenjtalk.run_frontend(text) == pyopenjtalk.run_frontend_detailed(text)[0]
    for mapping in (pyopenjtalk.g2p_mapping(text), pyopenjtalk.g2p_mapping_prosody(text)):
        assert "".join(text[slice(*word["char_span"])] for word in mapping) == text
        faces = [word for word in mapping if word["pos_group1"] == "顔文字"]
        assert len(faces) == 1
        assert faces[0]["char_span"] == (len(prefix), len(prefix) + 4)
        assert any(word["surface"] == "今日" for word in mapping)


@pytest.mark.parametrize(
    "text", ["(p-q)", "(a+b)", "(2026年)", "(ﾉ)日本語(ヾ)", "(pH)", "(NaCl)", "一(^_^)々"]
)
def test_non_faces_keep_original_analysis(text: str) -> None:
    """
    「(p-q)」「(NaCl)」などの数式・語句や、顔の境界をまたぐ「一(^_^)々」は無読化の対象から外れることを確認する。
    顔の文字集合に部分的に似ていても、語として解析された本文を消さない。
    """

    assert all(f["pos_group1"] != "顔文字" for f in pyopenjtalk.run_frontend(text))


@pytest.mark.parametrize(
    "prefix,face", [("仲良ｼ", "(^_^)"), ("仲良ｼ", "(^o^)人(^o^)"), ("仲良ｼ", "(^ω^)っ)ω^)")]
)
def test_shi_before_face_is_part_of_word(prefix: str, face: str) -> None:
    """
    「仲良ｼ(^_^)」の半角の「ｼ」は本文の語尾として読み、顔の腕に含めないことを確認する。
    正規化でカタカナの形が変わっても、顔の開始位置は元の語尾の後に置く。
    """

    mapping = pyopenjtalk.g2p_mapping(prefix + face)
    faces = [w for w in mapping if w["pos_group1"] == "顔文字"]
    assert len(faces) == 1
    assert faces[0]["char_span"] == (len(prefix), len(prefix + face))
    assert any("シ" in word["pron"] for word in mapping)


@pytest.mark.parametrize(
    "text", ["pc(^_^)です", "(^_^)ノート", "(^_^)つー○○○円", "(^_^)つー☆5個", "(^ω^)3人(^ω^)"]
)
def test_words_next_to_arms_are_preserved(text: str) -> None:
    """
    「(^_^)ノート」の「ノート」や「(^_^)つー○○○円」の数量を本文として残すことを確認する。
    腕や持ち物に見える文字が既知語・数詞の一部なら、無読化する範囲を顔だけへ狭める。
    """

    mapping = pyopenjtalk.g2p_mapping(text)
    faces = [w for w in mapping if w["pos_group1"] == "顔文字"]
    assert len(faces) >= 1
    assert all(
        not any(c in face["surface"] for c in ("pc", "ノート", "円", "5", "3人")) for face in faces
    )


@pytest.mark.parametrize("mode", ["None", "NFC", "NFKC"])
@pytest.mark.parametrize(
    "text",
    [
        "か\u3099(^_^)後",
        "㈱(^_^)後",
        "ｶﾞ(^_^)後",
        "\t前(^_^)\n後",
        "前(^_^)\u0301後",
        "「_(┐「ε:)_」",
    ],
)
def test_normalization_keeps_caller_positions(text: str, mode: NormalizeMode) -> None:
    """
    結合濁点、囲み文字、半角カナを顔文字の前に置いても、マッピングの文字位置が元の入力を指すことを確認する。
    顔に続く結合文字は顔に含め、本文の対応する引用符は独立した要素として残す。
    """

    mapping = pyopenjtalk.g2p_mapping(text, normalize_mode=mode)
    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    assert "".join(text[slice(*w["char_span"])] for w in mapping) == text
    for word in mapping:
        assert word["surface"] == jtalk.normalize_for_mecab(
            normalize_text(text[slice(*word["char_span"])], mode)
        )


def test_vanilla_keeps_face_reading() -> None:
    """
    「(´・ω・`)ノ」は既定では無読化し、use_vanilla=True では従来どおり「ω」や腕の「ノ」を読むことを確認する。
    顔文字の無読化は独自の後処理なので、既存の後処理を無効にする引数で効果を比較できる。
    """

    assert pyopenjtalk.g2p("(´・ω・`)ノ") == ""
    assert "o m e g a" in pyopenjtalk.g2p("(´・ω・`)ノ", use_vanilla=True)


@pytest.mark.parametrize(
    "surface,text,reading",
    [
        ("ω", "前(-'ω')後", "オメガ"),
        ("（＾＿＾）", "前(^_^)後", "カオ"),
        ("前（＾＿＾）後", "前(^_^)後", "ゼンタイ"),
        ("ノ", "前(^_^)ノ後", "ウデ"),
        ("人", "(^ω^)人(^ω^)", "ヒト"),
        ("人", "(^o^)人(^o^)", "ヒト"),
        ("ｏ", "(^o^)人(^o^)", "オー"),
        ("っ", "(^ω^)っ)ω^)", "テ"),
        ("σ", '(^ω^)σ")ω^)', "シグマ"),
        ("☆", "(^_^)つー☆", "ホシ"),
    ],
)
@pytest.mark.parametrize("mode", ["None", "NFC", "NFKC"])
@pytest.mark.parametrize("protect", [False, True])
def test_registered_face_parts_are_preserved(
    tmp_path: Path, surface: str, text: str, reading: str, mode: NormalizeMode, protect: bool
) -> None:
    """
    顔全体や「ω」「ノ」「人」をユーザー辞書に登録した場合は、登録語を顔として消さないことを確認する。
    読み保護を指定しない「人」には従来の文脈補正を適用し、それ以外は登録された読みを保つ。
    """

    csv = tmp_path / "user.csv"
    dic = tmp_path / "user.dic"
    csv.write_text(
        f"{surface},1345,1345,-20000,名詞,一般,*,*,*,*,{surface},{reading},{reading},0/3,C3\n",
        encoding="utf-8",
    )
    pyopenjtalk.mecab_dict_index(str(csv), str(dic))
    jtalk = pyopenjtalk.OpenJTalk(
        dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR,
        userdic=str(dic).encode(),
        userdic_reading_protection=[protect],
    )
    mapping = pyopenjtalk.g2p_mapping(text, jtalk=jtalk, normalize_mode=mode)
    assert any(
        word["read"] == ("ニン" if surface == "人" and not protect else reading) for word in mapping
    )
    registered = [m for m in jtalk.run_mecab_detailed(text)[1] if m["dictionary_index"] == 1]
    for face in (w for w in mapping if w["pos_group1"] == "顔文字"):
        assert all(
            m["char_span"][1] <= face["char_span"][0] or face["char_span"][1] <= m["char_span"][0]
            for m in registered
        )


@pytest.mark.parametrize(
    "text", ["三(^_^)軒", "一冊(^_^)配った", "コ(^_^)ーヒー", "(^_^)発車します"]
)
@pytest.mark.parametrize("read_as_pron", [False, True])
def test_pronunciation_options_keep_face_silent(text: str, read_as_pron: bool) -> None:
    """
    「三(^_^)軒」や「コ(^_^)ーヒー」は、発音を読みへ戻す設定でも顔を無音のままにすることを確認する。
    長音と四つ仮名を復元した音素がマッピングにも同じ順で入り、空の発音を書き戻す処理が顔へ読みを補わないようにする。
    """

    phones = pyopenjtalk.g2p(
        text,
        join=False,
        use_read_as_pron=read_as_pron,
        revert_long_vowels=True,
        revert_yotsugana=True,
    )
    mapping = pyopenjtalk.g2p_mapping(
        text, use_read_as_pron=read_as_pron, revert_long_vowels=True, revert_yotsugana=True
    )
    assert phones == [p for w in mapping for p in w["phonemes"] if p != "sp"]
    assert all(
        w["read"] == w["pron"] == "" and w["phonemes"] == []
        for w in mapping
        if w["pos_group1"] == "顔文字"
    )
