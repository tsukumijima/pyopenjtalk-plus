"""
番号の桁、アクセント句、休止が公開 API へ伝わることを確かめる。
"""

from pathlib import Path

import pytest

import pyopenjtalk
from pyopenjtalk.types import UserDictionaryEntry


def _g2p(text: str, *, kana: bool = False) -> str:
    result = pyopenjtalk.g2p(text, kana=kana)
    assert isinstance(result, str)
    return result


def test_phone_pair_accents() -> None:
    assert pyopenjtalk.g2p("070-3224-5679") == (
        "z e r o n a n a z e r o pau s a N n i i n i i y o N pau g o o r o k u n a n a ky u u"
    )
    assert "".join(pyopenjtalk.g2p_prosody("070-3224-5679")) == (
        "^ze[rona]na#ze]ro_sa[Nni]i#ni[iyo]N_go[oro]ku#na[nakyu]u$"
    )


@pytest.mark.parametrize(
    "text,pause_count",
    [
        ("07032245679", 0),
        ("0120123456", 0),
        ("0312345678", 0),
        ("03-1234-5678", 2),
        ("0467-12-3456", 2),
        ("04992-2-5555", 2),
        ("070ー3224ー5679", 2),
        ("市外局番213の486ー2435", 1),
        ("〇七〇-三二二四-五六七九", 2),
        ("〇七〇-〇〇二四-五六七九", 2),
        ("03(1234)5678", 2),
    ],
)
def test_phone_groups_keep_written_pauses(text: str, pause_count: int) -> None:
    assert _g2p(text).split().count("pau") == pause_count
    features = pyopenjtalk.run_frontend(text)
    assert all(feature["mora_size"] == 2 for feature in features if feature["pos_group1"] == "数")


def test_odd_phone_group_three_stays_separate() -> None:
    features = pyopenjtalk.run_frontend("0120-123-456")
    digits = [feature for feature in features if feature["pos_group1"] == "数"]
    assert [
        (feature["pron"], feature["chain_flag"], feature["acc"]) for feature in digits[4:7]
    ] == [
        ("イチ", 0, 3),
        ("ニー", 1, 1),
        ("サン", 0, 0),
    ]


def test_phone_context_does_not_change_quantity() -> None:
    assert _g2p("電話回線は3本", kana=True).endswith("サンボン")
    assert _g2p("電話番号110", kana=True).endswith("イチイチゼロ")


@pytest.mark.parametrize(
    "text,reading",
    [
        ("電話で100と200を足す", "デンワデヒャクトニヒャクヲタス"),
        ("電話料金は1.5円", "デンワリョーキンワイッテンゴエン"),
    ],
)
def test_phone_context_ends_before_quantities(text: str, reading: str) -> None:
    assert _g2p(text, kana=True) == reading
    assert "pau" not in _g2p(text).split()


@pytest.mark.parametrize(
    "text,reading",
    [
        ("〒104・8011", "イチゼロヨンハチゼロイチイチ"),
        ("電話番号03・3355・1881", "ゼロサンサンサンゴーゴーイチハチハチイチ"),
        ("受け付け電話番号は、03・3355・1881", "ゼロサンサンサンゴーゴーイチハチハチイチ"),
        ("電話兼ファクス03・3801・3552", "ゼロサンサンハチゼロイチサンゴーゴーニー"),
    ],
)
def test_explicit_number_context_keeps_middle_dot_groups(text: str, reading: str) -> None:
    assert _g2p(text, kana=True).replace("・", "").endswith(reading)
    assert _g2p(text).split().count("pau") == text.count("・") + text.count("、")


@pytest.mark.parametrize("text", ["070ー3224ー5679", "市外局番213の486ー2435", "〒123ー4567"])
def test_number_separator_pause_matches_mapping(text: str) -> None:
    mapping = pyopenjtalk.g2p_mapping(text)
    separators = [item for item in mapping if item["surface"] == "ー"]
    assert separators
    assert all(item["phonemes"] == ["pau"] for item in separators)
    assert [phone for item in mapping for phone in item["phonemes"]] == _g2p(text).split()
    prosody = pyopenjtalk.g2p_prosody(text)
    assert "unk" not in prosody
    assert prosody.count("_") == len(separators)
    # 「070ー3224」の「ー」は休止にしても、元の表層と文字位置を保つ
    for item in separators:
        start, end = item["char_span"]
        assert text[start:end] == "ー"


def test_freephone_prefix_precedes_mobile_prefix() -> None:
    joined = "".join(pyopenjtalk.g2p_prosody("08001234567"))
    separated = "".join(pyopenjtalk.g2p_prosody("0800-123-4567"))
    assert joined == separated.replace("_", "#")
    assert "pau" not in _g2p("08001234567").split()


@pytest.mark.parametrize("text", ["〒123-4567", "123-4567", "〒一二三-四五六七"])
def test_postal_number_pair_accents(text: str) -> None:
    assert _g2p(text).split().count("pau") == 1
    assert "".join(pyopenjtalk.g2p_prosody(text)).endswith("i[chini]i#sa[N_yo[Ngo]o#ro[kuna]na$")


def test_postal_number_without_separator() -> None:
    assert _g2p("郵便番号1234567", kana=True).endswith("イチニーサンヨンゴーロクナナ")
    assert "pau" not in _g2p("郵便番号1234567").split()


@pytest.mark.parametrize("counter,reading", [("円", "エン"), ("個", "コ")])
def test_counter_range_is_not_postal_number(counter: str, reading: str) -> None:
    text = f"価格は123-4567{counter}"
    assert _g2p(text, kana=True).endswith("ヨンセンゴヒャクロクジューナナ" + reading)


@pytest.mark.parametrize("counter,reading", [("円", "ロクエン"), ("個", "ロッコ")])
def test_identifier_context_does_not_extend_to_quantity(counter: str, reading: str) -> None:
    text = f"型番3248の商品を9876{counter}で購入"
    assert "サンニーヨンハチ" in _g2p(text, kana=True)
    assert "キューセンハッピャクナナジュー" + reading in _g2p(text, kana=True)


def test_vehicle_hyphen_keeps_existing_reading() -> None:
    assert pyopenjtalk.g2p("モハ205-3248", kana=True) == "モハニヒャクゴ−サンニーヨンハチ"
    assert "ni[hyakugo" in "".join(pyopenjtalk.g2p_prosody("モハ205-3248"))
    two = next(
        feature for feature in pyopenjtalk.run_frontend("モハ205-3248") if feature["string"] == "二"
    )
    assert two["acc"] == 4


@pytest.mark.parametrize(
    "text,reading,prosody",
    [
        ("802号室", "ハチマルニゴーシツ", "^ha[chima]ru#ni[go]oshItsu$"),
        ("国道409号線", "コクドーヨンマルキューゴーセン", "^ko[kudoo#yo[Nma]ru#kyu[ugooseN$"),
        ("12号室", "ジューニゴーシツ", "^ju[unigo]oshItsu$"),
        ("十二号室", "ジューニゴーシツ", "^ju[unigo]oshItsu$"),
        ("一二号室", "イチニゴーシツ", "^i[chinigo]oshItsu$"),
    ],
)
def test_room_and_road_numbers(text: str, reading: str, prosody: str) -> None:
    assert pyopenjtalk.g2p(text, kana=True) == reading
    assert "".join(pyopenjtalk.g2p_prosody(text)) == prosody


def test_explicit_digits_keep_two_zeroes_and_counter_reading() -> None:
    assert pyopenjtalk.g2p("一〇〇一号室", kana=True) == "イチマルマルイチゴーシツ"
    assert pyopenjtalk.g2p("16号車", kana=True) == "ジューロクゴーシャ"
    assert pyopenjtalk.g2p("109番地", kana=True) == "ヒャクキューバンチ"
    assert pyopenjtalk.g2p("一九九五年", kana=True) == "センキューヒャクキュージューゴネン"
    assert pyopenjtalk.g2p("〇円です。", kana=True) == "レーエンデス。"


@pytest.mark.parametrize(
    "written,positional",
    [
        ("二〇万円", "20万円"),
        ("二〇億円", "20億円"),
        ("二〇千個", "20千個"),
        ("一二〇万円", "120万円"),
        ("二〇〇〇円", "2000円"),
        ("三〇〇〇円", "3000円"),
    ],
)
def test_written_digit_sequence_with_place_unit_is_quantity(written: str, positional: str) -> None:
    assert pyopenjtalk.g2p(written) == pyopenjtalk.g2p(positional)
    assert pyopenjtalk.g2p_prosody(written) == pyopenjtalk.g2p_prosody(positional)


@pytest.mark.parametrize(
    "text,reading,prosody",
    [
        ("一二千個", "イチニーセンコ", "^i[chini]i#se]Nko$"),
        ("二三万人", "ニーサンマンニン", "^ni[isa]N#ma[NniN$"),
        ("一二百円", "イチニーヒャクエン", "^i[chini]i#hya[kueN$"),
    ],
)
def test_written_approximate_quantity_keeps_digit_reading(
    text: str, reading: str, prosody: str
) -> None:
    assert _g2p(text, kana=True) == reading
    assert "".join(pyopenjtalk.g2p_prosody(text)) == prosody


@pytest.mark.parametrize(
    "text,reading,prosody",
    [
        ("1001号機", "センイチゴーキ", "^se[Nichigo]oki$"),
        ("1001号室", "センイチゴーシツ", "^se[Nichigo]oshItsu$"),
        ("1032号機", "センサンジューニゴーキ", "^se[Nsa]Njuu#ni[go]oki$"),
        ("1021", "センニジューイチ", "^se[Nni]juu#i[chi$"),
        ("2139号機", "ニセンヒャクサンジューキューゴーキ", "^ni[se]N#hya[kUsa]Njuu#kyu[ugo]oki$"),
        ("2101号機", "ニセンヒャクイチゴーキ", "^ni[se]N#hya[kuichigo]oki$"),
        ("1000号機", "センゴーキ", "^se[Ngo]oki$"),
        ("9876号機", "キューハチナナロクゴーキ", "^kyu[uha]chi#na[narokugo]oki$"),
    ],
)
def test_compact_positional_identifiers(text: str, reading: str, prosody: str) -> None:
    assert pyopenjtalk.g2p(text, kana=True) == reading
    assert "".join(pyopenjtalk.g2p_prosody(text)) == prosody


@pytest.mark.parametrize(
    "text,reading",
    [
        (
            "控えの番号は、5823901746283915です",
            "ヒカエノバンゴーワ、ゴーハチニーサンキューゼロイチナナヨンロクニーハチサンキューイチゴーデス",
        ),
        ("1234567890123", "イチニーサンヨンゴーロクナナハチキューゼロイチニーサン"),
        (
            "2918374650192837465012を入力",
            "ニーキューイチハチサンナナヨンロクゴーゼロイチキューニーハチサンナナヨンロクゴーゼロイチニーヲニューリョク",
        ),
        (
            "2918374650192837465012通り",
            "ニーキューイチハチサンナナヨンロクゴーゼロイチキューニーハチサンナナヨンロクゴーゼロイチニートーリ",
        ),
    ],
)
def test_long_numbers_use_digit_reading(text: str, reading: str) -> None:
    """
    「兆」以上の位が要る13桁以上の数字列は、位取りすると長くまどろっこしくなるので、カード番号のように1桁ずつ読むことを確認する。
    区切りのない13桁以上の算用数字は数量として読まれることがまれで、位取りの「ヨンセンキューヒャクハチジューナナチョー…」は聞き取りにくい。
    17桁以上は「京」より上の位を数量でもふつう口にしないので、位取りの方が短い場合を除いて、助数詞が付いていても文脈によらず桁読みにする。
    """

    assert pyopenjtalk.g2p(text, kana=True) == reading


@pytest.mark.parametrize(
    "text,grouped",
    [
        ("1234567890123円", "1,234,567,890,123円"),
        ("1234567890123以上", "1,234,567,890,123以上"),
        ("約1234567890123", "約1,234,567,890,123"),
        ("1234567890123.45", "1,234,567,890,123.45"),
        ("1234567890123456通り", "1,234,567,890,123,456通り"),
        ("差額は-1234567890123円", "差額は-1,234,567,890,123円"),
    ],
)
def test_long_numbers_in_quantity_context_keep_positional_reading(text: str, grouped: str) -> None:
    """
    13〜16桁の数字列でも、助数詞・「約」・「以上」・小数点が付いて数量と分かる場合は、桁区切りのカンマを付けた場合と同じく位取りで読むことを確認する。
    「差額は-1234567890123円」のように前に負号があっても、直後の助数詞で数量と分かるので位取りで読む。
    桁読みにするのは、数量の手がかりがない番号のような数字列と、位取りの方が短い場合を除いた17桁以上の数字列に限る。
    """

    assert _g2p(text, kana=True) == _g2p(grouped, kana=True)
    assert "イッチョー" in _g2p(text, kana=True) or "センニヒャクサンジューヨンチョー" in _g2p(
        text, kana=True
    )


@pytest.mark.parametrize(
    "text,reading",
    [
        ("1000000000000円", "イッチョーエン"),
        ("1500000000000円", "イッチョーゴセンオクエン"),
        (
            "123456789012円",
            "センニヒャクサンジューヨンオクゴセンロッピャクナナジューハチマンキューセンジューニエン",
        ),
    ],
)
def test_long_round_numbers_keep_positional_reading(text: str, reading: str) -> None:
    """
    13桁以上でも「1兆円」のように位取りの方が短い数と、12桁以下の数は、これまでどおり位取りで読むことを確認する。
    桁読みにするのは、位取りのモーラ数が桁読みの2倍に3を足した数を超える場合に限る。
    """

    assert pyopenjtalk.g2p(text, kana=True) == reading


@pytest.mark.parametrize(
    "text,reading,prosody",
    [
        ("01号室", "ゼロイチゴーシツ", "^ze[roichigo]oshItsu$"),
        ("02番", "ゼロニバン", "^ze[roni]baN$"),
        ("〇一号室", "ゼロイチゴーシツ", "^ze[roichigo]oshItsu$"),
    ],
)
def test_leading_zero_is_pronounced(text: str, reading: str, prosody: str) -> None:
    assert pyopenjtalk.g2p(text, kana=True) == reading
    assert "".join(pyopenjtalk.g2p_prosody(text)) == prosody


@pytest.mark.parametrize(
    "text,reading",
    [("03本", "ゼロサンボン"), ("01個", "ゼロイッコ"), ("04人", "ゼロヨニン")],
)
def test_zero_padded_quantity_keeps_counter_pronunciation(text: str, reading: str) -> None:
    assert _g2p(text, kana=True) == reading
    assert "pau" not in _g2p(text).split()
    features = pyopenjtalk.run_frontend(text)
    # 「03本」の「ゼロ」が独立した句になっても、NHK アクセント辞典の「ゼ＼ロ」の核を保つ
    assert features[0]["acc"] == 1


@pytest.mark.parametrize("text", ["YDT-03型", "〇七三〇時", "010"])
def test_zero_padded_names_keep_digit_groups(text: str) -> None:
    features = pyopenjtalk.run_frontend(text)
    digits = [feature for feature in features if feature["pos_group1"] == "数"]
    assert [feature["chain_flag"] for feature in digits] == [
        index % 2 for index in range(len(digits))
    ]


def test_decimal_zero_keeps_positional_reading() -> None:
    assert pyopenjtalk.g2p("0.02ミリ", kana=True) == "レーテンゼロニーミリ"


@pytest.mark.parametrize("text", ["一〇・五", "10.5"])
def test_decimal_integer_part_is_not_identifier(text: str) -> None:
    assert _g2p(text, kana=True) == "ジュッテンゴ"
    assert "pau" not in _g2p(text).split()


@pytest.mark.parametrize("text", ["JAL3便", "ANA3便", "飛行機の3便", "3便に搭乗する"])
def test_confirmed_flight_number_is_flat(text: str) -> None:
    features = pyopenjtalk.run_frontend(text)
    three = next(feature for feature in features if feature["string"] == "三")
    assert three["acc"] == 0
    assert _g2p(text, kana=True).find("サンビン") >= 0


def test_multiphase_flight_name_has_flat_final_phrase() -> None:
    assert "".join(pyopenjtalk.g2p_prosody("飛行機の226便")).endswith("#ro[kubiN$")
    assert "".join(pyopenjtalk.g2p_prosody("荷物を3便に分ける")).find("#sa]NbiN") >= 0
    assert "#sa]NbiN" in "".join(pyopenjtalk.g2p_prosody("JALは一日3便運航する"))


def test_numeric_space_separates_values_without_pause() -> None:
    text = "EF65 1032号機"
    assert _g2p(text, kana=True) == "イーエフロクジューゴセンサンジューニゴーキ"
    assert "pau" not in _g2p(text).split()
    prosody = "".join(pyopenjtalk.g2p_prosody(text))
    assert "ro[kujuugo" in prosody
    assert "se[Nsa]Njuu#ni[go]oki" in prosody
    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    features, morphs = jtalk.run_mecab_detailed(text)
    assert features == jtalk.run_mecab(text)
    spaces = [morph for morph in morphs if morph["surface"].isspace()]
    assert spaces and all(morph["is_ignored"] is True for morph in spaces)
    mapping = pyopenjtalk.g2p_mapping(text)
    assert "pau" not in [phoneme for item in mapping for phoneme in item["phonemes"]]
    mapped_spaces = [item for item in mapping if item["surface"].isspace()]
    assert len(mapped_spaces) == 1
    assert mapped_spaces[0]["phonemes"] == ["sp"]
    assert mapped_spaces[0]["char_span"] == (4, 5)


def test_phone_spaces_preserve_digits_without_pause() -> None:
    assert _g2p("電話 03 1234 5678", kana=True) == "デンワゼロサンイチニーサンヨンゴーロクナナハチ"
    assert "pau" not in _g2p("電話 03 1234 5678").split()


def test_postal_spaces_preserve_digits_without_pause() -> None:
    assert _g2p("〒460 8511", kana=True) == "〒ヨンロクゼロハチゴーイチイチ"
    assert "pau" not in _g2p("〒460 8511").split()


def test_nonzero_area_code_phone_groups() -> None:
    assert _g2p("212-836-1725", kana=True) == "ニーイチニー−ハチサンロク−イチナナニーゴー"
    assert _g2p("212-836-1725").split().count("pau") == 2


@pytest.mark.parametrize(
    "text,phonemes",
    [
        ("東京 の空", "t o o ky o o n o s o r a"),
        ("ABC です", "e i b i i sh i i d e s U"),
        ("12 時間", "j u u n i j i k a N"),
        ("3 本", "s a N b o N"),
        ("私は 元気です", "w a t a sh i w a g e N k i d e s U"),
    ],
)
def test_other_spaces_keep_existing_phonemes(text: str, phonemes: str) -> None:
    assert _g2p(text) == phonemes


@pytest.mark.parametrize(
    "counter",
    [
        "年",
        "人",
        "時",
        "本",
        "分",
        "秒",
        "日",
        "個",
        "階",
    ],
)
@pytest.mark.parametrize("number", ["1", "2", "3", "4", "8", "10", "12", "2024"])
def test_spaced_counter_matches_adjacent_counter(counter: str, number: str) -> None:
    spaced = f"{number} {counter}"
    compact = f"{number}{counter}"
    assert pyopenjtalk.g2p(spaced) == pyopenjtalk.g2p(compact)
    assert [token for token in pyopenjtalk.g2p_prosody(spaced) if token != "sp"] == (
        pyopenjtalk.g2p_prosody(compact)
    )


@pytest.mark.parametrize(
    "number",
    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "０５", "十一"],
)
def test_spaced_calendar_month_matches_adjacent_month(number: str) -> None:
    spaced = f"{number} 月"
    compact = f"{number}月"
    assert pyopenjtalk.g2p(spaced) == pyopenjtalk.g2p(compact)
    assert [token for token in pyopenjtalk.g2p_prosody(spaced) if token != "sp"] == (
        pyopenjtalk.g2p_prosody(compact)
    )


@pytest.mark.parametrize(
    "text,reading",
    [("1 月ほど待った", "イチツキホドマッタ"), ("一 月が過ぎた", "イチツキガスギタ")],
)
def test_spaced_duration_month_keeps_tsuki(text: str, reading: str) -> None:
    assert _g2p(text, kana=True) == reading


@pytest.mark.parametrize("number", ["1", "2", "3"])
def test_spaced_counter_keeps_protected_user_dictionary(number: str, tmp_path: Path) -> None:
    user_csv = tmp_path / "protected_person.csv"
    user_dic = tmp_path / "protected_person.dic"
    user_csv.write_text("人,1345,1345,1,名詞,一般,*,*,*,*,人,ヒト,ヒト,0/2,C1\n", encoding="utf-8")
    pyopenjtalk.mecab_dict_index(str(user_csv), str(user_dic))
    try:
        pyopenjtalk.update_global_jtalk_with_user_dict(
            [UserDictionaryEntry(dic_path=str(user_dic), is_reading_protected=True)]
        )
        text = f"{number} 人"
        features = pyopenjtalk.run_frontend(text)
        person = next(feature for feature in features if feature["string"] == "人")
        assert person.get("is_reading_protected") is True
        assert person["read"] == "ヒト"
        assert person["pron"].replace("’", "") == "ヒト"
        assert person["mora_size"] == 2
        assert person["acc"] == 0
        assert person["pos_group1"] == "一般"
        assert person["pos_group3"] == "*"
        phonemes = _g2p(text).split()
        assert phonemes[-4:] == ["h", "I", "t", "o"]
        mapping = pyopenjtalk.g2p_mapping(text)
        assert mapping[-1]["surface"] == "人"
        assert mapping[-1]["char_span"] == (len(number) + 1, len(text))
        mecab_features, morphs = pyopenjtalk.run_mecab_detailed(text)
        assert pyopenjtalk.run_mecab(text) == mecab_features
        assert morphs[1]["is_ignored"] is True
        assert morphs[-1]["char_span"] == (len(number) + 1, len(text))
        paths = pyopenjtalk.run_mecab_nbest_features(text, max_paths=1)
        assert len(paths) == 1
        for path in paths:
            njd_features = pyopenjtalk.run_njd_from_mecab(path["features"])
            njd_person = next(feature for feature in njd_features if feature["string"] == "人")
            assert njd_person["read"] == "ヒト"
            assert njd_person["pos_group1"] == "一般"
    finally:
        pyopenjtalk.unset_user_dict()


@pytest.mark.parametrize("text", ["1 月に会う", "1 月5 日に会う"])
def test_spaced_calendar_month_in_date_context(text: str) -> None:
    assert _g2p(text) == _g2p(text.replace(" ", ""))
    assert [token for token in pyopenjtalk.g2p_prosody(text) if token != "sp"] == (
        pyopenjtalk.g2p_prosody(text.replace(" ", ""))
    )


@pytest.mark.parametrize(
    "text,reading",
    [
        ("2024 年", "ニセンニジューヨネン"),
        ("3 人", "サンニン"),
        ("10 時", "ジュージ"),
        ("3 本", "サンボン"),
        ("10 分", "ジュップン"),
        ("3 分の1", "サンブンノイチ"),
        ("１ 日本文化", "イチニホンブンカ"),
        ("５ 本書", "ゴホンショ"),
        ("図表３ 年齢", "ズヒョーサンネンレー"),
        ("２ 人づくりの基盤", "ニヒトズクリノキバン"),
        ("場面6 人前で話す", "バメンロクヒトマエデハナス"),
    ],
)
def test_spaced_counter_reading_and_word_boundaries(text: str, reading: str) -> None:
    assert pyopenjtalk.g2p(text, kana=True) == reading
    phonemes = pyopenjtalk.g2p(text)
    assert isinstance(phonemes, str)
    assert "pau" not in phonemes.split()


@pytest.mark.parametrize("text", ["3 人", "2 人", "3 日", "10 時", "０５ 月"])
def test_spaced_counter_mapping_keeps_input_positions(text: str) -> None:
    mapping = pyopenjtalk.g2p_mapping(text)
    phonemes = pyopenjtalk.g2p(text)
    assert isinstance(phonemes, str)
    assert [phone for word in mapping for phone in word["phonemes"] if phone != "sp"] == (
        phonemes.split()
    )
    # 「3 人」を「サンニン」、「2 人」を1語の「フタリ」にしても、数字の先頭から助数詞の末尾までを文字位置で示す
    assert mapping[0]["char_span"][0] == 0
    assert mapping[-1]["char_span"][1] == len(text)
    assert all(0 <= word["char_span"][0] < word["char_span"][1] <= len(text) for word in mapping)


def test_spaced_counter_mecab_ignored_space_is_preserved() -> None:
    features, morphs = pyopenjtalk.run_mecab_detailed("2024 年")
    # 「2024 年」の空白は MeCab の詳細結果では保持し、NJD へ渡す列からは除く
    assert all("記号,空白" not in feature for feature in features)
    spaces = [morph for morph in morphs if morph["is_ignored"]]
    assert len(spaces) == 1
    assert spaces[0]["char_span"] == (4, 5)


@pytest.mark.parametrize("space", ["", " "])
@pytest.mark.parametrize(
    "number,counter,reading,prosody",
    [
        ("1.5", "日", "イッテンゴニチ", "^i]clteN#go]nichi$"),
        ("0.2", "人", "レーテンニニン", "^re]eteN#ni[ni]N$"),
        ("0.1", "人", "レーテンイチニン", "^re]eteN#i[chini]N$"),
        ("1.2", "日", "イッテンニニチ", "^i]clteN#ni]nichi$"),
        ("1.5", "日間", "イッテンゴニチカン", "^i]clteN#go[nichi]kaN$"),
    ],
)
def test_decimal_counter_keeps_sino_japanese_reading(
    space: str, number: str, counter: str, reading: str, prosody: str
) -> None:
    # 「1.5 日」「0.2人」の小数部は「イツカ」「フタリ」にまとめず、漢語の「ゴニチ」「ニニン」と読む
    text = number + space + counter
    assert _g2p(text, kana=True) == reading
    assert "".join(token for token in pyopenjtalk.g2p_prosody(text) if token != "sp") == prosody
    assert "pau" not in _g2p(text).split()
    mapping = pyopenjtalk.g2p_mapping(text)
    assert mapping[-1]["char_span"] == (len(number) + len(space), len(text))
    assert [phone for word in mapping for phone in word["phonemes"] if phone != "sp"] == (
        _g2p(text).split()
    )


@pytest.mark.parametrize("written", ["二〇万円", "二〇億円", "一二〇万円"])
def test_written_quantity_across_space_matches_compact_form(written: str) -> None:
    # 「二〇 万円」は空白が数詞の結合を切っても、桁読みの「ニーマル」にせず「二〇万円」と同じ数量として読む
    spaced = written[:-2] + " " + written[-2:]
    assert _g2p(spaced) == _g2p(written)
    assert [token for token in pyopenjtalk.g2p_prosody(spaced) if token != "sp"] == (
        pyopenjtalk.g2p_prosody(written)
    )


@pytest.mark.parametrize("month", range(1, 13))
def test_zero_padded_calendar_month_reading_matches_compact_form(month: int) -> None:
    # 「04 月」「07 月」「09 月」は、空白なしの暦月と同じ「シ」「シチ」「ク」を使い、明示されたゼロを読む
    compact = f"{month:02d}月"
    spaced = f"{month:02d} 月"
    assert _g2p(spaced) == _g2p(compact)
    assert [token for token in pyopenjtalk.g2p_prosody(spaced) if token != "sp"] == (
        pyopenjtalk.g2p_prosody(compact)
    )


@pytest.mark.parametrize("text", ["2008年09月01日", "2008 年 09 月 01 日"])
def test_zero_padded_september_in_date_keeps_reading_and_accent(text: str) -> None:
    # 「2008 年 09 月 01 日」の9月は「ゼロクガツ」と読み、空白なしと同じモーラの位置で下がる
    assert _g2p(text, kana=True) == "ニセンハチネンゼロクガツゼロイチニチ"
    assert [token for token in pyopenjtalk.g2p_prosody(text) if token != "sp"] == (
        pyopenjtalk.g2p_prosody("2008年09月01日")
    )


@pytest.mark.parametrize("space", ["", " "])
@pytest.mark.parametrize("prefix", ["価格は", "〒"])
def test_decimal_range_is_not_postal_number(space: str, prefix: str) -> None:
    # 「価格は123-4567.89 円」の最後の組は小数なので、「ヨンセンゴヒャクロクジューナナテンハチキューエン」と読む
    text = prefix + "123-4567.89" + space + "円"
    assert _g2p(text, kana=True).endswith("ヨンセンゴヒャクロクジューナナテンハチキューエン")
    mapping = pyopenjtalk.g2p_mapping(text)
    point = next(
        word for word in mapping if word["char_span"] == (len(prefix) + 8, len(prefix) + 9)
    )
    assert point["phonemes"] == ["t", "e", "N"]
    assert [phone for word in mapping for phone in word["phonemes"] if phone != "sp"] == (
        _g2p(text).split()
    )


@pytest.mark.parametrize("space", ["", " "])
@pytest.mark.parametrize("prefix", ["〒印を", "郵便番号"])
def test_postal_context_does_not_change_sheet_quantity(space: str, prefix: str) -> None:
    # 「〒印を1234567 枚印刷する」は枚数なので、郵便の文脈があっても位取りと助数詞の結合を保つ
    text = prefix + "1234567" + space + "枚印刷する"
    assert _g2p(text, kana=True).endswith(
        "ヒャクニジューサンマンヨンセンゴヒャクロクジューナナマイインサツスル"
    )
    assert _g2p(text) == _g2p(text.replace(" ", ""))
    assert [token for token in pyopenjtalk.g2p_prosody(text) if token != "sp"] == (
        pyopenjtalk.g2p_prosody(text.replace(" ", ""))
    )


def test_sentence_middle_dots_do_not_make_next_number_decimal() -> None:
    # 「痛いです・・一日たっても」の中点は文の区切りなので、「一日」の整数としての発音と核を保つ
    text = "痛いです・・一日たっても"
    day = next(feature for feature in pyopenjtalk.run_frontend(text) if feature["string"] == "一日")
    assert day["pron"] == "イチニチ"
    assert day["acc"] == 4


def test_phone_number_ends_before_separate_quantity() -> None:
    # 「☎0967(44)0336 1泊」の「1泊」は空白で区切られた別の数量なので、市内局番の44を「ヨンヨン」のまま読む
    text = "☎0967(44)0336 1泊"
    assert "ヨンヨン" in _g2p(text, kana=True)
    assert "ヨンジューヨン" not in _g2p(text, kana=True)
    assert _g2p(text, kana=True).endswith("イッパク")


@pytest.mark.parametrize(
    "text,day,prosody",
    [
        ("5月1．2日", "フツカ", "fU[tsuka$"),
        ("5 月 1．2 日", "フツカ", "fU[tsuka$"),
        ("12月3.4日", "ヨッカ", "yo[clka$"),
        ("12 月 3.4 日", "ヨッカ", "yo[clka$"),
        ("五月一．二日", "フツカ", "fU[tsuka$"),
    ],
)
def test_calendar_day_enumeration_keeps_native_reading(text: str, day: str, prosody: str) -> None:
    # 「5月1．2日」「12月3.4日」の日付の列挙は、最後の日を「フツカ」「ヨッカ」の平板で読む
    assert _g2p(text, kana=True).endswith(day)
    feature = pyopenjtalk.run_frontend(text)[-1]
    assert feature["pron"].replace("’", "") == day
    assert feature["acc"] == 0
    assert "".join(pyopenjtalk.g2p_prosody(text)).endswith(prosody)
    mapping = pyopenjtalk.g2p_mapping(text)
    assert mapping[-1]["char_span"][1] == len(text)
    assert [phone for word in mapping for phone in word["phonemes"] if phone != "sp"] == (
        _g2p(text).split()
    )


@pytest.mark.parametrize(
    "text,reading",
    [
        ("2．3日", "ニーテンサンニチ"),
        ("今月1.5日働く", "コンゲツイッテンゴニチハタラク"),
        ("5ヶ月1.5日", "ゴカゲツイッテンゴニチ"),
        ("5月0.2人", "ゴガツレーテンニニン"),
    ],
)
def test_decimal_counter_outside_calendar_dates_keeps_sino_japanese_reading(
    text: str, reading: str
) -> None:
    # 月が前にない日数、「今月」の日数、期間や人数は小数の漢語読みを使う
    assert _g2p(text, kana=True) == reading


@pytest.mark.parametrize(
    "text",
    [
        "1,050円",
        "12,005人",
        "3,000,080円",
        "１，０５０円",
        "1,000",
        "10,050円",
        "1,234,567",
        "１，０００人",
        "1,050.5円",
        "100,000,000円",
        "価格は1,050円です",
        "参加者は12,005人です",
        "1,234円",
        "12,345",
        "123,456,789,012円",
        "1,002日",
        "1,024日",
        "１，０５０．５円",
        "1,050 円",
        "1,234,567 個",
    ],
)
def test_grouped_number_matches_ungrouped_reading_and_prosody(text: str) -> None:
    # 「1,050円」「12,005人」はカンマを外した位取りの数と同じ発音と核にし、数全体を続けて読む
    compact = text.replace(",", "").replace("，", "")
    assert _g2p(text, kana=True) == _g2p(compact, kana=True)
    assert _g2p(text) == _g2p(compact)
    assert pyopenjtalk.g2p_prosody(text) == pyopenjtalk.g2p_prosody(compact)
    assert "pau" not in _g2p(text).split()
    mapping = pyopenjtalk.g2p_mapping(text)
    assert [phone for word in mapping for phone in word["phonemes"] if phone != "sp"] == (
        _g2p(text).split()
    )
    covered_end = 0
    for word in mapping:
        start, end = word["char_span"]
        if start == end:
            continue
        assert start == covered_end
        covered_end = end
    assert covered_end == len(text)
    _, morphs = pyopenjtalk.run_mecab_detailed(text)
    commas = [morph for morph in morphs if morph["surface"] in (",", "，")]
    assert commas
    assert all(morph["is_ignored"] is False for morph in commas)


@pytest.mark.parametrize("text", ["1,2", "1,02", "1,05,000円", "01,02"])
def test_number_enumeration_keeps_comma_pause(text: str) -> None:
    # 「1,2」「01,02」の3桁区切りに当たらない列挙はカンマで休止し、ゼロ埋めの読みも保つ
    assert "pau" in _g2p(text).split()
    mapping = pyopenjtalk.g2p_mapping(text)
    assert [phone for word in mapping for phone in word["phonemes"] if phone != "sp"] == (
        _g2p(text).split()
    )


@pytest.mark.parametrize(
    "text",
    [
        "航空会社は3便を欠航した",
        "航空会社は3便を増便した",
        "航空会社は3便を運航した",
        "JALは3便を欠航した",
        "空港から3便を運航する",
    ],
)
def test_flight_quantity_keeps_counter_accent(text: str) -> None:
    # 「航空会社は3便を欠航した」は便の本数を数え、数量の「サ＼ンビン」の核を保つ
    three = next(feature for feature in pyopenjtalk.run_frontend(text) if feature["string"] == "三")
    assert three["acc"] == 1
    assert "#sa]NbiN" in "".join(pyopenjtalk.g2p_prosody(text))


@pytest.mark.parametrize("text", ["JAL3便を欠航した", "一日に乗るJAL226便"])
def test_flight_name_attached_to_airline_keeps_flat_accent(text: str) -> None:
    # 「JAL3便を欠航した」は会社名が番号に直接付く便名なので、欠航の文でも平板を保つ
    features = pyopenjtalk.run_frontend(text)
    last_digit = (
        next(index for index, feature in enumerate(features) if feature["string"] == "便") - 1
    )
    assert features[last_digit]["acc"] == 0


@pytest.mark.parametrize(
    "number,unit,reading",
    [("03", "千円", "ゼロサンゼンエン"), ("08", "百円", "ゼロハッピャクエン")],
)
def test_zero_padded_number_keeps_large_unit_sound_change(
    number: str, unit: str, reading: str
) -> None:
    # 「03千円」「08百円」は先頭のゼロを読み、末尾の数字と位の結合で濁音化・促音化する
    text = number + unit
    assert _g2p(text, kana=True) == reading
    assert "pau" not in _g2p(text).split()


@pytest.mark.parametrize(
    "suffix",
    [
        "以上かかってきた",
        "以下だった",
        "から200に増えた",
        "まで増えた",
        "未満だった",
        "超だった",
        "近くかかってきた",
        "ほどかかってきた",
        "くらいかかってきた",
        "ぐらいかかってきた",
        "より多い",
        "と比べて増えた",
        "に増えた",
        "に減った",
        "増えた",
        "減った",
        "に増加した",
        "に減少した",
        "と比較した",
        "を増やした",
        "を超えた",
        "を上回った",
        "を下回った",
    ],
)
@pytest.mark.parametrize("space", ["", " "])
def test_phone_quantity_matches_ordinary_number(suffix: str, space: str) -> None:
    # 「電話は100以上」「電話は100から200に増えた」は着信件数を表すので、「電話」の語がない場合と同じ「ヒャク」の読みと韻律にする
    quantity = "100" + space + suffix
    text = "電話は" + quantity
    assert _g2p(text, kana=True) == "デンワワ" + _g2p(quantity, kana=True)
    assert "".join(pyopenjtalk.g2p_prosody(text)).endswith(
        "".join(pyopenjtalk.g2p_prosody(quantity))[1:]
    )
    mapping = pyopenjtalk.g2p_mapping(text)
    assert [phone for word in mapping for phone in word["phonemes"] if phone != "sp"] == (
        _g2p(text).split()
    )
    covered_end = 0
    for word in mapping:
        start, end = word["char_span"]
        if start == end:
            continue
        assert start == covered_end
        covered_end = end
    assert covered_end == len(text)


@pytest.mark.parametrize(
    "text,reading,prosody",
    [
        ("110番に電話", "ヒャクトーバンニデンワ", "^hya[kUto]obaNni#de[Nwa$"),
        ("119に電話する", "イチイチキューニデンワスル", "^i[chii]chI#kyu]uni#de[Nwasuru$"),
        ("119 に電話する", "イチイチキューニデンワスル", "^i[chii]chI#kyu]uspni#de[Nwasuru$"),
        (
            "電話番号は0120-123-456",
            "デンワバンゴーワゼロイチニーゼロ−イチニーサン−ヨンゴーロク",
            "^de[Nwaba]Ngoowa#ze[roi]chi#ni[ize]ro_i[chini]i#sa[N_yo[Ngo]o#ro]ku$",
        ),
        ("電話が100件あった", "デンワガヒャッケンアッタ", "^de[Nwaga#hya]clkeN#a]clta$"),
    ],
)
def test_phone_destination_and_quantity_counter(text: str, reading: str, prosody: str) -> None:
    # 「119に電話する」は発信先の番号、「110番に電話」は「ヒャクトーバン」、「100件」は数量の音便とアクセント核を保つ
    assert _g2p(text, kana=True) == reading
    assert "".join(pyopenjtalk.g2p_prosody(text)) == prosody


def test_number_attached_to_name_before_phone_keeps_accent_phrase() -> None:
    # 「話者1に電話」の1は話者名の末尾なので、名前の一部として同じアクセント句で読む
    features = pyopenjtalk.run_frontend("話者1に電話した")
    one = next(feature for feature in features if feature["string"] == "一")
    assert one["chain_flag"] == 1


def test_grouped_phone_number_keeps_digit_reading_before_from() -> None:
    # 「電話番号は0120-123-456から」は3組に区切った番号なので、直後の「から」があっても発信元の番号として桁ごとに読む
    number = "電話番号は0120-123-456"
    assert _g2p(number + "から", kana=True) == _g2p(number, kana=True) + "カラ"
    assert _g2p(number + "から").split().count("pau") == 2


@pytest.mark.parametrize("label", ["電話番号", "番号", "電話番号は", "番号は"])
@pytest.mark.parametrize("space", ["", " "])
@pytest.mark.parametrize("particle", ["から", "まで", "より"])
def test_explicit_number_label_keeps_digit_reading(label: str, space: str, particle: str) -> None:
    # 「電話番号110から」は番号の見出しがあるので、「から」が続いても「イチイチゼロ」の読みとアクセントを保つ
    text = label + "110" + space + particle
    assert _g2p(text, kana=True) == _g2p(label, kana=True) + "イチイチゼロ" + _g2p(
        particle, kana=True
    )
    assert "i[chii]chi#ze]ro" in "".join(pyopenjtalk.g2p_prosody(text))
    assert "pau" not in _g2p(text).split()


@pytest.mark.parametrize("tail", ["する", "した", "をかける", " する", " をかける"])
def test_phone_call_without_following_noun_keeps_number_reading(tail: str) -> None:
    # 「119に電話する」「119に電話した」「119に電話をかける」は発信先を表すので、「イチイチキュー」と桁読みする
    assert _g2p("119に電話" + tail, kana=True).startswith("イチイチキューニデンワ")
    assert "pau" not in _g2p("119に電話" + tail).split()


@pytest.mark.parametrize("tail", ["連絡してください", "中です", "する"])
def test_phone_contact_keeps_destination_reading(tail: str) -> None:
    # 「119に電話連絡してください」「119に電話中です」も発信先を表すので、「電話」の後に名詞が続いても「イチイチキュー」と読む
    text = "119に電話" + tail
    assert _g2p(text, kana=True).startswith("イチイチキューニデンワ")
    assert "i[chii]chI#kyu]u" in "".join(pyopenjtalk.g2p_prosody(text))
    assert "pau" not in _g2p(text).split()
