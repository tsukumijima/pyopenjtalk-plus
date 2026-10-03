"""
番号の桁、アクセント句、休止が公開 API へ伝わることを確かめる。
"""

import pytest

import pyopenjtalk


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


def test_decimal_zero_keeps_positional_reading() -> None:
    assert pyopenjtalk.g2p("0.02ミリ", kana=True) == "レーテンゼロニーミリ"


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
