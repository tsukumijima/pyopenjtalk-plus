"""
番号の桁、アクセント句、休止が公開 API へ伝わることを確かめる。
"""

from collections.abc import Iterator
from pathlib import Path

import pytest

import pyopenjtalk
import pyopenjtalk.tsqyomi as tsqyomi
from pyopenjtalk.types import UserDictionaryEntry


@pytest.fixture(scope="module")
def tsqyomi_model() -> Iterator[None]:
    # 漢数字の数は tsqyomi を使う経路でも同じに読むことを確かめるため、既定モデルを1回だけロードする
    pytest.importorskip("onnxruntime")
    was_loaded = tsqyomi.is_model_loaded()
    if was_loaded is False:
        tsqyomi.load_model(["CPUExecutionProvider"])
    yield
    if was_loaded is False and tsqyomi.is_model_loaded():
        tsqyomi.unload_model()


def _g2p(text: str, *, kana: bool = False) -> str:
    result = pyopenjtalk.g2p(text, kana=kana)
    assert isinstance(result, str)
    return result


def _prosody(text: str, *, use_tsqyomi: bool = False) -> str:
    return "".join(pyopenjtalk.g2p_prosody(text, use_tsqyomi=use_tsqyomi))


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


@pytest.mark.parametrize(
    "written,digits",
    [
        ("〇一二〇-九九九-〇〇〇", "0120-999-000"),
        ("〇三-九九-〇〇", "03-99-00"),
        ("〇〇〇〇-九九九九", "0000-9999"),
    ],
)
def test_written_phone_groups_match_arabic_digits(written: str, digits: str) -> None:
    # 桁読みの漢数字で書いた電話番号の組も、算用数字と同じく「ゼロ」「キュー」と1桁ずつ読む
    # ハイフンで数字とつながる「九九」を掛け算の「クク」、「〇〇」を伏字の「マルマル」と読んだり、
    # ハイフンと「〇」がまとめて1つの記号にされて最後の組の読みが消えたりしないことを確かめる
    assert _g2p(written, kana=True) == _g2p(digits, kana=True)
    assert _prosody(written) == _prosody(digits)


@pytest.mark.parametrize(
    "text,reading",
    [
        ("〇〇町", "マルマルマチ"),
        ("〇〇です", "マルマルデス"),
        ("〇〇〇", "マルマルマル"),
        ("九九を覚える", "ククヲオボエル"),
    ],
)
def test_placeholder_zero_and_multiplication_table_stay_words(text: str, reading: str) -> None:
    # 数字とつながらない「〇〇」は伏字の「マルマル」、単独の「九九」は掛け算の「クク」のまま読む
    assert _g2p(text, kana=True) == reading


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
        ("電話番号03・3355・1881", "ゼロサンサンサンゴーゴーイチハチハチイチ"),
        ("受け付け電話番号は、03・3355・1881", "ゼロサンサンサンゴーゴーイチハチハチイチ"),
        ("電話兼ファクス03・3801・3552", "ゼロサンサンハチゼロイチサンゴーゴーニー"),
    ],
)
def test_explicit_number_context_keeps_middle_dot_groups(text: str, reading: str) -> None:
    assert _g2p(text, kana=True).replace("・", "").endswith(reading)
    assert _g2p(text).split().count("pau") == text.count("・") + text.count("、")


@pytest.mark.parametrize("text", ["070ー3224ー5679", "市外局番213の486ー2435", "123ー4567"])
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


def test_number_pair_without_postal_context_keeps_pause() -> None:
    # 「〒」や「郵便番号」のない「123-4567」は、郵便番号と決められないのでハイフンで休止する
    assert _g2p("123-4567").split().count("pau") == 1
    assert _prosody("123-4567").endswith("i[chini]i#sa[N_yo[Ngo]o#ro[kuna]na$")


@pytest.mark.parametrize(
    "text,reading,prosody",
    [
        ("〒123-4567", "イチニーサンノヨンゴーロクナナ", "i[chini]i#sa[Nno#yo[Ngo]o#ro[kuna]na$"),
        (
            "〒一二三-四五六七",
            "イチニーサンノヨンゴーロクナナ",
            "i[chini]i#sa[Nno#yo[Ngo]o#ro[kuna]na$",
        ),
        (
            "〒102-8661",
            "イチマルニーノハチロクロクイチ",
            "i[chima]ru#ni]ino#ha[chiro]ku#ro[kui]chi$",
        ),
        (
            "郵便番号102-8661",
            "イチマルニーノハチロクロクイチ",
            "i[chima]ru#ni]ino#ha[chiro]ku#ro[kui]chi$",
        ),
        (
            "〒一〇二-八六六一",
            "イチマルニーノハチロクロクイチ",
            "i[chima]ru#ni]ino#ha[chiro]ku#ro[kui]chi$",
        ),
        (
            "〒100-0001",
            "イチマルマルノゼロゼロゼロイチ",
            "i[chima]ru#ma[runo#ze[roze]ro#ze[roi]chi$",
        ),
        (
            "〒104・8011",
            "イチマルヨンノハチマルイチイチ",
            "i[chima]ru#yo]Nno#ha[chima]ru#i[chii]chi$",
        ),
    ],
)
def test_postal_number_reads_hyphen_as_no(text: str, reading: str, prosody: str) -> None:
    # 「〒102-8661」は郵便番号の読み方で、区切りを「ノ」、0 以外の数字の後に続く 0 を「マル」と読む
    # 組の頭の 0 (「0001」) は「ゼロ」のまま読み、アクセントは電話番号と同じく2桁ずつの句にする
    # 組の最後に1桁だけ残る数字は、電話番号と同じく単独の数字のアクセント (「ニ＼ー」「サン」「マル」) に「ノ」を付ける
    assert _g2p(text, kana=True).endswith(reading)
    assert "pau" not in _g2p(text).split()
    assert _prosody(text).endswith(prosody)


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
    "text,reading",
    [
        ("二〇二六年", "ニセンニジューロクネン"),
        ("一九九五年", "センキューヒャクキュージューゴネン"),
        ("12年", "ジューニネン"),
        ("一二号室", "イチニゴーシツ"),
        ("一〇二号室", "イチマルニゴーシツ"),
        ("一二年", "ジューニネン"),
        ("四五年", "ヨンジューゴネン"),
        ("平成一二年", "ヘーセージューニネン"),
        ("一月一二日", "イチガツジューニニチ"),
        ("第二三条", "ダイニジューサンジョー"),
        ("違反点数は一二点", "イハンテンスーワジューニテン"),
        ("二三日", "ニジューサンニチ"),
    ],
)
def test_written_digit_sequence_before_counter_keeps_positional_reading(
    text: str, reading: str
) -> None:
    """
    位の字なしの漢数字に助数詞が続く書き方は、これまでどおり位取りで読み、部屋番号だけを桁読みにすることを確認する。
    「四五年」(1945年)「平成一二年」の年の略記、「一月一二日」の日付、「第二三条」の条文番号、「一二点」の確定した点数は、公的な文書でも位の字を省いて書くので、隣り合う数でも概数の「ニサン」とは読まない。
    同じ理由で、概数とも年月日とも取れる「二三日」も、位取りの「ニジューサンニチ」と読む。
    """

    assert _g2p(text, kana=True) == reading


@pytest.mark.parametrize("use_tsqyomi", [False, True])
@pytest.mark.parametrize(
    "written,digits",
    [
        ("二十三円", "23円"),
        ("二十一時", "21時"),
        ("二十三秒", "23秒"),
        ("二十三回", "23回"),
        ("二十三歳", "23歳"),
        ("二十三個", "23個"),
        ("二十三人", "23人"),
        ("十三人", "13人"),
        ("十三億", "13億"),
        ("百十三円", "113円"),
        ("八十二分", "82分"),
        ("二十二分", "22分"),
        ("二十八時間", "28時間"),
        ("二十六日目", "26日目"),
        ("三十一日", "31日"),
        ("三百二十一時間", "321時間"),
        ("一万二千三百四十五人", "12345人"),
    ],
)
def test_kanji_number_matches_arabic_digits(
    written: str, digits: str, use_tsqyomi: bool, tsqyomi_model: None
) -> None:
    # 位の字を使った漢数字の数は、算用数字で書いた同じ数と同じ読みとアクセントにする
    # 「二十三円」の「十三」を tsqyomi が地名の候補と同じ1語に差し替えたり、「二十八時間」の「八時間」、
    # 「二十六日目」の「十六日」のように数の途中から始まる辞書の語を選んだりすると、数の区切りが崩れて「ニ￣:ジューサ＼ンエン」のような句になる
    assert _prosody(written, use_tsqyomi=use_tsqyomi) == _prosody(digits, use_tsqyomi=use_tsqyomi)
    assert pyopenjtalk.g2p(written, use_tsqyomi=use_tsqyomi) == pyopenjtalk.g2p(
        digits, use_tsqyomi=use_tsqyomi
    )


@pytest.mark.parametrize("use_tsqyomi", [False, True])
def test_kanji_twenty_three_yen_has_two_accent_phrases(
    use_tsqyomi: bool, tsqyomi_model: None
) -> None:
    # 「二十三円」は「23円」と同じく、「ニ＼ジュー」と「サ＼ンエン」の2つの句で読む
    assert _prosody("二十三円", use_tsqyomi=use_tsqyomi) == "^ni]juu#sa]NeN$"


@pytest.mark.parametrize(
    "written,digits",
    [
        ("七百十三件", "713件"),
        ("一万千百十円", "11110円"),
        ("二万五千百八十七です", "25187です"),
    ],
)
def test_kanji_place_characters_keep_arabic_digit_accent(written: str, digits: str) -> None:
    # 漢字で書いた「百」「十」も、算用数字から作る位の字と同じ結合規則で読み、「ナナ＼ヒャクジュー」の核を「ナナヒャ＼クジュー」へずらさない
    assert _prosody(written) == _prosody(digits)


@pytest.mark.parametrize("use_tsqyomi", [False, True])
@pytest.mark.parametrize(
    "written,digits",
    [
        ("四十日目", "40日目"),
        ("三分で着く", "3分で着く"),
        ("娘は三歳です", "娘は3歳です"),
        ("七本の矢", "7本の矢"),
        ("三組の夫婦", "3組の夫婦"),
    ],
)
def test_kanji_counter_words_are_not_read_as_place_names(
    written: str, digits: str, use_tsqyomi: bool, tsqyomi_model: None
) -> None:
    # 「四十日」「三分」「三歳」「七本」「三組」は、数詞と助数詞より地名の「シトカ」「サンブ」「ミトセ」「ナナモト」「ミクミ」の行を選んでいた
    # 地名の行を選ばれにくくし、算用数字で書いた同じ数と同じ読みとアクセントにする
    assert _prosody(written, use_tsqyomi=use_tsqyomi) == _prosody(digits, use_tsqyomi=use_tsqyomi)


@pytest.mark.parametrize("use_tsqyomi", [False, True])
@pytest.mark.parametrize(
    "text,reading",
    [
        ("8階", "ハチカイ"),
        ("八階", "ハチカイ"),
        ("18階", "ジューハチカイ"),
        ("十八階", "ジューハチカイ"),
        ("8階建て", "ハチカイダテ"),
        ("1階", "イッカイ"),
        ("6階", "ロッカイ"),
        ("10階", "ジュッカイ"),
        ("100階", "ヒャッカイ"),
        ("8階級", "ハチカイキュー"),
    ],
)
def test_eighth_floor_reads_hachi(
    text: str, reading: str, use_tsqyomi: bool, tsqyomi_model: None
) -> None:
    # 建物の「8階」は、エレベーターの案内と同じく促音にしない「ハチカイ」と読む
    # 「1階」「6階」「10階」「100階」は、これまでどおり「イッカイ」「ロッカイ」「ジュッカイ」「ヒャッカイ」と促音にする
    assert pyopenjtalk.g2p(text, kana=True, use_tsqyomi=use_tsqyomi) == reading


@pytest.mark.parametrize("use_tsqyomi", [False, True])
@pytest.mark.parametrize(
    "text,reading",
    [
        ("8個", "ハチコ"),
        ("八個", "ハチコ"),
        ("8回", "ハチカイ"),
        ("18回", "ジューハチカイ"),
        ("8件", "ハチケン"),
        ("8分", "ハチフン"),
        ("8羽", "ハチワ"),
        ("8票", "ハチヒョー"),
        ("18歩", "ジューハチホ"),
        ("十八歩", "ジューハチホ"),
        ("8本", "ハッポン"),
        ("八本", "ハッポン"),
        ("8発", "ハッパツ"),
        ("8分の1", "ハチブンノイチ"),
    ],
)
def test_eight_before_counter_reads_hachi_by_default(
    text: str, reading: str, use_tsqyomi: bool, tsqyomi_model: None
) -> None:
    # 8 の後で促音にも読める助数詞は、既定を促音にしない「ハチ」(「ハチコ」「ハチカイ」「ハチフン」) にする
    # 促音の方が明らかに多い「本」は「ハッポン」のまま、促音の形しかない「発」も「ハッパツ」のまま読む
    # tsqyomi が「歩」に促音の後の「ポ」の行を選んでも、「ハチ」の後では清音の「ホ」になる
    assert pyopenjtalk.g2p(text, kana=True, use_tsqyomi=use_tsqyomi) == reading


@pytest.mark.parametrize("use_tsqyomi", [False, True])
@pytest.mark.parametrize(
    "text,reading",
    [
        ("一審判決", "イッシンハンケツ"),
        ("第一審", "ダイイッシン"),
        ("一国のみ", "イッコクノミ"),
        ("一体性", "イッタイセー"),
        ("一書にいわく", "イッショニイワク"),
        ("一生涯", "イッショーガイ"),
        ("一戸建", "イッコダテ"),
        ("1死一塁", "イッシイチルイ"),
        ("1回転する", "イッカイテンスル"),
        ("1小節の長さ", "イッショーセツノナガサ"),
        ("第1党", "ダイイットー"),
        ("1佐", "イッサ"),
        ("第1譜", "ダイイップ"),
        ("1速の時", "イッソクノトキ"),
        ("1食分", "イッショクブン"),
        ("片側一車線", "カタガワイッシャセン"),
        ("第一講", "ダイイッコー"),
        ("一科目", "イッカモク"),
        ("捜査1課長", "ソーサイッカチョー"),
        ("分院1庁", "ブンインイッチョー"),
        ("一時", "イチジ"),
        ("1位", "イチイ"),
        ("1型", "イチガタ"),
        ("一部", "イチブ"),
        ("1キロ", "イチキロ"),
    ],
)
def test_one_before_kango_noun_is_geminated(
    text: str, reading: str, use_tsqyomi: bool, tsqyomi_model: None
) -> None:
    """
    「一」に助数詞でない漢語の名詞が続く「一審」「一国」「1死」「1回転」などで、「一」が「イッ」と促音になることを確認する。
    MeCab が「一」と後ろの漢語を分けて解析すると、後ろの語が助数詞の表にないので「イチシン」「イチコク」と読まれていた。
    後ろの漢語の頭が「カ」「サ」「タ」「ハ」行の無声の音のときだけ促音にし、「ハ」行は「第1譜」(「ダイイップ」) のように半濁音にする。
    濁音で始まる「一時」「一部」や、母音で始まる「1位」、片仮名の「1キロ」は今までどおり「イチ」と読む。
    """

    assert pyopenjtalk.g2p(text, kana=True, use_tsqyomi=use_tsqyomi) == reading


@pytest.mark.parametrize("use_tsqyomi", [False, True])
@pytest.mark.parametrize(
    "written,digits",
    [
        ("三階", "3階"),
        ("二区画", "2区画"),
        ("一区間", "1区間"),
        ("三口", "3口"),
    ],
)
def test_kanji_counter_words_match_arabic_digit_reading(
    written: str, digits: str, use_tsqyomi: bool, tsqyomi_model: None
) -> None:
    # 「三階」「二区」「一区」は地名の行、「三口」は一般名詞の「ミツクチ」の行を選び、算用数字の「3階」(「サンガイ」) などと読みが食い違っていた
    assert pyopenjtalk.g2p(written, kana=True, use_tsqyomi=use_tsqyomi) == pyopenjtalk.g2p(
        digits, kana=True, use_tsqyomi=use_tsqyomi
    )


@pytest.mark.parametrize(
    "text,reading",
    [
        ("三分袖のシャツ", "サンブソデノシャツ"),
        ("六分袖のシャツ", "ロクブソデノシャツ"),
    ],
)
def test_sleeve_length_keeps_bu_reading(text: str, reading: str) -> None:
    # 地名の「三分」「六分」の行を選ばれにくくしても、袖丈の「三分袖」「六分袖」は時間量の「サンプン」にせず「ブ」と読む
    assert _g2p(text, kana=True) == reading


@pytest.mark.parametrize("use_tsqyomi", [False, True])
@pytest.mark.parametrize(
    "text,reading",
    [
        ("二〇二六年", "ニセンニジューロクネン"),
        ("二千二十六年", "ニセンニジューロクネン"),
        ("二、三人", "ニ、サンニン"),
        ("十分間", "ジュップンカン"),
        ("十三駅", "ジューソーエキ"),
        ("二十歳", "ハタチ"),
        ("八百屋", "ヤオヤ"),
        ("五十嵐さん", "イガラシサン"),
        ("九十九里浜", "クジュークリハマ"),
        ("三分待つ", "サンプンマツ"),
    ],
)
def test_kanji_words_starting_with_numerals_keep_reading(
    text: str, reading: str, use_tsqyomi: bool, tsqyomi_model: None
) -> None:
    # 数字で始まる語 (「二十歳」「八百屋」「五十嵐」「九十九里浜」) や、地名の「十三」は、数を1字ずつに分けずに辞書の語として読む
    # 年号の「二〇二六年」、概数の「二、三人」、時間の「十分間」も今の読みを保つ
    assert pyopenjtalk.g2p(text, kana=True, use_tsqyomi=use_tsqyomi) == reading


@pytest.mark.parametrize(
    "text,reading,prosody",
    [
        ("三四日", "サンヨッカ", "^sa]Nyoclka$"),
        ("三四日かかる", "サンヨッカカカル", "^sa]Nyoclka#ka[ka]ru$"),
    ],
)
def test_written_approximate_count_of_days_uses_native_four(
    text: str, reading: str, prosody: str
) -> None:
    """
    「三四日」は「3〜4日」の概数で、後ろの「四日」を和語の「ヨッカ」と読み、「サンヨッカ」となることを確認する。
    「34日」という日付はないので、「二三日」と違って位取りで読む余地がなく、1語で登録して「サ」の直後で下げる。
    """

    assert _g2p(text, kana=True) == reading
    assert "".join(pyopenjtalk.g2p_prosody(text)) == prosody


@pytest.mark.parametrize(
    "text,reading",
    [
        ("一一〇番に通報した", "ヒャクトーバンニツーホーシタ"),
        ("110番に通報した", "ヒャクトーバンニツーホーシタ"),
        ("一一九番に電話した", "ヒャクジューキューバンニデンワシタ"),
        ("一一〇号室", "イチイチマルゴーシツ"),
    ],
)
def test_written_police_emergency_number_uses_hyakutoban(text: str, reading: str) -> None:
    """
    警察への通報番号を漢数字で書いた「一一〇番」が、算用数字の「110番」と同じく「ヒャクトーバン」と読まれることを確認する。
    漢数字の表記の行が辞書になく、数詞の処理で「百」「十」に展開されて「ヒャクジューバン」と読まれていた。
    消防の「一一九番」は国語辞典の見出しどおり位取りの「ヒャクジューキューバン」、部屋番号の「一一〇号室」は桁読みのまま変わらないことも確かめる。
    """

    assert _g2p(text, kana=True) == reading


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
    ("text", "expected_tail"),
    [
        ("型番AB-1200", "センニヒャク"),
        ("AB-12", "ジューニ"),
        ("AB-3248", "サンニーヨンハチ"),
    ],
)
def test_model_number_after_letters_uses_compact_positional_reading(
    text: str, expected_tail: str
) -> None:
    """
    英字とハイフンに続く型番の数字を、号機と同じく、位取りで短く読めるものは位取りで読むことを確認する。
    「型番AB-1200」は「イチニーゼロゼロ」と桁読みになっていたが、号機の「1000号機」と同じく「センニヒャク」と読む方が短く自然である。
    位取りすると長くなる「AB-3248」は桁読みのまま変わらないことも確かめる。
    """

    reading = pyopenjtalk.g2p(text, kana=True)
    assert isinstance(reading, str)
    assert reading.endswith(expected_tail)


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


@pytest.mark.parametrize(
    "text,prosody",
    [
        ("2.5次元", "^ n i ] i t e N # g o [ j i ] g e N $"),
        ("２．５次元ミュージカル", "^ n i ] i t e N # g o [ j i ] g e N # my u ] u j i k a r u $"),
    ],
)
def test_decimal_before_noun_keeps_accent_phrases(text: str, prosody: str) -> None:
    """
    小数に名詞が続く「2.5次元」は、整数部の「ニ＼ーテン」と、小数部と名詞の「ゴジ＼ゲン」の2つのアクセント句で読むことを確認する。
    小数部の数詞は後ろの名詞と1つの句になり、「次元」の核が残る。
    """

    assert " ".join(pyopenjtalk.g2p_prosody(text)) == prosody


@pytest.mark.parametrize("text", ["一〇・五", "10.5"])
def test_decimal_integer_part_is_not_identifier(text: str) -> None:
    assert _g2p(text, kana=True) == "ジュッテンゴ"
    assert "pau" not in _g2p(text).split()


@pytest.mark.parametrize(
    "text,reading",
    [
        ("〇・五", "レーテンゴ"),
        ("〇・〇三", "レーテンゼロサン"),
        ("三・五％", "サンテンゴパーセント"),
        ("一・五倍", "イッテンゴバイ"),
        ("三・五キロ", "サンテンゴキロ"),
        ("二・五割", "ニーテンゴワリ"),
        ("一・二年生", "イチ・ニネンセー"),
        ("三・四月", "サン・シガツ"),
        ("三・四人", "サン・ヨニン"),
        ("一〇・五", "ジュッテンゴ"),
        ("三〇・四〇代", "サンジュー・ヨンジューダイ"),
        ("二〇・三〇年代", "ニジュー・サンジューネンダイ"),
        ("一〇・二〇人", "ジュー・ニジューニン"),
    ],
)
def test_vertical_decimal_point_only_with_zero_or_unit(text: str, reading: str) -> None:
    """
    縦書きの小数点に使う漢数字の間の中黒は、小数と分かる場合だけ「テン」と読むことを確認する。
    前の数が「〇・五」「〇・〇三」のように「〇」で始まるときと、後ろに「％」「倍」「キロ」「割」のような量の単位が続くときは小数点として読む。
    前の数が「一〇・五」のように「〇」を書く桁読みの表記で、後ろの数が「〇」で終わらず、後ろに助数詞も続かないときも小数点として読む。
    「一・二年生」「三・四月」「三・四人」や、後ろの数が「〇」で終わる「三〇・四〇代」「二〇・三〇年代」「一〇・二〇人」は、数を並べる区切りとして別々の数に読む。
    """

    assert _g2p(text, kana=True) == reading


@pytest.mark.parametrize("use_vanilla", [False, True])
@pytest.mark.parametrize(
    "text,reading",
    [
        ("０．３ｃｍ白１．２ｃｍ", "レーテンサンセンチメートルシロイッテンニセンチメートル"),
        ("０．３　１．２", "レーテンサンイッテンニ"),
        (
            "０．３ｃｍビーズ　１．２ｃｍ",
            "レーテンサンセンチメートルビーズイッテンニセンチメートル",
        ),
        (
            "直径０．３ｃｍ白ウッドビーズ二十個　直径１．２ｃｍくるみボタン芯１個",
            "チョッケーレーテンサンセンチメートルシロウッドビーズニジュッコチョッケーイッテンニセンチメートルクルミボタンシンイッコ",
        ),
        ("０．３白１．２", "レーテンサンシロイッテンニ"),
        ("０．３ビーズ１．２", "レーテンサンビーズイッテンニ"),
        ("０．３ｃｍ　１．２ｃｍ", "レーテンサンセンチメートルイッテンニセンチメートル"),
        ("０．３　１．２　５．６", "レーテンサンイッテンニゴーテンロク"),
        ("0.3cm白1.2cm", "レーテンサンセンチメートルシロイッテンニセンチメートル"),
        ("０．３ｃｍ白２．４ｃｍ", "レーテンサンセンチメートルシロニーテンヨンセンチメートル"),
        ("０．３ｃｍ白５．６ｃｍ", "レーテンサンセンチメートルシロゴーテンロクセンチメートル"),
        # 小数の桁は整数の「8センチ」のような促音化を受けず、「ハチセンチ」と読む
        ("０．３ｃｍ白６．８ｃｍ", "レーテンサンセンチメートルシロロクテンハチセンチメートル"),
    ],
)
def test_multiple_decimals_separated_by_nouns_or_spaces(
    text: str, reading: str, use_vanilla: bool
) -> None:
    """
    「０．３ｃｍ白１．２ｃｍ」や「０．３　１．２」の2つ目の小数を「イッテンニ」と読むことを確認する。
    小数部の後ろに名詞や空白が続いても、次の小数点を読点として扱うと寸法の読みが途切れる。
    Python 側の後処理を適用しない場合も、NJD の数詞処理で小数点を「テン」に変換する。
    """

    assert pyopenjtalk.g2p(text, kana=True, use_vanilla=use_vanilla) == reading
    phonemes = pyopenjtalk.g2p(text, use_vanilla=use_vanilla)
    assert isinstance(phonemes, str)
    assert "pau" not in phonemes.split()


@pytest.mark.parametrize(
    "written,arabic,reading",
    [
        ("一点六一八", "1.618", "イッテンロクイチハチ"),
        ("十点一八", "10.18", "ジュッテンイチハチ"),
        ("六点一八メートル", "6.18m", "ロクテンイチハチメートル"),
        ("一点二三四五", "1.2345", "イッテンニーサンヨンゴー"),
        ("零点零五", "0.05", "レーテンゼロゴー"),
        ("九点五零零", "9.500", "キューテンゴーゼロゼロ"),
        ("〇点〇五", "0.05", "レーテンゼロゴー"),
    ],
)
def test_written_decimal_reads_fraction_digit_by_digit(
    written: str, arabic: str, reading: str
) -> None:
    """
    漢字の「点」で書いた小数「一点六一八」の小数部を、算用数字の「1.618」と同じく「ロクイチハチ」と1桁ずつ読むことを確認する。
    小数部を位取りの「ロッピャクジューハチ」と読むと、まったく別の数になる。
    小数部の「零」「〇」は算用数字の小数の「0」と同じく「ゼロ」、2桁ずつのアクセント句の分け方も算用数字の小数と同じにする。
    """

    assert _g2p(written, kana=True) == reading
    # 整数部と「テン」の句は辞書の「一点」「零点」の核を使うことがあるので、小数部の句だけを比べる
    assert (
        "".join(pyopenjtalk.g2p_prosody(written)).split("#", 1)[1]
        == "".join(pyopenjtalk.g2p_prosody(arabic)).split("#", 1)[1]
    )


@pytest.mark.parametrize(
    "text,reading",
    [
        ("三十六点五度", "サンジューロクテンゴド"),
        ("五十九点二", "ゴジューキューテンニ"),
        ("三点十五", "サンテンジューゴ"),
    ],
)
def test_written_decimal_with_single_digit_or_place_unit_fraction(text: str, reading: str) -> None:
    """
    小数部が1桁の「三十六点五度」はこれまでどおり読み、位の字を含む「三点十五」は書かれたとおりに位取りで読むことを確認する。
    「十」を含む小数部は1桁ずつの書き方ではないので、書き手が位取りの数として書いたものとみなす。
    """

    assert _g2p(text, kana=True) == reading


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
    # 郵便番号では、0 以外の数字の後に続く 0 を「マル」と読む
    assert _g2p("〒460 8511", kana=True) == "〒ヨンロクマルハチゴーイチイチ"
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


@pytest.mark.parametrize(
    "text,reading,phrase_count",
    [
        ("1・2年生", "イチ・ニネンセー", 2),
        ("一・二年生", "イチ・ニネンセー", 2),
        ("Ⅰ・Ⅱ", "イチ・ニ", 2),
        ("3・4月", "サン・シガツ", 2),
        ("三・四月", "サン・シガツ", 2),
        ("1・2・3", "イチ・ニ・サン", 3),
        ("1.5・2.5", "イッテンゴ・ニーテンゴ", 4),
    ],
)
def test_middle_dot_between_numbers_separates_numbers(
    text: str, reading: str, phrase_count: int, capfd: pytest.CaptureFixture[str]
) -> None:
    """
    数の間の中黒は数を並べる区切りなので、小数点の「テン」と読まず、前後を別々の数として読むことを確認する。
    「1・2年生」は「イッテンニネンセー」と小数のように読まれ、「3・4月」は読点と同じ休止が入っていた。
    中黒の前後は休止を置かずに別々のアクセント句にし、g2p() の音素と g2p_mapping() の音素も一致させる。
    カタカナの出力には、ほかの記号と同じく中黒の表記を残す。
    """

    assert _g2p(text, kana=True) == reading
    assert "pau" not in _g2p(text).split()
    prosody = pyopenjtalk.g2p_prosody(text)
    assert "_" not in prosody
    assert prosody.count("#") == phrase_count - 1
    mapping = pyopenjtalk.g2p_mapping(text)
    assert [phone for item in mapping for phone in item["phonemes"]] == _g2p(text).split()
    assert "".join(text[slice(*item["char_span"])] for item in mapping) == text
    # 中黒の印は JPCommon の品詞の表にないので、品詞の変換で警告が出ないことも確かめる
    assert "WARNING" not in capfd.readouterr().err


@pytest.mark.parametrize(
    "text,reading",
    [
        ("5・15事件", "ゴーイチゴージケン"),
        ("五・一五事件", "ゴーイチゴージケン"),
        ("2・26事件", "ニーニーロクジケン"),
        ("二・二六事件", "ニーニーロクジケン"),
        ("3・11", "サンテンイチイチ"),
        ("三・一一", "サンテンイチイチ"),
        ("9・11", "キューテンイチイチ"),
        ("九・一一", "キューテンイチイチ"),
        ("1・17", "イッテンイチナナ"),
        ("一・一七", "イッテンイチナナ"),
    ],
)
def test_customary_date_names_take_precedence_over_middle_dot_rule(text: str, reading: str) -> None:
    """
    慣用の読みがある日付の呼び名は、辞書の1語として、数を並べる中黒の規則より優先して読まれることを確認する。
    中黒の規則だけでは「5・15事件」が「ゴ・ジューゴジケン」、「3・11」が「サン・ジューイチ」と別々の数に読まれる。
    """

    assert _g2p(text, kana=True) == reading
    assert len(pyopenjtalk.run_frontend(text)) == 1


@pytest.mark.parametrize(
    "text,reading,pause_count",
    [
        ("1.5", "イッテンゴ", 0),
        ("３．５", "サンテンゴ", 0),
        ("1．2", "イッテンニ", 0),
        ("5月1．2日", "ゴガツイッテンフツカ", 0),
        ("2026/10/10", "ニセンニジューロク／ジュー／ジュー", 2),
        ("電話番号03・3355・1881", "デンワバンゴーゼロサン・サンサンゴーゴー・イチハチハチイチ", 2),
        ("〒104・8011", "〒イチマルヨンノハチマルイチイチ", 0),
        ("東京・大阪", "トーキョー・オーサカ", 1),
        ("A・B", "Ａ・Ｂ", 1),
        ("第1・第2", "ダイイチ・ダイニ", 1),
        ("1・A", "イチ・アンペア", 1),
        ("A・1", "Ａ・イチ", 1),
    ],
)
def test_middle_dot_rule_keeps_decimals_and_other_separators(
    text: str, reading: str, pause_count: int
) -> None:
    """
    数を並べる中黒の規則が、小数点・日付・電話番号・郵便番号の区切りと、数以外の語の間の中黒を変えないことを確認する。
    小数点は「.」「．」だけで、「1.5」「３．５」は「テン」と読む。
    電話番号の「03・3355・1881」は今までどおり中黒で休止し、郵便番号の「〒104・8011」は「ノ」と読む。
    「東京・大阪」「A・B」や、中黒の片側が数でない「第1・第2」「1・A」は、読点と同じ休止のまま読む。
    """

    assert _g2p(text, kana=True) == reading
    assert _g2p(text).split().count("pau") == pause_count


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


@pytest.mark.parametrize(
    "text,fullwidth",
    [
        ("03-1234-5678.", "03-1234-5678。"),
        ("一〇一,", "一〇一、"),
        ("一〇一.", "一〇一。"),
        ("四〇五,電話", "四〇五、電話"),
        ("ハイツ'三〇九,三〇八", "ハイツ'三〇九、三〇八"),
    ],
)
def test_halfwidth_punctuation_after_digits_is_not_separator(text: str, fullwidth: str) -> None:
    # 読点を「,」、句点を「.」で書いた文でも、後ろに数字のない「,」「.」は桁区切りや小数点でなく、
    # 句読点として「、」「。」と同じに読む (「一〇一,」を位取りの「ヒャクイチ」にしない)
    # 「〇」を書いた桁読みの漢数字の後の「,」も、組の区切りであって3桁区切りではない
    halfwidth_reading = _g2p(text, kana=True).translate(str.maketrans("，．", "、。"))
    assert halfwidth_reading.rstrip("、。") == _g2p(fullwidth, kana=True).rstrip("、。")
    assert _prosody(text).rstrip("_$") == _prosody(fullwidth).rstrip("_$")


@pytest.mark.parametrize(
    "text,reading",
    [
        ("三.五", "サンテンゴ"),
        ("3.5キロ", "サンテンゴキロ"),
        ("1,000円", "センエン"),
        ("一,〇〇〇", "セン"),
        ("1,050.5円", "センゴジュッテンゴエン"),
    ],
)
def test_real_decimal_point_and_digit_group_comma_keep_number(text: str, reading: str) -> None:
    # 後ろに数字が続く「.」は小数点、3桁の組が続く「,」は桁区切りとして、今までどおり1つの数に読む
    assert _g2p(text, kana=True) == reading
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
    ("text", "expected"),
    [
        ("100に電話料金を足す", "ヒャクニデンワリョーキンヲタス"),
        ("電話は100ある", "デンワワヒャクアル"),
        ("電話は100以上", "デンワワヒャクイジョー"),
        ("119に電話する", "イチイチキューニデンワスル"),
    ],
)
def test_phone_context_quantity_keeps_positional_reading(text: str, expected: str) -> None:
    """
    「電話」の前後にある数でも、電話番号でなく数量として使われるものは位取りの「ヒャク」と読まれることを確認する。
    発信先の番号を桁読みにする規則が広がり、「100に電話料金を足す」「電話は100ある」を「イチゼロゼロ」と読んでいた。
    発信先の「119に電話する」は桁読みのまま変わらないことも確かめる。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected


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


@pytest.mark.parametrize(
    "text,reading",
    [
        ("119にかける", "イチイチキューニカケル"),
        ("110に掛ける", "イチイチゼロニカケル"),
        ("110に電話", "イチイチゼロニデンワ"),
        ("118に電話", "イチイチハチニデンワ"),
        ("188に電話した", "イチハチハチニデンワシタ"),
        ("189に相談する", "イチハチキューニソーダンスル"),
        ("110へ通報した", "イチイチゼロエツーホーシタ"),
        ("104に電話", "イチマルヨンニデンワ"),
        ("177で連絡", "イチナナナナデレンラク"),
        ("110番に通報した", "ヒャクトーバンニツーホーシタ"),
        ("119番", "ヒャクジューキューバン"),
        ("100にかける", "ヒャクニカケル"),
        ("110円", "ヒャクジューエン"),
        ("得点は110", "トクテンワヒャクジュー"),
        ("189人", "ヒャクハチジューキューニン"),
        ("七一一九", "ナナイチイチキュー"),
    ],
)
def test_phone_destination_with_kakeru_uses_digit_reading(text: str, reading: str) -> None:
    """
    広く知られた 1XY の番号 (104・110・113・115・117・118・119・171・177・184・186・188・189) に、「電話」「通報」「連絡」「ダイヤル」「相談」「コール」「かける」「掛ける」が続くときは、発信先の番号として「イチイチゼロ」のように桁ごとに読まれることを確認する。
    0 は、「110」のように番号の末尾にあれば「ゼロ」、「104」のように真ん中にあれば部屋番号と同じく「マル」と読む。
    間の助詞は「に」「へ」「を」「で」を許す。「番」が付く「110番」は「ヒャクトーバン」、「119番」は位取りのまま読む。
    特別な番号でない「100にかける」や、助数詞の付く「110円」「189人」、得点の「110」は数量として位取りのまま読む。
    短縮ダイヤルの「#7119」は正規化が「七一一九」と桁読みの漢数字で渡すので、その「ナナイチイチキュー」の読みも確かめる。
    """

    assert pyopenjtalk.g2p(text, kana=True) == reading


@pytest.mark.parametrize(
    "text,reading",
    [
        (
            "電話番号は0120-205-123です",
            "デンワバンゴーワゼロイチニーゼロ−ニーマルゴー−イチニーサンデス",
        ),
        (
            "電話番号は03-3205-1234です",
            "デンワバンゴーワゼロサン−サンニーゼロゴー−イチニーサンヨンデス",
        ),
        (
            "電話番号は070-1234-5678です",
            "デンワバンゴーワゼロナナゼロ−イチニーサンヨン−ゴーロクナナハチデス",
        ),
        (
            "電話番号は050-1000-2000です",
            "デンワバンゴーワゼロゴーゼロ−イチゼロゼロゼロ−ニーゼロゼロゼロデス",
        ),
    ],
)
def test_phone_three_digit_group_reads_middle_zero_as_maru(text: str, reading: str) -> None:
    """
    電話番号を桁読みするとき、3桁の組の真ん中の 0 は部屋番号と同じく「マル」と読むことを確認する。
    「0120-205-123」の「205」は「ニーマルゴー」になる。
    組の先頭と末尾の 0 (「070」) と、4桁の組の 0 (「0120」「3205」「1000」) は、今までどおり「ゼロ」と読む。
    """

    assert pyopenjtalk.g2p(text, kana=True) == reading


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
