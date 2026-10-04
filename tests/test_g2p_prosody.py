"""Haqumei 互換の韻律記号付き音素 API (g2p_prosody / g2p_mapping_prosody) のテスト。"""

from pathlib import Path

import pytest

import pyopenjtalk
from pyopenjtalk.types import UserDictionaryEntry


def test_g2p_mapping_prosody_marks_pause_and_preserves_detailed_contract() -> None:
    """g2p_mapping_prosody において、文末の句点「。」など HTS ラベル上では pau として現れない記号であっても、入力テキスト上の全句読点と文字位置 (char_span) が正しく保持されることを確認する。"""

    mapping = pyopenjtalk.g2p_mapping_prosody("あ、い。")

    assert [entry["surface"] for entry in mapping] == ["あ", "、", "い", "。"]
    assert [entry["char_span"] for entry in mapping] == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert mapping[0]["phonemes"] == [{"kind": "Phoneme", "phoneme": "a", "pitch": "High"}]
    assert mapping[1]["phonemes"] == [{"kind": "Pause"}]
    assert mapping[2]["phonemes"] == [{"kind": "Phoneme", "phoneme": "i", "pitch": "High"}]
    assert mapping[3]["phonemes"] == [{"kind": "Pause"}]


def test_g2p_mapping_prosody_respects_non_pause_symbol_callback() -> None:
    """is_non_pause_symbol コールバックで括弧をポーズとして扱うように設定した場合でも、ラベル照合やマッピングが正しく動作することを確認する。"""

    mapping = pyopenjtalk.g2p_mapping_prosody("「あ」", is_non_pause_symbol=lambda _: False)

    assert [entry["phonemes"] for entry in mapping] == [
        [{"kind": "Pause"}],
        [{"kind": "Phoneme", "phoneme": "a", "pitch": "High"}],
        [{"kind": "Pause"}],
    ]
    assert pyopenjtalk.g2p_prosody("「あ」", is_non_pause_symbol=lambda _: False) == [
        "^",
        "_",
        "a",
        "_",
        "$",
    ]


def test_g2p_mapping_prosody_preserves_unknown_and_space() -> None:
    """HTS ラベルには直接現れない未知語（unk）や全角空白（sp）が、g2p_mapping_prosody においても詳細マッピング API と同様に正しい表層形と文字位置で出力されることを確認する。"""

    mapping = pyopenjtalk.g2p_mapping_prosody("𰻞𰻞　本")

    assert [(entry["surface"], entry["char_span"]) for entry in mapping] == [
        ("𰻞𰻞", (0, 2)),
        ("　", (2, 3)),
        ("本", (3, 4)),
    ]
    assert mapping[0]["is_unknown"] is True
    assert mapping[0]["phonemes"] == [{"kind": "Phoneme", "phoneme": "unk", "pitch": None}]
    assert mapping[1]["is_ignored"] is True
    assert mapping[1]["phonemes"] == [{"kind": "Phoneme", "phoneme": "sp", "pitch": None}]


def test_g2p_prosody_formats_pitch_and_accent_boundaries() -> None:
    """g2p_prosody において、BOS (^)、EOS ($)、アクセント句境界 (#) や、3種類のピッチ表記フォーマット (Default, Prefix, Numeric) がそれぞれ正しく出力されることを確認する。"""

    assert pyopenjtalk.g2p_prosody("青い空") == [
        "^",
        "a",
        "[",
        "o",
        "]",
        "i",
        "#",
        "s",
        "o",
        "]",
        "r",
        "a",
        "$",
    ]
    assert pyopenjtalk.g2p_prosody("青い空", format="Prefix") == [
        "^",
        "L_a",
        "H_o",
        "L_i",
        "#",
        "H_s",
        "H_o",
        "L_r",
        "L_a",
        "$",
    ]
    assert pyopenjtalk.g2p_prosody("青い空", format="Numeric") == [
        "^",
        "a:0",
        "o:1",
        "i:0",
        "#",
        "s:1",
        "o:1",
        "r:0",
        "a:0",
        "$",
    ]


@pytest.mark.parametrize("use_vanilla", [False, True])
@pytest.mark.parametrize(
    ("text", "expected_prosody", "expected_accents"),
    [
        ("12", "^ju[uni$", [3]),
        ("13", "^ju]usaN$", [1]),
        ("15", "^ju]ugo$", [1]),
        ("50", "^go[ju]u$", [2]),
        ("60", "^ro[kuju]u$", [3]),
        ("70", "^na[na]juu$", [2]),
        ("80", "^ha[chiju]u$", [3]),
        ("3時", "^sa]Nji$", [1]),
        ("10時", "^ju]uji$", [1]),
        ("11時", "^ju[uichi]ji$", [4]),
        ("12時", "^ju[uni]ji$", [3]),
        ("13時", "^ju[usa]Nji$", [3]),
        ("14時", "^ju]u#yo]ji$", [1, 1]),
        ("15時", "^ju]u#go]ji$", [1, 1]),
        ("16時", "^ju[uroku]ji$", [4]),
        ("17時", "^ju[ushIchi]ji$", [4]),
        ("18時", "^ju[uhachi]ji$", [4]),
        ("19時", "^ju]u#ku]ji$", [1, 1]),
        ("20時", "^ni[ju]uji$", [2]),
        ("21時", "^ni]juu#i[chi]ji$", [1, 2]),
        ("22時", "^ni]juu#ni]ji$", [1, 1]),
        ("23時", "^ni]juu#sa]Nji$", [1, 1]),
        ("24時", "^ni]juu#yo]ji$", [1, 1]),
        ("2人", "^fU[tari$", [3]),
        ("12人", "^ju[uni]niN$", [3]),
        ("13人", "^ju[usa]NniN$", [3]),
        ("14人", "^ju]u#yo[ni]N$", [1, 2]),
        ("19人", "^ju[ukyu]uniN$", [3]),
        ("12分", "^ju[uni]fuN$", [3]),
        ("13分", "^ju[usa]NpuN$", [3]),
        ("15分", "^ju]u#go]fuN$", [1, 1]),
        ("20歳", "^ni[ju]clsai$", [2]),
        ("11回", "^ju[uiclka]i$", [5]),
        ("12回", "^ju[unika]i$", [4]),
        ("15回", "^ju]u#go[ka]i$", [1, 2]),
        ("20回", "^ni[ju]clkai$", [2]),
        ("34回", "^sa]Njuu#yo[Nka]i$", [1, 3]),
        ("100回", "^hya[clka]i$", [3]),
        ("1000回", "^se]Nkai$", [1]),
        ("10000回", "^i[chimaNkai$", [0]),
        ("12円", "^ju[uni]eN$", [3]),
        ("100円", "^hya[kueN$", [0]),
        ("1000円", "^se[NeN$", [0]),
        ("10000円", "^i[chimaNeN$", [0]),
        ("12月", "^ju[unigatsu$", [5]),
        ("12日", "^ju[uninichi$", [5]),
        ("14日", "^ju]u#yo[clka$", [1, 0]),
        ("15日", "^ju]u#go]nichi$", [1, 1]),
        ("12個", "^ju[uni]ko$", [3]),
        ("13個", "^ju[usa]Nko$", [3]),
        ("17個", "^ju[unana]ko$", [4]),
        ("12本", "^ju[uni]hoN$", [3]),
        ("15本", "^ju]u#go[hoN$", [1, 0]),
        ("12枚", "^ju[uni]mai$", [3]),
        ("15枚", "^ju]u#go[mai$", [1, 0]),
        ("15台", "^ju]u#go[dai$", [1, 0]),
        ("9年", "^kyu]uneN$", [1]),
        ("19年", "^ju[ukyu]uneN$", [3]),
        ("15球", "^ju[ugokyuu$", [0]),
        ("15週", "^ju[ugoshuu$", [0]),
        ("15戦", "^ju[ugoseN$", [0]),
        ("15層", "^ju[ugosoo$", [0]),
        ("15倍", "^ju[ugobai$", [0]),
        ("15場所", "^ju[ugobasho$", [0]),
        ("15機種", "^ju[ugoki]shu$", [4]),
        ("15地区", "^ju[ugochi]ku$", [4]),
        ("12メートル", "^ju[unime]etoru$", [4]),
        ("15メートル", "^ju[ugome]etoru$", [4]),
        ("12階", "^ju[unikai$", [0]),
        ("15階", "^ju[ugokai$", [0]),
        ("12時間", "^ju[uniji]kaN$", [4]),
        ("15時間", "^ju[ugoji]kaN$", [4]),
        ("1石", "^i[clkoku$", [4]),
        ("8石", "^ha[chIkoku$", [4]),
        ("11石", "^ju[uiclkoku$", [6]),
        ("15石", "^ju]u#go]koku$", [1, 1]),
        ("20石", "^ni[ju]clkoku$", [2]),
        ("14日目", "^ju[uyoclkame$", [6]),
        ("15日目", "^ju[ugonichime$", [6]),
    ],
)
def test_g2p_prosody_numerals_follow_counter_accent_phrases(
    text: str, expected_prosody: str, expected_accents: list[int], use_vanilla: bool
) -> None:
    """
    NHK アクセント辞典の付録の数詞と助数詞の表に合わせ、数詞のアクセント句の区切りと核を確認する。
    「12時」は1句、「22時」は2句とし、「14時」「15分」は最初に掲載されている2句の形を使う。
    「時間」のような長い助数詞は「15時間」も1句になり、1桁の「3時」「2人」の核も保つ。
    「九年」「十九人」などは許容形の「キュー」という読みを保ち、その読みに対応する区切りと核を使う。
    「1石」「8石」「11石」は尾高型なので、文末の下降記号がなくてもアクセント核の位置を確認する。
    独自の後処理を無効にした場合も、C の数詞処理だけで同じ区切りと核になることを確認する。
    """

    prosody = pyopenjtalk.g2p_prosody(text, use_vanilla=use_vanilla)
    assert prosody.count("#") + 1 == len(expected_accents)
    assert "".join(prosody) == expected_prosody

    # 文末の尾高型は韻律記号に下降が現れないため、NJD のアクセント核も直接確認する
    features = pyopenjtalk.run_frontend(text, use_vanilla=use_vanilla)
    assert [
        feature["acc"]
        for index, feature in enumerate(features)
        if index == 0 or feature["chain_flag"] != 1
    ] == expected_accents


@pytest.mark.parametrize("use_vanilla", [False, True])
@pytest.mark.parametrize(
    ("text", "expected_reading"),
    [
        ("1アンダー", "ワンアンダー"),
        ("3アンダー", "スリーアンダー"),
        ("10アンダー", "テンアンダー"),
        ("8課", "ハチカ"),
        ("3階級", "サンカイキュー"),
        ("8階級", "ハチカイキュー"),
        ("1海里", "イチカイリ"),
        ("8海里", "ハチカイリ"),
        ("10家族", "ジュッカゾク"),
        ("10カップ", "ジュッカップ"),
        ("11株", "ジューイチカブ"),
        ("6カラット", "ロクカラット"),
        ("8缶", "ハチカン"),
        ("1気圧", "イチキアツ"),
        ("6機種", "ロッキシュ"),
        ("10機種", "ジュッキシュ"),
        ("8客", "ハチキャク"),
        ("98球", "キュージューハチキュー"),
        ("8球目", "ハチキューメ"),
        ("11切れ", "ジューイチキレ"),
        ("8斤", "ハチキン"),
        ("8区", "ハチク"),
        ("6区画", "ロックカク"),
        ("10区間", "ジュックカン"),
        ("8組", "ハチクミ"),
        ("1組", "イチクミ"),
        ("11組", "ジューイチクミ"),
        ("10系統", "ジュッケートー"),
        ("3件", "サンケン"),
        ("1000件", "センケン"),
        ("10000件", "イチマンケン"),
        ("98戸", "キュージューハチコ"),
        ("10工程", "ジュッコーテー"),
        ("10項目", "ジュッコーモク"),
        ("8皿", "ハチサラ"),
        ("1市", "イッシ"),
        ("1試合", "イッシアイ"),
        ("10試合", "ジュッシアイ"),
        ("11試合", "ジューイッシアイ"),
        ("1cc", "イチシーシー"),
        ("8シート", "ハチシート"),
        ("10シーベルト", "ジュッシーベルト"),
        ("1尺", "イッシャク"),
        ("2尺", "ニシャク"),
        ("1種目", "イッシュモク"),
        ("1石", "イッコク"),
        ("2石", "ニコク"),
        ("3石", "サンゴク"),
        ("6石", "ロッコク"),
        ("8石", "ハチコク"),
        ("10石", "ジュッコク"),
        ("11石", "ジューイッコク"),
        ("13石", "ジューサンゴク"),
        ("18石", "ジューハチコク"),
        ("20石", "ニジュッコク"),
        ("3寸", "サンズン"),
        ("90世帯", "キュージュッセタイ"),
        ("11選", "ジューイッセン"),
        ("1センチメートル", "イッセンチメートル"),
        ("8センチメートル", "ハッセンチメートル"),
        ("11束", "ジューイチタバ"),
        ("1玉", "ヒトタマ"),
        ("2玉", "フタタマ"),
        ("10チーム", "ジュッチーム"),
        ("10地区", "ジュッチク"),
        ("10地点", "ジュッチテン"),
        ("3DK", "サンディーケー"),
        ("10店舗", "ジュッテンポ"),
        ("8棟", "ハチムネ"),
        ("10棟", "ジュームネ"),
        ("11棟", "ジューイチムネ"),
        ("11とおり", "ジューイチトーリ"),
        ("1人前", "イチニンマエ"),
        ("2人前", "ニニンマエ"),
        ("8波", "ハチハ"),
        ("1箱", "ヒトハコ"),
        ("1場所", "ヒトバショ"),
        ("2場所", "フタバショ"),
        ("1柱", "ヒトハシラ"),
        ("3柱", "ミハシラ"),
        ("4柱", "ヨハシラ"),
        ("6柱", "ムハシラ"),
        ("10柱", "トハシラ"),
        ("13柱", "ジューサンハシラ"),
        ("1鉢", "ヒトハチ"),
        ("4鉢", "ヨンハチ"),
        ("10フィート", "ジュッフィート"),
        ("90袋", "キュージュップクロ"),
        ("4分咲き", "シブザキ"),
        ("四分咲き", "シブザキ"),
        ("1分袖", "イチブソデ"),
        ("4分袖", "シブソデ"),
        ("7分袖", "シチブソデ"),
        ("9分袖", "クブソデ"),
        ("8分目", "ハチブンメ"),
        ("10平方センチメートル", "ジューヘーホーセンチメートル"),
        ("10平方メートル", "ジューヘーホーメートル"),
        ("10ヘルツ", "ジュッヘルツ"),
        ("1部屋", "ヒトヘヤ"),
        ("2部屋", "フタヘヤ"),
        ("3部屋", "サンヘヤ"),
        ("10部屋", "ジュッヘヤ"),
        ("1夜", "イチヤ"),
        ("7夜", "シチヤ"),
        ("1役", "ヒトヤク"),
        ("2役", "フタヤク"),
        ("7里", "シチリ"),
        ("1000羽", "センバ"),
        ("10000羽", "イチマンバ"),
        ("10把", "ジュッパ"),
    ],
)
def test_numeral_counter_readings(text: str, expected_reading: str, use_vanilla: bool) -> None:
    """
    NHK アクセント辞典の付録の数詞と助数詞の表に載っている読みを確認する。
    数詞と助数詞が組み合わさったときの促音化と濁音化を確認する。
    「人前」「分袖」が複数の形態素へ分かれた場合も、助数詞全体の読みを確認する。
    辞書から引く「分目」「DK」などの読みも対象とする。
    「十」の促音は「ジュッ」を使い、独自の後処理を無効にした場合も同じ読みになる。
    """

    assert pyopenjtalk.g2p(text, kana=True, use_vanilla=use_vanilla) == expected_reading


@pytest.mark.parametrize("use_vanilla", [False, True])
@pytest.mark.parametrize(
    ("text", "expected_reading"),
    [
        ("7年生", "ナナネンセー"),
        ("17年生", "ジューナナネンセー"),
        ("1幕目", "イチマクメ"),
        ("2幕目", "ニマクメ"),
        ("3幕目", "サンマクメ"),
        ("4幕目", "ヨンマクメ"),
        ("5幕目", "ゴマクメ"),
        ("6幕目", "ロクマクメ"),
        ("7幕目", "ナナマクメ"),
        ("8幕目", "ハチマクメ"),
        ("9幕目", "キューマクメ"),
        ("10幕目", "ジューマクメ"),
        ("14幕目", "ジューヨンマクメ"),
    ],
)
def test_numeral_counter_readings_use_common_number_readings(
    text: str, expected_reading: str, use_vanilla: bool
) -> None:
    """
    学年の「七」は「ナナ」とし、幕番号には「イチ」「ニ」「サン」などの数字の読みを使う。
    NHK アクセント辞典の付録の数詞と助数詞の表には「シチネンセー」「ミマクメ」などが載るが、既定では現在の一般的な読みを使う。
    """

    assert pyopenjtalk.g2p(text, kana=True, use_vanilla=use_vanilla) == expected_reading


@pytest.mark.parametrize("use_vanilla", [False, True])
@pytest.mark.parametrize(
    ("text", "expected_reading"),
    [
        ("3時", "サンジ"),
        ("2人", "フタリ"),
        ("9年", "キューネン"),
        ("4年生", "ヨネンセー"),
        ("14年生", "ジューヨネンセー"),
        ("8鉢", "ハッパチ"),
        ("6鉢", "ロッパチ"),
        ("3本", "サンボン"),
        ("10本", "ジュッポン"),
        ("3階", "サンガイ"),
        ("8市", "ハチシ"),
        ("8種目", "ハチシュモク"),
        ("10幕目", "ジューマクメ"),
        ("3D", "スリーディー"),
        ("第1試合", "ダイイチシアイ"),
        ("相撲部屋", "スモーベヤ"),
        ("子供部屋", "コドモベヤ"),
    ],
)
def test_numeral_counter_readings_keep_existing_forms(
    text: str, expected_reading: str, use_vanilla: bool
) -> None:
    """
    助数詞ごとの読みと、単独の語の読みを確認する。
    「九年」「八市」「八種目」などは、NHK アクセント辞典に載っている許容形を使う。
    「階」は「サンガイ」、学年の「四」は「ヨ」という読みを確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True, use_vanilla=use_vanilla) == expected_reading


@pytest.mark.parametrize("use_vanilla", [False, True])
@pytest.mark.parametrize("number", ["2.1", "3.8", "2.11", "二ー点一", "一点八"])
def test_numeral_counter_readings_keep_decimal_digits(number: str, use_vanilla: bool) -> None:
    """
    小数点以下の桁は、単独で小数を読んだ場合と同じ読みを保つことを確認する。
    「センチメートル」の前で起こる促音化は、整数の場合に限る。
    """

    number_reading = pyopenjtalk.g2p(number, kana=True, use_vanilla=use_vanilla)
    assert isinstance(number_reading, str)
    assert (
        pyopenjtalk.g2p(number + "センチメートル", kana=True, use_vanilla=use_vanilla)
        == number_reading + "センチメートル"
    )


@pytest.mark.parametrize(
    ("text", "expected_reading", "expected_prosody"),
    [
        ("荷物は四千ｇだった", "ニモツワヨンセングラムダッタ", None),
        ("小麦粉200ｇを加える", "コムギコニヒャクグラムヲクワエル", None),
        ("5ｇ", "ゴグラム", "^go[gu]ramu$"),
        ("八十ｍ先の交差点", "ハチジューメートルサキノコーサテン", None),
        ("5ｍ", "ゴメートル", "^go[me]etoru$"),
        ("10ｔトラック", "ジュットントラック", None),
        ("5ｔ", "ゴトン", "^go]toN$"),
        ("2ｌのペットボトル", "ニリットルノペットボトル", None),
        ("2ｌ", "ニリットル", "^ni[ri]cltoru$"),
        ("１ｍｍ", "イチミリメートル", None),
        ("５ｍｇ", "ゴミリグラム", None),
    ],
)
def test_unit_letters_after_numerals_read_as_counters(
    text: str, expected_reading: str, expected_prosody: str | None
) -> None:
    """
    数字の直後の小文字の「ｇ」「ｍ」「ｔ」「ｌ」が、「グラム」「メートル」「トン」「リットル」という助数詞として読まれることを確認する。
    1文字の単位の英字は、MeCab が英字の記号として返すため、文末や「でした」の前では読まれずに落ちていた。
    アクセントは、数詞と助数詞が結合した標準的な形 (「ゴグ＼ラム」「ゴ＼トン」など) になることを確かめる。
    「ｍｍ」「ｍｇ」のような2文字の単位は、これまでどおり辞書の1語の助数詞として読まれることも確認する。
    """

    assert pyopenjtalk.g2p(text, kana=True) == expected_reading
    if expected_prosody is not None:
        assert "".join(pyopenjtalk.g2p_prosody(text)) == expected_prosody


@pytest.mark.parametrize(
    "text", ["ｍサイズ", "ｇ", "ａｂｃｄｅｆｇ", "ｍａｘ", "検定の結果、ｔ値が大きい"]
)
def test_unit_letters_without_numerals_are_not_counters(text: str) -> None:
    """
    数字に続かない「ｍサイズ」の「ｍ」や、英字の並びの中の「ｇ」、読点の後の「ｔ値」の「ｔ」は、単位の助数詞として読まれないことを確認する。
    単位の英字を助数詞として読み替えるのは、数詞の直後に限るためである。
    """

    features = pyopenjtalk.run_frontend(text)
    assert all(
        feature["pos_group2"] != "助数詞"
        for feature in features
        if feature["string"] in ("ｇ", "ｍ", "ｔ", "ｌ")
    ), features


@pytest.mark.parametrize("text", ["2m+1", "2ｍ＋1", "x＝2ｔ×3"])
def test_unit_letters_in_formulas_are_not_counters(text: str) -> None:
    """
    数式の「2m+1」や「x＝2ｔ×3」のように、数字の直後の英字に演算子が続く場合は、英字を単位の助数詞として読まないことを確認する。
    この英字は数式の変数で、「ニメートル＋イチ」と読むと式の意味が変わる。
    """

    features = pyopenjtalk.run_frontend(text)
    assert all(
        feature["pos_group2"] != "助数詞"
        for feature in features
        if feature["string"] in ("ｇ", "ｍ", "ｔ", "ｌ")
    ), features


def test_protected_symbol_letter_keeps_registered_accent(tmp_path: Path) -> None:
    """
    読み保護付きのユーザー辞書で記号として登録した「ｇ」は、数字の直後でも単位の「グラム」に置き換えず、登録した「ジー」の読みと平板のアクセント核を保つことを確認する。
    """

    user_csv = tmp_path / "protected_letter.csv"
    user_dic = tmp_path / "protected_letter.dic"
    user_csv.write_text(
        "ｇ,4,4,1,記号,アルファベット,*,*,*,*,ｇ,ジー,ジー,0/2,*\n", encoding="utf-8"
    )
    pyopenjtalk.mecab_dict_index(str(user_csv), str(user_dic))
    try:
        pyopenjtalk.update_global_jtalk_with_user_dict(
            [UserDictionaryEntry(dic_path=str(user_dic), is_reading_protected=True)]
        )
        features = pyopenjtalk.run_frontend("5ｇ")
        letter = next(feature for feature in features if feature["string"] == "ｇ")
        assert letter["pron"].replace("’", "") == "ジー"
        assert letter["acc"] == 0
        assert letter["pos"] == "記号"
        assert letter["pos_group3"] == "*"
    finally:
        pyopenjtalk.unset_user_dict()


def test_g2p_prosody_keeps_accent_boundary_across_space() -> None:
    """アクセント句の間に空白があっても、空白のない場合と同じ位置にアクセント句の境界 (#) が出ることを確認する。"""

    assert pyopenjtalk.g2p_prosody("青い 空") == [
        "^",
        "a",
        "[",
        "o",
        "]",
        "i",
        "#",
        "sp",
        "s",
        "o",
        "]",
        "r",
        "a",
        "$",
    ]


def test_g2p_prosody_marks_questions_exclamations_and_unknown_words() -> None:
    """g2p_prosody において、疑問符 (?)、感嘆符 (!)、未知語 ({unk}) の記号がそれぞれ正しく出力されることを確認する。"""

    question = pyopenjtalk.g2p_prosody("これはペンですか？")
    exclamation = pyopenjtalk.g2p_prosody("すごい！")
    unknown = pyopenjtalk.g2p_prosody("𰻞𰻞麺")

    assert question[-2:] == ["?", "$"]
    assert exclamation[-2:] == ["!", "$"]
    assert unknown[:4] == ["^", "{", "unk", "}"]


def test_g2p_prosody_returns_empty_list_for_empty_text() -> None:
    """空文字列を入力した場合、空リストが返されることを確認する。"""

    assert pyopenjtalk.g2p_prosody("") == []
