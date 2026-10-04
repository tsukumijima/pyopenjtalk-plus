"""
辞典や公式の読みで確認した既知の誤読を、正しいカナを期待値にして検証する。
正しく読めた例は strict な xfail により失敗として報告され、通常の回帰テストへ切り替える。
各例は対象の語を含む短い入力で発音を確認し、誤りの種類を理由欄に書く。
カナの長音表記や辞典で認められる別読みは、例ごとに期待値へ含める。
"""

import pytest

import pyopenjtalk


@pytest.fixture(scope="module")
def core() -> pyopenjtalk.OpenJTalk:
    """
    既定辞書を使う専用の解析器を返し、ユーザー辞書を使うほかのテストと解析器を分ける。
    """

    return pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # 標高を示す長さの単位「ｍ」を「メートル」と読む
        pytest.param(
            "三千百五十ｍ地点",
            ("メートル",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="英字と記号",
            ),
            id="latin_and_symbols-12",
        ),
        # 感染を指す infection を「インフェクション」と読む
        pytest.param(
            "Ｉｎｆｅｃｔｉｏｎ",
            ("インフェクション",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="英字と記号",
            ),
            id="latin_and_symbols-15",
        ),
        # 液体中の気泡現象を指す cavitation を「キャビテーション」と読む
        pytest.param(
            "ｃａｖｉｔａｔｉｏｎ",
            ("キャビテーション",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="英字と記号",
            ),
            id="latin_and_symbols-16",
        ),
        # 洗浄器などに利用される cavitation を「キャビテーション」と読む
        pytest.param(
            "ｃａｖｉｔａｔｉｏｎを利用したもの",
            ("キャビテーション",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="英字と記号",
            ),
            id="latin_and_symbols-17",
        ),
        # 表計算ソフトの EXCEL を「エクセル」と読む
        pytest.param(
            "ＥＸＣＥＬなどの表計算ソフト",
            ("エクセル",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="英字と記号",
            ),
            id="latin_and_symbols-44",
        ),
        # 数式の等号「＝」を「イコール」と読む
        pytest.param(
            "＝",
            ("イコール",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="英字と記号",
            ),
            id="latin_and_symbols-57",
        ),
        # 泉水の幅を示す「ｍ」を「メートル」と読む
        pytest.param(
            "幅百六十ｍの",
            ("メートル",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="英字と記号",
            ),
            id="latin_and_symbols-61",
        ),
        # 申告の英語見出しにある return を「リターン」と読む
        pytest.param(
            "ｒｅｔｕｒｎ",
            ("リターン",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="英字と記号",
            ),
            id="latin_and_symbols-82",
        ),
        # 直接原価計算の英語見出しにある direct を「ダイレクト」と読む
        pytest.param(
            "ｄｉｒｅｃｔ",
            ("ダイレクト",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="英字と記号",
            ),
            id="latin_and_symbols-83",
        ),
        # 料理の見出しにある Ginger を「ジンジャー」と読む
        pytest.param(
            "Ｇｉｎｇｅｒ",
            ("ジンジャー",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="英字と記号",
            ),
            id="latin_and_symbols-117",
        ),
        # HDD の記憶容量を示す「Ｇ」を単位接頭辞「ギガ」と読む
        pytest.param(
            "六十Ｇ　ＨＤＤ",
            ("ギガ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="英字と記号",
            ),
            id="latin_and_symbols-138",
        ),
        # 新興国グループの略称 BRICS を「ブリックス」と読む
        pytest.param(
            "BRICS",
            ("ブリックス",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="英字と記号",
            ),
            id="latin_and_symbols-280",
        ),
        # スケートの距離を示す 500m の単位を「メートル」と読む
        pytest.param(
            "女子500mで",
            ("メートル",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="英字と記号",
            ),
            id="latin_and_symbols-289",
        ),
    ],
)
def test_known_latin_and_symbols(
    core: pyopenjtalk.OpenJTalk, text: str, expected: tuple[str, ...]
) -> None:
    """
    感染を表す英単語や長さの単位記号が、辞典に記載されたカナで発音されることを確認する。
    """

    actual = pyopenjtalk.g2p(text, kana=True, use_tsqyomi=False, jtalk=core)
    # 対象の語の発音を比べ、同じ文にある別の語の誤りから独立させる
    assert any(reading in actual for reading in expected), (text, expected, actual)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # 歴史の期間を表す「ある時代」の名詞を「ジダイ」と読む
        pytest.param(
            "ある時代の終焉",
            ("ジダイ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-3",
        ),
        # 明治の前半を表す「前期」を「ゼンキ」と読む
        pytest.param(
            "明治前期",
            ("ゼンキ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-5",
        ),
        # 毎年上がる金額を指す「額」を「ガク」と読む
        pytest.param(
            "この額は毎年四月になると",
            ("ガク",),
            id="contextual_readings-6",
        ),
        # 還付金に相当する金額の「額」を「ガク」と読む
        pytest.param(
            "還付金の額相当額",
            (
                "ガクソートーガク",
                "ガクソウトウガク",
            ),
            id="contextual_readings-8",
        ),
        # 体の部位の「額」に汗をかく場面では「ヒタイ」と読む
        ## 単独の「額」は金額の「ガク」が多数派なので既定辞書は「ガク」を選び、体の部位の読みは文脈で読みを選ぶ tsqyomi に任せている
        pytest.param(
            "額に汗をかく",
            ("ヒタイ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-forehead-sweat",
        ),
        # 体の部位の「額」を拭く場面では「ヒタイ」と読む
        pytest.param(
            "額を拭く",
            ("ヒタイ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-forehead-wipe",
        ),
        # 南の地域の文化圏を表す「南方」を姓の読みと混同せず「ナンポー」と読む
        pytest.param(
            "南方文化圏",
            (
                "ナンポー",
                "ナンポウ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-19",
        ),
        # 鰹料理に並ぶ珍味「酒盗」を「シュトー」と読む
        pytest.param(
            "腹皮の塩焼き、酒盗、かつおみそ",
            (
                "シュトー",
                "シュトウ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-21",
        ),
        # 滝と間歇泉に並ぶ地形の「山」を「ヤマ」と読む
        pytest.param(
            "滝や山や間歇泉",
            ("ヤマ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-24",
        ),
        # 定理を扱う案を比較する「方が」を「ホーガ」と読む
        pytest.param(
            "定理を扱う方が自然",
            (
                "ホーガ",
                "ホウガ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-40",
        ),
        # 数学の対象「多様体」を「タヨータイ」と読む
        pytest.param(
            "ベクトル多様体は",
            (
                "タヨータイ",
                "タヨウタイ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-41",
        ),
        # 一次形式の表示の形を指す独立名詞「形」を「カタチ」または「カタ」と読む
        pytest.param(
            "上の形であたえられる",
            (
                "カタチ",
                "カタデ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-42",
        ),
        # 仮定できるか否かの問いを「イナカ」と読む
        pytest.param(
            "仮定することができるか否か",
            ("イナカ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-45",
        ),
        # 車でトンネルを通り抜ける「潜って」を「クグッテ」と読む
        pytest.param(
            "トンネルを潜って",
            ("クグッテ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-55",
        ),
        # 幼い子の苦しい旅を表す「辛かった」を「ツラカッタ」と読む
        pytest.param(
            "旅はさぞや辛かったでしょうが",
            ("ツラカッタ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-72",
        ),
        # 建設後の転用を述べる独立語「後に」を「ノチニ」または「アトニ」と読む
        pytest.param(
            "建設され後に転用",
            (
                "ノチニ",
                "アトニ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-92",
        ),
        # 家臣団の士を指す「家中」を「カチュー」と読む
        pytest.param(
            "家中の士を殺した",
            (
                "カチュー",
                "カチュウ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-95",
        ),
        # 大阪で生まれたことを表す「生を受ける」を「セー」と読む
        pytest.param(
            "生を受けました",
            (
                "セーヲ",
                "セイヲ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-100",
        ),
        # 徳目を表す一文字の「敬」を人名と混同せず「ケー」と読む
        pytest.param(
            "「敬」の一字",
            (
                "「ケー」",
                "「ケイ」",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-101",
        ),
        # 条目を解釈する動詞「解する」を「カイスル」と読む
        pytest.param(
            "条目と解する",
            ("カイスル",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-105",
        ),
        # 申し出る金額を指す「額」を「ガク」と読む
        pytest.param(
            "その額として相手方に申し出た金額",
            ("ガクトシテ",),
            id="contextual_readings-112",
        ),
        # 寄託した金額を指す「額」を「ガク」と読む
        pytest.param(
            "寄託した額を含む",
            ("ガクヲ",),
            id="contextual_readings-113",
        ),
        # 質問への回答者を指す「ご存知の方」を「カタ」と読む
        pytest.param(
            "御存知の方",
            ("カタ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-137",
        ),
        # 国際ルールに反しないやり方を表す「形」を「カタチ」または「カタ」と読む
        pytest.param(
            "国際ルールに反しない形で",
            (
                "カタチ",
                "カタデ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-141",
        ),
        # 調査研究を実施する動詞「行って」を「オコナッテ」と読む
        pytest.param(
            "情報収集や調査研究を行っている",
            ("オコナッテ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-144",
        ),
        # 稲作からの転作を述べる「米」を「コメ」と読む
        pytest.param(
            "米からの転作を奨励する",
            ("コメ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-145",
        ),
        # 会議の管理する範囲を表す「下に」を「モトニ」と読む
        pytest.param(
            "中央防災会議の下に",
            ("モトニ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-148",
        ),
        # 弾力的な運用を実施する「行って」を「オコナッテ」と読む
        pytest.param(
            "弾力的な運用を行っている",
            ("オコナッテ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-150",
        ),
        # 宗教の傾向を表す「宗教色」の接尾辞を「ショク」と読む
        pytest.param(
            "宗教色や",
            ("ショク",),
            id="contextual_readings-157",
        ),
        # ファンである人への敬称「方」を「カタ」と読む
        pytest.param(
            "ファンの方に是非",
            ("カタ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-159",
        ),
        # 志士の行方を自白させる「吐かせ」を「ハカセ」と読む
        pytest.param(
            "行方を吐かせようとする",
            ("ハカセ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-166",
        ),
        # 商品の一年あたりの回転数「年三回転」を「ネンサンカイテン」と読む
        pytest.param(
            "年三回転",
            ("ネンサンカイテン",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-171",
        ),
        # 販売促進の方法を表す「アノ手コノ手」を「アノテコノテ」と読む
        pytest.param(
            "アノ手コノ手",
            ("アノテコノテ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-175",
        ),
        # 映画の観賞を終えた「観終わって」を「ミオワッテ」と読む
        pytest.param(
            "映画観終わって",
            ("ミオワッテ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-180",
        ),
        # 演じた役に似た人生を表す慣用句「地で行く」を「ジデイク」と読む
        pytest.param(
            "地で行くような運命",
            (
                "ジデイク",
                "ジデユク",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-182",
        ),
        # シンクの下の空間を表す独立名詞「下」を「シタ」と読む
        pytest.param(
            "シンク下にスペース",
            ("シタニ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-183",
        ),
        # サラリーマンである人への敬称「方」を「カタ」と読む
        pytest.param(
            "サラリーマンの方で",
            ("カタ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-193",
        ),
        # 次の動作へつながる連用形「脱ぎ捨て、」を「ヌギステ」と読み、「テ」を読み足さない
        pytest.param(
            "濡れた上着を脱ぎ捨て、",
            ("ヌギステ、",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-201",
        ),
        # 年齢の概数を表す「歳位」の「位」を「クライ」または「グライ」と読む
        pytest.param(
            "六十歳位にしか見えなかった",
            (
                "クライ",
                "グライ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-207",
        ),
        # 募集の文字が目を引いた「目に留まった」を「トマッタ」と読む
        pytest.param(
            "目に留まった",
            ("トマッタ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-213",
        ),
        # 北海道の外の地域を指す「道外」を「ドーガイ」と読む
        pytest.param(
            "道外には",
            (
                "ドーガイ",
                "ドウガイ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-234",
        ),
        # 北海道の行政機関を指す「道」を「ドー」と読む
        pytest.param(
            "道は十五日までに",
            (
                "ドーワ",
                "ドウワ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-245",
        ),
        # 日本とアメリカの対立を表す「日米」を「ニチベー」と読む
        pytest.param(
            "日米の溝は深い",
            (
                "ニチベー",
                "ニチベイ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-252",
        ),
        # 王の統治する「国」を独立名詞の「クニ」と読む
        pytest.param(
            "長きにわたり国を治める",
            ("クニ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-258",
        ),
        # 筋道が整うことを表す「通って」を「トーッテ」と読む
        pytest.param(
            "筋道が通っており",
            (
                "トーッテ",
                "トオッテ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-259",
        ),
        # 特別な手配を指す仮名の「はからい」の語頭を「ハ」と読む
        pytest.param(
            "はからいが",
            ("ハカライ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-263",
        ),
        # 病院のベッドに空きがない「満床」を「マンショー」と読む
        pytest.param(
            "計６１床も満床で",
            (
                "マンショー",
                "マンショウ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み；日本歯科衛生士会「全身管理のための知識」（満床 まんしょう） https://www.jdha.or.jp/pdf/outline/zenshinkanri.pdf",
            ),
            id="contextual_readings-269",
        ),
        # 周囲の人の迷惑となる「端迷惑」を「ハタメーワク」と読む
        pytest.param(
            "端迷惑だ",
            (
                "ハタメーワク",
                "ハタメイワク",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-274",
        ),
        # 鳩を放つ空中を表す「空」を「ソラ」と読む
        pytest.param(
            "鳩が空に放たれた",
            ("ソラ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-275",
        ),
        # 裏の顔と対比する「表の顔」を「オモテ」と読む
        pytest.param(
            "表の顔と裏の顔",
            ("オモテ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-276",
        ),
        # 梅を見に来た訪問者を指す独立名詞「人」を「ヒト」と読む
        pytest.param(
            "見にきた人は写真を撮ったり",
            ("ヒトワ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-282",
        ),
        # 熊本の大雨を表す「雨が降りました」を「フリマシタ」と読む
        pytest.param(
            "とてもたくさんの雨が降りました",
            ("フリマシタ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-286",
        ),
        # 部屋の温度の低さを表す「寒すぎ」を「サムスギ」と読む
        pytest.param(
            "部屋は寒すぎかな",
            ("サムスギ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-294",
        ),
        # 小柄な人にも牛肉を切る能力があることを表す「捌けます」を「サバケ」と読む
        pytest.param(
            "ウィトマーは小柄だが、和牛も捌けます。",
            ("サバケ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-296",
        ),
        # 志を心に持つ「抱いて」を「イダイテ」と読む
        pytest.param(
            "高い志を抱いて",
            ("イダイテ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-297",
        ),
        # 畏れの感情を持つ「抱いた」を「イダイタ」と読む
        pytest.param(
            "畏れを抱いた",
            ("イダイタ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-299",
        ),
        # 曲芸で披露する技を表す「業」を「ワザ」と読む
        pytest.param(
            "曲芸の業を披露した",
            ("ワザ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="文脈で決まる読み",
            ),
            id="contextual_readings-301",
        ),
    ],
)
def test_known_contextual_readings(
    core: pyopenjtalk.OpenJTalk, text: str, expected: tuple[str, ...]
) -> None:
    """
    金額を表す「額」や人への敬称「方」などが、文中の語義に合う読みで発音されることを確認する。
    """

    actual = pyopenjtalk.g2p(text, kana=True, use_tsqyomi=False, jtalk=core)
    # 対象の語の発音を比べ、同じ文にある別の語の誤りから独立させる
    assert any(reading in actual for reading in expected), (text, expected, actual)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # イラストレーターの長沢節を指すため、人名を「ナガサワセツ」と読む
        pytest.param(
            "長沢節に傾倒",
            ("ナガサワセツ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="固有名詞；喜多方市美術館「長沢節展」（ながさわ・せつ） https://www.kcmofa.com/exhibit/2870/",
            ),
            id="proper_names-1",
        ),
        # 衣料ブランドの VAN を字母読みせず「ヴァン」と読む
        pytest.param(
            "ＶＡＮのメインオフィス",
            (
                "ヴァン",
                "バン",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="固有名詞；VAN 公式「ヴァンのボタンダウンシャツ」 https://www.van.co.jp/f/feature/shirts/",
            ),
            id="proper_names-2",
        ),
        # 浄水場のある地名「朝霞」を「あさか」の見出しに従い「アサカ」と読む
        pytest.param(
            "朝霞では",
            ("アサカ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="固有名詞",
            ),
            id="proper_names-30",
        ),
        # 仏教を保護した王「迦膩色迦」を「カニシカ」または「カニュシカ」と読む
        pytest.param(
            "迦膩色迦王",
            (
                "カニシカ",
                "カニュシカ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="固有名詞",
            ),
            id="proper_names-34",
        ),
        # 中国の宗教運動の名「太平道」を「タイヘードー」と読む
        pytest.param(
            "太平道が欲しい",
            (
                "タイヘードー",
                "タイヘイドウ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="固有名詞",
            ),
            id="proper_names-51",
        ),
        # 登場人物の理樹の名を公式の RIKI 表記に従って「リキ」と読む
        pytest.param(
            "理樹は複雑な気分",
            ("リキ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="固有名詞；公式アニメ「かぎなど」出演者紹介 (Riki Naoe) https://kaginado.com/en/staff-cast/",
            ),
            id="proper_names-64",
        ),
        # 理樹の心情を述べる文で、人名を「リキ」と読む
        pytest.param(
            "理樹の心情",
            ("リキ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="固有名詞；公式アニメ「かぎなど」出演者紹介 (Riki Naoe) https://kaginado.com/en/staff-cast/",
            ),
            id="proper_names-65",
        ),
        # 登場人物の小毬の名を公式の Komari 表記に従って「コマリ」と読む
        pytest.param(
            "小毬さんはどうしたんですか",
            ("コマリ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="固有名詞；公式アニメ「かぎなど」出演者紹介 (Komari Kamikita) https://kaginado.com/en/staff-cast/",
            ),
            id="proper_names-66",
        ),
        # 源義経の幼名「牛若丸」を「ウシワカマル」と読む
        pytest.param(
            "牛若丸",
            ("ウシワカマル",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="固有名詞",
            ),
            id="proper_names-90",
        ),
        # 映画出演の依頼先の姓「高峰」を「タカミネ」と読む
        pytest.param(
            "高峰に出演依頼をする",
            ("タカミネ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="固有名詞",
            ),
            id="proper_names-91",
        ),
        # 江戸時代の元号「寛文」を「カンブン」と読む
        pytest.param(
            "寛文九年四月",
            ("カンブン",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="固有名詞",
            ),
            id="proper_names-102",
        ),
        # 中国の史書「後漢書」を「ゴカンジョ」と読む
        pytest.param(
            "「後漢書」や「魏志」",
            ("ゴカンジョ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="固有名詞",
            ),
            id="proper_names-108",
        ),
        # 神の名「瓊々杵命」を「ニニギノミコト」と読む
        pytest.param(
            "瓊々杵命にすすめたが命は",
            ("ニニギノミコト",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="固有名詞",
            ),
            id="proper_names-110",
        ),
        # 伊豆諸島の地名「三宅島」を「ミヤケジマ」と読む
        pytest.param(
            "三宅島の縄文遺跡",
            ("ミヤケジマ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="固有名詞",
            ),
            id="proper_names-111",
        ),
        # ライト設計の邸宅「落水荘」を「ラクスイソー」と読む
        pytest.param(
            "落水荘",
            (
                "ラクスイソー",
                "ラクスイソウ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="固有名詞",
            ),
            id="proper_names-125",
        ),
        # 建築家の名「毅曠」を公式の読み「キコー」で読む
        pytest.param(
            "毛綱毅曠",
            (
                "キコー",
                "キコウ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="固有名詞；北海道立美術館等所蔵作品データベース「毛綱毅曠」（もづな きこう） https://artmuseum.pref.hokkaido.lg.jp/database/artist/2775",
            ),
            id="proper_names-127",
        ),
        # 大分県の自治体名「日田市」を「ヒタ」と読む
        pytest.param(
            "大分県日田市",
            ("ヒタ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="固有名詞",
            ),
            id="proper_names-186",
        ),
        # 水の流れが名所の地名「秋月」を「アキズキ」と読む
        pytest.param(
            "秋月の一番の魅力",
            (
                "アキズキ",
                "アキヅキ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="固有名詞",
            ),
            id="proper_names-187",
        ),
        # 寺院名「聖林寺」を公式の読み「ショーリンジ」で読む
        pytest.param(
            "聖林寺の境内",
            (
                "ショーリンジ",
                "ショウリンジ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="固有名詞；聖林寺 公式「聖林寺について」（しょうりんじ） https://www.shorinji-temple.jp/",
            ),
            id="proper_names-208",
        ),
        # 大分の城下町を指す「豊後竹田」を「タケタ」と読む
        pytest.param(
            "豊後竹田",
            ("タケタ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="固有名詞",
            ),
            id="proper_names-212",
        ),
        # 製品名の Diamond の部分を「ダイヤモンド」と読む
        pytest.param(
            "ＤｉａｍｏｎｄＭａｘ",
            ("ダイヤモンド",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="固有名詞",
            ),
            id="proper_names-215",
        ),
        # 北海道の自治体「幌延町」を公式の読み「ホロノベチョー」で読む
        pytest.param(
            "留萌管内幌延町",
            (
                "ホロノベチョー",
                "ホロノベチョウ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="固有名詞；幌延町 公式資料（ほろのべちょう） https://www.town.horonobe.lg.jp/www4/section/soumu/le009f000001c9qd-att/le009f000001cbbx.pdf",
            ),
            id="proper_names-221",
        ),
        # 中国河北省の都市「唐山市」を「トーザン」と読む
        pytest.param(
            "唐山市",
            (
                "トーザン",
                "トウザン",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="固有名詞",
            ),
            id="proper_names-236",
        ),
        # 中国の姓「胡」に続く副主席を「コフクシュセキ」と読む
        pytest.param(
            "胡副主席",
            ("コフクシュセキ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="固有名詞",
            ),
            id="proper_names-238",
        ),
        # 宮崎県の自治体名「三股町」を公式の「ミマタチョー」で読む
        pytest.param(
            "宮崎県三股町",
            (
                "ミマタチョー",
                "ミマタチョウ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="固有名詞；三股町 公式「にほんご教室」案内（みまたちょう） https://www.town.mimata.lg.jp/upload/file/10kyouiku/02syougai/にほんごきょうしつチラシ.pdf",
            ),
            id="proper_names-243",
        ),
        # 広島の地名「呉」を「クレ」と読む
        pytest.param(
            "広島・呉",
            ("クレ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="固有名詞",
            ),
            id="proper_names-244",
        ),
        # 慣用句に現れる清水寺の名を「キヨミズ」と読む
        pytest.param(
            "清水の舞台から",
            ("キヨミズ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="固有名詞",
            ),
            id="proper_names-270",
        ),
        # 出身地を表す「遊佐」を「ユザ」と読む
        pytest.param(
            "遊佐の出身だ",
            ("ユザ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="固有名詞",
            ),
            id="proper_names-279",
        ),
        # 俳優の小芝風花の名を公式の読み「フーカ」で読む
        pytest.param(
            "小芝風花さん",
            (
                "フーカ",
                "フウカ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="固有名詞；厚生労働省 公式プレスリリース「小芝風花」（こしば ふうか） https://prtimes.jp/main/html/rd/p/000000011.000047982.html",
            ),
            id="proper_names-281",
        ),
        # 美術家の美智の名を公式の YOSHITOMO 表記に従って「ヨシトモ」と読む
        pytest.param(
            "奈良美智さん",
            ("ヨシトモ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="固有名詞；奈良美智財団 公式カタログ (YOSHITOMO NARA) https://www.yoshitomonara.org/ja/",
            ),
            id="proper_names-283",
        ),
        # 横綱の四股名「大の里」を公式の読み「オーノサト」で読む
        pytest.param(
            "大の里",
            (
                "オーノサト",
                "オオノサト",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="固有名詞；日本相撲協会 公式「大の里」（おおのさと） https://sumo.or.jp/Yokozuna/profile/128/",
            ),
            id="proper_names-284",
        ),
        # スノーボード選手の心椛の名を公式の読み「ココモ」で読む
        pytest.param(
            "村瀬心椛選手",
            ("ココモ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="固有名詞；日本オリンピック委員会 公式「村瀬心椛」（むらせ ここも） https://joc.or.jp/games/olympic/beijing_winter/sports/snowboard/team/murasekokomo.html",
            ),
            id="proper_names-290",
        ),
    ],
)
def test_known_proper_names(
    core: pyopenjtalk.OpenJTalk, text: str, expected: tuple[str, ...]
) -> None:
    """
    人名・地名・史書名が、辞典の見出しや本人・自治体などが示す公式の読みで発音されることを確認する。
    """

    actual = pyopenjtalk.g2p(text, kana=True, use_tsqyomi=False, jtalk=core)
    # 対象の語の発音を比べ、同じ文にある別の語の誤りから独立させる
    assert any(reading in actual for reading in expected), (text, expected, actual)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # 名詞「申立て」の後に接続助詞の「テ」を読み足さず「モーシタテ」と読む
        pytest.param(
            "不服申立て制度",
            (
                "モーシタテセード",
                "モウシタテセイド",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-9",
        ),
        # 水が固まった「氷」の語頭を濁らせず「コオリ」と読む
        pytest.param(
            "水面で固化、つまり氷になる",
            (
                "コオリ",
                "コーリ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-18",
        ),
        # 別名を指す「ふたつ名」を「フタツナ」と読む
        pytest.param(
            "ふたつ名がござんしてね",
            ("フタツナ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-26",
        ),
        # 相似の関係にある形を「ソージケー」と読む
        pytest.param(
            "相似形であるともだち",
            (
                "ソージケー",
                "ソウジケイ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-62",
        ),
        # 街を歩き回る動詞「彷徨う」を「サマヨウ」と読む
        pytest.param(
            "擬態の街を彷徨う",
            ("サマヨウ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-67",
        ),
        # 浄土教の三つの経典を指す「浄土三部経」を「ジョードサンブキョー」と読む
        pytest.param(
            "浄土三部経と呼ばれる",
            (
                "サンブキョー",
                "サンブキョウ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-73",
        ),
        # 籠手本体と括弧内の振り仮名をともに「コテ」と読む
        pytest.param(
            "籠手（こて）手甲（てっこう）",
            ("コテ（コテ）",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-81",
        ),
        # 資金を事前に渡す制度の「前渡」を「マエワタシ」と読む
        pytest.param(
            "定額資金前渡制度",
            ("マエワタシ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-84",
        ),
        # 口腔内を清潔に保つ「保清」を「ホセー」と読む
        pytest.param(
            "口腔内保清を心がける",
            (
                "ホセー",
                "ホセイ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-86",
        ),
        # 幕府の役職「横目付」を「ヨコメツケ」と読む
        pytest.param(
            "横目付に",
            ("ヨコメツケ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-96",
        ),
        # 式の始まりを告げる「開式」を「カイシキ」と読む
        pytest.param(
            "間もなく開式でございます",
            ("カイシキ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-106",
        ),
        # 呼吸に関わる筋肉の専門語「呼吸筋」を「コキューキン」と読む
        pytest.param(
            "肩の呼吸筋の動き",
            (
                "コキューキン",
                "コキュウキン",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-114",
        ),
        # 生物学の相同関係を表す「相同性」を「ソードーセー」と読む
        pytest.param(
            "相同性は低い",
            (
                "ソードーセー",
                "ソウドウセイ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-115",
        ),
        # 超能力などの現象を指す「超常現象」を「チョージョーゲンショー」と読む
        pytest.param(
            "超常現象の解明",
            (
                "チョージョーゲンショー",
                "チョウジョウゲンショウ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-119",
        ),
        # 実物の大きさの型紙を表す「実物大型紙」を「ジツブツダイカタガミ」と読む
        pytest.param(
            "実物大型紙",
            ("ジツブツダイカタガミ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-123",
        ),
        # 祭りを求めて巡る人を指す「香具師」を「ヤシ」と読む
        pytest.param(
            "香具師っていうのは",
            ("ヤシ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-129",
        ),
        # 腰まで掛ける寝具の「上掛け」を「ウワガケ」と読む
        pytest.param(
            "上掛けが彼女の腰まで",
            ("ウワガケ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-136",
        ),
        # 見直しを行う期間の「見直」を「ミナオシ」と読む
        pytest.param(
            "見直期間等",
            ("ミナオシ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-146",
        ),
        # 利用者からの意思表示の「申出」を「モーシデ」と読む
        pytest.param(
            "利用者からの申出により",
            (
                "モーシデ",
                "モウシデ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-149",
        ),
        # 漁業を営む家を指す「漁家」を「ギョカ」と読む
        pytest.param(
            "南薩摩の漁家では",
            ("ギョカ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-177",
        ),
        # 皮膚の疾患「母斑」を「ボハン」と読む
        pytest.param(
            "扁平母斑と",
            ("ボハン",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-178",
        ),
        # 伊藤母斑の病名にある「母斑」を「ボハン」と読む
        pytest.param(
            "伊藤母斑と",
            ("ボハン",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-179",
        ),
        # 飲み終えた容器を指す「厄介者」を「ヤッカイモノ」と読む
        pytest.param(
            "厄介者である",
            ("ヤッカイモノ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-192",
        ),
        # 囲碁の勝敗「中押勝」を「チューオシガチ」と読む
        pytest.param(
            "黒中押勝",
            (
                "チューオシガチ",
                "チュウオシガチ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-206",
        ),
        # 近著と対比する以前の著書「前著」を「ゼンチョ」と読む
        pytest.param(
            "組んだ前著",
            ("ゼンチョ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-218",
        ),
        # 大勢の人を表す「大人数」を「オーニンズー」と読む
        pytest.param(
            "大人数が暮らす",
            (
                "オーニンズー",
                "オオニンズウ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-232",
        ),
        # 将来の年代層を指す「世代」を辞典にある「セダイ」などの読みで読む
        pytest.param(
            "赤字を放置すれば、後の世代は生活レベル",
            (
                "セダイ",
                "セタイ",
                "セーダイ",
                "セイダイ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-249",
        ),
        # 楽器の高い音を扱う「高音部」を「コーオンブ」と読む
        pytest.param(
            "高音部では",
            (
                "コーオンブ",
                "コウオンブ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-256",
        ),
        # 盆栽の枝が育つ動詞「生う」を「オウ」と読む
        pytest.param(
            "美しく生う",
            ("オウ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-260",
        ),
        # 波の浸食によってできた崖「海蝕崖」を「カイショクガイ」と読む
        pytest.param(
            "海蝕崖の壮大さに",
            ("カイショクガイ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-262",
        ),
        # 植物の花の一部「唇弁」を「シンベン」と読む
        pytest.param(
            "唇弁の内側に",
            ("シンベン",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-267",
        ),
        # 血液の流出先「内頸静脈」の「内」を「ナイ」と読む
        pytest.param(
            "内頸静脈へ流出する",
            (
                "ナイケー",
                "ナイケイ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-268",
        ),
        # ゴールより前の地点「手前」を「テマエ」と読む
        pytest.param(
            "ゴール手前で",
            ("テマエ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-273",
        ),
        # 炎が空へ上がろうとする「沖せん」を「チューセン」と読む
        pytest.param(
            "空に沖せんとした",
            (
                "チューセン",
                "チュウセン",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="辞書にない語・辞書の読みの誤り",
            ),
            id="dictionary_readings-300",
        ),
    ],
)
def test_known_dictionary_readings(
    core: pyopenjtalk.OpenJTalk, text: str, expected: tuple[str, ...]
) -> None:
    """
    「保清」「呼吸筋」などを辞典の語義に合うまとまりで読み、語の分割による誤読を検出する。
    """

    actual = pyopenjtalk.g2p(text, kana=True, use_tsqyomi=False, jtalk=core)
    # 対象の語の発音を比べ、同じ文にある別の語の誤りから独立させる
    assert any(reading in actual for reading in expected), (text, expected, actual)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # 六つの角を持つ穴の形を「ロッカクケー」など辞典の促音形で読む
        pytest.param(
            "６角形の穴あけ",
            (
                "ロッカクケー",
                "ロッカクケイ",
                "ロッカッケー",
                "ロッカッケイ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="数詞と助数詞",
            ),
            id="numbers_and_counters-31",
        ),
        # 学位取得者の人数比率の「１人」を付録に従って「ヒトリ」と読む
        pytest.param(
            "理工系学位取得者数１人に対する",
            ("ヒトリ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="数詞と助数詞",
            ),
            id="numbers_and_counters-153",
        ),
        # 入学者三千九百二十二人の末尾を「ニニン」と読む
        pytest.param(
            "入学者数は、三千九百二十二人となっている",
            (
                "ニジューニニン",
                "ニジュウニニン",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="数詞と助数詞",
            ),
            id="numbers_and_counters-154",
        ),
        # 玉鋼で作る二梃の道具を助数詞「チョー」で数える
        pytest.param(
            "二梃だそうですな",
            (
                "ニチョー",
                "ニチョウ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="数詞と助数詞",
            ),
            id="numbers_and_counters-170",
        ),
        # 調査を発表した日付の「１日」を「ツイタチ」と読む
        pytest.param(
            "日本銀行は１日、３月の企業短期経済観測調査",
            ("ツイタチ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="数詞と助数詞",
            ),
            id="numbers_and_counters-237",
        ),
        # 証券会社が銘柄を外した日付「一日」を「ツイタチ」と読む
        pytest.param(
            "ゴールドマン・サックス証券が一日、ＵＦＪについて",
            ("ツイタチ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="数詞と助数詞",
            ),
            id="numbers_and_counters-247",
        ),
    ],
)
def test_known_numbers_and_counters(
    core: pyopenjtalk.OpenJTalk, text: str, expected: tuple[str, ...]
) -> None:
    """
    日付の「１日」や人数の「二十二人」が、辞典の助数詞の表に記載された読みで発音されることを確認する。
    """

    actual = pyopenjtalk.g2p(text, kana=True, use_tsqyomi=False, jtalk=core)
    # 対象の語の発音を比べ、同じ文にある別の語の誤りから独立させる
    assert any(reading in actual for reading in expected), (text, expected, actual)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # すすり上げる鼻水を指す「水鼻」を「ミズバナ」と読む
        pytest.param(
            "水鼻をすすりあげ",
            (
                "ミズバナ",
                "ミズッパナ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="連濁と音便",
            ),
            id="rendaku_and_contractions-28",
        ),
        # 会社の規模を表す「大会社」を「ダイガイシャ」と読む
        pytest.param(
            "大会社の発行する株式",
            ("ダイガイシャ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="連濁と音便",
            ),
            id="rendaku_and_contractions-59",
        ),
        # コルネットに続く接尾辞「菓子」を「ガシ」と読む
        pytest.param(
            "コルネット菓子",
            ("ガシ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="連濁と音便",
            ),
            id="rendaku_and_contractions-97",
        ),
        # 参考の域を出ない動詞「脱しない」を促音のある「ダッシナイ」と読む
        pytest.param(
            "参考の域を脱しない",
            ("ダッシナイ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="連濁と音便",
            ),
            id="rendaku_and_contractions-174",
        ),
        # 魚料理を並べる接尾辞「尽くし」を「ズクシ」と読む
        pytest.param(
            "えのは（ヤマメ）尽くし",
            (
                "ズクシ",
                "ヅクシ",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="連濁と音便",
            ),
            id="rendaku_and_contractions-211",
        ),
        # 挿し木に根がつく動詞「根付く」を「ネズク」と読む
        pytest.param(
            "根付くらしい",
            (
                "ネズク",
                "ネヅク",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="連濁と音便",
            ),
            id="rendaku_and_contractions-264",
        ),
    ],
)
def test_known_rendaku_and_contractions(
    core: pyopenjtalk.OpenJTalk, text: str, expected: tuple[str, ...]
) -> None:
    """
    「水鼻」「根付く」などが辞典に記載された濁音や促音のある読みで発音されることを確認する。
    """

    actual = pyopenjtalk.g2p(text, kana=True, use_tsqyomi=False, jtalk=core)
    # 対象の語の発音を比べ、同じ文にある別の語の誤りから独立させる
    assert any(reading in actual for reading in expected), (text, expected, actual)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # 旧仮名で表す「無知を装ふ」の動詞を「ヨソオウ」または「ヨソー」と読む
        pytest.param(
            "無知を装ふことによつて",
            (
                "ヨソオウ",
                "ヨソー",
            ),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="長音と表記",
            ),
            id="historical_spelling-133",
        ),
    ],
)
def test_known_historical_spelling(
    core: pyopenjtalk.OpenJTalk, text: str, expected: tuple[str, ...]
) -> None:
    """
    旧仮名の「装ふ」が辞典に対応する動詞として読まれ、現代の発音を得られることを確認する。
    """

    actual = pyopenjtalk.g2p(text, kana=True, use_tsqyomi=False, jtalk=core)
    # 対象の語の発音を比べ、同じ文にある別の語の誤りから独立させる
    assert any(reading in actual for reading in expected), (text, expected, actual)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # 料理名「薩摩揚げ」の末尾に「ゲ」を読み足さず「サツマアゲ」と読む
        pytest.param(
            "揚物　薩摩揚げ",
            ("サツマアゲ",),
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="その他",
            ),
            id="extra_moras-20",
        ),
    ],
)
def test_known_extra_moras(
    core: pyopenjtalk.OpenJTalk, text: str, expected: tuple[str, ...]
) -> None:
    """
    料理名「薩摩揚げ」の語末までを発音し、同じ「ゲ」を二度読み足す誤りを検出する。
    """

    actual = pyopenjtalk.g2p(text, kana=True, use_tsqyomi=False, jtalk=core)
    assert isinstance(actual, str)
    # 語末までを比べ、余分な「ゲ」が続いた場合も読みの誤りとして検出する
    assert actual.endswith(expected), (text, expected, actual)
