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


@pytest.mark.parametrize(
    "text,expected_prosody",
    [
        ("守らない", "^ma[mora]nai$"),
        ("書かない", "^ka[ka]nai$"),
        ("書かせない", "^ka[kase]nai$"),
        ("出来ない", "^de[ki]nai$"),
        ("書かないで", "^ka[ka]naide$"),
        ("書かせる", "^ka[kase]ru$"),
    ],
)
def test_accented_verb_negative_form_falls_before_nai(text: str, expected_prosody: str) -> None:
    """
    起伏型の動詞のナイ形 (「守らない」「書かない」) が、「ない」の直前の拍の後で下がる「マモラ＼ナイ」「カカ＼ナイ」と読まれることを確認する。
    使役の「せ」が付いた「書かせない」は「カカセ＼ナイ」、起伏型の「出来る」のナイ形は「デキ＼ナイ」と読む。
    「ないで」も同じく「カカ＼ナイデ」、使役の基本形は「カカセ＼ル」と読む。
    """

    assert "".join(pyopenjtalk.g2p_prosody(text)) == expected_prosody


@pytest.mark.parametrize(
    "text,expected_prosody",
    [
        ("盛らない", "^mo[ranai$"),
        ("行かない", "^i[kanai$"),
        ("聞かせない", "^kI[kasenai$"),
        ("歌わせない", "^u[tawasenai$"),
        ("盛らないで", "^mo[ranaide$"),
        ("行かないで", "^i[kanaide$"),
        ("聞かせる", "^kI[kaseru$"),
        ("歌わせる", "^u[tawaseru$"),
        ("上がらず", "^a[gara]zu$"),
    ],
)
def test_flat_verb_negative_form_stays_flat(text: str, expected_prosody: str) -> None:
    """
    平板型の動詞のナイ形 (「盛らない」「行かない」) が、平板のまま「モラナイ」「イカナイ」と読まれることを確認する。
    MeCab が「聞か」「せ」「ない」に分ける使役の「聞かせない」「歌わせない」も、「聞かせる」と同じく平板の「キカセナイ」「ウタワセナイ」と読む。
    「ないで」と使役の基本形も平板の「モラナイデ」「キカセル」と読む。
    「ない」の規則は未然形の行の核を変えずに直すので、打ち消しの「ず」は今までどおり「アガラ＼ズ」と「ず」の前で下げる。
    """

    assert "".join(pyopenjtalk.g2p_prosody(text)) == expected_prosody


@pytest.mark.parametrize(
    "text,expected_prosody",
    [
        ("守ろう", "^ma[moro]o$"),
        ("守ろうとする", "^ma[moro]otosuru$"),
        ("書こうとする", "^ka[ko]otosuru$"),
        ("盛ろう", "^mo[ro]o$"),
    ],
)
def test_volitional_form_falls_before_u(text: str, expected_prosody: str) -> None:
    """
    動詞の意向形が「う」の直前の拍の後で下がり (「マモロ＼ー」「モロ＼ー」)、起伏型の動詞は後ろに「とする」が続いても同じ形を保つことを確認する。
    「う」は付属語として核を決める型なので、平板型の「盛る」の意向形も「モロ＼ー」と下がる。
    """

    assert "".join(pyopenjtalk.g2p_prosody(text)) == expected_prosody


@pytest.mark.parametrize("text", ["かかるだろう", "調和しないだろう"])
def test_daroo_is_not_treated_as_volitional_form(text: str) -> None:
    """
    推量の「だろう」の「う」は意向形と違い、句の末尾の「ロ」の後で下がる形にならないことを確認する。
    「う」の結合規則を辞書の行で変えると、句の頭の動詞に従って「だろう」の「う」にも同じ規則が当たり、「かかるだろう」が1つの句の「カカルダロ＼ー」になる。
    """

    assert "".join(pyopenjtalk.g2p_prosody(text)).endswith("ro]o$") is False


@pytest.mark.parametrize(
    ("text", "expected_prosody"),
    [
        ("そぞろ雨", "^so[zoro]ame$"),
        ("だいおういか", "^da[io]oika$"),
        ("かませ犬", "^ka[mase]inu$"),
        ("子ども受け", "^ko[domouke$"),
        ("上司受け", "^jo[oshiuke$"),
        ("ろうそく受け", "^ro[osoku]uke$"),
        ("金魚売り", "^ki[Ngyo]uri$"),
        ("青田売り", "^a[otauri$"),
        ("ドル売り", "^do[ruuri$"),
        ("あや織物", "^a[yaori]mono$"),
        ("交ぜ織物", "^ma[zeori]mono$"),
        ("こがたあかいえ蚊", "^ko[gataakaie]ka$"),
        ("はまだら蚊", "^ha[madara]ka$"),
        ("安物買い", "^ya[sumono]kai$"),
        ("本会議", "^ho[Nka]igi$"),
        ("上一段活用", "^ka[miichidaNka]tsuyoo$"),
        ("五段活用", "^go[daNka]tsuyoo$"),
        ("人名用漢字", "^ji[Nmeeyooka]Nji$"),
        ("一枚看板", "^i[chimaika]NbaN$"),
        ("立て看板", "^ta[teka]NbaN$"),
        ("高音部記号", "^ko[ooNbuki]goo$"),
        ("１０種競技", "^ju[clshUkyo]ogi$"),
        ("反面教師", "^ha[NmeNkyo]oshi$"),
        ("消費者金融", "^sho[ohIshaki]Nyuu$"),
        ("企業内組合", "^ki[gyoonaiku]miai$"),
        ("くろごけぐも", "^ku[rogoke]gumo$"),
        ("せあかごけぐも", "^se[akagoke]gumo$"),
        ("あなぐも", "^a[nagumo$"),
        ("放射性元素", "^ho[oshaseege]Nso$"),
        ("本建築", "^ho[Nke]NchIku$"),
        ("定時制高校", "^te[ejiseeko]okoo$"),
        ("買いことば", "^ka[iko]toba$"),
        ("事実婚", "^ji[jitsUkoN$"),
        ("時ざけ", "^to[kizake$"),
        ("突然死", "^to[tsuze]Nshi$"),
        ("起床時刻", "^kI[shooji]koku$"),
        ("同時刻", "^do[oji]koku$"),
        ("到着時刻", "^to[ochakuji]koku$"),
        ("火山性地震", "^ka[zaNseeji]shiN$"),
        ("直下型地震", "^cho[clkagataji]shiN$"),
        ("一斉射撃", "^i[clseesha]geki$"),
        ("六十余州", "^ro[kujuuyo]shuu$"),
        ("掛けじょうゆ", "^ka[kejo]oyu$"),
        ("からしじょうゆ", "^ka[rashijo]oyu$"),
        ("高山植物", "^ko[ozaNshoku]butsu$"),
        ("平安神宮", "^he[eaNjiNgu]u$"),
        ("明治神宮", "^me[ejijiNgu]u$"),
        ("二重生活", "^ni[juuse]ekatsu$"),
        ("直接選挙", "^cho[kUsetsuse]Nkyo$"),
        ("よろず相談", "^yo[rozuso]odaN$"),
        ("４番打者", "^yo[baNda]sha$"),
        ("四番打者", "^yo[baNda]sha$"),
        ("だし巻きたまご", "^da[shimakIta]mago$"),
        ("だし巻き卵", "^da[shimakIta]mago$"),
        ("溶きたまご", "^to[kIta]mago$"),
        ("後日談", "^go[jitsu]daN$"),
        ("基準外賃金", "^ki[juNgaichi]NgiN$"),
        ("Ｘデー", "^e[clkUsu]dee$"),
        ("らせんとじ", "^ra[seNtoji$"),
        ("飲み友達", "^no[mito]modachi$"),
        ("押しかけ女房", "^o[shIkakenyo]oboo$"),
        ("五月人形", "^go[gatsuni]Ngyoo$"),
        ("戦時賠償", "^se[Njiba]ishoo$"),
        ("あんパン", "^a[Npa]N$"),
        ("遊び半分", "^a[sobiha]NbuN$"),
        ("おもしろ半分", "^o[moshiroha]NbuN$"),
        ("面白半分", "^o[moshiroha]NbuN$"),
        ("夜間飛行", "^ya[kaNhi]koo$"),
        ("しろさばふぐ", "^shi[rosabafu]gu$"),
        ("放射性物質", "^ho[oshaseebu]clshItsu$"),
        ("英米文学", "^e[ebeebu]Ngaku$"),
        ("中間報告", "^chu[ukaNho]okoku$"),
        ("とんがり帽子", "^to[Ngaribo]oshi$"),
        ("やりたい放題", "^ya[ritaiho]odai$"),
        ("押しボタン", "^o[shibo]taN$"),
        ("言い間違い", "^i[imachigai$"),
        ("聞き間違い", "^kI[kimachigai$"),
        ("見間違い", "^mi[machigai$"),
        ("足摺岬", "^a[shizurimi]saki$"),
        ("芋掘り名人", "^i[mohorime]ejiN$"),
        ("千一夜物語", "^se[Nichiyamonoga]tari$"),
        ("脅し文句", "^o[doshimo]Nku$"),
        ("決まり文句", "^ki[marimo]Nku$"),
        ("斜め読み", "^na[nameyomi$"),
        ("まとめ読み", "^ma[tomeyomi$"),
        ("えぞしまりす", "^e[zoshima]risu$"),
        ("たいわんりす", "^ta[iwaNri]su$"),
        ("一時利用", "^i[chijiri]yoo$"),
        ("深夜料金", "^shi[Nyaryo]okiN$"),
        ("欧州連合", "^o[oshuure]Ngoo$"),
        ("刺しゅう糸", "^shI[shu]uito$"),
        ("ミシン糸", "^mi[shi]Nito$"),
        ("賛美歌", "^sa[Nbi]ka$"),
        ("讃美歌", "^sa[Nbi]ka$"),
        ("カードキー", "^ka[ado]kii$"),
        ("毒ぐも", "^do[kugumo$"),
        ("いじめっ子", "^i[jimeclko$"),
        ("パリっ子", "^pa[riclko$"),
        ("化学式", "^ka[gaku]shIki$"),
        ("出初め式", "^de[zome]shIki$"),
        ("機関銃", "^kI[ka]Njuu$"),
        ("野蛮人", "^ya[ba]NjiN$"),
        ("決勝戦", "^ke[clsho]oseN$"),
        ("筆立て", "^fu[detate$"),
        ("聞き違い", "^kI[kichigai$"),
        ("読み違い", "^yo[michigai$"),
        ("永代橋", "^e[eta]ibashi$"),
        ("減圧弁", "^ge[Natsu]beN$"),
        ("静脈弁", "^jo[omyaku]beN$"),
        ("人工弁", "^ji[Nko]obeN$"),
        ("赤ペン", "^a[kapeN$"),
        ("筆ペン", "^fu[depeN$"),
        ("よもぎ蒸し", "^yo[mogimushi$"),
        ("てれ屋", "^te[re]ya$"),
        ("照れ屋", "^te[re]ya$"),
        ("溶岩流", "^yo[oga]Nryuu$"),
        ("石焼き芋", "^i[shiyaki]imo$"),
        ("越前がに", "^e[chize]Ngani$"),
        ("上海がに", "^sha[Nha]igani$"),
        ("たかあしがに", "^ta[kaashi]gani$"),
        ("松葉がに", "^ma[tsuba]gani$"),
        ("やしがに", "^ya[shigani$"),
        ("碁盤じま", "^go[baNjima$"),
        ("洪積層", "^ko[oseki]soo$"),
        ("つきよたけ", "^tsU[kiyo]take$"),
        ("なめたけ", "^na[metake$"),
        ("ナメタケ", "^na[metake$"),
        ("てんぐだけ", "^te[Ngu]dake$"),
        ("ポルチーニだけ", "^po[ruchiini]dake$"),
        ("わらいだけ", "^wa[rai]dake$"),
        ("取れたて", "^to[retate$"),
        ("採れたて", "^to[retate$"),
        ("獲れたて", "^to[retate$"),
        ("無洗米", "^mu[seNmai$"),
        ("肉まんじゅう", "^ni[kuma]Njuu$"),
        ("雨もよう", "^a[memo]yoo$"),
        ("荒れもよう", "^a[remo]yoo$"),
        ("荒れ模様", "^a[remo]yoo$"),
        ("芋ようかん", "^i[moyo]okaN$"),
        ("くりようかん", "^ku[riyo]okaN$"),
        ("練りようかん", "^ne[riyo]okaN$"),
    ],
)
def test_compound_nouns_read_as_one_accent_phrase(text: str, expected_prosody: str) -> None:
    """
    複合名詞が、アクセント辞典の見出しか接尾辞の型の例語と同じ1つのアクセント句で読まれ、同じ拍の後で下がることを確認する。
    「立て看板」「人名用漢字」は、前部と後部の品詞の境で2つの句に分かれていたが、「タテカ＼ンバン」「ジンメーヨーカ＼ンジ」と1句で読む。
    「刺しゅう糸」「化学式」は、後部の接尾辞の結合規則が前部によって割れる語で、「シシューイ＼ト」「カガク＼シキ」と読む。
    「越前がに」「てんぐだけ」は、後部が助詞の「が」「に」や「だけ」に分かれていたが、「エチゼ＼ンガニ」「テング＼ダケ」と読む。
    「上一段活用」の「カミ」、「4番打者」の「ヨバン」、「讃美歌」の「サンビカ」のように、1語にしたことで読みも直る語を含む。
    """

    assert "".join(pyopenjtalk.g2p_prosody(text)) == expected_prosody


@pytest.mark.parametrize(
    ("text", "expected_prosody"),
    [
        ("羽黒高等学校", "^ha[gurokootooga]clkoo$"),
        ("羽黒高校", "^ha[guroko]okoo$"),
        ("羽黒高校に通う", "^ha[guroko]okooni#ka[you$"),
        ("今治東高等学校", "^i[ma]bari#hi[gashIkootooga]clkoo$"),
        ("今治東高校", "^i[ma]bari#hi[gashIko]okoo$"),
        ("苫小牧工業高等専門学校", "^to[mako]mai#ko[ogyookootooseNmoNga]clkoo$"),
        ("苫小牧工業高専", "^to[makomaikoogyooko]oseN$"),
        ("苫小牧工業高専の学生", "^to[makomaikoogyooko]oseNno#ga[kUsee$"),
        ("東京都立上野高等学校", "^to[okyooto]ritsu#u[enokootooga]clkoo$"),
        ("東京都立上野高校", "^to[okyooto]ritsu#u[enoko]okoo$"),
        ("三重県立大学に通う", "^mi[ekeNritsuda]igakuni#ka[you$"),
    ],
)
def test_school_names_written_as_koukou_and_kousen(text: str, expected_prosody: str) -> None:
    """
    普段の文章で書く「〇〇高校」「〇〇工業高専」が、正式名の「〇〇高等学校」「〇〇工業高等専門学校」と同じ読みの1語として引かれ、句の分け方とアクセントが崩れないことを確認する。
    辞書に正式名の行しかないと「高校」「高専」が別の語に切れるので、正式名の行から普段の表記の行を作っている。
    「羽黒高校」は正式名と同じく1句で、「高校」の「コ」の後で下がる「ハグロコ＼ーコー」と読み、後ろに「に通う」が続いても同じ形を保つ。
    「今治東高校」は、正式名が起伏型の地名の「今治」で句を分けているので、同じ位置で分けて「イマ＼バリ・ヒガシコ＼ーコー」と読む。
    「苫小牧工業高専」は、正式名では「苫小牧」で句を分けるが、略した表記はひと続きに言うので1句の「トマコマイコーギョーコ＼ーセン」と読む。
    「東京都立上野高校」のように「〇〇立」が付く名前は、正式名も表記も「〇〇立」と学校名で句を分け、「トーキョート＼リツ・ウエノコ＼ーコー」と読む。
    後ろが「大学」だけの「三重県立大学」は、構成語の「三重」「県立」「大学」に分かれず1語の行で引かれ、1句の「ミエケンリツダ＼イガク」と読む。
    """

    assert "".join(pyopenjtalk.g2p_prosody(text)) == expected_prosody


@pytest.mark.parametrize(
    ("text", "expected_prosody"),
    [
        ("大井競馬場", "^o]oi#ke[ebajoo$"),
        ("大井競馬場に行く", "^o]oi#ke[ebajooni#i[ku$"),
        ("船橋競馬場", "^fu[na]bashI#ke[ebajoo$"),
        ("下関競艇場", "^shi[mono]sekI#kyo[oteejoo$"),
        ("中山競馬場", "^na[kayamakeebajoo$"),
    ],
)
def test_racecourse_names_keep_place_name_accent(text: str, expected_prosody: str) -> None:
    """
    競馬場・競艇場の名前で、先頭の地名が起伏型なら地名のアクセントを保ち、地名と平板型の「ケーバジョー」「キョーテージョー」の2つの句に分かれることを確認する。
    「大井競馬場」は、辞書に行がないと「大井」「競馬」「場」が1句の平板型の「オーイケーバジョー」にまとまり、地名の「オ＼ーイ」の核が消えていた。
    1語の行で平板型にしていた「船橋競馬場」も、「フナ＼バシ・ケーバジョー」と読む。
    「中山競馬場」のように先頭の地名が平板型なら、1句のままでも地名のアクセントが保たれるので、1句の平板型で読む。
    """

    assert "".join(pyopenjtalk.g2p_prosody(text)) == expected_prosody


@pytest.mark.parametrize("use_vanilla", [False, True])
@pytest.mark.parametrize(
    ("text", "expected_prosody"),
    [
        ("東京タワーえーとスカイツリー", "^to[okyoota]waa#e[eto#sU[kaitsuri]i$"),
        ("今日はえーと晴れです", "^kyo]owa#e[eto#ha[re]desU$"),
        ("えーと東京", "^e[eto#to[okyoo$"),
    ],
)
def test_filler_starts_new_accent_phrase_before_accent_is_computed(
    text: str, expected_prosody: str, use_vanilla: bool
) -> None:
    """
    文の途中のフィラーが、前の語の句の核を壊さず、フィラーと直後の名詞がそれぞれ新しいアクセント句になることを確認する。
    以前は NJD が「えーと」を前の句につないで句全体の核を計算した後で、後処理が「スカイ」から句を切っていた。
    そのため「東京タワーえーとスカイツリー」では、核 (15) が切った後の句 (「東京タワーえーと」の10モーラ) の外に残り、「タワー」の後で下がらなかった。
    句の切れ目は核の計算の前に決めるので、単独の「東京タワー」と同じ「トーキョータ＼ワー」になる。
    """

    assert "".join(pyopenjtalk.g2p_prosody(text, use_vanilla=use_vanilla)) == expected_prosody


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
        ("3遍", "^sa[Nbe]N$", [3]),
        ("2遍", "^ni[he]N$", [2]),
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
    回数の「3遍」「2遍」は辞典の「サンベ＼ン」「ニヘ＼ン」のように、「遍」の最初の拍の直後で下げる。
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
    ("text", "expected_prosody", "expected_accents"),
    [
        # 「第」や「約」に11〜19が続くときも、1桁の「第3番」と同じく数詞から新しいアクセント句を始める
        ("第11番", "^da]i#ju[uichi]baN$", [1, 4]),
        ("第13番", "^da]i#ju[usa]NbaN$", [1, 3]),
        ("第11幕", "^da]i#ju[uichi]maku$", [1, 4]),
        ("第十二幕", "^da]i#ju[uni]maku$", [1, 3]),
        ("第11日", "^da]i#ju[uichinichi$", [1, 6]),
        ("第15日", "^da]i#ju]u#go]nichi$", [1, 1, 1]),
        ("約11人", "^ya]ku#ju[uichi]niN$", [1, 4]),
        # 辞書や数詞の処理が1語にした日付の語 (「一日」「二日」「千日」) に続く「目」も、「十一日目」と同じく尾高型で結合する
        ("1日目", "^i[chinichime$", [5]),
        ("2日目", "^fU[tsukame$", [4]),
        ("24日目", "^ni]juu#yo[clkame$", [1, 4]),
        ("千日目", "^se[Nnichime$", [5]),
        ("3日目の朝", "^mi[clkame]no#a]sa$", [4, 1]),
        # 「九日」は「ココノカ＼」と尾高型で読む
        ("9日", "^ko[konoka$", [4]),
        # 順番を表す「第2日」「第10日」の「ニチ」も、「十二日」と同じく一・二・六・七・八・十の後で尾高型にする
        ("第2日", "^da]i#ni[nichi$", [1, 3]),
        ("第10日", "^da]i#ju[unichi$", [1, 4]),
        # 「五」に続く「題」「段」「問」は、「本」「枚」と同じく数詞の桁にかかわらず後部を平板型にする
        ("5題", "^go[dai$", [0]),
        ("15題", "^ju]u#go[dai$", [1, 0]),
        ("25題", "^ni]juu#go[dai$", [1, 0]),
        ("5段", "^go[daN$", [0]),
        ("5問", "^go[moN$", [0]),
        ("3問", "^sa]NmoN$", [1]),
        # 「三」に続く「番」も平板型にする
        # 11〜19を1句で読む「13番」は「ジューサ＼ンバン」のまま核を置く
        ("3番", "^sa[NbaN$", [0]),
        ("23番", "^ni]juu#sa[NbaN$", [1, 0]),
        ("第3番", "^da]i#sa[NbaN$", [1, 0]),
        ("13番", "^ju[usa]NbaN$", [3]),
        # 「場所」は和語の「ヒト」「フタ」の後だけ前部末型で、3以上の数の後は平板型にする
        ("1場所", "^hI[to]basho$", [2]),
        ("2場所", "^fU[ta]basho$", [2]),
        ("3場所", "^sa[Nbasho$", [0]),
        ("5場所", "^go[basho$", [0]),
        ("10場所", "^ju[ubasho$", [0]),
        ("20場所", "^ni[juubasho$", [0]),
        # 袖丈の「分袖」は「分」(「ブ」) と「袖」を1つの句で読み、「ブ」の後で下げる
        ("1分袖", "^i[chibu]sode$", [3]),
        ("10分袖", "^ju[ubu]sode$", [3]),
        ("それは何分袖です。", "^so[rewa#na[Nbu]sodedesU_$", [0, 3, 0]),
        # 「尺」は二・五・六の後、「節」は六の後、「足」は一の後で尾高型にする
        # 11〜19 の一の位も同じ型にする
        ("2尺", "^ni[shaku$", [3]),
        ("15尺", "^ju]u#go[shaku$", [1, 3]),
        ("16尺", "^ju[urokUshaku$", [6]),
        ("3尺", "^sa]Njaku$", [1]),
        ("6節", "^ro[kUsetsu$", [4]),
        ("16節", "^ju[urokUsetsu$", [6]),
        ("1足", "^i[clsoku$", [4]),
        # 1語で数を表す「十」「百」「千」「何」に続く「人」も、2桁以上の数と同じく前部末型にする
        ("10人", "^ju]uniN$", [1]),
        ("100人", "^hya[ku]niN$", [2]),
        ("1000人", "^se]NniN$", [1]),
        ("何人", "^na]NniN$", [1]),
        # 「幕目」は「日目」と同じく、別のノードに分かれた「目」まで尾高型で結合する
        # 数詞は「イチ」「ヨン」の読みのまま保つ
        ("7幕目", "^na[namakume$", [5]),
        ("1幕目", "^i[chimakume$", [5]),
        ("4幕目", "^yo[Nmakume$", [5]),
        # 「選」は数詞の後で前部末型にし、「十五選」は「十五分」と同じく「ジュ＼ー・ゴ＼セン」と2つの句に分ける
        ("3選", "^sa]NseN$", [1]),
        ("5選", "^go]seN$", [1]),
        ("15選", "^ju]u#go]seN$", [1, 1]),
        # 「十五度目」は一の位が「ゴ」でも1つのアクセント句で読む
        ("15度目", "^ju[ugodome$", [5]),
        # 「二十選」「二十等」は「二十回」と同じく、「ジュッ」の前の「ジュ」の後で下げる
        ("20選", "^ni[ju]clseN$", [2]),
        ("20等", "^ni[ju]cltoo$", [2]),
        # 名詞,一般 の行で解析される「機種」「地区」は、数詞の後で助数詞の核を保ち、「階」と「級」に分かれる「階級」は「カ」の後で下げる
        ## 「1階級」「8階級」は促音にせず、「3階級」は濁らずに読む
        ("1機種", "^i[chIki]shu$", [3]),
        ("1地区", "^i[chIchi]ku$", [3]),
        ("1階級", "^i[chIka]ikyuu$", [3]),
        ("3階級", "^sa[Nka]ikyuu$", [3]),
        ("8階級", "^ha[chIka]ikyuu$", [3]),
        # 「万円」は、1桁の数と単独の「十」「百」「千」の後では平板型で読む
        ("2万円", "^ni[maNeN$", [0]),
        ("9万円", "^kyu[umaNeN$", [0]),
        ("10万円", "^ju[umaNeN$", [0]),
        ("十万円", "^ju[umaNeN$", [0]),
        ("100万円", "^hya[kumaNeN$", [0]),
        ("1000万円", "^se[NmaNeN$", [0]),
        # 位の字に数が付く「二十」「何百」「一千」と、「数十」「十数」の後では「マ」の後で下げる
        ("20万円", "^ni[juuma]NeN$", [4]),
        ("30万円", "^sa[Njuuma]NeN$", [5]),
        ("数十万円", "^su[ujuuma]NeN$", [5]),
        ("十数万円", "^ju[usuuma]NeN$", [5]),
        ("何百万円", "^na[Nbyakuma]NeN$", [5]),
        ("一千万円", "^i[clseNma]NeN$", [5]),
        # 「二十」の後で句を分ける「二十五万円」は、後ろの句を1桁の数と同じく平板型で読む
        ("25万円", "^ni]juu#go[maNeN$", [1, 0]),
    ],
)
def test_g2p_prosody_counter_accents_after_numerals(
    text: str, expected_prosody: str, expected_accents: list[int], use_vanilla: bool
) -> None:
    """
    数詞に続く助数詞のアクセントを、一般的なアクセント辞典の数詞と助数詞の表に合わせることを確認する。
    「第11番」は「ダ＼イ・ジューイチ＼バン」のように「第」の後で句を切り、11〜19 も1桁の数と同じ句の分け方にする。
    独自の後処理を無効にした場合も、C の数詞処理だけで同じ区切りと核になることを確認する。
    """

    prosody = pyopenjtalk.g2p_prosody(text, use_vanilla=use_vanilla)
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
        ("3遍", "サンベン"),
        ("4遍", "ヨンヘン"),
        ("6遍", "ロッペン"),
        ("1000遍", "センベン"),
        ("何遍", "ナンベン"),
        ("3編", "サンペン"),
    ],
)
def test_numeral_counter_readings(text: str, expected_reading: str, use_vanilla: bool) -> None:
    """
    NHK アクセント辞典の付録の数詞と助数詞の表に載っている読みを確認する。
    数詞と助数詞が組み合わさったときの促音化と濁音化を確認する。
    「人前」「分袖」が複数の形態素へ分かれた場合も、助数詞全体の読みを確認する。
    回数の「遍」は「本」と同じく三・千・何の後で濁って「サンベン」「センベン」「ナンベン」と読み、「三編」の「サンペン」のように半濁音にはしない。
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
        # 「同一」「統一」の「一」は数詞ではないので、助数詞として読む「機種」「地区」「階級」が続いても「イツ」と読む
        ("同一機種", "ドーイツキシュ"),
        ("同一地区", "ドーイツチク"),
        ("同一階級", "ドーイツカイキュー"),
        ("統一地区", "トーイツチク"),
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


@pytest.mark.parametrize(
    "text,expected_prosody",
    [
        ("工業地区", "^ko[ogyoochi]ku$"),
        ("文教地区", "^bu[Nkyoochi]ku$"),
        ("居場所", "^i[basho$"),
    ],
)
def test_compound_nouns_ending_with_chiku_and_basho(text: str, expected_prosody: str) -> None:
    """
    後部に「地区」「場所」が付く複合名詞が、数詞の後の助数詞としてではなく、複合語の型で読まれることを確認する。
    「工業地区」「文教地区」は「地区」の核を保つ「コーギョーチ＼ク」、「居場所」は平板の「イバショ」と読む。
    """

    assert "".join(pyopenjtalk.g2p_prosody(text)) == expected_prosody


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


def test_counter_after_decimal_keeps_plain_reading() -> None:
    """
    小数の「1.1本」の「本」は、整数の「1本」のような促音化を受けず「イッテンイチホン」と読まれることを確認する。
    """

    assert pyopenjtalk.g2p("1.1本", kana=True) == "イッテンイチホン"


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


@pytest.mark.parametrize(
    "text,reading",
    [
        ("6kW", "ロッキロワット"),
        ("6kHz", "ロッキロヘルツ"),
        ("6kWh", "ロッキロワットアワー"),
        ("1kW", "イチキロワット"),
        ("8kW", "ハチキロワット"),
    ],
)
def test_kilo_unit_letters_geminate_six(text: str, reading: str) -> None:
    """
    大文字を含む英字の単位の「ｋＷ」「ｋＨｚ」でも、片仮名の「キロワット」と同じく6を「ロッ」と促音にして「ロッキロワット」と読むことを確認する。
    これまでは助数詞の表に小文字の「ｋｗ」しかなく、「ロクキロワット」と読んでいた。
    """

    assert pyopenjtalk.g2p(text, kana=True) == reading


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


@pytest.mark.parametrize("mora_count,accent", [(55, 50), (55, 49), (55, 0), (55, 1), (105, 100)])
def test_long_registered_word_keeps_pitch_fall_after_label_clipping(
    tmp_path: Path, mora_count: int, accent: int
) -> None:
    """
    ユーザー辞書に55モーラで50モーラ目に核を持つ語を登録すると、50モーラ目まで高く、後ろの5モーラは低く出力されることを確認する。
    HTS ラベルの核とモーラ位置が49に丸められても下降位置を保ち、平板・頭高・100モーラ目の核と3種類の韻律形式も検証する。
    """

    user_csv = tmp_path / "long_word.csv"
    user_dic = tmp_path / "long_word.dic"
    reading = "カ" * mora_count
    user_csv.write_text(
        f"長音検査,1345,1345,-5000,名詞,一般,*,*,*,*,長音検査,{reading},{reading},{accent}/{mora_count},C1\n",
        encoding="utf-8",
    )
    pyopenjtalk.mecab_dict_index(str(user_csv), str(user_dic))
    jtalk = pyopenjtalk.OpenJTalk(
        dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR,
        userdic=str(user_dic).encode(),
        userdic_reading_protection=[True],
    )
    expected = [
        "High"
        if (index == 1 if accent == 1 else index >= 2 and (accent == 0 or index <= accent))
        else "Low"
        for index in range(1, mora_count + 1)
    ]
    features = pyopenjtalk.run_frontend("長音検査", jtalk=jtalk)
    assert (features[0]["acc"], features[0]["mora_size"]) == (accent, mora_count)
    mapping = pyopenjtalk.g2p_mapping_prosody("長音検査", jtalk=jtalk)
    assert [
        phone["pitch"]
        for entry in mapping
        for phone in entry["phonemes"]
        if phone["kind"] == "Phoneme" and phone["phoneme"] == "a"
    ] == expected
    assert [
        token
        for token in pyopenjtalk.g2p_prosody("長音検査", format="Prefix", jtalk=jtalk)
        if token.endswith("_a")
    ] == ["H_a" if pitch == "High" else "L_a" for pitch in expected]
    assert [
        token
        for token in pyopenjtalk.g2p_prosody("長音検査", format="Numeric", jtalk=jtalk)
        if token.startswith("a:")
    ] == ["a:1" if pitch == "High" else "a:0" for pitch in expected]
    tokens = pyopenjtalk.g2p_prosody("長音検査", jtalk=jtalk)
    assert tokens.count("]") == int(accent > 0)
    if accent > 0:
        assert tokens[: tokens.index("]")].count("a") == accent


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
