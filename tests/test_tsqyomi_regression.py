"""tsqyomi の実推論テスト。"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

import pyopenjtalk
import pyopenjtalk.tsqyomi as tsqyomi
import pyopenjtalk.tsqyomi.diagnostics as tsqyomi_diagnostics
import pyopenjtalk.tsqyomi.inference as tsqyomi_inference
import pyopenjtalk.tsqyomi.model as tsqyomi_model
from pyopenjtalk.tsqyomi.inference import select_mecab_features_with_tsqyomi
from pyopenjtalk.types import MeCabMorph


@dataclass(frozen=True)
class _TargetExpectation:
    """
    1対象について期待する診断と発音。

    tsqyomi が診断へ記録した表層について、適用結果と発音を検証する。
    位置は text 内の surface 出現番号 occurrence で指定する。
    同一 surface が複数ある場合は 0, 1, … と数える。
    char_span は前置き付き文本など occurrence だけでは足りない症例向けの上書き。
    """

    surface: str
    expected_pronunciation: str | None = None
    expected_outcome: str = "applied"
    expected_segment_text: str | None = None
    was_preserved: bool = False
    occurrence: int = 0
    char_span: tuple[int, int] | None = None


def _resolve_char_span(text: str, target: _TargetExpectation) -> tuple[int, int]:
    """TargetExpectation から text 上の char_span を解決する。"""

    if target.char_span is not None:
        assert text[target.char_span[0] : target.char_span[1]] == target.surface
        return target.char_span

    start = 0
    index = text.index(target.surface, start)
    for _ in range(target.occurrence):
        start = index + 1
        index = text.index(target.surface, start)
    span = (index, index + len(target.surface))
    assert text[span[0] : span[1]] == target.surface
    return span


def _resolve_targets(
    text: str,
    targets: tuple[_TargetExpectation, ...],
) -> tuple[_TargetExpectation, ...]:
    """case.text から各 target の char_span を解決する。"""

    return tuple(replace(target, char_span=_resolve_char_span(text, target)) for target in targets)


@dataclass(frozen=True)
class _Case:
    """1文の読み推論回帰症例。"""

    text: str
    expected_kana: str
    targets: tuple[_TargetExpectation, ...] = ()
    expect_no_diagnostics: bool = False


# revision 1157e36e (v3/model.onnx) で CPU 推論した期待値
## `_TargetExpectation` は tsqyomi が診断記録に残した、読み選択または保護が成立した表層を検証する
## `expected_kana` は v3 の現状出力を固定する（未達症例では誤った全文読みを含む）
## コメントアウトした `_TargetExpectation` は本来の期待読みで、達成後に有効化する TODO
## TODO 文言: 語彙未収載かつ文脈上ほんとうに競合読みがある表層だけ「v3 メタデータに「表層」を足したら `_TargetExpectation` でも検証する」
## 競合読みが文脈上存在せず辞書既定で到達済みの表層は TODO にしない
_CASES: tuple[_Case, ...] = (
    _Case(
        text="人気の店です。",
        expected_kana="ニンキノミセデス。",
        targets=(
            _TargetExpectation(
                surface="人気",
                expected_pronunciation="ニンキ",
                expected_segment_text="人気の店です。",
            ),
        ),
    ),
    _Case(
        text="人気のない店",
        expected_kana="ヒトケノナイミセ",
        targets=(
            _TargetExpectation(
                surface="人気",
                expected_pronunciation="ヒトケ",
                expected_segment_text="人気のない店",
            ),
        ),
    ),
    _Case(
        text="休日で人気の少ない茶室に入ると、座卓には最中が置かれていた。",
        expected_kana="キュージツデヒトケノスクナイチャシツニハイルト、ザタクニワモナカガオカレテイタ。",
        targets=(
            _TargetExpectation(
                surface="人気",
                expected_pronunciation="ヒトケ",
            ),
            # TODO: v4 で「入る」を読み分け対象に追加したら `_TargetExpectation` でも検証する
            # _TargetExpectation(
            #     surface="入る",
            #     expected_pronunciation="ハイル",
            # ),
            _TargetExpectation(
                surface="最中",
                expected_pronunciation="モナカ",
            ),
        ),
    ),
    _Case(
        text="もうこの程度で十分です",
        expected_kana="モーコノテードデジューブンデス",
        targets=(
            _TargetExpectation(
                surface="十分",
                expected_pronunciation="ジューブン",
            ),
        ),
    ),
    _Case(
        text="この踊りは私の一番の十八番です",
        expected_kana="コノオドリワワタシノイチバンノオハコデス",
        targets=(
            _TargetExpectation(
                surface="十八番",
                expected_pronunciation="オハコ",
            ),
        ),
    ),
    _Case(
        text="何人いますか",
        expected_kana="ナンニンイマスカ",
        targets=(
            _TargetExpectation(
                surface="何人",
                expected_pronunciation="ナンニン",
                expected_outcome="dictionary_default_protected",
                was_preserved=True,
            ),
        ),
    ),
    _Case(
        text="会議を行った",
        expected_kana="カイギヲオコナッタ",
        targets=(
            _TargetExpectation(
                surface="行っ",
                expected_pronunciation="オコナッ",
            ),
        ),
    ),
    _Case(
        text="駅へ行った",
        expected_kana="エキエイッタ",
        targets=(
            _TargetExpectation(
                surface="行っ",
                expected_pronunciation="イッ",
            ),
        ),
    ),
    _Case(
        text="学校に通っている",
        expected_kana="ガッコーニカヨッテイル",
        targets=(
            _TargetExpectation(
                surface="通っ",
                expected_pronunciation="カヨッ",
            ),
        ),
    ),
    _Case(
        text="門を通って入る",
        expected_kana="モンヲトーッテハイル",
        targets=(
            _TargetExpectation(
                surface="通っ",
                expected_pronunciation="トーッ",
            ),
            # TODO: v4 で「入る」を読み分け対象に追加したら `_TargetExpectation` でも検証する
            # _TargetExpectation(
            #     surface="入る",
            #     expected_pronunciation="ハイル",
            # ),
        ),
    ),
    _Case(
        text="この通りで待つ",
        expected_kana="コノトーリデマツ",
        targets=(
            _TargetExpectation(
                surface="通り",
                expected_pronunciation="トーリ",
            ),
        ),
    ),
    _Case(
        text="予想通りで驚いた",
        expected_kana="ヨソードーリデオドロイタ",
        targets=(
            _TargetExpectation(
                surface="通り",
                expected_pronunciation="ドーリ",
            ),
        ),
    ),
    _Case(
        text="商売上",
        expected_kana="ショーバイジョー",
        targets=(
            _TargetExpectation(
                surface="上",
                expected_pronunciation="ジョー",
            ),
        ),
    ),
    _Case(
        text="÷÷÷÷人気",
        expected_kana="÷÷÷÷ニンキ",
        targets=(
            _TargetExpectation(
                surface="人気",
                expected_pronunciation="ニンキ",
            ),
        ),
    ),
    _Case(
        text="あと一寸です",
        expected_kana="アトチョットデス",
        targets=(
            _TargetExpectation(
                surface="一寸",
                expected_pronunciation="チョット",
            ),
        ),
    ),
    _Case(
        text="いじけるなんて大人気ないな君は。",
        expected_kana="イジケルナンテオトナゲナイナキミワ。",
        targets=(
            _TargetExpectation(
                surface="大人気",
                expected_pronunciation="オトナゲ",
            ),
        ),
    ),
    _Case(
        text="この中で何曲歌える？",
        expected_kana="コノナカデナンキョクウタエル？",
        targets=(
            _TargetExpectation(
                surface="中",
                expected_pronunciation="ナカ",
            ),
            _TargetExpectation(
                surface="何",
                expected_pronunciation="ナン",
                expected_outcome="dictionary_default_protected",
                was_preserved=True,
            ),
        ),
    ),
    _Case(
        text="人の金で食う飯は美味い。",
        expected_kana="ヒトノカネデクウメシワウマイ。",
        targets=(
            _TargetExpectation(
                surface="金",
                expected_pronunciation="カネ",
            ),
        ),
    ),
    _Case(
        text="仕事の最中に最中を食べるな！",
        expected_kana="シゴトノサイチューニモナカヲタベルナ！",
        targets=(
            _TargetExpectation(
                surface="最中",
                expected_pronunciation="サイチュー",
            ),
            _TargetExpectation(
                surface="最中",
                occurrence=1,
                expected_pronunciation="モナカ",
            ),
        ),
    ),
    _Case(
        # TODO: 「大分県」を「大分」にしても読めるようにしたい (現状「県」の suffix がないと読めない)
        text="大分県にもう大分長いこと住んでいるな。",
        expected_kana="オーイタケンニモーダイブナガイコトスンデイルナ。",
        targets=(
            _TargetExpectation(
                surface="大分",
                expected_pronunciation="オーイタ",
            ),
            _TargetExpectation(
                surface="大分",
                occurrence=1,
                expected_pronunciation="ダイブ",
            ),
        ),
    ),
    _Case(
        text="彼に敬意を表します。",
        expected_kana="カレニケーイヲヒョーシマス。",
        targets=(
            _TargetExpectation(
                surface="表し",
                expected_pronunciation="ヒョーシ",
            ),
        ),
    ),
    _Case(
        text="新しく金が発見された地に赴くにも金がかかる。",
        expected_kana="アタラシクキンガハッケンサレタチニオモムクニモカネガカカル。",
        targets=(
            _TargetExpectation(
                surface="金",
                expected_pronunciation="キン",
            ),
            _TargetExpectation(
                surface="金",
                occurrence=1,
                expected_pronunciation="カネ",
            ),
        ),
    ),
    _Case(
        text="泥を被るという被害を被った。",
        expected_kana="ドロヲカブルトイウヒガイヲカブッタ。",
        targets=(
            _TargetExpectation(
                surface="被る",
                expected_pronunciation="カブル",
            ),
            # TODO: 期待は「コウムッ」だが現状「カブッ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="被っ",
            #     expected_pronunciation="コウムッ",
            # ),
        ),
    ),
    _Case(
        text="竹田はかつて岡藩の城下町であった。",
        expected_kana="タケタワカツテオカハンノジョーカマチデアッタ。",
        targets=(
            _TargetExpectation(
                surface="竹田",
                expected_pronunciation="タケタ",
            ),
        ),
    ),
    _Case(
        text="素振りをする素振りを見せた。",
        expected_kana="スブリヲスルソブリヲミセタ。",
        targets=(
            _TargetExpectation(
                surface="素振り",
                expected_pronunciation="スブリ",
            ),
            _TargetExpectation(
                surface="素振り",
                occurrence=1,
                expected_pronunciation="ソブリ",
            ),
        ),
    ),
    _Case(
        text="角の生えた鬼に向かって角が立たない言い回し。",
        expected_kana="ツノノハエタオニニムカッテカドガタタナイイーマワシ。",
        targets=(
            _TargetExpectation(
                surface="角",
                expected_pronunciation="ツノ",
            ),
            _TargetExpectation(
                surface="角",
                occurrence=1,
                expected_pronunciation="カド",
            ),
        ),
    ),
    _Case(
        text="辛いことだが仕方がない。",
        expected_kana="ツライコトダガシカタガナイ。",
        targets=(
            _TargetExpectation(
                surface="辛い",
                expected_pronunciation="ツライ",
            ),
        ),
    ),
    _Case(
        text="深夜の路地は人気が無くて怖い。",
        expected_kana="シンヤノロジワヒトケガナクテコワイ。",
        targets=(
            _TargetExpectation(
                surface="人気",
                expected_pronunciation="ヒトケ",
            ),
        ),
    ),
    _Case(
        text="金の時計を買うために、一生懸命に金を貯めた。",
        expected_kana="キンノトケーヲカウタメニ、イッショーケンメーニカネヲタメタ。",
        targets=(
            _TargetExpectation(
                surface="金",
                expected_pronunciation="キン",
            ),
            _TargetExpectation(
                surface="金",
                occurrence=1,
                expected_pronunciation="カネ",
            ),
        ),
    ),
    _Case(
        text="カブトムシの立派な角に止まった小さな虫を、指で軽く弾く。",
        expected_kana="カブトムシノリッパナツノニトマッタチーサナムシヲ、ユビデカルクハジク。",
        targets=(
            _TargetExpectation(
                surface="角",
                expected_pronunciation="ツノ",
            ),
            _TargetExpectation(
                surface="弾く",
                expected_pronunciation="ハジク",
            ),
        ),
    ),
    _Case(
        text="庭に植えた紅葉の木が立派に育ってきた。",
        expected_kana="ニワニウエタモミジノキガリッパニソダッテキタ。",
        targets=(
            _TargetExpectation(
                surface="紅葉",
                expected_pronunciation="モミジ",
            ),
            _TargetExpectation(
                surface="木",
                expected_pronunciation="キ",
            ),
        ),
    ),
    _Case(
        text="診療は月・水・金です。",
        expected_kana="シンリョーワゲツ・スイ・キンデス。",
        targets=(
            _TargetExpectation(
                surface="月",
                expected_pronunciation="ゲツ",
            ),
            _TargetExpectation(
                surface="水",
                expected_pronunciation="スイ",
            ),
            _TargetExpectation(
                surface="金",
                expected_pronunciation="キン",
            ),
        ),
    ),
    _Case(
        text="会議は火・木に開きます。",
        expected_kana="カイギワカ・モクニアキマス。",
        targets=(
            _TargetExpectation(
                surface="火",
                expected_pronunciation="カ",
            ),
            _TargetExpectation(
                surface="木",
                expected_pronunciation="モク",
            ),
            # TODO: 本来は「ヒラキ」だが現状「アキ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="開き",
            #     expected_pronunciation="ヒラキ",
            # ),
        ),
    ),
    _Case(
        text="営業は土・日です。",
        expected_kana="エーギョーワド・ニチデス。",
        targets=(
            _TargetExpectation(
                surface="土",
                expected_pronunciation="ド",
            ),
            _TargetExpectation(
                surface="日",
                expected_pronunciation="ニチ",
            ),
        ),
    ),
    _Case(
        text="誕生月にお祝いします。",
        expected_kana="タンジョーズキニオイワイシマス。",
        targets=(
            # NOTE: 連語エントリ「誕生月」で辞書が直接「ズキ」を返すため、tsqyomi の介入対象にならない
            # 現状、辞書へ列挙できない造語の「〜月」は候補列挙の表記不一致で未解決のまま残ると思われる
        ),
    ),
    _Case(
        text="締め切り月に提出します。",
        expected_kana="シメキリゲツニテーシュツシマス。",
        targets=(
            # TODO: 本来は「ヅキ」だが辞書単独では「ツキ」、現状 tsqyomi は「ゲツ」を選ぶ
            # _TargetExpectation(
            #     surface="月",
            #     expected_pronunciation="ヅキ",
            # ),
        ),
    ),
    _Case(
        text="パーティー日は会場を貸し切ります。",
        expected_kana="パーティービワカイジョーヲカシキリマス。",
        targets=(
            _TargetExpectation(
                surface="日",
                expected_pronunciation="ビ",
                expected_outcome="dictionary_default_protected",
                was_preserved=True,
            ),
        ),
    ),
    _Case(
        text="サービス日はポイントが二倍になります。",
        expected_kana="サービスビワポイントガニバイニナリマス。",
        targets=(
            _TargetExpectation(
                surface="日",
                expected_pronunciation="ビ",
                expected_outcome="dictionary_default_protected",
                was_preserved=True,
            ),
        ),
    ),
    _Case(
        text="外来日は休みです。",
        expected_kana="ガイライビワヤスミデス。",
        targets=(
            _TargetExpectation(
                surface="日",
                expected_pronunciation="ビ",
                expected_outcome="dictionary_default_protected",
                was_preserved=True,
            ),
        ),
    ),
    _Case(
        text="定休日は木・金となります。",
        expected_kana="テーキュービワモク・キントナリマス。",
        targets=(
            _TargetExpectation(
                surface="日",
                expected_pronunciation="ビ",
                expected_outcome="dictionary_default_protected",
                was_preserved=True,
            ),
            _TargetExpectation(
                surface="木",
                expected_pronunciation="モク",
            ),
            _TargetExpectation(
                surface="金",
                expected_pronunciation="キン",
            ),
        ),
    ),
    _Case(
        text="漫画家です。",
        expected_kana="マンガカデス。",
        targets=(
            _TargetExpectation(
                surface="家",
                expected_pronunciation="カ",
                expected_outcome="dictionary_default_protected",
                was_preserved=True,
            ),
        ),
    ),
    _Case(
        text="専門家です。",
        expected_kana="センモンカデス。",
        targets=(
            _TargetExpectation(
                surface="家",
                expected_pronunciation="カ",
                expected_outcome="dictionary_default_protected",
                was_preserved=True,
            ),
        ),
    ),
    _Case(
        text="山田家です。",
        expected_kana="ヤマダケデス。",
        targets=(
            _TargetExpectation(
                surface="家",
                expected_pronunciation="ケ",
                expected_outcome="dictionary_default_protected",
                was_preserved=True,
            ),
        ),
    ),
    _Case(
        text="将軍家です。",
        expected_kana="ショーグンケデス。",
        targets=(
            _TargetExpectation(
                surface="家",
                expected_outcome="no_exact_morph_range",
            ),
        ),
    ),
    _Case(
        text="子宝に恵まれ、代々家が栄えるように",
        expected_kana="コダカラニメグマレ、ダイダイイエガサカエルヨーニ",
        targets=(
            _TargetExpectation(
                surface="家",
                expected_pronunciation="イエ",
            ),
        ),
    ),
    _Case(
        text="月が明るい夜です。",
        expected_kana="ツキガアカルイヨルデス。",
        targets=(
            _TargetExpectation(
                surface="月",
                expected_pronunciation="ツキ",
            ),
        ),
    ),
    _Case(
        text="1月は寒いです。",
        expected_kana="イチガツワサムイデス。",
        targets=(
            # 数字と一体化した「1月」は月だけの形態素範囲を持たないため、全文読みと診断結果を固定
            _TargetExpectation(
                surface="月",
                expected_outcome="no_exact_morph_range",
            ),
        ),
    ),
    _Case(
        text="日が長くなりました。",
        expected_kana="ヒガナガクナリマシタ。",
        targets=(
            _TargetExpectation(
                surface="日",
                expected_pronunciation="ヒ",
            ),
        ),
    ),
    _Case(
        text="1日で終わります。",
        expected_kana="イチニチデオワリマス。",
        targets=(
            _TargetExpectation(
                surface="日",
                expected_pronunciation="ニチ",
            ),
        ),
    ),
    _Case(
        text="あちらの方がお見えになった理由は、皆まで言わずとも分かる。",
        expected_kana="アチラノホーガオミエニナッタリユーワ、ミナマデイワズトモワカル。",
        targets=(
            # TODO: 本来は「カタ」だが現状「ホー」が選ばれてしまう
            # _TargetExpectation(
            #     surface="方",
            #     expected_pronunciation="カタ",
            # ),
        ),
    ),
    _Case(
        text="この方はどちらの方からお越しになりましたか？",
        expected_kana="コノカタワドチラノカタカラオコシニナリマシタカ？",
        targets=(
            _TargetExpectation(
                surface="方",
                expected_pronunciation="カタ",
            ),
            # TODO: 本来は「ホー」だが現状「カタ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="方",
            #     occurrence=1,
            #     expected_pronunciation="ホー",
            # ),
        ),
    ),
    _Case(
        text="この絵は筆を使わずに描いたの？",
        expected_kana="コノエワフデヲツカワズニエガイタノ？",
        targets=(
            # TODO: 本来は「カイ」だが現状「エガイ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="描い",
            #     expected_pronunciation="カイ",
            # ),
        ),
    ),
    _Case(
        text="この作家の心理描写の描きかたには定評がある",
        expected_kana="コノサッカノシンリビョーシャノエガキカタニワテーヒョーガアル",
        targets=(
            _TargetExpectation(
                surface="描き",
                expected_pronunciation="エガキ",
            ),
        ),
    ),
    _Case(
        text="この美しい紅葉の絶景を独り占めすることなど、何人たりとも許されない。",
        expected_kana="コノウツクシイコーヨーノゼッケーヲヒトリジメスルコトナド、ナンニンタリトモユルサレナイ。",
        targets=(
            _TargetExpectation(
                surface="紅葉",
                expected_pronunciation="コーヨー",
            ),
            # NOTE: 「何人（ナンピト）」は登場頻度が稀で人間でも読み間違えるため、現時点では読み分け対象に含めていない
        ),
    ),
    _Case(
        text="ギターを弾く銀髪の彼は何人だ？",
        expected_kana="ギターヲヒクギンパツノカレワナンニンダ？",
        targets=(
            _TargetExpectation(
                surface="弾く",
                expected_pronunciation="ヒク",
            ),
            # TODO: 本来は「ナニジン」だが現状「ナンニン」が選ばれてしまう
            # _TargetExpectation(
            #     surface="何人",
            #     expected_pronunciation="ナニジン",
            # ),
        ),
    ),
    _Case(
        text="スケートの羽生選手と将棋の羽生棋士。",
        expected_kana="スケートノハニューセンシュトショーギノハニューキシ。",
        targets=(
            _TargetExpectation(
                surface="羽生",
                expected_pronunciation="ハニュー",
            ),
            # TODO: 本来は「ハブ」だが現状「ハニュー」が選ばれてしまう
            # _TargetExpectation(
            #     surface="羽生",
            #     occurrence=1,
            #     expected_pronunciation="ハブ",
            # ),
        ),
    ),
    _Case(
        text="予約なしでも入れるホテルを駅前で探した。",
        expected_kana="ヨヤクナシデモイレルホテルヲエキマエデサガシタ。",
        targets=(
            # TODO: v3 メタデータに「入れる」を足したら `_TargetExpectation` でも検証する
            # _TargetExpectation(
            #     surface="入れる",
            #     expected_pronunciation="ハイレル",
            # ),
        ),
    ),
    _Case(
        text="京都府宇治市の小倉駅ですか、それとも福岡県北九州市の小倉駅ですか。",
        expected_kana="キョートフウジシノコクラエキデスカ、ソレトモフクオカケンキタキューシューシノコクラエキデスカ。",
        targets=(
            # TODO: 本来は「オグラ」だが現状「コクラ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="小倉",
            #     expected_pronunciation="オグラ",
            # ),
            _TargetExpectation(
                surface="小倉",
                occurrence=1,
                expected_pronunciation="コクラ",
            ),
        ),
    ),
    _Case(
        text="人気の絶えない観光地だが、一本裏道に入ると急に人気がなくなる。",
        expected_kana="ニンキノタエナイカンコーチダガ、イッポンウラミチニハイルトキューニヒトケガナクナル。",
        targets=(
            _TargetExpectation(
                surface="人気",
                expected_pronunciation="ニンキ",
            ),
            # TODO: v4 で「入る」を読み分け対象に追加したら `_TargetExpectation` でも検証する
            # _TargetExpectation(
            #     surface="入る",
            #     expected_pronunciation="ハイル",
            # ),
            _TargetExpectation(
                surface="人気",
                occurrence=1,
                expected_pronunciation="ヒトケ",
            ),
        ),
    ),
    _Case(
        text="八戸は県内第二の人口を有しており、家屋が八戸しか無いわけでは断じて無い。",
        expected_kana="ハチコワケンナイダイニノジンコーヲユーシテオリ、カオクガハチコシカナイワケデワダンジテナイ。",
        targets=(
            # TODO: 本来は「ハチノヘ」だが現状「ハチコ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="八戸",
            #     expected_pronunciation="ハチノヘ",
            # ),
            _TargetExpectation(
                surface="八戸",
                occurrence=1,
                expected_pronunciation="ハチコ",
            ),
        ),
    ),
    _Case(
        text="国立の大学であれば学費が安いらしい",
        expected_kana="コクリツノダイガクデアレバガクヒガヤスイラシイ",
        targets=(
            _TargetExpectation(
                surface="国立",
                expected_pronunciation="コクリツ",
            ),
        ),
    ),
    _Case(
        text="今日は中央線で国立に向かう",
        expected_kana="キョーワチューオーセンデクニタチニムカウ",
        targets=(
            _TargetExpectation(
                surface="国立",
                expected_pronunciation="クニタチ",
            ),
        ),
    ),
    _Case(
        text="天気いいし、皆で表に出て遊ぼ？",
        expected_kana="テンキイイシ、ミナデオモテニデテアソボ？",
        targets=(
            _TargetExpectation(
                surface="表",
                expected_pronunciation="オモテ",
            ),
        ),
    ),
    _Case(
        text="子供が相手を殴ってしまった。警察が動くような大事になる前に、相手の親と話し合うべきだ",
        expected_kana="コドモガアイテヲナグッテシマッタ。ケーサツガウゴクヨーナダイジニナルマエニ、アイテノオヤトハナシアウベキダ",
        targets=(
            # TODO: 本来は「オオゴト」だが現状「ダイジ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="大事",
            #     expected_pronunciation="オオゴト",
            # ),
        ),
    ),
    _Case(
        text="将棋で玉を動かす。",
        expected_kana="ショーギデギョクヲウゴカス。",
        targets=(
            _TargetExpectation(
                surface="玉",
                expected_pronunciation="ギョク",
            ),
        ),
    ),
    _Case(
        text="愛しのあの子の愛し方がわからない。",
        expected_kana="アイシノアノコノアイシカタガワカラナイ。",
        targets=(
            # TODO: 本来は「イトシ」だが現状「アイシ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="愛し",
            #     expected_pronunciation="イトシ",
            # ),
            _TargetExpectation(
                surface="愛し",
                occurrence=1,
                expected_pronunciation="アイシ",
            ),
        ),
    ),
    _Case(
        text="歌が上手な彼女は、交渉事でも常に一枚上手であり、舞台の上手で堂々と振る舞った。",
        expected_kana="ウタガジョーズナカノジョワ、コーショーゴトデモツネニイチマイウワテデアリ、ブタイノジョーズデドードートフルマッタ。",
        targets=(
            _TargetExpectation(
                surface="上手",
                expected_pronunciation="ジョーズ",
            ),
            # NOTE: 「一枚上手」は一つの複合語として収録済み
            # NOTE: 舞台用語の「カミテ」は登場頻度が稀で人間でも読み間違えるため、現時点では読み分け対象に含めていない
        ),
    ),
    _Case(
        text="決め球として沈む球を使う。",
        expected_kana="キメダマトシテシズムタマヲツカウ。",
        targets=(
            # NOTE: 「決め球」は一つの複合語として収録済み
            _TargetExpectation(
                surface="球",
                occurrence=1,
                expected_pronunciation="タマ",
            ),
        ),
    ),
    _Case(
        text="この将棋では金か角を打てば勝ち。",
        expected_kana="コノショーギデワカネカカドヲウテバカチ。",
        targets=(
            # TODO: 本来は「キン」だが現状「カネ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="金",
            #     expected_pronunciation="キン",
            # ),
            # TODO: 本来は「カク」だが現状「カド」が選ばれてしまう
            # _TargetExpectation(
            #     surface="角",
            #     expected_pronunciation="カク",
            # ),
        ),
    ),
    _Case(
        text="風車とは、風を受けて回る羽根のついたおもちゃである。",
        expected_kana="フーシャトワ、カゼヲウケテマワルハネノツイタオモチャデアル。",
        targets=(
            # TODO: 本来は「カザグルマ」だが現状「フーシャ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="風車",
            #     expected_pronunciation="カザグルマ",
            # ),
        ),
    ),
    _Case(
        text="ひらがなだけ",
        expected_kana="ヒラガナダケ",
        expect_no_diagnostics=True,
    ),
)


@dataclass(frozen=True)
class _EmbeddedSentenceCase:
    """時間量表層を埋め込んだ文と、文中で維持すべき表層。"""

    text: str
    embedded_surface: str


_DURATION_SENTENCE_TEMPLATES: tuple[str, ...] = (
    "{surface}かかります。",
    "あと{surface}です。",
    "約{surface}待ってください。",
    "{surface}程度かかります。",
)
_DURATION_MINUTE_AFTER_TEMPLATE = "{surface}後に届きます。"
_MINUTE_TENS_SURFACES: tuple[str, ...] = (
    "二十分",
    "三十分",
    "四十分",
    "五十分",
    "六十分",
    "七十分",
    "八十分",
    "九十分",
)
_HOUR_DURATION_SURFACES: tuple[str, ...] = (
    "十時間",
    "二時間",
    "三時間",
    "四時間",
    "五時間",
    "六時間",
    "七時間",
    "八時間",
    "九時間",
    "何時間",
)
_SIMPLE_MINUTE_SURFACES: tuple[str, ...] = ("数分", "何分")
_HUNDRED_MINUTE_SURFACES: tuple[str, ...] = (
    "百二十分",
    "百三十分",
    "百四十分",
    "百五十分",
)
_COMPOUND_MINUTE_AFTER_SURFACES: tuple[str, ...] = ("数分後",)
_NAN_COUNTER_SENTENCE_CASES: tuple[_EmbeddedSentenceCase, ...] = (
    _EmbeddedSentenceCase("何人いますか", "何人"),
    _EmbeddedSentenceCase("何人と会いますか。", "何人"),
    _EmbeddedSentenceCase("何軒か見学できますか。", "何軒"),
    _EmbeddedSentenceCase("何個かありますか。", "何個"),
    _EmbeddedSentenceCase("この中で何曲歌える？", "何曲"),
    _EmbeddedSentenceCase("何分かかりますか。", "何分"),
    _EmbeddedSentenceCase("何分後に届きます。", "何分"),
)
_NAN_CLOCK_SENTENCE_CASES: tuple[_EmbeddedSentenceCase, ...] = (
    _EmbeddedSentenceCase("何時何分です。", "何時何分"),
    _EmbeddedSentenceCase("何時間何分かかります。", "何時間"),
    _EmbeddedSentenceCase("何時後に届きます。", "何時"),
    _EmbeddedSentenceCase("何時まで営業しますか。", "何時"),
    _EmbeddedSentenceCase("何時まで後。", "何時"),
    _EmbeddedSentenceCase("あと十時間後です。", "十時間"),
)
_JANUARY_DURATION_SENTENCE_CASES: tuple[_EmbeddedSentenceCase, ...] = (
    _EmbeddedSentenceCase("一月前かかります。", "一月"),
    _EmbeddedSentenceCase("あと一月前です。", "一月"),
    _EmbeddedSentenceCase("一月程度かかります。", "一月"),
    _EmbeddedSentenceCase("一月前に申し込みました。", "一月"),
)


def _build_duration_embedded_sentence_cases() -> tuple[_EmbeddedSentenceCase, ...]:
    """時間量辞書表層を代表文型へ機械的に埋め込んだ症例列を返す。"""

    cases: list[_EmbeddedSentenceCase] = []

    def append_template_cases(surfaces: tuple[str, ...], *, include_after: bool) -> None:
        for surface in surfaces:
            for template in _DURATION_SENTENCE_TEMPLATES:
                cases.append(
                    _EmbeddedSentenceCase(
                        text=template.format(surface=surface),
                        embedded_surface=surface,
                    )
                )
            if include_after is True:
                cases.append(
                    _EmbeddedSentenceCase(
                        text=_DURATION_MINUTE_AFTER_TEMPLATE.format(surface=surface),
                        embedded_surface=surface,
                    )
                )

    append_template_cases(_MINUTE_TENS_SURFACES, include_after=True)
    append_template_cases(_HOUR_DURATION_SURFACES, include_after=False)
    append_template_cases(_SIMPLE_MINUTE_SURFACES, include_after=False)
    append_template_cases(_HUNDRED_MINUTE_SURFACES, include_after=False)
    append_template_cases(_COMPOUND_MINUTE_AFTER_SURFACES, include_after=False)

    cases.append(
        _EmbeddedSentenceCase(
            text="数分後に届きます。",
            embedded_surface="数分後",
        )
    )
    cases.extend(_NAN_COUNTER_SENTENCE_CASES)
    cases.extend(_NAN_CLOCK_SENTENCE_CASES)
    cases.extend(_JANUARY_DURATION_SENTENCE_CASES)
    return tuple(cases)


_DURATION_EMBEDDED_SENTENCE_CASES = _build_duration_embedded_sentence_cases()


def _assert_tsqyomi_kana_matches_mecab_baseline(text: str) -> str:
    """MeCab 既定読みと tsqyomi 有効時のカタカナ出力が一致することを検証する。"""

    baseline = pyopenjtalk.g2p(text, kana=True, use_tsqyomi=False, use_vanilla=True)
    with_tsqyomi = pyopenjtalk.g2p(text, kana=True, use_tsqyomi=True, use_vanilla=True)
    assert isinstance(baseline, str)
    assert isinstance(with_tsqyomi, str)
    assert baseline == with_tsqyomi
    return with_tsqyomi


def _load_tsqyomi_v3() -> None:
    """固定 revision の v3 モデルをロードする。"""

    pytest.importorskip("onnxruntime")
    if tsqyomi.is_model_loaded() is False:
        tsqyomi.load_model(["CPUExecutionProvider"])


@pytest.fixture(scope="session")
def tsqyomi_v3() -> Iterator[None]:
    """セッション全体で v3 モデルを1回ロードする。"""

    _load_tsqyomi_v3()
    yield
    if tsqyomi.is_model_loaded():
        tsqyomi.unload_model()


def _run_with_diagnostics(text: str) -> tuple[str, list[tsqyomi_diagnostics.TargetDiagnostic]]:
    """g2p() の結果と診断記録を同時に返す。"""

    tsqyomi_diagnostics.start_recording()
    try:
        kana_result = pyopenjtalk.g2p(text, kana=True, use_tsqyomi=True, use_vanilla=True)
    except Exception:
        tsqyomi_diagnostics.stop_recording()
        raise
    assert isinstance(kana_result, str)
    return kana_result, tsqyomi_diagnostics.stop_recording()


def _find_diagnostic(
    diagnostics: list[tsqyomi_diagnostics.TargetDiagnostic],
    expectation: _TargetExpectation,
) -> tsqyomi_diagnostics.TargetDiagnostic:
    """表層と位置から診断1件を特定する。"""

    matched = [
        diagnostic
        for diagnostic in diagnostics
        if diagnostic.surface == expectation.surface
        and diagnostic.char_span == expectation.char_span
    ]
    assert len(matched) == 1
    return matched[0]


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.text)
def test_reading_regression(case: _Case, tsqyomi_v3: None) -> None:
    """v3 モデルの読み選択とカタカナ出力が固定した期待値と一致する。"""

    kana, diagnostics = _run_with_diagnostics(case.text)

    assert kana == case.expected_kana
    if case.expect_no_diagnostics:
        assert diagnostics == []
        return

    for expectation in _resolve_targets(case.text, case.targets):
        diagnostic = _find_diagnostic(diagnostics, expectation)
        assert diagnostic.outcome == expectation.expected_outcome
        assert diagnostic.selected_pronunciation == expectation.expected_pronunciation
        assert diagnostic.was_preserved is expectation.was_preserved
        if expectation.expected_segment_text is not None:
            assert diagnostic.segment_text == expectation.expected_segment_text


def test_load_model_is_idempotent(tsqyomi_v3: None) -> None:
    """ロード済みのモデルを繰り返し取得しない。"""

    loaded_model = tsqyomi.get_loaded_model()
    tsqyomi.load_model(["CPUExecutionProvider"])
    assert tsqyomi.get_loaded_model() is loaded_model


def test_unload_model_can_reload(tsqyomi_v3: None) -> None:
    """unload_model() 後に再ロードできる。"""

    assert tsqyomi.is_model_loaded() is True
    tsqyomi.unload_model()
    assert tsqyomi.is_model_loaded() is False
    _load_tsqyomi_v3()
    assert tsqyomi.is_model_loaded() is True


def test_long_text_passes_only_target_sentence_to_model(tsqyomi_v3: None) -> None:
    """長い前置きでは対象を含む末尾文だけをモデルへ渡す。"""

    prefix = "これはひらがなだけのぶんしょうです。" * 50
    target_sentence = "人気のない店"
    text = prefix + target_sentence

    _kana, diagnostics = _run_with_diagnostics(text)

    ninki = _find_diagnostic(
        diagnostics,
        replace(
            _TargetExpectation(
                surface="人気",
                expected_pronunciation="ヒトケ",
                expected_segment_text=target_sentence,
            ),
            char_span=(900, 902),
        ),
    )
    assert ninki.segment_text == target_sentence


def test_adjacent_targets_use_candidate_connection_cost(tsqyomi_v3: None) -> None:
    """隣接する2対象では後側形態素の link_cost に候補間接続辺を反映する。"""

    text = "人気最中です"
    target_spans = ((0, 2), (2, 4))
    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    analysis = jtalk.analyze_mecab_candidates(text, target_spans)
    paths_by_span = {
        target_span: tuple(path for path in analysis["paths"] if path["char_span"] == target_span)
        for target_span in target_spans
    }
    connection_costs = {
        (connection["left_node_id"], connection["right_node_id"]): connection["cost"]
        for connection in analysis["connections"]
    }

    _features, morphs = select_mecab_features_with_tsqyomi(text, jtalk)

    left_pronunciation = morphs[0]["features"][9]
    right_pronunciation = morphs[1]["features"][9]
    left_path = next(
        path
        for path in paths_by_span[target_spans[0]]
        if path["pronunciation"] == left_pronunciation
    )
    right_path = next(
        path
        for path in paths_by_span[target_spans[1]]
        if path["pronunciation"] == right_pronunciation
    )
    expected_link_cost = connection_costs[(left_path["node_ids"][-1], right_path["node_ids"][0])]
    assert morphs[1]["link_cost"] == expected_link_cost


def test_high_level_dictionary_protection_skips_model_inference(
    tmp_path: Path,
    tsqyomi_v3: None,
) -> None:
    """読み保護ユーザー辞書ではモデル推論を止める。"""

    unprotected_csv = tmp_path / "unprotected.csv"
    protected_csv = tmp_path / "protected.csv"
    unprotected_dic = tmp_path / "unprotected.dic"
    protected_dic = tmp_path / "protected.dic"
    unprotected_csv.write_text(
        "人気,,,1,名詞,一般,*,*,*,*,人気,ニンキ,ニンキ,0/3,*\n",
        encoding="utf-8",
    )
    protected_csv.write_text(
        "人気,,,1,名詞,一般,*,*,*,*,人気,ヒトケ,ヒトケ,0/3,*\n",
        encoding="utf-8",
    )
    pyopenjtalk.mecab_dict_index(str(unprotected_csv), str(unprotected_dic))
    pyopenjtalk.mecab_dict_index(str(protected_csv), str(protected_dic))

    try:
        pyopenjtalk.update_global_jtalk_with_user_dict(
            [
                {
                    "dic_path": str(unprotected_dic),
                    "is_reading_protected": False,
                },
                {
                    "dic_path": str(protected_dic),
                    "is_reading_protected": True,
                },
            ]
        )
        kana, diagnostics = _run_with_diagnostics("人気の店です。")
        assert kana == "ニンキノミセデス。"
        assert len(diagnostics) == 1
        assert diagnostics[0].outcome == "reading_protected"
        assert diagnostics[0].selected_pronunciation is None
    finally:
        pyopenjtalk.unset_user_dict()


def test_include_morphs_false_skips_morph_rebuild(tsqyomi_v3: None) -> None:
    """include_morphs=False では形態素差し替えを省略し feature だけ更新する。"""

    replace_calls = 0
    original_replace = tsqyomi_inference._replace_morph

    def counting_replace(*args: Any, **kwargs: Any) -> MeCabMorph:
        """形態素差し替えの呼び出し回数を記録する。"""

        nonlocal replace_calls
        replace_calls += 1
        return original_replace(*args, **kwargs)

    text = "一寸です"
    jtalk = pyopenjtalk.OpenJTalk(dn_mecab=pyopenjtalk.OPEN_JTALK_DICT_DIR)
    tsqyomi_inference._replace_morph = counting_replace
    try:
        features, morphs = select_mecab_features_with_tsqyomi(
            text,
            jtalk,
            include_morphs=False,
        )
    finally:
        tsqyomi_inference._replace_morph = original_replace

    assert morphs == []
    assert replace_calls == 0
    assert any("チョット" in feature for feature in features)


def test_v3_onnx_contract_matches_loaded_model(tsqyomi_v3: None) -> None:
    """ロード済み v3 セッションがメタデータ契約を満たす。"""

    model = tsqyomi.get_loaded_model()
    tsqyomi.TsqyomiModel.validate_onnx_contract(model.session, model.metadata)


def test_model_revision_is_pinned() -> None:
    """テストが参照するモデル revision が実装側の固定値と一致する。"""

    assert tsqyomi_model._MODEL_REVISION == "1157e36e1bf81a4cc01ed911b7dc691106c1ccdb"
    assert tsqyomi_model._MODEL_FILES["model"] == "v3/model.onnx"


def test_g2p_mapping_aligns_tsqyomi_reading_to_morph_char_span(tsqyomi_v3: None) -> None:
    """g2p_mapping() の char_span と phoneme 列が、tsqyomi の選択結果と一致する。"""

    text = "深夜の路地は人気が無くて怖い。"
    mapping = pyopenjtalk.g2p_mapping(text, use_tsqyomi=True, use_vanilla=True)
    ninki = next(entry for entry in mapping if entry["surface"] == "人気")
    assert ninki["char_span"] == (6, 8)
    assert ninki["phonemes"] == ["h", "I", "t", "o", "k", "e"]


def test_run_frontend_detailed_reflects_tsqyomi_pronunciation(tsqyomi_v3: None) -> None:
    """run_frontend_detailed() の NJD feature が tsqyomi による選択発音を反映する。"""

    text = "大分県にもう大分長いこと住んでいるな。"
    _features, morphs = pyopenjtalk.run_frontend_detailed(text, use_tsqyomi=True, use_vanilla=True)
    oita_first = next(morph for morph in morphs if morph["char_span"] == (0, 2))
    oita_second = next(morph for morph in morphs if morph["char_span"] == (6, 8))
    assert oita_first["features"][9] == "オーイタ"
    assert oita_second["features"][9] == "ダイブ"


def test_extract_fullcontext_succeeds_with_tsqyomi(tsqyomi_v3: None) -> None:
    """extract_fullcontext() が tsqyomi 有効時でもラベル列を返す。"""

    text = "竹田はかつて岡藩の城下町であった。"
    labels = pyopenjtalk.extract_fullcontext(text, use_tsqyomi=True, use_vanilla=True)
    assert len(labels) >= 1
    assert all(isinstance(label, str) for label in labels)


@pytest.mark.parametrize(
    ("text", "expected_surfaces"),
    [
        (
            "もし明朝体が重いようならまたおいで。",
            ("明朝", "体"),
        ),
        (
            "将棋において玉の扱いは重要。",
            ("玉", "要"),
        ),
    ],
)
def test_compound_scored_surface_reports_no_exact_morph_range(
    tsqyomi_v3: None,
    text: str,
    expected_surfaces: tuple[str, ...],
) -> None:
    """辞書形態素境界と target span が一致しない複合表層は差し替えを行わない。"""

    _kana, diagnostics = _run_with_diagnostics(text)
    assert tuple(diagnostic.surface for diagnostic in diagnostics) == expected_surfaces
    assert all(diagnostic.outcome == "no_exact_morph_range" for diagnostic in diagnostics)
    assert all(diagnostic.selected_pronunciation is None for diagnostic in diagnostics)


def test_enabled_tsqyomi_changes_g2p_output_from_baseline(tsqyomi_v3: None) -> None:
    """tsqyomi 有効時は無効時と異なるカタカナ出力になる対象文を通す。"""

    text = "深夜の路地は人気が無くて怖い。"
    without_tsqyomi = pyopenjtalk.g2p(text, kana=True, use_tsqyomi=False)
    with_tsqyomi, _diagnostics = _run_with_diagnostics(text)
    assert without_tsqyomi != with_tsqyomi
    assert with_tsqyomi == "シンヤノロジワヒトケガナクテコワイ。"


@pytest.mark.parametrize(
    ("text", "expected_kana"),
    (
        ("何時間かかりますか。", "ナンジカンカカリマスカ。"),
        ("毎日何時間も働きます。", "マイニチナンジカンモハタラキマス。"),
        ("振込み後何時間で届きますか。", "フリコミゴナンジカンデトドキマスカ。"),
        ("何時間後に届きますか。", "ナンジカンゴニトドキマスカ。"),
        ("何時間以内に返信がありますか。", "ナンジカンイナイニヘンシンガアリマスカ。"),
        ("何時間程度かかりますか。", "ナンジカンテードカカリマスカ。"),
        ("何時間でも待ちます。", "ナンジカンデモマチマス。"),
        ("二十分前に始めます。", "ニジュップンマエニハジメマス。"),
        ("一時間かかります。", "イチジカンカカリマス。"),
        ("二十四時間営業です。", "ニジューヨンジカンエーギョーデス。"),
        ("二時間かかります。", "ニジカンカカリマス。"),
        ("三時間かかります。", "サンジカンカカリマス。"),
        ("四時間かかります。", "ヨンジカンカカリマス。"),
        ("五時間かかります。", "ゴジカンカカリマス。"),
        ("六時間かかります。", "ロクジカンカカリマス。"),
        ("七時間かかります。", "ナナジカンカカリマス。"),
        ("八時間かかります。", "ハチジカンカカリマス。"),
        ("九時間かかります。", "キュウジカンカカリマス。"),
        ("十時間かかります。", "ジュージカンカカリマス。"),
        ("あと二時間です。", "アトニジカンデス。"),
        ("あと十時間後です。", "アトトトキカンゴデス。"),
        ("何時まで営業しますか。", "ナンジマデエーギョーシマスカ。"),
        ("いつまで営業しますか。", "イツマデエーギョーシマスカ。"),
        ("門を通った時に止める間もなく進んだ。", "モンヲトーッタトキニトメルマモナクススンダ。"),
    ),
)
def test_hour_duration_expressions_keep_dictionary_owned_readings(
    tsqyomi_v3: None,
    text: str,
    expected_kana: str,
) -> None:
    """時間量の内部介入を止め、文脈選択が必要な周辺語の既存結果も維持する。"""

    baseline = pyopenjtalk.g2p(text, kana=True, use_tsqyomi=False, use_vanilla=True)
    with_tsqyomi, _diagnostics = _run_with_diagnostics(text)

    assert baseline == expected_kana
    assert with_tsqyomi == expected_kana


@pytest.mark.parametrize(
    ("text", "expected_baseline", "expected_with_tsqyomi"),
    (
        ("体中が痛い。", "カラダチューガイタイ。", "カラダジューガイタイ。"),
        ("五分後に始めます。", "ゴブゴニハジメマス。", "ゴフンゴニハジメマス。"),
        ("十分後に始めます。", "ジュップンゴニハジメマス。", "ジップンゴニハジメマス。"),
        ("何分かかりますか。", "ナンフンカカリマスカ。", "ナンフンカカリマスカ。"),
        (
            "四十分後に戻ります。",
            "ヨンジュップンゴニモドリマス。",
            "ヨンジュップンゴニモドリマス。",
        ),
    ),
)
def test_non_hour_expressions_remain_available_to_tsqyomi(
    tsqyomi_v3: None,
    text: str,
    expected_baseline: str,
    expected_with_tsqyomi: str,
) -> None:
    """時間量の保護規則を広げず、既存の文脈選択による読み修正を維持する。"""

    baseline = pyopenjtalk.g2p(text, kana=True, use_tsqyomi=False, use_vanilla=True)
    with_tsqyomi, _diagnostics = _run_with_diagnostics(text)

    assert baseline == expected_baseline
    assert with_tsqyomi == expected_with_tsqyomi


@pytest.mark.parametrize(
    ("text", "expected_kana"),
    (
        ("三十分前に到着します。", "サンジュップンマエニトーチャクシマス。"),
        ("約三十分待ってください。", "ヤクサンジュップンマッテクダサイ。"),
        ("あと三十分です。", "アトサンジュップンデス。"),
        ("残り三十分。", "ノコリサンジュップン。"),
        ("三十分程度かかります。", "サンジュップンテードカカリマス。"),
        ("毎日三十分運動します。", "マイニチサンジュップンウンドーシマス。"),
        ("二十分後に戻ります。", "ニジュップンゴニモドリマス。"),
        ("四十分後に戻ります。", "ヨンジュップンゴニモドリマス。"),
        ("百二十分です。", "ヒャクニジュップンデス。"),
        ("百三十分かかりました。", "ヒャクサンジュップンカカリマシタ。"),
        ("百四十分かかりました。", "ヒャクヨンジュップンカカリマシタ。"),
        ("百五十分かかりました。", "ヒャクゴジュップンカカリマシタ。"),
        ("数分後に届きます。", "スーフンゴニトドキマス。"),
        ("数分後", "スーフンゴ"),
        ("何分後に届きます。", "ナンプンゴニトドキマス。"),
        ("四十分後に戻ります。", "ヨンジュップンゴニモドリマス。"),
    ),
)
def test_minute_duration_expressions_keep_dictionary_owned_readings(
    tsqyomi_v3: None,
    text: str,
    expected_kana: str,
) -> None:
    """複数数詞 + 分 や 数分 + 後 など、辞書既定読みを維持すべき分単位の時間量を保護する。"""

    baseline = pyopenjtalk.g2p(text, kana=True, use_tsqyomi=False, use_vanilla=True)
    with_tsqyomi, _diagnostics = _run_with_diagnostics(text)

    assert baseline == expected_kana
    assert with_tsqyomi == expected_kana


@pytest.mark.parametrize(
    "case",
    _DURATION_EMBEDDED_SENTENCE_CASES,
    ids=lambda case: f"{case.embedded_surface}:{case.text}",
)
def test_duration_dictionary_surfaces_embedded_in_sentences_match_mecab_baseline_with_tsqyomi(
    tsqyomi_v3: None,
    case: _EmbeddedSentenceCase,
) -> None:
    """時間量表層を代表文型へ埋め込んでも、tsqyomi 有効時に MeCab 既定読みが変わらない。"""

    with_tsqyomi_kana = _assert_tsqyomi_kana_matches_mecab_baseline(case.text)
    assert case.embedded_surface in case.text
    assert len(with_tsqyomi_kana) > 0


@pytest.mark.parametrize(
    ("text", "surface", "char_span", "expected_outcome"),
    (
        (
            "百二十分です。",
            "十分",
            (2, 4),
            "dictionary_default_protected",
        ),
        (
            "あと三十分。",
            "十分",
            (3, 5),
            "dictionary_default_protected",
        ),
        (
            "数分後に届きます。",
            "後",
            (2, 3),
            "dictionary_default_protected",
        ),
        (
            "何分",
            "何分",
            (0, 2),
            "dictionary_default_protected",
        ),
        (
            "何軒か見学できますか。",
            "何",
            (0, 1),
            "dictionary_default_protected",
        ),
        (
            "あと一月前です。",
            "一月",
            (2, 4),
            "dictionary_default_protected",
        ),
        (
            "何時後に届きます。",
            "何時",
            (0, 2),
            "dictionary_default_protected",
        ),
        (
            "何時まで後。",
            "何時",
            (0, 2),
            "dictionary_default_protected",
        ),
        (
            "何時まで後。",
            "後",
            (4, 5),
            "dictionary_default_protected",
        ),
        (
            "四十分後に戻ります。",
            "後",
            (3, 4),
            "dictionary_default_protected",
        ),
    ),
)
def test_minute_duration_protection_records_dictionary_default_outcome(
    tsqyomi_v3: None,
    text: str,
    surface: str,
    char_span: tuple[int, int],
    expected_outcome: str,
) -> None:
    """保護対象表層ではモデル適用前に辞書既定読み保護の診断が記録される。"""

    _kana, diagnostics = _run_with_diagnostics(text)
    matched = [
        diagnostic
        for diagnostic in diagnostics
        if diagnostic.surface == surface and diagnostic.char_span == char_span
    ]
    assert len(matched) == 1
    assert matched[0].outcome == expected_outcome
    assert matched[0].was_preserved is True


@pytest.mark.parametrize(
    ("text", "expected_baseline", "expected_with_tsqyomi"),
    (
        ("五分かかります。", "ゴブカカリマス。", "ゴフンカカリマス。"),
        ("十分かかります。", "ジューブンカカリマス。", "ジップンカカリマス。"),
        ("五分後に始めます。", "ゴブゴニハジメマス。", "ゴフンゴニハジメマス。"),
        ("十分後に始めます。", "ジュップンゴニハジメマス。", "ジップンゴニハジメマス。"),
        ("体中が痛い。", "カラダチューガイタイ。", "カラダジューガイタイ。"),
    ),
)
def test_nan_disambiguation_and_minute_heteronyms_remain_available_to_tsqyomi(
    tsqyomi_v3: None,
    text: str,
    expected_baseline: str,
    expected_with_tsqyomi: str,
) -> None:
    """代名詞用法の「何」や単独「五分/十分」は、tsqyomi の文脈選択を残す。"""

    baseline = pyopenjtalk.g2p(text, kana=True, use_tsqyomi=False, use_vanilla=True)
    with_tsqyomi, _diagnostics = _run_with_diagnostics(text)

    assert baseline == expected_baseline
    assert with_tsqyomi == expected_with_tsqyomi
