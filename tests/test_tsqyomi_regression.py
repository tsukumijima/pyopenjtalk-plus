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
from pyopenjtalk.tsqyomi.diagnostics import TargetDiagnosticOutcome
from pyopenjtalk.tsqyomi.inference import select_mecab_features_with_tsqyomi
from pyopenjtalk.types import MeCabMorph, UserDictionaryEntry


def _load_tsqyomi_default_model() -> None:
    """固定リビジョンの既定モデルをロードする。"""

    pytest.importorskip("onnxruntime")
    if tsqyomi.is_model_loaded() is False:
        tsqyomi.load_model(["CPUExecutionProvider"])


@pytest.fixture(scope="session")
def tsqyomi_default_model() -> Iterator[None]:
    """セッション全体で既定モデルを1回ロードする。"""

    _load_tsqyomi_default_model()
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
    expected_outcome: TargetDiagnosticOutcome = "applied"
    expected_segment_text: str | None = None
    was_preserved: bool = False
    occurrence: int = 0
    char_span: tuple[int, int] | None = None


@dataclass(frozen=True)
class _DictionaryReadingExpectation:
    """
    辞書経路へ委ねる語について期待する表層位置。

    この表層範囲へ tsqyomi が介入していないことと、tsqyomi 無効時の最良形態素が
    期待する発音へ到達することを確認する。
    """

    surface: str
    expected_pronunciation: str
    occurrence: int = 0
    char_span: tuple[int, int] | None = None


def _resolve_char_span(
    text: str,
    target: _TargetExpectation | _DictionaryReadingExpectation,
) -> tuple[int, int]:
    """期待値から text 上の char_span を解決する。"""

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
class _ReadingCase:
    """1文について期待する文全体のカナと、tsqyomi または辞書が読みを決める表層ごとの期待値。"""

    text: str
    expected_kana: str
    targets: tuple[_TargetExpectation, ...] = ()
    dictionary_targets: tuple[_DictionaryReadingExpectation, ...] = ()
    expect_no_diagnostics: bool = False


# v5/model.onnx で CPU 推論した期待値
## `_TargetExpectation` は tsqyomi が診断記録に残した、読み選択または保護が成立した表層を検証する
## `_DictionaryReadingExpectation` は辞書経路が読みを確定し、tsqyomi が介入しない表層を検証する
## `expected_kana` は v5 の現状出力を固定する
## コメントアウトした `_TargetExpectation` は本来の期待読みで、達成後に有効化する TODO
## 語彙未収載かつ文脈上本当に競合読みがある表層だけ、メタデータに「表層」を足したら `_TargetExpectation` でも検証する
## 競合読みが文脈上存在せず辞書既定で到達済みの表層は targets に含めない
_READING_CASES: tuple[_ReadingCase, ...] = (
    _ReadingCase(
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
    _ReadingCase(
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
    _ReadingCase(
        text="休日で人気の少ない茶室に入ると、座卓には最中が置かれていた。",
        expected_kana="キュージツデヒトケノスクナイチャシツニハイルト、ザタクニワモナカガオカレテイタ。",
        targets=(
            _TargetExpectation(
                surface="人気",
                expected_pronunciation="ヒトケ",
            ),
            _TargetExpectation(
                surface="最中",
                expected_pronunciation="モナカ",
            ),
        ),
        dictionary_targets=(
            _DictionaryReadingExpectation(
                surface="入る",
                expected_pronunciation="ハイル",
            ),
        ),
    ),
    _ReadingCase(
        text="もうこの程度で十分です",
        expected_kana="モーコノテードデジューブンデス",
        targets=(
            _TargetExpectation(
                surface="十分",
                expected_pronunciation="ジューブン",
            ),
        ),
    ),
    _ReadingCase(
        text="この踊りは私の一番の十八番です",
        expected_kana="コノオドリワワタシノイチバンノオハコデス",
        targets=(
            _TargetExpectation(
                surface="十八番",
                expected_pronunciation="オハコ",
            ),
        ),
    ),
    _ReadingCase(
        text="何人いますか",
        expected_kana="ナンニンイマスカ",
        targets=(
            _TargetExpectation(
                surface="何人",
                expected_pronunciation="ナンニン",
            ),
        ),
    ),
    _ReadingCase(
        text="会議を行った",
        expected_kana="カイギヲオコナッタ",
        targets=(
            _TargetExpectation(
                surface="行っ",
                expected_pronunciation="オコナッ",
            ),
        ),
    ),
    _ReadingCase(
        text="駅へ行った",
        expected_kana="エキエイッタ",
        targets=(
            _TargetExpectation(
                surface="行っ",
                expected_pronunciation="イッ",
            ),
        ),
    ),
    _ReadingCase(
        text="学校に通っている",
        expected_kana="ガッコーニカヨッテイル",
        targets=(
            _TargetExpectation(
                surface="通っ",
                expected_pronunciation="カヨッ",
            ),
        ),
    ),
    _ReadingCase(
        text="門を通って入る",
        expected_kana="モンヲトーッテハイル",
        targets=(
            _TargetExpectation(
                surface="通っ",
                expected_pronunciation="トーッ",
            ),
        ),
        dictionary_targets=(
            _DictionaryReadingExpectation(
                surface="入る",
                expected_pronunciation="ハイル",
            ),
        ),
    ),
    _ReadingCase(
        text="部屋に入る",
        expected_kana="ヘヤニハイル",
        dictionary_targets=(
            _DictionaryReadingExpectation(
                surface="入る",
                expected_pronunciation="ハイル",
            ),
        ),
    ),
    _ReadingCase(
        text="気に入る",
        expected_kana="キニイル",
        dictionary_targets=(
            _DictionaryReadingExpectation(
                surface="気に入る",
                expected_pronunciation="キニイル",
            ),
        ),
    ),
    _ReadingCase(
        text="悦に入った",
        expected_kana="エツニイッタ",
        dictionary_targets=(
            _DictionaryReadingExpectation(
                surface="悦に入っ",
                expected_pronunciation="エツニイッ",
            ),
        ),
    ),
    _ReadingCase(
        text="この通りで待つ",
        expected_kana="コノトーリデマツ",
        targets=(
            _TargetExpectation(
                surface="通り",
                expected_pronunciation="トーリ",
            ),
        ),
    ),
    _ReadingCase(
        text="予想通りで驚いた",
        expected_kana="ヨソードーリデオドロイタ",
        targets=(
            _TargetExpectation(
                surface="通り",
                expected_pronunciation="ドーリ",
            ),
        ),
    ),
    _ReadingCase(
        text="商売上",
        expected_kana="ショーバイジョー",
        targets=(
            _TargetExpectation(
                surface="上",
                expected_pronunciation="ジョー",
            ),
        ),
    ),
    _ReadingCase(
        text="÷÷÷÷人気",
        expected_kana="÷÷÷÷ニンキ",
        targets=(
            _TargetExpectation(
                surface="人気",
                expected_pronunciation="ニンキ",
            ),
        ),
    ),
    _ReadingCase(
        text="あと一寸です",
        expected_kana="アトチョットデス",
        targets=(
            _TargetExpectation(
                surface="一寸",
                expected_pronunciation="チョット",
            ),
        ),
    ),
    _ReadingCase(
        text="いじけるなんて大人気ないな君は。",
        expected_kana="イジケルナンテオトナゲナイナキミワ。",
        targets=(
            _TargetExpectation(
                surface="大人気",
                expected_pronunciation="オトナゲ",
            ),
        ),
    ),
    _ReadingCase(
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
    _ReadingCase(
        text="人の金で食う飯は美味い。",
        expected_kana="ヒトノカネデクウメシワウマイ。",
        targets=(
            _TargetExpectation(
                surface="金",
                expected_pronunciation="カネ",
            ),
        ),
    ),
    _ReadingCase(
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
    _ReadingCase(
        text="大分にもう大分長いこと住んでいるな。",
        expected_kana="オーイタニモーダイブナガイコトスンデイルナ。",
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
    _ReadingCase(
        text="彼に敬意を表します。",
        expected_kana="カレニケーイヲヒョーシマス。",
        targets=(
            _TargetExpectation(
                surface="表し",
                expected_pronunciation="ヒョーシ",
            ),
        ),
    ),
    _ReadingCase(
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
    _ReadingCase(
        text="泥を被るという被害を被った。",
        expected_kana="ドロヲカブルトイウヒガイヲコームッタ。",
        targets=(
            _TargetExpectation(
                surface="被る",
                expected_pronunciation="カブル",
            ),
            _TargetExpectation(
                surface="被っ",
                expected_pronunciation="コームッ",
            ),
        ),
    ),
    _ReadingCase(
        text="竹田はかつて岡藩の城下町であった。",
        expected_kana="タケタワカツテオカハンノジョーカマチデアッタ。",
        targets=(
            _TargetExpectation(
                surface="竹田",
                expected_pronunciation="タケタ",
            ),
        ),
    ),
    _ReadingCase(
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
    _ReadingCase(
        text="角の生えた鬼に向かって角が立たない言い回し。",
        expected_kana="ツノノハエタオニニムカッテツノガタタナイイーマワシ。",
        targets=(
            _TargetExpectation(
                surface="角",
                expected_pronunciation="ツノ",
            ),
            # TODO: 本来は「カド」だが現状「ツノ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="角",
            #     occurrence=1,
            #     expected_pronunciation="カド",
            # ),
        ),
    ),
    _ReadingCase(
        text="辛いことだが仕方がない。",
        expected_kana="ツライコトダガシカタガナイ。",
        targets=(
            _TargetExpectation(
                surface="辛い",
                expected_pronunciation="ツライ",
            ),
        ),
    ),
    _ReadingCase(
        text="深夜の路地は人気が無くて怖い。",
        expected_kana="シンヤノロジワヒトケガナクテコワイ。",
        targets=(
            _TargetExpectation(
                surface="人気",
                expected_pronunciation="ヒトケ",
            ),
        ),
    ),
    _ReadingCase(
        text="金の時計を買うために、一生懸命に金を貯めた。",
        expected_kana="カネノトケーヲカウタメニ、イッショーケンメーニカネヲタメタ。",
        targets=(
            # TODO: 本来は「キン」だが現状「カネ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="金",
            #     expected_pronunciation="キン",
            # ),
            _TargetExpectation(
                surface="金",
                occurrence=1,
                expected_pronunciation="カネ",
            ),
        ),
    ),
    _ReadingCase(
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
    _ReadingCase(
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
    _ReadingCase(
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
    _ReadingCase(
        text="会議は火・木に開きます。",
        expected_kana="カイギワカ・モクニヒラキマス。",
        targets=(
            _TargetExpectation(
                surface="火",
                expected_pronunciation="カ",
            ),
            _TargetExpectation(
                surface="木",
                expected_pronunciation="モク",
            ),
            _TargetExpectation(
                surface="開き",
                expected_pronunciation="ヒラキ",
            ),
        ),
    ),
    _ReadingCase(
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
    _ReadingCase(
        text="誕生月にお祝いします。",
        expected_kana="タンジョーズキニオイワイシマス。",
    ),
    _ReadingCase(
        text="締め切り月に提出します。",
        expected_kana="シメキリヅキニテーシュツシマス。",
        targets=(
            _TargetExpectation(
                surface="月",
                expected_pronunciation="ヅキ",
            ),
        ),
    ),
    _ReadingCase(
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
    _ReadingCase(
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
    _ReadingCase(
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
    _ReadingCase(
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
    # 前の名詞や数詞に付く接尾辞・助数詞の「体」「床」「頭」では、辞書が既定で選ぶ「タイ」「ショー」「トー」がモデルの読み候補にない
    ## モデルは候補の「カラダ」「ユカ」「アタマ」などのどれかを選ぶが、どれも誤りなので、辞書の既定の読みを維持することを確かめる
    _ReadingCase(
        text="受容体の働きを調べる。",
        expected_kana="ジュヨータイノハタラキヲシラベル。",
        targets=(
            _TargetExpectation(
                surface="体",
                expected_pronunciation="タイ",
                expected_outcome="dictionary_default_protected",
                was_preserved=True,
            ),
        ),
    ),
    _ReadingCase(
        text="構造体の強度を確かめる。",
        expected_kana="コーゾータイノキョードヲタシカメル。",
        targets=(
            _TargetExpectation(
                surface="体",
                expected_pronunciation="タイ",
                expected_outcome="dictionary_default_protected",
                was_preserved=True,
            ),
        ),
    ),
    _ReadingCase(
        text="二体の人形を飾る。",
        expected_kana="ニタイノニンギョーヲカザル。",
        targets=(
            _TargetExpectation(
                surface="体",
                expected_pronunciation="タイ",
                expected_outcome="dictionary_default_protected",
                was_preserved=True,
            ),
        ),
    ),
    _ReadingCase(
        text="病院は三十八万床ある。",
        expected_kana="ビョーインワサンジューハチマンショーアル。",
        targets=(
            _TargetExpectation(
                surface="床",
                expected_pronunciation="ショー",
                expected_outcome="dictionary_default_protected",
                was_preserved=True,
            ),
        ),
    ),
    _ReadingCase(
        text="五頭の牛を飼う。",
        expected_kana="ゴトーノウシヲカウ。",
        targets=(
            _TargetExpectation(
                surface="頭",
                expected_pronunciation="トー",
                expected_outcome="dictionary_default_protected",
                was_preserved=True,
            ),
        ),
    ),
    # 長さの「十間」は辞書の1語の行が「ジッケン」と読み、「間」だけの形態素がないので tsqyomi は介入しない
    _ReadingCase(
        text="十間先の家",
        expected_kana="ジッケンサキノイエ",
        targets=(
            _TargetExpectation(
                surface="間",
                expected_outcome="no_exact_morph_range",
            ),
            _TargetExpectation(
                surface="家",
                expected_pronunciation="イエ",
            ),
        ),
    ),
    # 日付に続く「午前」が「午」と「前」に分かれると、「前」が tsqyomi の対象になって「ゼン」と「マエ」の間で揺れる
    ## 辞書が「午前」を1語で選び、「前」だけの形態素がないので tsqyomi が介入しないことを確かめる
    _ReadingCase(
        text="十一日午前十時に開く。",
        expected_kana="ジューイチニチゴゼンジュージニヒラク。",
        targets=(
            _TargetExpectation(
                surface="前",
                expected_outcome="no_exact_morph_range",
            ),
            _TargetExpectation(
                surface="時",
                expected_pronunciation="ジ",
            ),
        ),
    ),
    # 単独の名詞の「体」「床」「頭」では、モデルが文脈から選ぶ読み分けが働き続けることを確かめる
    _ReadingCase(
        text="体を動かす。",
        expected_kana="カラダヲウゴカス。",
        targets=(
            _TargetExpectation(
                surface="体",
                expected_pronunciation="カラダ",
            ),
        ),
    ),
    _ReadingCase(
        text="床に座る。",
        expected_kana="ユカニスワル。",
        targets=(
            _TargetExpectation(
                surface="床",
                expected_pronunciation="ユカ",
            ),
        ),
    ),
    _ReadingCase(
        text="頭が痛い。",
        expected_kana="アタマガイタイ。",
        targets=(
            _TargetExpectation(
                surface="頭",
                expected_pronunciation="アタマ",
            ),
        ),
    ),
    _ReadingCase(
        text="間に合う。",
        expected_kana="マニアウ。",
        targets=(
            _TargetExpectation(
                surface="間",
                expected_outcome="no_exact_morph_range",
            ),
        ),
    ),
    _ReadingCase(
        text="午後の授業",
        expected_kana="ゴゴノジュギョー",
        targets=(
            _TargetExpectation(
                surface="後",
                expected_outcome="no_exact_morph_range",
            ),
        ),
    ),
    _ReadingCase(
        text="漫画家です。",
        expected_kana="マンガカデス。",
        targets=(
            _TargetExpectation(
                surface="家",
                expected_pronunciation="カ",
            ),
        ),
    ),
    _ReadingCase(
        text="専門家です。",
        expected_kana="センモンカデス。",
        targets=(
            _TargetExpectation(
                surface="家",
                expected_pronunciation="カ",
            ),
        ),
    ),
    _ReadingCase(
        text="山田家です。",
        expected_kana="ヤマダケデス。",
        targets=(
            _TargetExpectation(
                surface="家",
                expected_pronunciation="ケ",
            ),
        ),
    ),
    _ReadingCase(
        text="将軍家です。",
        expected_kana="ショーグンケデス。",
        targets=(
            _TargetExpectation(
                surface="家",
                expected_outcome="no_exact_morph_range",
            ),
        ),
    ),
    _ReadingCase(
        text="子宝に恵まれ、代々家が栄えるように",
        expected_kana="コダカラニメグマレ、ダイダイイエガサカエルヨーニ",
        targets=(
            _TargetExpectation(
                surface="家",
                expected_pronunciation="イエ",
            ),
        ),
    ),
    _ReadingCase(
        text="月が明るい夜です。",
        expected_kana="ツキガアカルイヨルデス。",
        targets=(
            _TargetExpectation(
                surface="月",
                expected_pronunciation="ツキ",
            ),
        ),
    ),
    _ReadingCase(
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
    _ReadingCase(
        text="日が長くなりました。",
        expected_kana="ヒガナガクナリマシタ。",
        targets=(
            _TargetExpectation(
                surface="日",
                expected_pronunciation="ヒ",
            ),
        ),
    ),
    _ReadingCase(
        text="1日で終わります。",
        expected_kana="イチニチデオワリマス。",
        targets=(
            _TargetExpectation(
                surface="日",
                expected_pronunciation="ニチ",
            ),
        ),
    ),
    _ReadingCase(
        text="あちらの方がお見えになった理由は、皆まで言わずとも分かる。",
        expected_kana="アチラノカタガオミエニナッタリユーワ、ミナマデイワズトモワカル。",
        targets=(
            _TargetExpectation(
                surface="方",
                expected_pronunciation="カタ",
            ),
        ),
    ),
    _ReadingCase(
        text="この方はどちらの方からお越しになりましたか？",
        expected_kana="コノカタワドチラノホーカラオコシニナリマシタカ？",
        targets=(
            _TargetExpectation(
                surface="方",
                expected_pronunciation="カタ",
            ),
            _TargetExpectation(
                surface="方",
                occurrence=1,
                expected_pronunciation="ホー",
            ),
        ),
    ),
    _ReadingCase(
        text="この絵は筆を使わずに描いたの？",
        expected_kana="コノエワフデヲツカワズニカイタノ？",
        targets=(
            _TargetExpectation(
                surface="描い",
                expected_pronunciation="カイ",
            ),
        ),
    ),
    _ReadingCase(
        text="この作家の心理描写の描きかたには定評がある",
        expected_kana="コノサッカノシンリビョーシャノエガキカタニワテーヒョーガアル",
        targets=(
            _TargetExpectation(
                surface="描き",
                expected_pronunciation="エガキ",
            ),
        ),
    ),
    _ReadingCase(
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
    _ReadingCase(
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
    _ReadingCase(
        text="スケートの羽生選手と将棋の羽生棋士。",
        expected_kana="スケートノハブセンシュトショーギノハブキシ。",
        targets=(
            # TODO: 本来は「ハニュー」だが現状「ハブ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="羽生",
            #     expected_pronunciation="ハニュー",
            # ),
            _TargetExpectation(
                surface="羽生",
                occurrence=1,
                expected_pronunciation="ハブ",
            ),
        ),
    ),
    _ReadingCase(
        text="予約なしでも入れるホテルを駅前で探した。",
        expected_kana="ヨヤクナシデモハイレルホテルヲエキマエデサガシタ。",
        targets=(
            _TargetExpectation(
                surface="入れる",
                expected_pronunciation="ハイレル",
            ),
        ),
    ),
    _ReadingCase(
        text="京都府宇治市の小倉駅ですか、それとも福岡県北九州市の小倉駅ですか。",
        expected_kana="キョートフウジシノオグラエキデスカ、ソレトモフクオカケンキタキューシューシノコクラエキデスカ。",
        targets=(
            _TargetExpectation(
                surface="小倉",
                expected_pronunciation="オグラ",
            ),
            _TargetExpectation(
                surface="小倉",
                occurrence=1,
                expected_pronunciation="コクラ",
            ),
        ),
    ),
    _ReadingCase(
        text="人気の絶えない観光地だが、一本裏道に入ると急に人気がなくなる。",
        expected_kana="ヒトケノタエナイカンコーチダガ、イッポンウラミチニハイルトキューニヒトケガナクナル。",
        targets=(
            # TODO: 本来は「ニンキ」だが現状「ヒトケ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="人気",
            #     expected_pronunciation="ニンキ",
            # ),
            _TargetExpectation(
                surface="人気",
                occurrence=1,
                expected_pronunciation="ヒトケ",
            ),
        ),
        dictionary_targets=(
            _DictionaryReadingExpectation(
                surface="入る",
                expected_pronunciation="ハイル",
            ),
        ),
    ),
    _ReadingCase(
        text="八戸は県内第二の人口を有しており、家屋が八戸しか無いわけでは断じて無い。",
        expected_kana="ハチノヘワケンナイダイニノジンコーヲユーシテオリ、カオクガハチノヘシカナイワケデワダンジテナイ。",
        targets=(
            _TargetExpectation(
                surface="八戸",
                expected_pronunciation="ハチノヘ",
            ),
            # TODO: 本来は「ハチコ」だが現状「ハチノヘ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="八戸",
            #     occurrence=1,
            #     expected_pronunciation="ハチコ",
            # ),
        ),
    ),
    _ReadingCase(
        text="国立の大学であれば学費が安いらしい",
        expected_kana="コクリツノダイガクデアレバガクヒガヤスイラシイ",
        targets=(
            _TargetExpectation(
                surface="国立",
                expected_pronunciation="コクリツ",
            ),
        ),
    ),
    _ReadingCase(
        text="今日は中央線で国立に向かう",
        expected_kana="キョーワチューオーセンデクニタチニムカウ",
        targets=(
            _TargetExpectation(
                surface="国立",
                expected_pronunciation="クニタチ",
            ),
        ),
    ),
    _ReadingCase(
        text="天気いいし、皆で表に出て遊ぼ？",
        expected_kana="テンキイイシ、ミナデオモテニデテアソボ？",
        targets=(
            _TargetExpectation(
                surface="表",
                expected_pronunciation="オモテ",
            ),
        ),
    ),
    _ReadingCase(
        text="子供が相手を殴ってしまった。警察が動くような大事になる前に、相手の親と話し合うべきだ",
        expected_kana="コドモガアイテヲナグッテシマッタ。ケーサツガウゴクヨーナオーゴトニナルマエニ、アイテノオヤトハナシアウベキダ",
        targets=(
            _TargetExpectation(
                surface="大事",
                expected_pronunciation="オーゴト",
            ),
        ),
    ),
    _ReadingCase(
        text="将棋で玉を動かす。",
        expected_kana="ショーギデギョクヲウゴカス。",
        targets=(
            _TargetExpectation(
                surface="玉",
                expected_pronunciation="ギョク",
            ),
        ),
    ),
    _ReadingCase(
        text="愛しのあの子の愛し方がわからない。",
        expected_kana="イトシノアノコノイトシカタガワカラナイ。",
        targets=(
            _TargetExpectation(
                surface="愛し",
                expected_pronunciation="イトシ",
            ),
            # TODO: 本来は「アイシ」だが現状「イトシ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="愛し",
            #     occurrence=1,
            #     expected_pronunciation="アイシ",
            # ),
        ),
    ),
    _ReadingCase(
        text="歌が上手な彼女は、交渉事でも常に一枚上手であり、舞台の上手で堂々と振る舞った。",
        expected_kana="ウタガジョーズナカノジョワ、コーショーゴトデモツネニイチマイウワテデアリ、ブタイノカミテデドードートフルマッタ。",
        targets=(
            _TargetExpectation(
                surface="上手",
                expected_pronunciation="ジョーズ",
            ),
            # NOTE: 「一枚上手」は一つの複合語として収録済み
            _TargetExpectation(
                surface="上手",
                occurrence=2,
                expected_pronunciation="カミテ",
            ),
        ),
    ),
    _ReadingCase(
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
    _ReadingCase(
        text="この将棋では金か角を打てば勝ち。",
        expected_kana="コノショーギデワキンカカクヲウテバカチ。",
        targets=(
            _TargetExpectation(
                surface="金",
                expected_pronunciation="キン",
            ),
            _TargetExpectation(
                surface="角",
                expected_pronunciation="カク",
            ),
        ),
    ),
    _ReadingCase(
        text="風車とは、風を受けて回る羽根のついたおもちゃである。",
        expected_kana="カザグルマトワ、カゼヲウケテマワルハネノツイタオモチャデアル。",
        targets=(
            _TargetExpectation(
                surface="風車",
                expected_pronunciation="カザグルマ",
            ),
        ),
    ),
    _ReadingCase(
        text="ひらがなだけ",
        expected_kana="ヒラガナダケ",
        expect_no_diagnostics=True,
    ),
    # 以下は、既定モデルが読み分ける語彙と読みの組を、それぞれ1文以上で検証する症例
    ## メタデータに読みを追加しただけで実際には選ばれない組や、モデルの更新で読めなくなった組を見つけるため、語彙の全体を網羅する
    ## 文の多くは実在の文章から対象を含む1文を切り出し、読みの手がかりを残したまま短くしたもので、表層と読みの順に並べる
    _ReadingCase(
        text="新年早々しわくちゃのお札だとちょっといやですよね。",
        expected_kana="シンネンソーソーシワクチャノオフダダトチョットイヤデスヨネ。",
        targets=(
            # TODO: 本来は「オサツ」だが現状「オフダ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="お札",
            #     expected_pronunciation="オサツ",
            # ),
        ),
    ),
    _ReadingCase(
        text="特にテレビで紹介されたようなお札は需要が高まるんじゃないかなぁ。",
        expected_kana="トクニテレビデショーカイサレタヨーナオサツワジュヨーガタカマルンジャナイカナー。",
        targets=(
            _TargetExpectation(
                surface="お札",
                expected_pronunciation="オサツ",
            ),
        ),
    ),
    _ReadingCase(
        text="お札を買って、ついでにおみくじも！",
        expected_kana="オフダヲカッテ、ツイデニオミクジモ！",
        targets=(
            _TargetExpectation(
                surface="お札",
                expected_pronunciation="オフダ",
            ),
        ),
    ),
    _ReadingCase(
        text="彼はその後に角かくしの花嫁姿を一度も見ていないわけではないのだが、木田さんとの結婚が決まると、子供の時に見たその人の真っ白な化粧の印象が甦ってならなかった。",
        expected_kana="カレワソノアトニツノカクシノハナヨメスガタヲイチドモミテイナイワケデワナイノダガ、キダサントノケッコンガキマルト、コドモノトキニミタソノヒトノマッシロナケショーノインショーガヨミガエッテナラナカッタ。",
        targets=(
            _TargetExpectation(
                surface="その後",
                expected_pronunciation="ソノアト",
            ),
        ),
    ),
    _ReadingCase(
        text="その後、ヒヌカンとトートーメーにジールシンチをした旨を報告し",
        expected_kana="ソノゴ、ヒヌカントトートーメーニジールシンチヲシタムネヲホーコクシ",
        targets=(
            _TargetExpectation(
                surface="その後",
                expected_pronunciation="ソノゴ",
            ),
        ),
    ),
    _ReadingCase(
        text="そうすると一割一分くらいになってくるわけです。",
        expected_kana="ソースルトイチワリイチブクライニナッテクルワケデス。",
        targets=(
            _TargetExpectation(
                surface="一分",
                expected_pronunciation="イチブ",
            ),
        ),
    ),
    _ReadingCase(
        text="京阪本線淀駅より徒歩一分。",
        expected_kana="ケーハンホンセンヨドエキヨリトホイップン。",
        targets=(
            _TargetExpectation(
                surface="一分",
                expected_pronunciation="イップン",
            ),
        ),
    ),
    _ReadingCase(
        text="一味なのに黄色（黄金？",
        expected_kana="イチミナノニキイロ（オーゴン？",
        targets=(
            _TargetExpectation(
                surface="一味",
                expected_pronunciation="イチミ",
            ),
        ),
    ),
    _ReadingCase(
        text="はい……なんか一味足りない気が",
        expected_kana="ハイ……ナンカヒトアジタリナイキガ",
        targets=(
            _TargetExpectation(
                surface="一味",
                expected_pronunciation="ヒトアジ",
            ),
        ),
    ),
    _ReadingCase(
        text="汽笛一声の新橋−横浜間に続き、大阪−神戸間にも「陸（おか）蒸気」を走らせることが計画され、発着の大阪駅は当初、都心に近い堂島付近が候補地にあがった。",
        expected_kana="キテキイッセーノシンバシ−ヨコハマカンニツズキ、オーサカ−コーベカンニモ「リク（オカ）ジョーキ」ヲハシラセルコトガケーカクサレ、ハッチャクノオーサカエキワトーショ、トシンニチカイドージマフキンガコーホチニアガッタ。",
        targets=(
            _TargetExpectation(
                surface="一声",
                expected_pronunciation="イッセー",
            ),
        ),
    ),
    _ReadingCase(
        text="その一声聞くと嬉しくなるよねー。",
        expected_kana="ソノヒトコエキクトウレシクナルヨネー。",
        targets=(
            _TargetExpectation(
                surface="一声",
                expected_pronunciation="ヒトコエ",
            ),
        ),
    ),
    _ReadingCase(
        text="一寸延ばし、五分延ばしみたいな形でやっていくような形で来ておられる。",
        expected_kana="チョットノバシ、ゴブノバシミタイナカタチデヤッテイクヨーナカタチデキテオラレル。",
        targets=(
            # TODO: 本来は「イッスン」だが現状「チョット」が選ばれてしまう
            # _TargetExpectation(
            #     surface="一寸",
            #     expected_pronunciation="イッスン",
            # ),
        ),
    ),
    _ReadingCase(
        text="鎌鼬の太刀は総馬の体一寸（約三センチメートル）手前で止められた。",
        expected_kana="カマイタチノタチワソーマノカラダイッスン（ヤクサンセンチメートル）テマエデトメラレタ。",
        targets=(
            _TargetExpectation(
                surface="一寸",
                expected_pronunciation="イッスン",
            ),
        ),
    ),
    _ReadingCase(
        text="読んでおくべき一文だと思います。",
        expected_kana="ヨンデオクベキイチブンダトオモイマス。",
        targets=(
            _TargetExpectation(
                surface="一文",
                expected_pronunciation="イチブン",
            ),
        ),
    ),
    _ReadingCase(
        text="でも、こういう一文の得にもならないようなことを考えるのは好きなので、楽しくもあるんですよね。",
        expected_kana="デモ、コーユウイチモンノトクニモナラナイヨーナコトヲカンガエルノワスキナノデ、タノシクモアルンデスヨネ。",
        targets=(
            _TargetExpectation(
                surface="一文",
                expected_pronunciation="イチモン",
            ),
        ),
    ),
    _ReadingCase(
        text="八時から十一時だな。",
        expected_kana="ハチジカラジューイチジダナ。",
        targets=(
            _TargetExpectation(
                surface="一時",
                expected_pronunciation="イチジ",
            ),
        ),
    ),
    _ReadingCase(
        text="一時も目が離せない展開になりそうです。",
        expected_kana="イットキモメガハナセナイテンカイニナリソーデス。",
        targets=(
            _TargetExpectation(
                surface="一時",
                expected_pronunciation="イットキ",
            ),
        ),
    ),
    _ReadingCase(
        text="東京にとっては、ほっとする一時であろう。",
        expected_kana="トーキョーニトッテワ、ホットスルヒトトキデアロー。",
        targets=(
            _TargetExpectation(
                surface="一時",
                expected_pronunciation="ヒトトキ",
            ),
        ),
    ),
    _ReadingCase(
        text="これを受け行政院（内閣に相当）は批准文書を十二月二日にダブリューティーオー事務局へ送付、来年一月一日の台湾加盟が実現する運びとなる。",
        expected_kana="コレヲウケギョーセーイン（ナイカクニソートー）ワヒジュンブンショヲジューニガツフツカニダブリューティーオージムキョクエソーフ、ライネンイチガツツイタチノタイワンカメーガジツゲンスルハコビトナル。",
        targets=(
            _TargetExpectation(
                surface="一月",
                expected_pronunciation="イチガツ",
            ),
        ),
    ),
    _ReadingCase(
        text="一月遅れてやっと到着しました。",
        expected_kana="ヒトツキオクレテヤットトーチャクシマシタ。",
        targets=(
            _TargetExpectation(
                surface="一月",
                expected_pronunciation="ヒトツキ",
            ),
        ),
    ),
    _ReadingCase(
        text="集団の中でも、ムードメーカーになることが多く、影のご意見番として一目置かれるはず。",
        expected_kana="シューダンノナカデモ、ムードメーカーニナルコトガオーク、カゲノゴイケンバントシテイチモクオカレルハズ。",
        targets=(
            _TargetExpectation(
                surface="一目",
                expected_pronunciation="イチモク",
            ),
        ),
    ),
    _ReadingCase(
        text="一目十万本といわれる日本一の。",
        expected_kana="イチモクジューマンホントイワレルニッポンイチノ。",
        targets=(
            # TODO: 本来は「ヒトメ」だが現状「イチモク」が選ばれてしまう
            # _TargetExpectation(
            #     surface="一目",
            #     expected_pronunciation="ヒトメ",
            # ),
        ),
    ),
    _ReadingCase(
        text="嘉子にも一目会って帰りたい。",
        expected_kana="ヨシコニモヒトメアッテカエリタイ。",
        targets=(
            _TargetExpectation(
                surface="一目",
                expected_pronunciation="ヒトメ",
            ),
        ),
    ),
    _ReadingCase(
        text="飯田）計画の一端であると。",
        expected_kana="イーダ）ケーカクノイッタンデアルト。",
        targets=(
            _TargetExpectation(
                surface="一端",
                expected_pronunciation="イッタン",
            ),
        ),
    ),
    _ReadingCase(
        text="小学生は一端のお手伝いさんですが…。",
        expected_kana="ショーガクセーワイッパシノオテツダイサンデスガ…。",
        targets=(
            _TargetExpectation(
                surface="一端",
                expected_pronunciation="イッパシ",
            ),
        ),
    ),
    _ReadingCase(
        text="あら、一見さんですか！",
        expected_kana="アラ、イチゲンサンデスカ！",
        targets=(
            _TargetExpectation(
                surface="一見",
                expected_pronunciation="イチゲン",
            ),
        ),
    ),
    _ReadingCase(
        text="そのせいで、一見すると、実際よりも不細工に見える。",
        expected_kana="ソノセイデ、イッケンスルト、ジッサイヨリモブサイクニミエル。",
        targets=(
            _TargetExpectation(
                surface="一見",
                expected_pronunciation="イッケン",
            ),
        ),
    ),
    _ReadingCase(
        text="数江が一言のもとに否定した。",
        expected_kana="カズエガヒトコトノモトニヒテーシタ。",
        targets=(
            # TODO: 本来は「イチゴン」だが現状「ヒトコト」が選ばれてしまう
            # _TargetExpectation(
            #     surface="一言",
            #     expected_pronunciation="イチゴン",
            # ),
        ),
    ),
    _ReadingCase(
        text="総馬の一言は数江の逆鱗にふれた。",
        expected_kana="ソーマノヒトコトワカズエノゲキリンニフレタ。",
        targets=(
            _TargetExpectation(
                surface="一言",
                expected_pronunciation="ヒトコト",
            ),
        ),
    ),
    _ReadingCase(
        text="そして改造に一足５０００円ー…。",
        expected_kana="ソシテカイゾーニイッソクゴセンエンー…。",
        targets=(
            _TargetExpectation(
                surface="一足",
                expected_pronunciation="イッソク",
            ),
        ),
    ),
    _ReadingCase(
        text="総馬がうれしそうに言ったとき、二人の間合いは一足一刀になった。",
        expected_kana="ソーマガウレシソーニイッタトキ、フタリノマアイワヒトアシイットーニナッタ。",
        targets=(
            # TODO: 本来は「イッソク」だが現状「ヒトアシ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="一足",
            #     expected_pronunciation="イッソク",
            # ),
        ),
    ),
    _ReadingCase(
        text="一足先にレストランで大祝賀。",
        expected_kana="ヒトアシサキニレストランデダイシュクガ。",
        targets=(
            _TargetExpectation(
                surface="一足",
                expected_pronunciation="ヒトアシ",
            ),
        ),
    ),
    _ReadingCase(
        text="福山への一途な思いを強調。",
        expected_kana="フクヤマエノイチズナオモイヲキョーチョー。",
        targets=(
            _TargetExpectation(
                surface="一途",
                expected_pronunciation="イチズ",
            ),
        ),
    ),
    _ReadingCase(
        text="外部環境の変化は加速の一途をたどり、それに伴い「Ｍ＆Ａ」も活況です。",
        expected_kana="ガイブカンキョーノヘンカワカソクノイットヲタドリ、ソレニトモナイ「Ｍ＆Ａ」モカッキョーデス。",
        targets=(
            _TargetExpectation(
                surface="一途",
                expected_pronunciation="イット",
            ),
        ),
    ),
    _ReadingCase(
        text="三国じゃないし……四人だし……",
        expected_kana="サンゴクジャナイシ……ヨニンダシ……",
        targets=(
            _TargetExpectation(
                surface="三国",
                expected_pronunciation="サンゴク",
            ),
        ),
    ),
    _ReadingCase(
        text="はるか三国山脈の北部丹後山付近に源流をもつ利根川は、埼玉のこのあたりでは川幅も広く、ゆったりとした流れになる。",
        expected_kana="ハルカミクニサンミャクノホクブタンゴヤマフキンニゲンリューヲモツトネガワワ、サイタマノコノアタリデワカワハバモヒロク、ユッタリトシタナガレニナル。",
        targets=(
            _TargetExpectation(
                surface="三国",
                expected_pronunciation="ミクニ",
            ),
        ),
    ),
    _ReadingCase(
        text="三次バックアップまで完了してるわよ",
        expected_kana="サンジバックアップマデカンリョーシテルワヨ",
        targets=(
            _TargetExpectation(
                surface="三次",
                expected_pronunciation="サンジ",
            ),
        ),
    ),
    _ReadingCase(
        text="内訳は福山、尾道、三次の各市のそれぞれ１人。",
        expected_kana="ウチワケワフクヤマ、オノミチ、ミヨシノカクシノソレゾレヒトリ。",
        targets=(
            _TargetExpectation(
                surface="三次",
                expected_pronunciation="ミヨシ",
            ),
        ),
    ),
    _ReadingCase(
        text="新年の初販売は三田阪急！",
        expected_kana="シンネンノハツハンバイワサンダハンキュー！",
        targets=(
            _TargetExpectation(
                surface="三田",
                expected_pronunciation="サンダ",
            ),
        ),
    ),
    _ReadingCase(
        text="本日ＴＢＳ「ひるおび！」に三田寛子が出演します。",
        expected_kana="ホンジツティービーエス「ヒルオビ！」ニミタヒロコガシュツエンシマス。",
        targets=(
            _TargetExpectation(
                surface="三田",
                expected_pronunciation="ミタ",
            ),
        ),
    ),
    _ReadingCase(
        text="東京市芝三田小山町に生まれる。",
        expected_kana="トーキョーシシバサンダオヤママチニウマレル。",
        targets=(
            # TODO: 本来は「ミタ」だが現状「サンダ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="三田",
            #     expected_pronunciation="ミタ",
            # ),
        ),
    ),
    _ReadingCase(
        text="えー、次の汽車は特別急行夏の大三角行きです。",
        expected_kana="エー、ツギノキシャワトクベツキューコーナツノダイミスミイキデス。",
        targets=(
            # TODO: 本来は「サンカク」だが現状「ミスミ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="三角",
            #     expected_pronunciation="サンカク",
            # ),
        ),
    ),
    _ReadingCase(
        text="たしか……夏の大三角見るんだよね。",
        expected_kana="タシカ……ナツノダイサンカクミルンダヨネ。",
        targets=(
            _TargetExpectation(
                surface="三角",
                expected_pronunciation="サンカク",
            ),
        ),
    ),
    _ReadingCase(
        text="汽船で三角港から島原港へ行く。",
        expected_kana="キセンデミスミコーカラシマバラコーエイク。",
        targets=(
            _TargetExpectation(
                surface="三角",
                expected_pronunciation="ミスミ",
            ),
        ),
    ),
    _ReadingCase(
        text="自分の方が上であると。",
        expected_kana="ジブンノホーガウエデアルト。",
        targets=(
            _TargetExpectation(
                surface="上",
                expected_pronunciation="ウエ",
            ),
        ),
    ),
    _ReadingCase(
        text="上下姿に朱の着物、白扇を腰に威儀を正して御殿へとのぼっていった。",
        expected_kana="カミシモスガタニシュノキモノ、ハクセンヲコシニイギヲタダシテゴテンエトノボッテイッタ。",
        targets=(
            _TargetExpectation(
                surface="上下",
                expected_pronunciation="カミシモ",
            ),
        ),
    ),
    _ReadingCase(
        text="体が左右、上下に揺れる。",
        expected_kana="カラダガサユー、ジョーゲニユレル。",
        targets=(
            _TargetExpectation(
                surface="上下",
                expected_pronunciation="ジョーゲ",
            ),
        ),
    ),
    _ReadingCase(
        text="両横綱は日馬富士が貴ノ岩を押し倒して１敗を堅持し、鶴竜は小結魁聖を上手ひねりで退けて４勝目を挙げた。",
        expected_kana="リョーヨコズナワハルマフジガタカノイワヲオシタオシテイッパイヲケンジシ、ツルリューワコムスビサキガケヒジリヲウワテヒネリデシリゾケテヨンショーメヲアゲタ。",
        targets=(
            _TargetExpectation(
                surface="上手",
                expected_pronunciation="ウワテ",
            ),
        ),
    ),
    _ReadingCase(
        text="お父さんがね、上方漫才大好きなんだ",
        expected_kana="オトーサンガネ、カミガタマンザイダイスキナンダ",
        targets=(
            _TargetExpectation(
                surface="上方",
                expected_pronunciation="カミガタ",
            ),
        ),
    ),
    _ReadingCase(
        text="上方から観察していたんですね。",
        expected_kana="ジョーホーカラカンサツシテイタンデスネ。",
        targets=(
            _TargetExpectation(
                surface="上方",
                expected_pronunciation="ジョーホー",
            ),
        ),
    ),
    _ReadingCase(
        text="日本統治下で断種、堕胎",
        expected_kana="ニホントーチカデダンシュ、ダタイ",
        targets=(
            _TargetExpectation(
                surface="下",
                expected_pronunciation="カ",
            ),
        ),
    ),
    _ReadingCase(
        text="タイムパラドックスものの映画としては、下の下。",
        expected_kana="タイムパラドックスモノノエーガトシテワ、シタノシタ。",
        targets=(
            # TODO: 本来は「ゲ」だが現状「シタ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="下",
            #     expected_pronunciation="ゲ",
            # ),
        ),
    ),
    _ReadingCase(
        text="澄み渡った秋空の下…",
        expected_kana="スミワタッタアキゾラノシタ…",
        targets=(
            _TargetExpectation(
                surface="下",
                expected_pronunciation="シタ",
            ),
        ),
    ),
    _ReadingCase(
        text="よって現行の関税割当制度の下では、国産ＮＣは国産価格より五十一パーセント安い輸入ＮＣとまでなら併用されるメリットを有することになる。",
        expected_kana="ヨッテゲンコーノカンゼーワリアテセードノモトデワ、コクサンエヌシーワコクサンカカクヨリゴジューイチパーセントヤスイユニューエヌシートマデナラヘーヨーサレルメリットヲユースルコトニナル。",
        targets=(
            _TargetExpectation(
                surface="下",
                expected_pronunciation="モト",
            ),
        ),
    ),
    _ReadingCase(
        text="今回はしめみゅをメインで見ようと思ってて、そしたらしめみゅで横花下手にいたもんだからありがたかった。",
        expected_kana="コンカイワシメミユヲメインデミヨートオモッテテ、ソシタラシメミュデヨコハナヘタニイタモンダカラアリガタカッタ。",
        targets=(
            # TODO: 本来は「シモテ」だが現状「ヘタ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="下手",
            #     expected_pronunciation="シモテ",
            # ),
        ),
    ),
    _ReadingCase(
        text="２カメ（舞台下手）の担当は鳴海さん。",
        expected_kana="ニカメ（ブタイシモテ）ノタントーワナルミサン。",
        targets=(
            _TargetExpectation(
                surface="下手",
                expected_pronunciation="シモテ",
            ),
        ),
    ),
    _ReadingCase(
        text="言葉が足りなかろうと、他人と交流を交わすのが下手であろうと、思いを伝えることを恐れることはすまい。",
        expected_kana="コトバガタリナカロート、タニントコーリューヲカワスノガヘタデアロート、オモイヲツタエルコトヲオソレルコトワスマイ。",
        targets=(
            _TargetExpectation(
                surface="下手",
                expected_pronunciation="ヘタ",
            ),
        ),
    ),
    _ReadingCase(
        text="２００６年の長野県知事選挙で田中は下野し、田中に代わり村井仁が知事に就任した。",
        expected_kana="ニセンロクネンノナガノケンチジセンキョデタナカワゲヤシ、タナカニカワリムライヒトシガチジニシューニンシタ。",
        targets=(
            _TargetExpectation(
                surface="下野",
                expected_pronunciation="ゲヤ",
            ),
        ),
    ),
    _ReadingCase(
        text="初夏を迎え山田池公園ではシモツケ（下野）の花が咲き出してきました。",
        expected_kana="ショカヲムカエヤマダイケコーエンデワシモツケ（シモツケ）ノハナガサキダシテキマシタ。",
        targets=(
            _TargetExpectation(
                surface="下野",
                expected_pronunciation="シモツケ",
            ),
        ),
    ),
    _ReadingCase(
        text="大陸中に黄巾党の討伐命令が回ってるのよ。",
        expected_kana="タイリクジューニコーキントーノトーバツメーレーガマワッテルノヨ。",
        targets=(
            _TargetExpectation(
                surface="中",
                expected_pronunciation="ジュー",
            ),
        ),
    ),
    _ReadingCase(
        text="支度中の邦子さん",
        expected_kana="シタクチューノクニコサン",
        targets=(
            _TargetExpectation(
                surface="中",
                expected_pronunciation="チュー",
            ),
        ),
    ),
    _ReadingCase(
        text="先発ピッチャーは中日。",
        expected_kana="センパツピッチャーワチューニチ。",
        targets=(
            _TargetExpectation(
                surface="中日",
                expected_pronunciation="チューニチ",
            ),
        ),
    ),
    _ReadingCase(
        text="中日（８日目）や国民の祝日は、好角家の著名人がゲストに招かれることが多い。",
        expected_kana="チューニチ（ヨーカメ）ヤコクミンノシュクジツワ、コーカクカノチョメージンガゲストニマネカレルコトガオーイ。",
        targets=(
            # TODO: 本来は「ナカビ」だが現状「チューニチ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="中日",
            #     expected_pronunciation="ナカビ",
            # ),
        ),
    ),
    _ReadingCase(
        text="連休中日の日曜日、若松謙維さんと仙北市ー大仙市へと走らせていただきました。",
        expected_kana="レンキューナカビノニチヨービ、ワカマツカネシゲサントセンボクシーダイセンシエトハシラセテイタダキマシタ。",
        targets=(
            _TargetExpectation(
                surface="中日",
                expected_pronunciation="ナカビ",
            ),
        ),
    ),
    _ReadingCase(
        text="中間考査があるからこそよ。",
        expected_kana="チューカンコーサガアルカラコソヨ。",
        targets=(
            _TargetExpectation(
                surface="中間",
                expected_pronunciation="チューカン",
            ),
        ),
    ),
    _ReadingCase(
        text="西教寺には光秀の寄進状が残されており、そこに記された１８人のうち一名は武士ではなく中間であった。",
        expected_kana="サイキョージニワミツヒデノキシンジョーガノコサレテオリ、ソコニシルサレタジューハチニンノウチイチメーワブシデワナクチューゲンデアッタ。",
        targets=(
            _TargetExpectation(
                surface="中間",
                expected_pronunciation="チューゲン",
            ),
        ),
    ),
    _ReadingCase(
        text="日本語は、主に日本国内で使用される。",
        expected_kana="ニホンゴワ、オモニニホンコクナイデシヨーサレル。",
        targets=(
            _TargetExpectation(
                surface="主",
                expected_pronunciation="オモ",
            ),
        ),
    ),
    _ReadingCase(
        text="日曜日の礼拝では、信者たちが声をそろえて主の祈りを唱えた。",
        expected_kana="ニチヨービノレーハイデワ、シンジャタチガコエヲソロエテシュノイノリヲトナエタ。",
        targets=(
            _TargetExpectation(
                surface="主",
                expected_pronunciation="シュ",
            ),
        ),
    ),
    _ReadingCase(
        text="この池には主と呼ばれる大きな鯉が住んでいて、釣り人の間では昔から有名だ。",
        expected_kana="コノイケニワヌシトヨバレルオーキナコイガスンデイテ、ツリジンノアイダデワムカシカラユーメーダ。",
        targets=(
            _TargetExpectation(
                surface="主",
                expected_pronunciation="ヌシ",
            ),
        ),
    ),
    _ReadingCase(
        text="なお、豊川六段の二歩はのちにフジテレビ系列トリビアの泉でも取り上げられ、「７８へぇ」の得点を獲得。",
        expected_kana="ナオ、トヨカワロクダンノニフワノチニフジテレビケーレツトリビアノイズミデモトリアゲラレ、「ナナジューハチヘー」ノトクテンヲカクトク。",
        targets=(
            _TargetExpectation(
                surface="二歩",
                expected_pronunciation="ニフ",
            ),
        ),
    ),
    _ReadingCase(
        text="二歩戻らないようにしないとね！",
        expected_kana="ニホモドラナイヨーニシナイトネ！",
        targets=(
            _TargetExpectation(
                surface="二歩",
                expected_pronunciation="ニホ",
            ),
        ),
    ),
    _ReadingCase(
        text="二重の扉とかがあるそうですし。",
        expected_kana="ニジューノトビラトカガアルソーデスシ。",
        targets=(
            _TargetExpectation(
                surface="二重",
                expected_pronunciation="ニジュー",
            ),
        ),
    ),
    _ReadingCase(
        text="比較的まぶたが薄めの方は、まつげカールをかけると二重になることもあります。",
        expected_kana="ヒカクテキマブタガウスメノカタワ、マツゲカールヲカケルトフタエニナルコトモアリマス。",
        targets=(
            _TargetExpectation(
                surface="二重",
                expected_pronunciation="フタエ",
            ),
        ),
    ),
    _ReadingCase(
        text="午後５時十五分から、べるが通り、十五ｋｍ、１時間三十三分三十二秒。",
        expected_kana="ゴゴゴジジューゴフンカラ、ベルガトーリ、ジューゴキロメートル、イチジカンサンジューサンプンサンジューニビョー。",
        targets=(
            _TargetExpectation(
                surface="五分",
                expected_pronunciation="ゴフン",
                expected_outcome="dictionary_default_protected",
                was_preserved=True,
            ),
        ),
    ),
    _ReadingCase(
        text="１セット、ハンデをもらっているのに、五分にも持っていけないとは…。",
        expected_kana="ヒトセット、ハンデヲモラッテイルノニ、ゴフンニモモッテイケナイトワ…。",
        targets=(
            # TODO: 本来は「ゴブ」だが現状「ゴフン」が選ばれてしまう
            # _TargetExpectation(
            #     surface="五分",
            #     expected_pronunciation="ゴブ",
            # ),
        ),
    ),
    _ReadingCase(
        text="お屋形様と京に行ったんじゃ……",
        expected_kana="オヤカタサマトキョーニイッタンジャ……",
        targets=(
            _TargetExpectation(
                surface="京",
                expected_pronunciation="キョー",
            ),
        ),
    ),
    _ReadingCase(
        text="今年はスーパーコンピューター「京」の稼働も控えており、中核施設が勢揃いすることになります。",
        expected_kana="コトシワスーパーコンピューター「ケー」ノカドーモヒカエテオリ、チューカクシセツガセーゾロイスルコトニナリマス。",
        targets=(
            _TargetExpectation(
                surface="京",
                expected_pronunciation="ケー",
            ),
        ),
    ),
    _ReadingCase(
        text="もうひとつは人事交流です。",
        expected_kana="モーヒトツワジンジコーリューデス。",
        targets=(
            _TargetExpectation(
                surface="人事",
                expected_pronunciation="ジンジ",
            ),
        ),
    ),
    _ReadingCase(
        text="中途採用が思うように進まないらしく、同じ部署の採用担当（人事ではない）からヒアリングを受ける。",
        expected_kana="チュートサイヨーガオモウヨーニススマナイラシク、オナジブショノサイヨータントー（ヒトゴトデワナイ）カラヒアリングヲウケル。",
        targets=(
            # TODO: 本来は「ジンジ」だが現状「ヒトゴト」が選ばれてしまう
            # _TargetExpectation(
            #     surface="人事",
            #     expected_pronunciation="ジンジ",
            # ),
        ),
    ),
    _ReadingCase(
        text="おまえも人事ではないぞ？",
        expected_kana="オマエモヒトゴトデワナイゾ？",
        targets=(
            _TargetExpectation(
                surface="人事",
                expected_pronunciation="ヒトゴト",
            ),
        ),
    ),
    _ReadingCase(
        text="−それが、今日なのだ。",
        expected_kana="−ソレガ、キョーナノダ。",
        targets=(
            _TargetExpectation(
                surface="今日",
                expected_pronunciation="キョー",
            ),
        ),
    ),
    _ReadingCase(
        text="以後、２年に１回ずつこの会議は開催され今日に至っている。",
        expected_kana="イゴ、ニネンニイッカイズツコノカイギワカイサイサレコンニチニイタッテイル。",
        targets=(
            _TargetExpectation(
                surface="今日",
                expected_pronunciation="コンニチ",
            ),
        ),
    ),
    _ReadingCase(
        text="Ｆ組で中田浩二のマルセイユ（仏）は１−０でヘーレンフェイン（オランダ）を下して２連勝。",
        expected_kana="Ｆグミデナカタコージノマルセイユ（フツ）ワイチ−ゼロデヘーレンフェイン（オランダ）ヲクダシテニレンショー。",
        targets=(
            _TargetExpectation(
                surface="仏",
                expected_pronunciation="フツ",
            ),
        ),
    ),
    _ReadingCase(
        text="つねに仏のような心でいたい。",
        expected_kana="ツネニホトケノヨーナココロデイタイ。",
        targets=(
            _TargetExpectation(
                surface="仏",
                expected_pronunciation="ホトケ",
            ),
        ),
    ),
    _ReadingCase(
        text="京都チャンネルで最後のＯＡとなった２００８−９年の京都の仏を収録！",
        expected_kana="キョートチャンネルデサイゴノオーエイトナッタニーゼロゼロハチ−キューネンノキョートノフツヲシューロク！",
        targets=(
            # TODO: 本来は「ホトケ」だが現状「フツ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="仏",
            #     expected_pronunciation="ホトケ",
            # ),
        ),
    ),
    _ReadingCase(
        text="このような保税地域以外の指定された場所に外国貨物を置くことを他所蔵置という。",
        expected_kana="コノヨーナホゼーチイキイガイノシテーサレタバショニガイコクカモツヲオクコトヲタショゾーチトイウ。",
        targets=(
            _TargetExpectation(
                surface="他所",
                expected_pronunciation="タショ",
            ),
        ),
    ),
    _ReadingCase(
        text="他所じゃオクラの苗を見るけど？",
        expected_kana="ヨソジャオクラノナエヲミルケド？",
        targets=(
            _TargetExpectation(
                surface="他所",
                expected_pronunciation="ヨソ",
            ),
        ),
    ),
    _ReadingCase(
        text="「仮名目録」って難しそうな巻物でございます。",
        expected_kana="「カナモクロク」ッテムズカシソーナマキモノデゴザイマス。",
        targets=(
            _TargetExpectation(
                surface="仮名",
                expected_pronunciation="カナ",
            ),
        ),
    ),
    _ReadingCase(
        text="脱がされたキングさん（仮名）。",
        expected_kana="ヌガサレタキングサン（カメー）。",
        targets=(
            _TargetExpectation(
                surface="仮名",
                expected_pronunciation="カメー",
            ),
        ),
    ),
    _ReadingCase(
        text="元八郎は無理に体を割りいれた。",
        expected_kana="モトハチローワムリニカラダヲワリイレタ。",
        targets=(
            _TargetExpectation(
                surface="体",
                expected_pronunciation="カラダ",
            ),
        ),
    ),
    _ReadingCase(
        text="でも、砥上にしてみれば、それは体のいい言い訳にしか聞こえないに違いない。",
        expected_kana="デモ、トガミニシテミレバ、ソレワカラダノイイイーワケニシカキコエナイニチガイナイ。",
        targets=(
            # TODO: 本来は「テー」だが現状「カラダ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="体",
            #     expected_pronunciation="テー",
            # ),
        ),
    ),
    _ReadingCase(
        text="何か食物をくれ」",
        expected_kana="ナニカショクモツヲクレ」",
        targets=(
            _TargetExpectation(
                surface="何",
                expected_pronunciation="ナニ",
            ),
        ),
    ),
    _ReadingCase(
        text="何をしても許されるものではない。",
        expected_kana="ナンヲシテモユルサレルモノデワナイ。",
        targets=(
            # TODO: 本来は「ナニ」だが現状「ナン」が選ばれてしまう
            # _TargetExpectation(
            #     surface="何",
            #     expected_pronunciation="ナニ",
            # ),
        ),
    ),
    _ReadingCase(
        text="アイドルの人達には「お前の母ちゃん何人だあ！",
        expected_kana="アイドルノヒトタチニワ「オマエノカーチャンナニジンダア！",
        targets=(
            _TargetExpectation(
                surface="何人",
                expected_pronunciation="ナニジン",
            ),
        ),
    ),
    _ReadingCase(
        text="何人も、裁判所において裁判を受ける権利を奪はれない。",
        expected_kana="ナンニンモ、サイバンショニオイテサイバンヲウケルケンリヲダツハレナイ。",
        targets=(
            # TODO: 本来は「ナニビト」だが現状「ナンニン」が選ばれてしまう
            # _TargetExpectation(
            #     surface="何人",
            #     expected_pronunciation="ナニビト",
            # ),
        ),
    ),
    _ReadingCase(
        text="何分、古すぎた。",
        expected_kana="ナンプン、フルスギタ。",
        targets=(
            # TODO: 本来は「ナニブン」だが現状「ナンフン」が選ばれてしまう
            # _TargetExpectation(
            #     surface="何分",
            #     expected_pronunciation="ナニブン",
            # ),
        ),
    ),
    _ReadingCase(
        text="でも何時も急ぎ足で駅に向かう事になるんですよね。",
        expected_kana="デモイツモイソギアシデエキニムカウコトニナルンデスヨネ。",
        targets=(
            _TargetExpectation(
                surface="何時",
                expected_pronunciation="イツ",
            ),
        ),
    ),
    _ReadingCase(
        text="マル夜は何時から何時まで寝ますか？",
        expected_kana="マルヨルワナンジカラナンジマデネマスカ？",
        targets=(
            _TargetExpectation(
                surface="何時",
                expected_pronunciation="ナンジ",
            ),
        ),
    ),
    _ReadingCase(
        text="最近はいつ何時何が起こるかわかりません",
        expected_kana="サイキンワイツナンドキナニガオコルカワカリマセン",
        targets=(
            _TargetExpectation(
                surface="何時",
                expected_pronunciation="ナンドキ",
            ),
        ),
    ),
    _ReadingCase(
        text="お好きな色は何色ですか？",
        expected_kana="オスキナイロワナニイロデスカ？",
        targets=(
            _TargetExpectation(
                surface="何色",
                expected_pronunciation="ナニイロ",
            ),
        ),
    ),
    _ReadingCase(
        text="セルフは何色かありましたよ。",
        expected_kana="セルフワナンショクカアリマシタヨ。",
        targets=(
            _TargetExpectation(
                surface="何色",
                expected_pronunciation="ナンショク",
            ),
        ),
    ),
    _ReadingCase(
        text="従来の映画と比較して見ても、江戸時代の戯作者の作物から、急に自然派文学に接した心地がする",
        expected_kana="ジューライノエーガトヒカクシテミテモ、エドジダイノゲサクシャノサクブツカラ、キューニシゼンハブンガクニセッシタココチガスル",
        targets=(
            _TargetExpectation(
                surface="作物",
                expected_pronunciation="サクブツ",
            ),
        ),
    ),
    _ReadingCase(
        text="スイカ以外にはどんな作物を？",
        expected_kana="スイカイガイニワドンナサクモツヲ？",
        targets=(
            _TargetExpectation(
                surface="作物",
                expected_pronunciation="サクモツ",
            ),
        ),
    ),
    _ReadingCase(
        text="よどみに浮ぶうたかたは、かつ消えかつ結びて久しくとどまりたる例なし。",
        expected_kana="ヨドミニウカブウタカタワ、カツキエカツムスビテヒサシクトドマリタルレーナシ。",
        targets=(
            # TODO: 本来は「タメシ」だが現状「レー」が選ばれてしまう
            # _TargetExpectation(
            #     surface="例",
            #     expected_pronunciation="タメシ",
            # ),
        ),
    ),
    _ReadingCase(
        text="子役の受賞は例がない。",
        expected_kana="コヤクノジュショーワレーガナイ。",
        targets=(
            _TargetExpectation(
                surface="例",
                expected_pronunciation="レー",
            ),
        ),
    ),
    _ReadingCase(
        text="パリ周辺にはシャルル、ドゴール、オルリーの二国際空港があるが、離着陸便の増加に伴い利用者数が急増。",
        expected_kana="パリシューヘンニワシャルル、ドゴール、オルリーノニコクサイクーコーガアルガ、リチャクリクビンノゾーカニトモナイリヨーシャスーガキューゾー。",
        targets=(
            _TargetExpectation(
                surface="便",
                expected_pronunciation="ビン",
            ),
        ),
    ),
    _ReadingCase(
        text="柔らかい無形便の排出（文献十八）",
        expected_kana="ヤワラカイムケーベンノハイシュツ（ブンケンジューハチ）",
        targets=(
            _TargetExpectation(
                surface="便",
                expected_pronunciation="ベン",
            ),
        ),
    ),
    _ReadingCase(
        text="二つの式を連立させて、xとyの値をそれぞれ求めなさい。",
        expected_kana="フタツノシキヲレンリツサセテ、ｘトｙノアタイヲソレゾレモトメナサイ。",
        targets=(
            _TargetExpectation(
                surface="値",
                expected_pronunciation="アタイ",
            ),
        ),
    ),
    _ReadingCase(
        text="決算発表を受けて、この会社の株の値が一日で二割も上がった。",
        expected_kana="ケッサンハッピョーヲウケテ、コノカイシャノカブノネガイチニチデニワリモアガッタ。",
        targets=(
            _TargetExpectation(
                surface="値",
                expected_pronunciation="ネ",
            ),
        ),
    ),
    _ReadingCase(
        text="「水鏡」で耳をふさいだのは「わたし」の側だ。",
        expected_kana="「ミズカガミ」デミミヲフサイダノワ「ワタシ」ノソバダ。",
        targets=(
            # TODO: 本来は「ガワ」だが現状「ソバ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="側",
            #     expected_pronunciation="ガワ",
            # ),
        ),
    ),
    _ReadingCase(
        text="「北風の国には行かず、俺の側に留まると−。",
        expected_kana="「キタカゼノクニニワイカズ、オレノソバニトドマルト−。",
        targets=(
            _TargetExpectation(
                surface="側",
                expected_pronunciation="ソバ",
            ),
        ),
    ),
    _ReadingCase(
        text="飲んでる側から汗で水分出ちゃいます",
        expected_kana="ノンデルガワカラアセデスイブンデチャイマス",
        targets=(
            # TODO: 本来は「ソバ」だが現状「ガワ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="側",
            #     expected_pronunciation="ソバ",
            # ),
        ),
    ),
    _ReadingCase(
        text="十三世紀、元の皇帝フビライは日本に服属を求める使者を何度も送った。",
        expected_kana="ジューサンセーキ、モトノコーテーフビライワニホンニフクゾクヲモトメルシシャヲナンドモオクッタ。",
        targets=(
            # TODO: 本来は「ゲン」だが現状「モト」が選ばれてしまう
            # _TargetExpectation(
            #     surface="元",
            #     expected_pronunciation="ゲン",
            # ),
        ),
    ),
    _ReadingCase(
        text="通貨の円と元とウォンの語源を教えてください。",
        expected_kana="ツーカノエントモトトウォンノゴゲンヲオシエテクダサイ。",
        targets=(
            # TODO: 本来は「ゲン」だが現状「モト」が選ばれてしまう
            # _TargetExpectation(
            #     surface="元",
            #     expected_pronunciation="ゲン",
            # ),
        ),
    ),
    _ReadingCase(
        text="掃除が終わったら、借りた道具はすべて元の場所に戻しておいてください。",
        expected_kana="ソージガオワッタラ、カリタドーグワスベテモトノバショニモドシテオイテクダサイ。",
        targets=(
            _TargetExpectation(
                surface="元",
                expected_pronunciation="モト",
            ),
        ),
    ),
    _ReadingCase(
        text="カゴに入れれば洗ってくれる。",
        expected_kana="カゴニイレレバアラッテクレル。",
        targets=(
            _TargetExpectation(
                surface="入れれ",
                expected_pronunciation="イレレ",
            ),
        ),
    ),
    _ReadingCase(
        text="今はただ自害せん」と言い残して入水したとも言われている。",
        expected_kana="イマワタダジガイセン」トイーノコシテジュスイシタトモイワレテイル。",
        targets=(
            _TargetExpectation(
                surface="入水",
                expected_pronunciation="ジュスイ",
            ),
        ),
    ),
    _ReadingCase(
        text="っつー事でカメラ持って入水！",
        expected_kana="ッツーゴトデカメラモッテニュースイ！",
        targets=(
            _TargetExpectation(
                surface="入水",
                expected_pronunciation="ニュースイ",
            ),
        ),
    ),
    _ReadingCase(
        text="それは、公文も同じですね。",
        expected_kana="ソレワ、クモンモオナジデスネ。",
        targets=(
            _TargetExpectation(
                surface="公文",
                expected_pronunciation="クモン",
            ),
        ),
    ),
    _ReadingCase(
        text="しかし、日本ではこの公文書への意識やシステム、法整備が諸外国に比べてかなり遅れているそうです。",
        expected_kana="シカシ、ニホンデワコノコーブンショエノイシキヤシステム、ホーセービガショガイコクニクラベテカナリオクレテイルソーデス。",
        targets=(
            _TargetExpectation(
                surface="公文",
                expected_pronunciation="コーブン",
            ),
        ),
    ),
    _ReadingCase(
        text="さすがに、ひとりでコレに参加するほどの兵ではないデス…王子様、ごめんなさい。",
        expected_kana="サスガニ、ヒトリデコレニサンカスルホドノヘーデワナイデス…オージサマ、ゴメンナサイ。",
        targets=(
            # TODO: 本来は「ツワモノ」だが現状「ヘー」が選ばれてしまう
            # _TargetExpectation(
            #     surface="兵",
            #     expected_pronunciation="ツワモノ",
            # ),
        ),
    ),
    _ReadingCase(
        text="見事落札したのは、オークションにかけられた莉奈グッズは全て競り落としていると語る歴戦の兵だった。",
        expected_kana="ミゴトラクサツシタノワ、オークションニカケラレタリナグッズワスベテセリオトシテイルトカタルレキセンノツワモノダッタ。",
        targets=(
            _TargetExpectation(
                surface="兵",
                expected_pronunciation="ツワモノ",
            ),
        ),
    ),
    _ReadingCase(
        text="兵たちが動いたことで人垣がくずれた。",
        expected_kana="ヘータチガウゴイタコトデヒトガキガクズレタ。",
        targets=(
            _TargetExpectation(
                surface="兵",
                expected_pronunciation="ヘー",
            ),
        ),
    ),
    _ReadingCase(
        text="えーい、いいや内輪ネタだけど。",
        expected_kana="エーイ、イイヤウチワネタダケド。",
        targets=(
            _TargetExpectation(
                surface="内輪",
                expected_pronunciation="ウチワ",
            ),
        ),
    ),
    _ReadingCase(
        text="接触型シールタイプであれば、ベアリングの外輪と内輪の両方にオイルシールが接触している",
        expected_kana="セッショクガタシールタイプデアレバ、ベアリングノガイリントナイリンノリョーホーニオイルシールガセッショクシテイル",
        targets=(
            _TargetExpectation(
                surface="内輪",
                expected_pronunciation="ナイリン",
            ),
        ),
    ),
    _ReadingCase(
        text="これもけっこう凹むんだ…。",
        expected_kana="コレモケッコーヘコムンダ…。",
        targets=(
            _TargetExpectation(
                surface="凹む",
                expected_pronunciation="ヘコム",
            ),
        ),
    ),
    _ReadingCase(
        text="ビッダーズに出店したい！",
        expected_kana="ビッダーズニシュッテンシタイ！",
        targets=(
            _TargetExpectation(
                surface="出店",
                expected_pronunciation="シュッテン",
            ),
        ),
    ),
    _ReadingCase(
        text="あなたは好きな出店がありますか？",
        expected_kana="アナタワスキナデミセガアリマスカ？",
        targets=(
            _TargetExpectation(
                surface="出店",
                expected_pronunciation="デミセ",
            ),
        ),
    ),
    _ReadingCase(
        text="いわば「出所」したような感じなのかな。",
        expected_kana="イワバ「シュッショ」シタヨーナカンジナノカナ。",
        targets=(
            _TargetExpectation(
                surface="出所",
                expected_pronunciation="シュッショ",
            ),
        ),
    ),
    _ReadingCase(
        text="噂の出所は毛利蘭辺りだろう。",
        expected_kana="ウワサノデドコロワモーリランアタリダロー。",
        targets=(
            _TargetExpectation(
                surface="出所",
                expected_pronunciation="デドコロ",
            ),
        ),
    ),
    _ReadingCase(
        text="出水市内で各家庭に分かれて民泊活動をしています。",
        expected_kana="イズミシナイデカクカテーニワカレテミンパクカツドーヲシテイマス。",
        targets=(
            _TargetExpectation(
                surface="出水",
                expected_pronunciation="イズミ",
            ),
        ),
    ),
    _ReadingCase(
        text="明治時代に入って石炭採掘（常磐炭田、磐城炭鉱を参照）が始まると、坑内から温泉が多く出水した。",
        expected_kana="メージジダイニハイッテセキタンサイクツ（トキワタンデン、イワキタンコーヲサンショー）ガハジマルト、コーナイカラオンセンガオークシュッスイシタ。",
        targets=(
            _TargetExpectation(
                surface="出水",
                expected_pronunciation="シュッスイ",
            ),
        ),
    ),
    _ReadingCase(
        text="桜江町を中心にパイピング現象により堤防下から出水、谷住郷、鹿賀、川越などで冠水。",
        expected_kana="サクラエマチヲチューシンニパイピングゲンショーニヨリテーボーカカライズミ、タニジューゴー、カガ、カワゴシナドデカンスイ。",
        targets=(
            # TODO: 本来は「シュッスイ」だが現状「イズミ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="出水",
            #     expected_pronunciation="シュッスイ",
            # ),
        ),
    ),
    _ReadingCase(
        text="待って下さい、わたしだって最低限の分別はあります。",
        expected_kana="マッテクダサイ、ワタシダッテサイテーゲンノフンベツワアリマス。",
        targets=(
            _TargetExpectation(
                surface="分別",
                expected_pronunciation="フンベツ",
            ),
        ),
    ),
    _ReadingCase(
        text="来場者にゴミの分別、排出削減を促す。",
        expected_kana="ライジョーシャニゴミノブンベツ、ハイシュツサクゲンヲウナガス。",
        targets=(
            _TargetExpectation(
                surface="分別",
                expected_pronunciation="ブンベツ",
            ),
        ),
    ),
    _ReadingCase(
        text="なるほど、初心で拗らせまくってた多恵子さんを押しまくったのね",
        expected_kana="ナルホド、ショシンデコジラセマクッテタタエコサンヲオシマクッタノネ",
        targets=(
            # TODO: 本来は「ウブ」だが現状「ショシン」が選ばれてしまう
            # _TargetExpectation(
            #     surface="初心",
            #     expected_pronunciation="ウブ",
            # ),
        ),
    ),
    _ReadingCase(
        text="まったく初心なんだから。",
        expected_kana="マッタクウブナンダカラ。",
        targets=(
            _TargetExpectation(
                surface="初心",
                expected_pronunciation="ウブ",
            ),
        ),
    ),
    _ReadingCase(
        text="初心に戻るのもいいですね。",
        expected_kana="ショシンニモドルノモイイデスネ。",
        targets=(
            _TargetExpectation(
                surface="初心",
                expected_pronunciation="ショシン",
            ),
        ),
    ),
    _ReadingCase(
        text="一体何の利益があるかということです。",
        expected_kana="イッタイナンノリエキガアルカトイウコトデス。",
        targets=(
            _TargetExpectation(
                surface="利益",
                expected_pronunciation="リエキ",
            ),
        ),
    ),
    _ReadingCase(
        text="その為、安産祈願にご利益があるとして氏子崇敬者の信仰を集めて来ました。",
        expected_kana="ソノタメ、アンザンキガンニゴリヤクガアルトシテウジコスーケーシャノシンコーヲアツメテキマシタ。",
        targets=(
            _TargetExpectation(
                surface="利益",
                expected_pronunciation="リヤク",
            ),
        ),
    ),
    _ReadingCase(
        text="前首相は退任後も党内で強い影響力を持ち続けている。",
        expected_kana="ゼンシュショーワタイニンゴモトーナイデツヨイエーキョーリョクヲモチツズケテイル。",
        targets=(
            _TargetExpectation(
                surface="前",
                expected_pronunciation="ゼン",
            ),
        ),
    ),
    _ReadingCase(
        text="待ち合わせは午後六時に駅の改札の前でお願いします。",
        expected_kana="マチアワセワゴゴロクジニエキノカイサツノマエデオネガイシマス。",
        targets=(
            _TargetExpectation(
                surface="前",
                expected_pronunciation="マエ",
            ),
        ),
    ),
    _ReadingCase(
        text="ふんわりと、真綿で包まれたように優しい抱擁。",
        expected_kana="フンワリト、マワタデツツマレタヨーニヤサシイホーヨー。",
        targets=(
            _TargetExpectation(
                surface="包ま",
                expected_pronunciation="ツツマ",
            ),
        ),
    ),
    _ReadingCase(
        text="もうひとつは「岩手県北上市、水神温泉山照園」。",
        expected_kana="モーヒトツワ「イワテケンキタカミシ、スイジンヌルイズミヤマアキラエン」。",
        targets=(
            _TargetExpectation(
                surface="北上",
                expected_pronunciation="キタカミ",
            ),
        ),
    ),
    _ReadingCase(
        text="宿を出発し九州道を北上。",
        expected_kana="ヤドヲシュッパツシキューシュードーヲホクジョー。",
        targets=(
            _TargetExpectation(
                surface="北上",
                expected_pronunciation="ホクジョー",
            ),
        ),
    ),
    _ReadingCase(
        text="一九九五年にはこのエスコの部隊を正式な部門として発足しまして、その後、大阪、名古屋、その他全国十三支店への展開を進めてきてございます。",
        expected_kana="センキューヒャクキュージューゴネンニワコノエスコノブタイヲセーシキナブモントシテホッソクシマシテ、ソノゴ、オーサカ、ナゴヤ、ソノタゼンコクジューソーシテンエノテンカイヲススメテキテゴザイマス。",
        targets=(
            # TODO: 本来は「ジューサン」だが現状「ジューソー」が選ばれてしまう
            # _TargetExpectation(
            #     surface="十三",
            #     expected_pronunciation="ジューサン",
            # ),
        ),
    ),
    _ReadingCase(
        text="十五、十四、十三、…」",
        expected_kana="ジューゴ、ジューヨン、ジューサン、…」",
        targets=(
            _TargetExpectation(
                surface="十三",
                expected_pronunciation="ジューサン",
            ),
        ),
    ),
    _ReadingCase(
        text="大阪十三の立呑み「特一」さん。",
        expected_kana="オーサカジューソーノタチノミ「トクイチ」サン。",
        targets=(
            _TargetExpectation(
                surface="十三",
                expected_pronunciation="ジューソー",
            ),
        ),
    ),
    _ReadingCase(
        text="まずは、歌舞伎十八番の内「矢の根」で幕開き。",
        expected_kana="マズワ、カブキジューハチバンノウチ「ヤノネ」デマクアキ。",
        targets=(
            _TargetExpectation(
                surface="十八番",
                expected_pronunciation="ジューハチバン",
            ),
        ),
    ),
    _ReadingCase(
        text="あとは手早く混ぜるだけで、人気のクリーム味、ペペロンチーノ、トマト味もわずか十分で完成！",
        expected_kana="アトワテバヤクマゼルダケデ、ニンキノクリームアジ、ペペロンチーノ、トマトアジモワズカジュップンデカンセー！",
        targets=(
            _TargetExpectation(
                surface="十分",
                expected_pronunciation="ジュップン",
            ),
        ),
    ),
    _ReadingCase(
        text="西の空を見上げると、大根を薄く切ったような半月が空高くくっきりと残っており。",
        expected_kana="ニシノソラヲミアゲルト、ダイコンヲウスクキッタヨーナハンゲツガソラタカククッキリトノコッテオリ。",
        targets=(
            _TargetExpectation(
                surface="半月",
                expected_pronunciation="ハンゲツ",
            ),
        ),
    ),
    _ReadingCase(
        text="これも、買ってから約半月。",
        expected_kana="コレモ、カッテカラヤクハンツキ。",
        targets=(
            _TargetExpectation(
                surface="半月",
                expected_pronunciation="ハンツキ",
            ),
        ),
    ),
    _ReadingCase(
        text="引退を機に、彼は俳優として歩んだ半生を一冊の本にまとめた。",
        expected_kana="インタイヲキニ、カレワハイユートシテアユンダハンセーヲイッサツノホンニマトメタ。",
        targets=(
            _TargetExpectation(
                surface="半生",
                expected_pronunciation="ハンセー",
            ),
        ),
    ),
    _ReadingCase(
        text="乾麺よりも食感が良いので、最近はスーパーで半生のうどんを買うことが多い。",
        expected_kana="カンメンヨリモショッカンガヨイノデ、サイキンワスーパーデハンナマノウドンヲカウコトガオーイ。",
        targets=(
            _TargetExpectation(
                surface="半生",
                expected_pronunciation="ハンナマ",
            ),
        ),
    ),
    _ReadingCase(
        text="これでみんな「やきもの博士」！",
        expected_kana="コレデミンナ「ヤキモノハカセ」！",
        targets=(
            _TargetExpectation(
                surface="博士",
                expected_pronunciation="ハカセ",
            ),
        ),
    ),
    _ReadingCase(
        text="菜緒はすでにアメリカで博士課程を修了しているの。",
        expected_kana="ナオワスデニアメリカデハクシカテーヲシューリョーシテイルノ。",
        targets=(
            _TargetExpectation(
                surface="博士",
                expected_pronunciation="ハクシ",
            ),
        ),
    ),
    _ReadingCase(
        text="取手は着脱可能ですから！",
        expected_kana="トッテワチャクダツカノーデスカラ！",
        targets=(
            _TargetExpectation(
                surface="取手",
                expected_pronunciation="トッテ",
            ),
        ),
    ),
    _ReadingCase(
        text="公明党取手市議団４名で参加です。",
        expected_kana="コーメートートリデシギダンヨンメーデサンカデス。",
        targets=(
            _TargetExpectation(
                surface="取手",
                expected_pronunciation="トリデ",
            ),
        ),
    ),
    _ReadingCase(
        text="片運転台のキハ’サンゴーゼロ形は同形式２両で編成を組み、車両番号は取手向きが偶数、下館向きが奇数となっている。",
        expected_kana="カタウンテンダイノキハサンゴーゼロガタワドーケーシキニリョーデヘンセーヲクミ、シャリョーバンゴーワトッテムキガグースー、シモダテムキガキスートナッテイル。",
        targets=(
            # TODO: 本来は「トリデ」だが現状「トッテ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="取手",
            #     expected_pronunciation="トリデ",
            # ),
        ),
    ),
    _ReadingCase(
        text="誠くん、口数は多くないし",
        expected_kana="マコトクン、クチカズワオークナイシ",
        targets=(
            _TargetExpectation(
                surface="口数",
                expected_pronunciation="クチカズ",
            ),
        ),
    ),
    _ReadingCase(
        text="抽優なら全員獲得できたようですが、抽優での応募総口数は２００口超え。",
        expected_kana="抽優ナラゼンインカクトクデキタヨーデスガ、抽優デノオーボソークチスーワニヒャックチコエ。",
        targets=(
            _TargetExpectation(
                surface="口数",
                expected_pronunciation="クチスー",
            ),
        ),
    ),
    _ReadingCase(
        text="返礼品も魅力的で、どうしても３万円のコースが欲しくて、口数が増えるというお知らせが来たときは本当にテンションあがりました！",
        expected_kana="ヘンレーヒンモミリョクテキデ、ドーシテモサンマンエンノコースガホシクテ、クチカズガフエルトイウオシラセガキタトキワホントーニテンションアガリマシタ！",
        targets=(
            # TODO: 本来は「クチスー」だが現状「クチカズ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="口数",
            #     expected_pronunciation="クチスー",
            # ),
        ),
    ),
    _ReadingCase(
        text="ホームベーカリーからパン生地らしきものを取り出し、包丁で米を叩いた。",
        expected_kana="ホームベーカリーカラパンキジラシキモノヲトリダシ、ホーチョーデコメヲハタイタ。",
        targets=(
            # TODO: 本来は「タタイ」だが現状「ハタイ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="叩い",
            #     expected_pronunciation="タタイ",
            # ),
        ),
    ),
    _ReadingCase(
        text="一通りの挨拶が済むと、前足で軽く叩いてみる。",
        expected_kana="ヒトトーリノアイサツガスムト、マエアシデカルクタタイテミル。",
        targets=(
            _TargetExpectation(
                surface="叩い",
                expected_pronunciation="タタイ",
            ),
        ),
    ),
    _ReadingCase(
        text="大枚叩いて買ったんだが。",
        expected_kana="タイマイタタイテカッタンダガ。",
        targets=(
            # TODO: 本来は「ハタイ」だが現状「タタイ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="叩い",
            #     expected_pronunciation="ハタイ",
            # ),
        ),
    ),
    _ReadingCase(
        text="翠姉さんが名代で出るって",
        expected_kana="ミドリネーサンガミョーダイデデルッテ",
        targets=(
            _TargetExpectation(
                surface="名代",
                expected_pronunciation="ミョーダイ",
            ),
        ),
    ),
    _ReadingCase(
        text="しっかりとしたお品でした。",
        expected_kana="シッカリトシタオシナデシタ。",
        targets=(
            _TargetExpectation(
                surface="品",
                expected_pronunciation="シナ",
            ),
        ),
    ),
    _ReadingCase(
        text="図Ｉ３。国産製材品と他材料製品の生産量指数",
        expected_kana="ズＩサン。コクサンセーザイヒントタザイリョーセーヒンノセーサンリョーシスー",
        targets=(
            _TargetExpectation(
                surface="品",
                expected_pronunciation="ヒン",
            ),
        ),
    ),
    _ReadingCase(
        text="マッシュルームは石づきを切り落としてから土を落とし、縦に幅５ミリメートルに切る。",
        expected_kana="マッシュルームワイシズキヲキリオトシテカラツチヲオトシ、タテニハバゴミリメートルニキル。",
        targets=(
            _TargetExpectation(
                surface="土",
                expected_pronunciation="ツチ",
            ),
        ),
    ),
    _ReadingCase(
        text="最近の夏は、本当に体に堪えます。",
        expected_kana="サイキンノナツワ、ホントーニカラダニコタエマス。",
        targets=(
            _TargetExpectation(
                surface="堪え",
                expected_pronunciation="コタエ",
            ),
        ),
    ),
    _ReadingCase(
        text="大人でも鑑賞に堪えうる映画なのかも",
        expected_kana="オトナデモカンショーニタエウルエーガナノカモ",
        targets=(
            _TargetExpectation(
                surface="堪え",
                expected_pronunciation="タエ",
            ),
        ),
    ),
    _ReadingCase(
        text="でも、くすりが塗れないよー",
        expected_kana="デモ、クスリガヌレナイヨー",
        targets=(
            _TargetExpectation(
                surface="塗れ",
                expected_pronunciation="ヌレ",
            ),
        ),
    ),
    _ReadingCase(
        text="別に何かが劇的に変化しているというわけではないのだ。",
        expected_kana="ベツニナニカガゲキテキニヘンカシテイルトイウワケデワナイノダ。",
        targets=(
            _TargetExpectation(
                surface="変化",
                expected_pronunciation="ヘンカ",
            ),
        ),
    ),
    _ReadingCase(
        text="また、伊勢物語の古注釈書である伊勢物語抄（冷泉家流伊勢抄）では、陰陽記にある説として百年生きた狐狸などが変化したものを「つくもがみ」としている。",
        expected_kana="マタ、イセモノガタリノコチューシャクショデアルイセモノガタリショー（レーセンケリューイセショー）デワ、インヨーキニアルセツトシテヒャクネンイキタコリナドガヘンカシタモノヲ「ツクモガミ」トシテイル。",
        targets=(
            # TODO: 本来は「ヘンゲ」だが現状「ヘンカ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="変化",
            #     expected_pronunciation="ヘンゲ",
            # ),
        ),
    ),
    _ReadingCase(
        text="雨乞いに効があったといわれる鼓を先祖にもつ野狐があらわれて…勘太郎が野狐の五変化にいどむ舞台にご期待ください。",
        expected_kana="アマゴイニコーガアッタトイワレルツズミヲセンゾニモツノギツネガアラワレテ…カンタローガノギツネノゴヘンゲニイドムブタイニゴキタイクダサイ。",
        targets=(
            _TargetExpectation(
                surface="変化",
                expected_pronunciation="ヘンゲ",
            ),
        ),
    ),
    _ReadingCase(
        text="学校外よりも学校の方が楽しい、または楽しくなくても休むほどではない場合も多いため。",
        expected_kana="ガッコーガイヨリモガッコーノホーガタノシイ、マタワタノシクナクテモヤスムホドデワナイバアイモオーイタメ。",
        targets=(
            _TargetExpectation(
                surface="外",
                expected_pronunciation="ガイ",
            ),
        ),
    ),
    _ReadingCase(
        text="雨が上がったので、子供たちは外に出て鬼ごっこを始めた。",
        expected_kana="アメガアガッタノデ、コドモタチワソトニデテオニゴッコヲハジメタ。",
        targets=(
            _TargetExpectation(
                surface="外",
                expected_pronunciation="ソト",
            ),
        ),
    ),
    _ReadingCase(
        text="この外に千円引上げますれば更に三十六億程度減るだろうということであります。",
        expected_kana="コノソトニセンエンヒキアゲマスレバサラニサンジューロクオクテードヘルダロートイウコトデアリマス。",
        targets=(
            # TODO: 本来は「ホカ」だが現状「ソト」が選ばれてしまう
            # _TargetExpectation(
            #     surface="外",
            #     expected_pronunciation="ホカ",
            # ),
        ),
    ),
    _ReadingCase(
        text="第二号社会保険診療報酬課税の特例等に関する請願外六百四十件の請願を議題といたします。",
        expected_kana="ダイ二号社会保険診療報酬課税ノトクレートーニカンスルセーガンホカロッピャクヨンジュッケンノセーガンヲギダイトイタシマス。",
        targets=(
            _TargetExpectation(
                surface="外",
                expected_pronunciation="ホカ",
            ),
        ),
    ),
    _ReadingCase(
        text="そういう外面的、物質的なことにこそ客観性があると考えています。",
        expected_kana="ソーユウガイメンテキ、ブッシツテキナコトニコソキャッカンセーガアルトカンガエテイマス。",
        targets=(
            _TargetExpectation(
                surface="外面",
                expected_pronunciation="ガイメン",
            ),
        ),
    ),
    _ReadingCase(
        text="優等生を演じ外面の良い長女。",
        expected_kana="ユートーセーヲエンジソトズラノヨイチョージョ。",
        targets=(
            _TargetExpectation(
                surface="外面",
                expected_pronunciation="ソトズラ",
            ),
        ),
    ),
    _ReadingCase(
        text="かつて発見した物品が大事に至りそうになった時にエドワード、シルヴィアと共に解決したのが縁での友人である。",
        expected_kana="カツテハッケンシタブッピンガオーゴトニイタリソーニナッタトキニエドワード、シルヴィアトトモニカイケツシタノガエンデノユージンデアル。",
        targets=(
            # TODO: 本来は「ダイジ」だが現状「オーゴト」が選ばれてしまう
            # _TargetExpectation(
            #     surface="大事",
            #     expected_pronunciation="ダイジ",
            # ),
        ),
    ),
    _ReadingCase(
        text="理を説くには時が大事である。",
        expected_kana="リヲトクニワトキガダイジデアル。",
        targets=(
            _TargetExpectation(
                surface="大事",
                expected_pronunciation="ダイジ",
            ),
        ),
    ),
    _ReadingCase(
        text="マシュマロは子供に大人気",
        expected_kana="マシュマロワコドモニダイニンキ",
        targets=(
            _TargetExpectation(
                surface="大人気",
                expected_pronunciation="ダイニンキ",
            ),
        ),
    ),
    _ReadingCase(
        text="でも、今になってこのことを公にすれば、関係のない大勢の人たちを巻き込んでしまう。",
        expected_kana="デモ、イマニナッテコノコトヲオーヤケニスレバ、カンケーノナイオーゼーノヒトタチヲマキコンデシマウ。",
        targets=(
            _TargetExpectation(
                surface="大勢",
                expected_pronunciation="オーゼー",
            ),
        ),
    ),
    _ReadingCase(
        text="大勢は日本時間の午後にも判明する見通し。",
        expected_kana="タイセーワニホンジカンノゴゴニモハンメースルミトーシ。",
        targets=(
            _TargetExpectation(
                surface="大勢",
                expected_pronunciation="タイセー",
            ),
        ),
    ),
    _ReadingCase(
        text="トランペットや大太鼓となればなおさらです。",
        expected_kana="トランペットヤオーダイコトナレバナオサラデス。",
        targets=(
            _TargetExpectation(
                surface="大太鼓",
                expected_pronunciation="オーダイコ",
            ),
        ),
    ),
    _ReadingCase(
        text="大鉦鼓は、御遊または宮中の盛儀における舞楽で庭上で用いられ、大太鼓とならんで置かれる。",
        expected_kana="ダイショーコワ、ギョユーマタワキューチューノセーギニオケルブガクデニワジョーデモチイラレ、ダダイコトナランデオカレル。",
        targets=(
            _TargetExpectation(
                surface="大太鼓",
                expected_pronunciation="ダダイコ",
            ),
        ),
    ),
    _ReadingCase(
        text="アパートの部屋の鍵って入居者が変わる度、大家さんは必ず変えてくれているのでしょうか？",
        expected_kana="アパートノヘヤノカギッテニューキョシャガカワルタビ、オーヤサンワカナラズカエテクレテイルノデショーカ？",
        targets=(
            _TargetExpectation(
                surface="大家",
                expected_pronunciation="オーヤ",
            ),
        ),
    ),
    _ReadingCase(
        text="掲示板では成功している大家、不動産投資家に対するインタビューなども掲載されている。",
        expected_kana="ケージバンデワセーコーシテイルタイカ、フドーサントーシカニタイスルインタビューナドモケーサイサレテイル。",
        targets=(
            # TODO: 本来は「オーヤ」だが現状「タイカ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="大家",
            #     expected_pronunciation="オーヤ",
            # ),
        ),
    ),
    _ReadingCase(
        text="交響曲と歌曲の大家として知られる。",
        expected_kana="コーキョーキョクトカキョクノタイカトシテシラレル。",
        targets=(
            _TargetExpectation(
                surface="大家",
                expected_pronunciation="タイカ",
            ),
        ),
    ),
    _ReadingCase(
        text="無事お勉強が終わると今日は大船で暑気払いです。",
        expected_kana="ブジオベンキョーガオワルトキョーワオーフナデショキバライデス。",
        targets=(
            _TargetExpectation(
                surface="大船",
                expected_pronunciation="オーフナ",
            ),
        ),
    ),
    _ReadingCase(
        text="倭館の前の船着き場は東西七十八間（約百四十二メートル）、南北百二十二間（約二百二十二メートル）を誇る広大なもので、水深もあり、かなりの大船でも着けることができた。",
        expected_kana="ワカンノマエノフナツキバワトーザイナナジューハッケン（ヤクヒャクヨンジューニメートル）、ナンボクヒャクニジューニケン（ヤクニヒャクニジューニメートル）ヲホコルコーダイナモノデ、スイシンモアリ、カナリノオーブネデモツケルコトガデキタ。",
        targets=(
            _TargetExpectation(
                surface="大船",
                expected_pronunciation="オーブネ",
            ),
        ),
    ),
    _ReadingCase(
        text="限られた予算の中で、如何にして成果を上げるかが問われている。",
        expected_kana="カギラレタヨサンノナカデ、イカニシテセーカヲアゲルカガトワレテイル。",
        targets=(
            _TargetExpectation(
                surface="如何",
                expected_pronunciation="イカ",
            ),
        ),
    ),
    _ReadingCase(
        text="寒い日が続いておりますが、皆様如何お過ごしでしょうか。",
        expected_kana="サムイヒガツズイテオリマスガ、ミナサマイカガオスゴシデショーカ。",
        targets=(
            _TargetExpectation(
                surface="如何",
                expected_pronunciation="イカガ",
            ),
        ),
    ),
    _ReadingCase(
        text="規則に違反した場合は、理由の如何を問わず退場処分とします。",
        expected_kana="キソクニイハンシタバアイワ、リユーノイカンヲトワズタイジョーショブントシマス。",
        targets=(
            _TargetExpectation(
                surface="如何",
                expected_pronunciation="イカン",
            ),
        ),
    ),
    _ReadingCase(
        text="手品や奇術の多くは唐から伝わり猿楽の芸の一つであり、如何様とも呼ばれ、それを行うものを如何様師とも呼称していた。",
        expected_kana="テジナヤキジュツノオークワトーカラツタワリサルガクノゲーノヒトツデアリ、イカヨートモヨバレ、ソレヲオコナウモノヲイカサマシトモコショーシテイタ。",
        targets=(
            # TODO: 本来は「イカサマ」だが現状「イカヨー」が選ばれてしまう
            # _TargetExpectation(
            #     surface="如何様",
            #     expected_pronunciation="イカサマ",
            # ),
        ),
    ),
    _ReadingCase(
        text="どうか、如何様にもご命令を",
        expected_kana="ドーカ、イカヨーニモゴメーレーヲ",
        targets=(
            _TargetExpectation(
                surface="如何様",
                expected_pronunciation="イカヨー",
            ),
        ),
    ),
    _ReadingCase(
        text="「やさしい川崎の地名（上）」、「川崎町と南河原村の字名」には関連するエピソードも！",
        expected_kana="「ヤサシイカワサキノチメー（ウエ）」、「カワサキマチトミナミカワラムラノアザメー」ニワカンレンスルエピソードモ！",
        targets=(
            _TargetExpectation(
                surface="字",
                expected_pronunciation="アザ",
            ),
        ),
    ),
    _ReadingCase(
        text="１字姓プラス１字名",
        expected_kana="イチジセープラスイチジメー",
        targets=(
            _TargetExpectation(
                surface="字",
                expected_pronunciation="ジ",
            ),
        ),
    ),
    _ReadingCase(
        text="午後はピーエム一時より、近くの桟橋に集合していた東京湾を代表する５台のガイド船に、講師を含めて分乗し実釣となりました。",
        expected_kana="ゴゴワピーエムイチジヨリ、チカクノサンバシニシューゴーシテイタトーキョーワンヲダイヒョースルゴダイノガイドセンニ、コーシヲフクメテブンジョーシジッツリトナリマシタ。",
        targets=(
            _TargetExpectation(
                surface="実",
                expected_pronunciation="ジッ",
            ),
        ),
    ),
    _ReadingCase(
        text="そうこうしていると、１１時過ぎてからは、親戚、実の従兄弟や叔父とかの家に、タケノコをもらいに、乗り慣れてはいない、軽トラで、僕と母とで向かいました。",
        expected_kana="ソーコーシテイルト、ジューイチジスギテカラワ、シンセキ、ミノイトコヤオジトカノイエニ、タケノコヲモライニ、ノリナレテワイナイ、ケートラデ、ボクトハハトデムカイマシタ。",
        targets=(
            # TODO: 本来は「ジツ」だが現状「ミ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="実",
            #     expected_pronunciation="ジツ",
            # ),
        ),
    ),
    _ReadingCase(
        text="で、その原因が又二つございまして、一つは実は坑内の実労働時間の短縮というところに一つの原因がございます。",
        expected_kana="デ、ソノゲンインガマタフタツゴザイマシテ、ヒトツワジツワコーナイノジツロードージカンノタンシュクトイウトコロニヒトツノゲンインガゴザイマス。",
        targets=(
            _TargetExpectation(
                surface="実",
                expected_pronunciation="ジツ",
                occurrence=1,
            ),
        ),
    ),
    _ReadingCase(
        text="うちの庭にあった梅の木は、花が咲かなくなって実がならなくなったと思ったら、枯れてしまった。",
        expected_kana="ウチノニワニアッタウメノキワ、ハナガサカナクナッテミガナラナクナッタトオモッタラ、カレテシマッタ。",
        targets=(
            _TargetExpectation(
                surface="実",
                expected_pronunciation="ミ",
            ),
        ),
    ),
    _ReadingCase(
        text="なんだか明日までこの寒気が居座るそうです。",
        expected_kana="ナンダカアシタマデコノサムケガイスワルソーデス。",
        targets=(
            # TODO: 本来は「カンキ」だが現状「サムケ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="寒気",
            #     expected_pronunciation="カンキ",
            # ),
        ),
    ),
    _ReadingCase(
        text="次の寒気に期待しましょう。",
        expected_kana="ツギノカンキニキタイシマショー。",
        targets=(
            _TargetExpectation(
                surface="寒気",
                expected_pronunciation="カンキ",
            ),
        ),
    ),
    _ReadingCase(
        text="想像しただけで寒気がする温度ですが、毎日当たり前のように降り注ぐ太陽エネルギーの存在を再認識させられたのです。",
        expected_kana="ソーゾーシタダケデサムケガスルオンドデスガ、マイニチアタリマエノヨーニフリソソグタイヨーエネルギーノソンザイヲサイニンシキサセラレタノデス。",
        targets=(
            _TargetExpectation(
                surface="寒気",
                expected_pronunciation="サムケ",
            ),
        ),
    ),
    _ReadingCase(
        text="とおっしゃっていましたが、世界平和の祈りと呼吸法による人類即神也の印は対になっているように思います。",
        expected_kana="トオッシャッテイマシタガ、セカイヘーワノイノリトコキューホーニヨルジンルイソクシンマタノシルシワツイニナッテイルヨーニオモイマス。",
        targets=(
            _TargetExpectation(
                surface="対",
                expected_pronunciation="ツイ",
            ),
        ),
    ),
    _ReadingCase(
        text="猫のミーメとは指輪の小人のミーメと同じ名。",
        expected_kana="ネコノミーメトワユビワノコビトノミーメトオナジナ。",
        targets=(
            _TargetExpectation(
                surface="小人",
                expected_pronunciation="コビト",
            ),
        ),
    ),
    _ReadingCase(
        text="基本的には小学生が」小人」となります。",
        expected_kana="キホンテキニワショーガクセーガ」ショーニン」トナリマス。",
        targets=(
            _TargetExpectation(
                surface="小人",
                expected_pronunciation="ショーニン",
            ),
        ),
    ),
    _ReadingCase(
        text="大人２５００円。小人。",
        expected_kana="オトナニセンゴヒャクエン。コビト。",
        targets=(
            # TODO: 本来は「ショーニン」だが現状「コビト」が選ばれてしまう
            # _TargetExpectation(
            #     surface="小人",
            #     expected_pronunciation="ショーニン",
            # ),
        ),
    ),
    _ReadingCase(
        text="山陰の松島とも呼ばれる。",
        expected_kana="サンインノマツシマトモヨバレル。",
        targets=(
            _TargetExpectation(
                surface="山陰",
                expected_pronunciation="サンイン",
            ),
        ),
    ),
    _ReadingCase(
        text="やがてゆっくりと水平移動し、隣の山陰に消えていった。",
        expected_kana="ヤガテユックリトスイヘーイドーシ、トナリノヤマカゲニキエテイッタ。",
        targets=(
            _TargetExpectation(
                surface="山陰",
                expected_pronunciation="ヤマカゲ",
            ),
        ),
    ),
    _ReadingCase(
        text="それは千九百七十七（昭和五十二）年、別会社を大同団結して、広島市に思い切って近代的な工場建設に踏み切った時のことである。",
        expected_kana="ソレワセンキューヒャクナナジューナナ（ショーワゴジューニ）ネン、ベツガイシャヲダイドーダンケツシテ、ヒロシマシニオモイキッテキンダイテキナコージョーケンセツニフミキッタトキノコトデアル。",
        targets=(
            _TargetExpectation(
                surface="工場",
                expected_pronunciation="コージョー",
            ),
        ),
    ),
    _ReadingCase(
        text="夫は小さな町工場で働き、私は近くのスーパーで働いた。",
        expected_kana="オットワチーサナマチコーバデハタラキ、ワタシワチカクノスーパーデハタライタ。",
        targets=(
            _TargetExpectation(
                surface="工場",
                expected_pronunciation="コーバ",
            ),
        ),
    ),
    _ReadingCase(
        text="これらの商品展開は、すべて顧客の困っていることは何か、それにどう応えればよいかを軸に工夫に工夫を重ねてきたことの結果ばかりである。",
        expected_kana="コレラノショーヒンテンカイワ、スベテコキャクノコマッテイルコトワナニカ、ソレニドーコタエレバヨイカヲジクニクフーニクフーヲカサネテキタコトノケッカバカリデアル。",
        targets=(
            _TargetExpectation(
                surface="工夫",
                expected_pronunciation="クフー",
            ),
        ),
    ),
    _ReadingCase(
        text="募集で集まった工夫を監視するように、思春に命じていたのよ",
        expected_kana="ボシューデアツマッタコーフヲカンシスルヨーニ、シシュンニメージテイタノヨ",
        targets=(
            _TargetExpectation(
                surface="工夫",
                expected_pronunciation="コーフ",
            ),
        ),
    ),
    _ReadingCase(
        text="水門補修で使っている工夫の中に、それらしい人間が混ざっていないか調べて欲しいとのことでしたが……",
        expected_kana="スイモンホシューデツカッテイルクフーノナカニ、ソレラシイニンゲンガマザッテイナイカシラベテホシイトノコトデシタガ……",
        targets=(
            # TODO: 本来は「コーフ」だが現状「クフー」が選ばれてしまう
            # _TargetExpectation(
            #     surface="工夫",
            #     expected_pronunciation="コーフ",
            # ),
        ),
    ),
    _ReadingCase(
        text="市場で海鮮食べた記憶が。",
        expected_kana="イチバデカイセンタベタキオクガ。",
        targets=(
            _TargetExpectation(
                surface="市場",
                expected_pronunciation="イチバ",
            ),
        ),
    ),
    _ReadingCase(
        text="（１）業務用乳製品の市場規模",
        expected_kana="（イチ）ギョームヨーニューセーヒンノシジョーキボ",
        targets=(
            _TargetExpectation(
                surface="市場",
                expected_pronunciation="シジョー",
            ),
        ),
    ),
    _ReadingCase(
        text="平野キャラを楽しむにもばっちりです。",
        expected_kana="ヒラノキャラヲタノシムニモバッチリデス。",
        targets=(
            _TargetExpectation(
                surface="平野",
                expected_pronunciation="ヒラノ",
            ),
        ),
    ),
    _ReadingCase(
        text="しかし、平野と木々ばかりだな。",
        expected_kana="シカシ、ヘーヤトキギバカリダナ。",
        targets=(
            _TargetExpectation(
                surface="平野",
                expected_pronunciation="ヘーヤ",
            ),
        ),
    ),
    _ReadingCase(
        text="感慨が現実に関与を許す様な甘い考えをとらない一番目族である吾輩には、とても及びもつかぬことではあるが、優柔不断な主人は、年中感慨の中にいる。",
        expected_kana="カンガイガゲンジツニカンヨヲユルスヨーナアマイカンガエヲトラナイイチバンメゾクデアルワガハイニワ、トテモオヨビモツカヌコトデワアルガ、ユージューフダンナシュジンワ、ネンジューカンガイノナカニイル。",
        targets=(
            _TargetExpectation(
                surface="年中",
                expected_pronunciation="ネンジュー",
            ),
        ),
    ),
    _ReadingCase(
        text="年中、年長のお友だち２人でブリオという木製の汽車のおもちゃで遊びました。",
        expected_kana="ネンチュー、ネンチョーノオトモダチフタリデブリオトイウモクセーノキシャノオモチャデアソビマシタ。",
        targets=(
            _TargetExpectation(
                surface="年中",
                expected_pronunciation="ネンチュー",
            ),
        ),
    ),
    _ReadingCase(
        text="寝ている者を気遣ってそっと床を抜け出し、洗面に立った。",
        expected_kana="ネテイルモノヲキズカッテソットトコヲヌケダシ、センメンニタッタ。",
        targets=(
            _TargetExpectation(
                surface="床",
                expected_pronunciation="トコ",
            ),
        ),
    ),
    _ReadingCase(
        text="床を用意させますから、少しお休み下さい",
        expected_kana="ユカヲヨーイサセマスカラ、スコシオヤスミクダサイ",
        targets=(
            # TODO: 本来は「トコ」だが現状「ユカ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="床",
            #     expected_pronunciation="トコ",
            # ),
        ),
    ),
    _ReadingCase(
        text="アストロッドは観念し、床に視線を落とした。",
        expected_kana="アストロッドワカンネンシ、ユカニシセンヲオトシタ。",
        targets=(
            _TargetExpectation(
                surface="床",
                expected_pronunciation="ユカ",
            ),
        ),
    ),
    _ReadingCase(
        text="そして、口付けを強請るように首筋に腕を回して引き寄せれば。",
        expected_kana="ソシテ、クチズケヲネダルヨーニクビスジニウデヲマワシテヒキヨセレバ。",
        targets=(
            _TargetExpectation(
                surface="強請る",
                expected_pronunciation="ネダル",
            ),
        ),
    ),
    _ReadingCase(
        text="だってぇ……わたしに気付かないで、ウキウキ歩いてるものだから、強請るざいりょ……いえ、気になってぇー",
        expected_kana="ダッテェ……ワタシニキズカナイデ、ウキウキアルイテルモノダカラ、ネダルザイリョ……イエ、キニナッテーー",
        targets=(
            # TODO: 本来は「ユスル」だが現状「ネダル」が選ばれてしまう
            # _TargetExpectation(
            #     surface="強請る",
            #     expected_pronunciation="ユスル",
            # ),
        ),
    ),
    _ReadingCase(
        text="へっ、狙いをつけて弾を撃ちゃいいんだろ。",
        expected_kana="ヘッ、ネライヲツケテタマヲウチャイインダロ。",
        targets=(
            _TargetExpectation(
                surface="弾",
                expected_pronunciation="タマ",
            ),
        ),
    ),
    _ReadingCase(
        text="−２本田のＦＫ弾に歓喜したミラン。",
        expected_kana="−ニホンダノエフケイダンニカンキシタミラン。",
        targets=(
            _TargetExpectation(
                surface="弾",
                expected_pronunciation="ダン",
            ),
        ),
    ),
    _ReadingCase(
        text="みんな必死の形相で忙しそうに働いていた。",
        expected_kana="ミンナヒッシノギョーソーデイソガシソーニハタライテイタ。",
        targets=(
            _TargetExpectation(
                surface="形相",
                expected_pronunciation="ギョーソー",
            ),
        ),
    ),
    _ReadingCase(
        text="同様に、全体と個物、形相と質料、時間と空間、内在と超越等の対概念も、各々が持つ相互補完性において矛盾的自己同一的であるといわれる。",
        expected_kana="ドーヨーニ、ゼンタイトコブツ、ケーソートシツリョー、ジカントクーカン、ナイザイトチョーエツトーノタイガイネンモ、オノオノガモツソーゴホカンセーニオイテムジュンテキジコドーイツテキデアルトイワレル。",
        targets=(
            _TargetExpectation(
                surface="形相",
                expected_pronunciation="ケーソー",
            ),
        ),
    ),
    _ReadingCase(
        text="うｐした後に気づいたのですが、ドナルドっぷりが凄くて自分で驚きました。",
        expected_kana="ウｐシタゴニキズイタノデスガ、ドナルドップリガスゴクテジブンデオドロキマシタ。",
        targets=(
            # TODO: 本来は「アト」だが現状「ゴ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="後",
            #     expected_pronunciation="アト",
            # ),
        ),
    ),
    _ReadingCase(
        text="私たちも後を追って歩き出す。",
        expected_kana="ワタシタチモアトヲオッテアルキダス。",
        targets=(
            _TargetExpectation(
                surface="後",
                expected_pronunciation="アト",
            ),
        ),
    ),
    _ReadingCase(
        text="登るにつれて後を振り返ると高度感が素晴らしい。",
        expected_kana="ノボルニツレテアトヲフリカエルトコードカンガスバラシイ。",
        targets=(
            # TODO: 本来は「ウシロ」だが現状「アト」が選ばれてしまう
            # _TargetExpectation(
            #     surface="後",
            #     expected_pronunciation="ウシロ",
            # ),
        ),
    ),
    _ReadingCase(
        text="映画感想、「百年後…」",
        expected_kana="エーガカンソー、「ヒャクネンゴ…」",
        targets=(
            _TargetExpectation(
                surface="後",
                expected_pronunciation="ゴ",
                expected_outcome="dictionary_default_protected",
                was_preserved=True,
            ),
        ),
    ),
    _ReadingCase(
        text="実力を評価してもらえる、仕事で抜擢される機会が多くなり、それをきっかけに、後の人生が飛躍する運勢を持つことを示しています。",
        expected_kana="ジツリョクヲヒョーカシテモラエル、シゴトデバッテキサレルキカイガオークナリ、ソレヲキッカケニ、ノチノジンセーガヒヤクスルウンセーヲモツコトヲシメシテイマス。",
        targets=(
            _TargetExpectation(
                surface="後",
                expected_pronunciation="ノチ",
            ),
        ),
    ),
    _ReadingCase(
        text="第二部の対戦結果は後ほどに。",
        expected_kana="ダイニブノタイセンケッカワアトホドニ。",
        targets=(
            # TODO: 本来は「ノチ」だが現状「アト」が選ばれてしまう
            # _TargetExpectation(
            #     surface="後",
            #     expected_pronunciation="ノチ",
            # ),
        ),
    ),
    _ReadingCase(
        text="同年の上洛時にも御供し、記録に「瀬戸口与助。",
        expected_kana="ドーネンノジョーラクジニモオトモシ、キロクニ「セトグチヨスケ。",
        targets=(
            _TargetExpectation(
                surface="御供",
                expected_pronunciation="オトモ",
            ),
        ),
    ),
    _ReadingCase(
        text="子供たちが下校の頃、御供を拾おうとする老若男女で賑わいます。",
        expected_kana="コドモタチガゲコーノコロ、オトモヲヒロオートスルローニャクナンニョデニギワイマス。",
        targets=(
            # TODO: 本来は「ゴクー」だが現状「オトモ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="御供",
            #     expected_pronunciation="ゴクー",
            # ),
        ),
    ),
    _ReadingCase(
        text="袈裟を着た影現寺住職も拝殿から御供を撒いています。",
        expected_kana="ケサヲキタカゲゲンテラジューショクモハイデンカラゴクーヲマイテイマス。",
        targets=(
            _TargetExpectation(
                surface="御供",
                expected_pronunciation="ゴクー",
            ),
        ),
    ),
    _ReadingCase(
        text="御手洗に行った際は石鹸での手洗いと手指消毒を行います。",
        expected_kana="オテアライニイッタサイワセッケンデノテアライトシュシショードクヲオコナイマス。",
        targets=(
            _TargetExpectation(
                surface="御手洗",
                expected_pronunciation="オテアライ",
            ),
        ),
    ),
    _ReadingCase(
        text="御手洗シリーズとしてはイマイチ破壊力に欠ける。",
        expected_kana="ミタライシリーズトシテワイマイチハカイリョクニカケル。",
        targets=(
            _TargetExpectation(
                surface="御手洗",
                expected_pronunciation="ミタライ",
            ),
        ),
    ),
    _ReadingCase(
        text="日韓で心中しても意味がない。",
        expected_kana="ニッカンデシンジューシテモイミガナイ。",
        targets=(
            _TargetExpectation(
                surface="心中",
                expected_pronunciation="シンジュー",
            ),
        ),
    ),
    _ReadingCase(
        text="すると一幡の外祖父、比企能員は千幡との分割相続となったことに憤り、外戚の権威を笠に着て独歩の志心中に抱き、謀反を企てて千幡とその外戚以下を滅ぼそうとした。",
        expected_kana="スルトイチハタノガイソフ、ヒキヨシカズワセンハタトノブンカツソーゾクトナッタコトニイキドーリ、ガイセキノケンイヲカサニキテドッポノココロザシシンジューニイダキ、ムホンヲクワダテテセンハタトソノガイセキイカヲホロボソートシタ。",
        targets=(
            # TODO: 本来は「シンチュー」だが現状「シンジュー」が選ばれてしまう
            # _TargetExpectation(
            #     surface="心中",
            #     expected_pronunciation="シンチュー",
            # ),
        ),
    ),
    _ReadingCase(
        text="心中お察しいたします……",
        expected_kana="シンチューオサッシイタシマス……",
        targets=(
            _TargetExpectation(
                surface="心中",
                expected_pronunciation="シンチュー",
            ),
        ),
    ),
    _ReadingCase(
        text="数江が怒りの叫びをあげた。",
        expected_kana="カズエガイカリノサケビヲアゲタ。",
        targets=(
            _TargetExpectation(
                surface="怒り",
                expected_pronunciation="イカリ",
            ),
        ),
    ),
    _ReadingCase(
        text="私が怒らず注意を促しただけでも、これまた怒りくるって大泣き。",
        expected_kana="ワタシガオコラズチューイヲウナガシタダケデモ、コレマタオコリクルッテオーナキ。",
        targets=(
            # TODO: 本来は「イカリ」だが現状「オコリ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="怒り",
            #     expected_pronunciation="イカリ",
            # ),
        ),
    ),
    _ReadingCase(
        text="どういうことをすると彼女が怒るのか、とか、彼女が何を大事にして暮らしているのか、とか。",
        expected_kana="ドーユウコトヲスルトカノジョガオコルノカ、トカ、カノジョガナニヲダイジニシテクラシテイルノカ、トカ。",
        targets=(
            _TargetExpectation(
                surface="怒る",
                expected_pronunciation="オコル",
            ),
        ),
    ),
    _ReadingCase(
        text="かわいい子を見るとね、からかいたくなるのが人間の性だよ",
        expected_kana="カワイイコヲミルトネ、カラカイタクナルノガニンゲンノサガダヨ",
        targets=(
            _TargetExpectation(
                surface="性",
                expected_pronunciation="サガ",
            ),
        ),
    ),
    _ReadingCase(
        text="それにただ傍観するのはどうも性に会わないもので。",
        expected_kana="ソレニタダボーカンスルノワドーモセーニアワナイモノデ。",
        targets=(
            # TODO: 本来は「ショー」だが現状「セー」が選ばれてしまう
            # _TargetExpectation(
            #     surface="性",
            #     expected_pronunciation="ショー",
            # ),
        ),
    ),
    _ReadingCase(
        text="スーツ、ネクタイ姿で就職活動するのは苦手だし、最先端の建築物を作るより自分はリフォーム業のほうが性に合っているような気がするから…。",
        expected_kana="スーツ、ネクタイスガタデシューショクカツドースルノワニガテダシ、サイセンタンノケンチクブツヲツクルヨリジブンワリフォームギョーノホーガショーニアッテイルヨーナキガスルカラ…。",
        targets=(
            _TargetExpectation(
                surface="性",
                expected_pronunciation="ショー",
            ),
        ),
    ),
    _ReadingCase(
        text="受動性、少ない発言",
        expected_kana="ジュドーセー、スクナイハツゲン",
        targets=(
            _TargetExpectation(
                surface="性",
                expected_pronunciation="セー",
            ),
        ),
    ),
    _ReadingCase(
        text="最初は預かるだけのつもりだったが、毎日世話をするうちに子猫に情が移ってしまった。",
        expected_kana="サイショワアズカルダケノツモリダッタガ、マイニチゼワヲスルウチニコネコニジョーガウツッテシマッタ。",
        targets=(
            _TargetExpectation(
                surface="情",
                expected_pronunciation="ジョー",
            ),
        ),
    ),
    _ReadingCase(
        text="敵の大将は、命乞いをする若武者を武士の情で見逃してやった。",
        expected_kana="テキノタイショーワ、イノチゴイヲスルワカムシャヲブシノナサケデミノガシテヤッタ。",
        targets=(
            _TargetExpectation(
                surface="情",
                expected_pronunciation="ナサケ",
            ),
        ),
    ),
    _ReadingCase(
        text="祖父も九十を過ぎて、最近は少し惚けてきたのか同じ話を何度も繰り返す。",
        expected_kana="ソフモキュージューヲスギテ、サイキンワスコシボケテキタノカオナジバナシヲナンドモクリカエス。",
        targets=(
            _TargetExpectation(
                surface="惚け",
                expected_pronunciation="ボケ",
            ),
        ),
    ),
    _ReadingCase(
        text="証拠はそろっているのだから、知らないふりをして惚けるのはやめなさい。",
        expected_kana="ショーコワソロッテイルノダカラ、シラナイフリヲシテトボケルノワヤメナサイ。",
        targets=(
            _TargetExpectation(
                surface="惚ける",
                expected_pronunciation="トボケル",
            ),
        ),
    ),
    _ReadingCase(
        text="「あなたを愛している、アストロッド」",
        expected_kana="「アナタヲアイシテイル、アストロッド」",
        targets=(
            _TargetExpectation(
                surface="愛し",
                expected_pronunciation="アイシ",
            ),
        ),
    ),
    _ReadingCase(
        text="問題ごとに決められた手数内で課題をクリアする。",
        expected_kana="モンダイゴトニキメラレタテカズナイデカダイヲクリアスル。",
        targets=(
            _TargetExpectation(
                surface="手数",
                expected_pronunciation="テカズ",
            ),
        ),
    ),
    _ReadingCase(
        text="このような事務は手数が大変です。",
        expected_kana="コノヨーナジムワテスーガタイヘンデス。",
        targets=(
            _TargetExpectation(
                surface="手数",
                expected_pronunciation="テスー",
            ),
        ),
    ),
    _ReadingCase(
        text="という疑念も関組長は抱いている。",
        expected_kana="トイウギネンモセキクミチョーワイダイテイル。",
        targets=(
            _TargetExpectation(
                surface="抱い",
                expected_pronunciation="イダイ",
            ),
        ),
    ),
    _ReadingCase(
        text="太陽を抱く月は本当に文句なく面白いです！",
        expected_kana="タイヨーヲダクツキワホントーニモンクナクオモシロイデス！",
        targets=(
            # TODO: 本来は「イダク」だが現状「ダク」が選ばれてしまう
            # _TargetExpectation(
            #     surface="抱く",
            #     expected_pronunciation="イダク",
            # ),
        ),
    ),
    _ReadingCase(
        text="鮫なら捌けると思いますから",
        expected_kana="サメナラサバケルトオモイマスカラ",
        targets=(
            _TargetExpectation(
                surface="捌ける",
                expected_pronunciation="サバケル",
            ),
        ),
    ),
    _ReadingCase(
        text="センタースクリーンが左右に開き捌ける。",
        expected_kana="センタースクリーンガサユーニヒラキハケル。",
        targets=(
            _TargetExpectation(
                surface="捌ける",
                expected_pronunciation="ハケル",
            ),
        ),
    ),
    _ReadingCase(
        text="レーベル名がクレプスキュールの捩り。",
        expected_kana="レーベルメーガクレプスキュールノモジリ。",
        targets=(
            _TargetExpectation(
                surface="捩り",
                expected_pronunciation="モジリ",
            ),
        ),
    ),
    _ReadingCase(
        text="確かに現ＭＢＸ９のＩアームはロアアームの柔らかい材質を受け止めることが出来ずグニャグニャ捩れます。",
        expected_kana="タシカニゲンエムビーエックスキューノＩアームワロアアームノヤワラカイザイシツヲウケトメルコトガデキズグニャグニャネジレマス。",
        targets=(
            _TargetExpectation(
                surface="捩れ",
                expected_pronunciation="ネジレ",
            ),
        ),
    ),
    _ReadingCase(
        text="バシバシ取り捲りました。",
        expected_kana="バシバシトリマクリマシタ。",
        targets=(
            _TargetExpectation(
                surface="捲り",
                expected_pronunciation="マクリ",
            ),
        ),
    ),
    _ReadingCase(
        text="３月ということで家に飾っているコナンカレンダーもようやく１枚捲りました。",
        expected_kana="サンガツトイウコトデイエニカザッテイルコナンカレンダーモヨーヤクイチマイマクリマシタ。",
        targets=(
            # TODO: 本来は「メクリ」だが現状「マクリ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="捲り",
            #     expected_pronunciation="メクリ",
            # ),
        ),
    ),
    _ReadingCase(
        text="駆真は違和感に首を捻った。",
        expected_kana="カシンワイワカンニクビヲヒネッタ。",
        targets=(
            _TargetExpectation(
                surface="捻っ",
                expected_pronunciation="ヒネッ",
            ),
        ),
    ),
    _ReadingCase(
        text="まず捻れたのは間違いなく体型だが、何よりも変わったのは考え方だと思う。",
        expected_kana="マズネジレタノワマチガイナクタイケーダガ、ナニヨリモカワッタノワカンガエカタダトオモウ。",
        targets=(
            _TargetExpectation(
                surface="捻れ",
                expected_pronunciation="ネジレ",
            ),
        ),
    ),
    _ReadingCase(
        text="現在のＳＣＰ４１０移動モジュールは、１２個の１ミリメートルの通気穴が開き単純な掛け金が蓋に取り付けられた２０ｘ２０ｘ２０センチメートルの透明なアクリルグラス容器です。",
        expected_kana="ゲンザイノエスシーピーヨンヒャクジューイドーモジュールワ、ジューニコノイチミリメートルノツーキアナガヒラキタンジュンナカケガネガフタニトリツケラレタニジューｘニジューｘニジュッセンチメートルノトーメーナアクリルグラスヨーキデス。",
        targets=(
            _TargetExpectation(
                surface="掛け金",
                expected_pronunciation="カケガネ",
            ),
        ),
    ),
    _ReadingCase(
        text="節税額は「掛け金バツその人の税率」。",
        expected_kana="セツゼーガクワ「カケキンバツソノヒトノゼーリツ」。",
        targets=(
            _TargetExpectation(
                surface="掛け金",
                expected_pronunciation="カケキン",
            ),
        ),
    ),
    _ReadingCase(
        text="底に。塩をひと摘み。",
        expected_kana="ソコニ。シオヲヒトツミ。",
        targets=(
            # TODO: 本来は「ツマミ」だが現状「ツミ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="摘み",
            #     expected_pronunciation="ツマミ",
            # ),
        ),
    ),
    _ReadingCase(
        text="ピーマンやトウガラシはしばらく生きているが、霜にやられる前に葉を摘んでおいて佃煮にする。",
        expected_kana="ピーマンヤトーガラシワシバラクイキテイルガ、シモニヤラレルマエニハヲツンデオイテツクダニニスル。",
        targets=(
            _TargetExpectation(
                surface="摘ん",
                expected_pronunciation="ツン",
            ),
        ),
    ),
    _ReadingCase(
        text="そのまま地面に頭を擦りつけてろっ",
        expected_kana="ソノママジメンニアタマヲスリツケテロッ",
        targets=(
            _TargetExpectation(
                surface="擦り",
                expected_pronunciation="スリ",
            ),
        ),
    ),
    _ReadingCase(
        text="モンスターとの関わりを息子に擦りつけたりするなど狡猾。",
        expected_kana="モンスタートノカカワリヲムスコニコスリツケタリスルナドコーカツ。",
        targets=(
            # TODO: 本来は「ナスリ」だが現状「コスリ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="擦り",
            #     expected_pronunciation="ナスリ",
            # ),
        ),
    ),
    _ReadingCase(
        text="擦り付け舗装していただきました。",
        expected_kana="スリツケホソーシテイタダキマシタ。",
        targets=(
            _TargetExpectation(
                surface="擦り付け",
                expected_pronunciation="スリツケ",
            ),
        ),
    ),
    _ReadingCase(
        text="他プレイヤーと同じマスを通過することで呪い状態を擦り付けることができる。",
        expected_kana="タプレイヤートオナジマスヲツーカスルコトデノロイジョータイヲコスリツケルコトガデキル。",
        targets=(
            # TODO: 本来は「ナスリツケル」だが現状「コスリツケル」が選ばれてしまう
            # _TargetExpectation(
            #     surface="擦り付ける",
            #     expected_pronunciation="ナスリツケル",
            # ),
        ),
    ),
    _ReadingCase(
        text="他人に擦り付けるような連中が成長できるわけがない。",
        expected_kana="タニンニナスリツケルヨーナレンチューガセーチョーデキルワケガナイ。",
        targets=(
            _TargetExpectation(
                surface="擦り付ける",
                expected_pronunciation="ナスリツケル",
            ),
        ),
    ),
    _ReadingCase(
        text="彼は船尾楼の壁を背の支えにして、その場にずるずると座り込んだ。",
        expected_kana="カレワセンビローノカベヲセノササエニシテ、ソノバニズルズルトスワリコンダ。",
        targets=(
            _TargetExpectation(
                surface="支え",
                expected_pronunciation="ササエ",
            ),
        ),
    ),
    _ReadingCase(
        text="弓道部の主将は、静かに弓を引き絞って矢を放った。",
        expected_kana="キュードーブノシュショーワ、シズカニユミヲヒキシボッテヤヲハナッタ。",
        targets=(
            _TargetExpectation(
                surface="放っ",
                expected_pronunciation="ハナッ",
            ),
        ),
    ),
    _ReadingCase(
        text="あの子は一人で考えたいみたいだから、しばらく放っておいてあげよう。",
        expected_kana="アノコワヒトリデカンガエタイミタイダカラ、シバラクホーッテオイテアゲヨー。",
        targets=(
            _TargetExpectation(
                surface="放っ",
                expected_pronunciation="ホーッ",
            ),
        ),
    ),
    _ReadingCase(
        text="放出駅が自分の育った街だから恩返しをしたい！",
        expected_kana="ハナテンエキガジブンノソダッタマチダカラオンガエシヲシタイ！",
        targets=(
            _TargetExpectation(
                surface="放出",
                expected_pronunciation="ハナテン",
            ),
        ),
    ),
    _ReadingCase(
        text="貸衣装大放出セールというやつです。",
        expected_kana="カシイショーダイホーシュツセールトイウヤツデス。",
        targets=(
            _TargetExpectation(
                surface="放出",
                expected_pronunciation="ホーシュツ",
            ),
        ),
    ),
    _ReadingCase(
        text="平安時代の貴族たちは、和歌を添えた文を交わして恋心を伝え合った。",
        expected_kana="ヘーアンジダイノキゾクタチワ、ワカヲソエタフミヲカワシテコイゴコロヲツタエアッタ。",
        targets=(
            _TargetExpectation(
                surface="文",
                expected_pronunciation="フミ",
            ),
        ),
    ),
    _ReadingCase(
        text="次の英語の文を読んで、下線部を日本語に訳しなさい。",
        expected_kana="ツギノエーゴノブンヲヨンデ、カセンブヲニホンゴニヤクシナサイ。",
        targets=(
            _TargetExpectation(
                surface="文",
                expected_pronunciation="ブン",
            ),
        ),
    ),
    _ReadingCase(
        text="日本では、足袋のサイズを文で表しますが、これは一文銭を並べて長さを測ったことに由来しています。",
        expected_kana="ニホンデワ、タビノサイズヲフミデアラワシマスガ、コレワイチモンセンヲナベテナガサヲハカッタコトニユライシテイマス。",
        targets=(
            # TODO: 本来は「モン」だが現状「フミ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="文",
            #     expected_pronunciation="モン",
            # ),
        ),
    ),
    _ReadingCase(
        text="貨幣価値は１００文（四十ぶんの一両）とされ、当百銭とも呼ばれたが、天保通宝の質量は寛永通宝一文銭５から６枚分しか無く、１００文の価値を見出すのは困難であった。",
        expected_kana="カヘーカチワヒャクモン（ヨンジューブンノイチリョー）トサレ、トーヒャクセントモヨバレタガ、テンポウツーホーノシツリョーワカンエーツーホーイチモンセンゴカラロクマイブンシカナク、ヒャクモンノカチヲミイダスノワコンナンデアッタ。",
        targets=(
            _TargetExpectation(
                surface="文",
                expected_pronunciation="モン",
                occurrence=2,
            ),
        ),
    ),
    _ReadingCase(
        text="どうにかそれを無視し、文書作成ソフトを開く。",
        expected_kana="ドーニカソレヲムシシ、ブンショサクセーソフトヲヒラク。",
        targets=(
            _TargetExpectation(
                surface="文書",
                expected_pronunciation="ブンショ",
            ),
        ),
    ),
    _ReadingCase(
        text="水野家文書などの貴重な古文書も所蔵している。",
        expected_kana="ミズノケモンジョナドノキチョーナコモンジョモショゾーシテイル。",
        targets=(
            _TargetExpectation(
                surface="文書",
                expected_pronunciation="モンジョ",
            ),
        ),
    ),
    _ReadingCase(
        text="レイの静止も断ってシン出撃。",
        expected_kana="レイノセーシモタッテシンシュツゲキ。",
        targets=(
            # TODO: 本来は「コトワッ」だが現状「タッ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="断っ",
            #     expected_pronunciation="コトワッ",
            # ),
        ),
    ),
    _ReadingCase(
        text="元八郎は断った。",
        expected_kana="モトハチローワコトワッタ。",
        targets=(
            _TargetExpectation(
                surface="断っ",
                expected_pronunciation="コトワッ",
            ),
        ),
    ),
    _ReadingCase(
        text="迷いを断って（８月１６日）「女優の魅力は」との問いに、「いろいろな人生を歩めること」と言う。",
        expected_kana="マヨイヲタッテ（ハチガツジューロクニチ）「ジョユーノミリョクワ」トノトイニ、「イロイロナジンセーヲアユメルコト」トイウ。",
        targets=(
            _TargetExpectation(
                surface="断っ",
                expected_pronunciation="タッ",
            ),
        ),
    ),
    _ReadingCase(
        text="「大きくて、凛々しくて…」だったのに、周りにいた方々が写らないようにしたら、こんな写真しか撮れませんでした。",
        expected_kana="「オーキクテ、リリシクテ…」ダッタノニ、マワリニイタカタガタガウツラナイヨーニシタラ、コンナシャシンシカトレマセンデシタ。",
        targets=(
            _TargetExpectation(
                surface="方々",
                expected_pronunciation="カタガタ",
            ),
        ),
    ),
    _ReadingCase(
        text="その話を私が方々でしてみると、驚くことに、ポルシェの燃費なんてどうでもいいのさ、ポルシェの魅力は別のところにあるのさ、と口にする方々の多いこと！",
        expected_kana="ソノハナシヲワタシガホーボーデシテミルト、オドロクコトニ、ポルシェノネンピナンテドーデモイイノサ、ポルシェノミリョクワベツノトコロニアルノサ、トグチニスルカタガタノオーイコト！",
        targets=(
            _TargetExpectation(
                surface="方々",
                expected_pronunciation="ホーボー",
            ),
        ),
    ),
    _ReadingCase(
        text="確かに日中は暑いですよね。",
        expected_kana="タシカニニッチューワアツイデスヨネ。",
        targets=(
            _TargetExpectation(
                surface="日中",
                expected_pronunciation="ニッチュー",
            ),
        ),
    ),
    _ReadingCase(
        text="日向でも十分に生育するのだ。",
        expected_kana="ヒナタデモジューブンニセーイクスルノダ。",
        targets=(
            _TargetExpectation(
                surface="日向",
                expected_pronunciation="ヒナタ",
            ),
        ),
    ),
    _ReadingCase(
        text="和銅６年（７１３年）−大隅国が日向国から分離。",
        expected_kana="ワドーロクネン（ナナヒャクジューサンネン）−オースミコクガヒューガコクカラブンリ。",
        targets=(
            _TargetExpectation(
                surface="日向",
                expected_pronunciation="ヒューガ",
            ),
        ),
    ),
    _ReadingCase(
        text="安枝さんは同ディーブイディー発売イベントが２月２７日にも大阪の信長書店、日本橋店で開催される。",
        expected_kana="ヤスエサンワドーディーブイディーハツバイイベントガニガツニジューシチニチニモオーサカノノブナガショテン、ニッポンバシテンデカイサイサレル。",
        targets=(
            _TargetExpectation(
                surface="日本橋",
                expected_pronunciation="ニッポンバシ",
            ),
        ),
    ),
    _ReadingCase(
        text="保土ヶ谷や品川や日本橋が出て来る。",
        expected_kana="ホドガヤヤシナガワヤニホンバシガデテクル。",
        targets=(
            _TargetExpectation(
                surface="日本橋",
                expected_pronunciation="ニホンバシ",
            ),
        ),
    ),
    _ReadingCase(
        text="明朝、街道の関門前に来るよう、カルボは言う。",
        expected_kana="ミンチョー、カイドーノカンモンマエニクルヨー、カルボワイウ。",
        targets=(
            # TODO: 本来は「ミョーチョー」だが現状「ミンチョー」が選ばれてしまう
            # _TargetExpectation(
            #     surface="明朝",
            #     expected_pronunciation="ミョーチョー",
            # ),
        ),
    ),
    _ReadingCase(
        text="明朝より総攻撃を仕掛ける。",
        expected_kana="ミョーチョーヨリソーコーゲキヲシカケル。",
        targets=(
            _TargetExpectation(
                surface="明朝",
                expected_pronunciation="ミョーチョー",
            ),
        ),
    ),
    _ReadingCase(
        text="１、フォントをＭＳゴシック、または、ＭＳ明朝にする。",
        expected_kana="イチ、フォントヲエムエスゴシック、マタワ、エムエスミンチョーニスル。",
        targets=(
            _TargetExpectation(
                surface="明朝",
                expected_pronunciation="ミンチョー",
            ),
        ),
    ),
    _ReadingCase(
        text="十二時少し前だった。",
        expected_kana="ジューニジスコシマエダッタ。",
        targets=(
            _TargetExpectation(
                surface="時",
                expected_pronunciation="ジ",
            ),
        ),
    ),
    _ReadingCase(
        text="自分だけが、一列の時",
        expected_kana="ジブンダケガ、イチレツノトキ",
        targets=(
            _TargetExpectation(
                surface="時",
                expected_pronunciation="トキ",
                was_preserved=True,
            ),
        ),
    ),
    _ReadingCase(
        text="すっかり長居してしまいました。そろそろお暇いたします。",
        expected_kana="スッカリナガイシテシマイマシタ。ソロソロオイトマイタシマス。",
        targets=(
            _TargetExpectation(
                surface="暇",
                expected_pronunciation="イトマ",
            ),
        ),
    ),
    _ReadingCase(
        text="今週末は特に予定もなく暇なので、久しぶりに映画でも見に行こうと思う。",
        expected_kana="コンシューマツワトクニヨテーモナクヒマナノデ、ヒサシブリニエーガデモミニイコートオモウ。",
        targets=(
            _TargetExpectation(
                surface="暇",
                expected_pronunciation="ヒマ",
            ),
        ),
    ),
    _ReadingCase(
        text="「うめぇな！」最上の褒め言葉です。",
        expected_kana="「ウメーナ！」サイジョーノホメコトバデス。",
        targets=(
            _TargetExpectation(
                surface="最上",
                expected_pronunciation="サイジョー",
            ),
        ),
    ),
    _ReadingCase(
        text="手の届く範囲の最上を狙うって感じのってところかしら",
        expected_kana="テノトドクハンイノモガミヲネラウッテカンジノッテトコロカシラ",
        targets=(
            # TODO: 本来は「サイジョー」だが現状「モガミ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="最上",
            #     expected_pronunciation="サイジョー",
            # ),
        ),
    ),
    _ReadingCase(
        text="寛政４年、最上徳内が渡樺し調査。",
        expected_kana="カンセーヨネン、モガミトクナイガワタリカバシチョーサ。",
        targets=(
            _TargetExpectation(
                surface="最上",
                expected_pronunciation="モガミ",
            ),
        ),
    ),
    _ReadingCase(
        text="緑は最低値、赤は最高値を表す。",
        expected_kana="ミドリワサイテーチ、アカワサイコーチヲアラワス。",
        targets=(
            _TargetExpectation(
                surface="最高値",
                expected_pronunciation="サイコーチ",
            ),
        ),
    ),
    _ReadingCase(
        text="とはいえ、主要３指数最高値圏にあり、今後短期的な調整局面には注意が必要です。",
        expected_kana="トワイエ、シュヨーサンシスーサイコーチケンニアリ、コンゴタンキテキナチョーセーキョクメンニワチューイガヒツヨーデス。",
        targets=(
            # TODO: 本来は「サイタカネ」だが現状「サイコーチ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="最高値",
            #     expected_pronunciation="サイタカネ",
            # ),
        ),
    ),
    _ReadingCase(
        text="相場の最高値と最安値とをいふ。",
        expected_kana="ソーバノサイタカネトサイヤスネトヲイウ。",
        targets=(
            _TargetExpectation(
                surface="最高値",
                expected_pronunciation="サイタカネ",
            ),
        ),
    ),
    _ReadingCase(
        text="申込書の所定の欄に、氏名と提出の月日を記入してください。",
        expected_kana="モーシコミショノショテーノランニ、シメートテーシュツノガッピヲキニューシテクダサイ。",
        targets=(
            _TargetExpectation(
                surface="月日",
                expected_pronunciation="ガッピ",
            ),
        ),
    ),
    _ReadingCase(
        text="故郷を離れてから長い月日が流れたが、あの日の景色は今も忘れられない。",
        expected_kana="コキョーヲハナレテカラナガイツキヒガナガレタガ、アノヒノケシキワイマモワスレラレナイ。",
        targets=(
            _TargetExpectation(
                surface="月日",
                expected_pronunciation="ツキヒ",
            ),
        ),
    ),
    _ReadingCase(
        text="これは、顔見知りによるプロバイダ−運営であり、木目細かなサ−ビス及び利用のホロ−を期待できることを特徴とするものです。",
        expected_kana="コレワ、カオミシリニヨルプロバイダ−ウンエーデアリ、キメコマカナサ−ビスオヨビリヨーノホロ−ヲキタイデキルコトヲトクチョートスルモノデス。",
        targets=(
            _TargetExpectation(
                surface="木目",
                expected_pronunciation="キメ",
            ),
        ),
    ),
    _ReadingCase(
        text="明るく木目がきれいなお店です。",
        expected_kana="アカルクモクメガキレイナオミセデス。",
        targets=(
            _TargetExpectation(
                surface="木目",
                expected_pronunciation="モクメ",
            ),
        ),
    ),
    _ReadingCase(
        text="こうなるともう末期です。",
        expected_kana="コーナルトモーマッキデス。",
        targets=(
            _TargetExpectation(
                surface="末期",
                expected_pronunciation="マッキ",
            ),
        ),
    ),
    _ReadingCase(
        text="レジで財布から札を取り出し、一枚ずつ数えて店員に渡した。",
        expected_kana="レジデサイフカラサツヲトリダシ、イチマイズツカゾエテテンインニワタシタ。",
        targets=(
            _TargetExpectation(
                surface="札",
                expected_pronunciation="サツ",
            ),
        ),
    ),
    _ReadingCase(
        text="初詣に行った神社で家内安全の札を受け、家の神棚に祀った。",
        expected_kana="ハツモーデニイッタジンジャデカナイアンゼンノフダヲウケ、イエノカミダナニマツッタ。",
        targets=(
            _TargetExpectation(
                surface="札",
                expected_pronunciation="フダ",
            ),
        ),
    ),
    _ReadingCase(
        text="今日は本当にありがとうございました。また来ますね。",
        expected_kana="キョーワホントーニアリガトーゴザイマシタ。マタキマスネ。",
        targets=(
            _TargetExpectation(
                surface="来",
                expected_pronunciation="キ",
                expected_outcome="dictionary_default_protected",
                was_preserved=True,
            ),
        ),
    ),
    _ReadingCase(
        text="雨が強くなってきたので、彼は明日の集まりには来ないそうだ。",
        expected_kana="アメガツヨクナッテキタノデ、カレワアシタノアツマリニワコナイソーダ。",
        targets=(
            _TargetExpectation(
                surface="来",
                expected_pronunciation="コ",
                expected_outcome="dictionary_default_protected",
                was_preserved=True,
            ),
        ),
    ),
    _ReadingCase(
        text="来る来年も、どうぞよろしくお願いいたします！",
        expected_kana="キタルライネンモ、ドーゾヨロシクオネガイイタシマス！",
        targets=(
            _TargetExpectation(
                surface="来る",
                expected_pronunciation="キタル",
            ),
        ),
    ),
    _ReadingCase(
        text="彼はこちらの方まで来ると、",
        expected_kana="カレワコチラノホーマデクルト、",
        targets=(
            _TargetExpectation(
                surface="来る",
                expected_pronunciation="クル",
            ),
        ),
    ),
    _ReadingCase(
        text="なお当時の阪神の打撃コーチは上記の柏原であった。",
        expected_kana="ナオトージノハンシンノダゲキコーチワジョーキノカシワバラデアッタ。",
        targets=(
            _TargetExpectation(
                surface="柏原",
                expected_pronunciation="カシワバラ",
            ),
        ),
    ),
    _ReadingCase(
        text="みんなで滋賀県米原市柏原を盛り上げようー！",
        expected_kana="ミンナデシガケンマイバラシカシワラヲモリアゲヨーー！",
        targets=(
            # TODO: 本来は「カシワバラ」だが現状「カシワラ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="柏原",
            #     expected_pronunciation="カシワバラ",
            # ),
        ),
    ),
    _ReadingCase(
        text="柏原市にとっても大きな課題。",
        expected_kana="カシワラシニトッテモオーキナカダイ。",
        targets=(
            _TargetExpectation(
                surface="柏原",
                expected_pronunciation="カシワラ",
            ),
        ),
    ),
    _ReadingCase(
        text="竹で作った壁の保護柵です。",
        expected_kana="タケデツクッタカベノホゴサクデス。",
        targets=(
            _TargetExpectation(
                surface="柵",
                expected_pronunciation="サク",
            ),
        ),
    ),
    _ReadingCase(
        text="兎も角、彼のおかげで、国連事務総長から来た余計な柵がなくなったからな……",
        expected_kana="トモカク、カレノオカゲデ、コクレンジムソーチョーカラキタヨケーナシガラミガナクナッタカラナ……",
        targets=(
            _TargetExpectation(
                surface="柵",
                expected_pronunciation="シガラミ",
            ),
        ),
    ),
    _ReadingCase(
        text="なんていうか、ななちゃんがときどき作る料理とは、何かが根本的に違うような味だった。",
        expected_kana="ナンテイウカ、ナナチャンガトキドキツクルリョーリトワ、ナニカガコンポンテキニチガウヨーナアジダッタ。",
        targets=(
            _TargetExpectation(
                surface="根本",
                expected_pronunciation="コンポン",
            ),
        ),
    ),
    _ReadingCase(
        text="「負けリリーフ」伊藤、根本、小宮山",
        expected_kana="「マケリリーフ」イトー、ネモト、コミヤマ",
        targets=(
            _TargetExpectation(
                surface="根本",
                expected_pronunciation="ネモト",
            ),
        ),
    ),
    _ReadingCase(
        text="この後八戒さんに気孔で一撃、悟浄の鎖鎌で八裂きにされ、止めは三蔵の銃で天昇する予定です（汗）焔…食う？",
        expected_kana="コノアトハッカイサンニキコーデイチゲキ、ゴジョーノクサリガマデヤツザキニサレ、トメワサンゾーノジューデテンノボルスルヨテーデス（アセ）ホノオ…クウ？",
        targets=(
            # TODO: 本来は「トドメ」だが現状「トメ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="止め",
            #     expected_pronunciation="トドメ",
            # ),
        ),
    ),
    _ReadingCase(
        text="午前中はこの程度に止めまして、午後二時から再開いたしたいと存じます。",
        expected_kana="ゴゼンチューワコノテードニトドメマシテ、ゴゴニジカラサイカイイタシタイトゾンジマス。",
        targets=(
            _TargetExpectation(
                surface="止め",
                expected_pronunciation="トドメ",
            ),
        ),
    ),
    _ReadingCase(
        text="力尽くでも止められるものなら止めたい。",
        expected_kana="チカラコトゴトクデモトメラレルモノナラトメタイ。",
        targets=(
            _TargetExpectation(
                surface="止め",
                expected_pronunciation="トメ",
            ),
        ),
    ),
    _ReadingCase(
        text="僕はこれは奇抜なのは止めて！",
        expected_kana="ボクワコレワキバツナノワヤメテ！",
        targets=(
            _TargetExpectation(
                surface="止め",
                expected_pronunciation="ヤメ",
            ),
        ),
    ),
    _ReadingCase(
        text="手伝うの止めて帰ろうかな……",
        expected_kana="テツダウノトメテカエローカナ……",
        targets=(
            # TODO: 本来は「ヤメ」だが現状「トメ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="止め",
            #     expected_pronunciation="ヤメ",
            # ),
        ),
    ),
    _ReadingCase(
        text="みんな、小町先輩を止めろーッ！",
        expected_kana="ミンナ、コマチセンパイヲヤメローッ！",
        targets=(
            # TODO: 本来は「トメロ」だが現状「ヤメロ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="止めろ",
            #     expected_pronunciation="トメロ",
            # ),
        ),
    ),
    _ReadingCase(
        text="こんなこと、正気の沙汰とは思えないっ」",
        expected_kana="コンナコト、ショーキノサタトワオモエナイッ」",
        targets=(
            _TargetExpectation(
                surface="正気",
                expected_pronunciation="ショーキ",
            ),
        ),
    ),
    _ReadingCase(
        text="将棋の終盤、彼は相手の玉の頭に歩を打って詰みに追い込んだ。",
        expected_kana="ショーギノシューバン、カレワアイテノギョクノアタマニフヲウッテツミニオイコンダ。",
        targets=(
            _TargetExpectation(
                surface="歩",
                expected_pronunciation="フ",
            ),
        ),
    ),
    _ReadingCase(
        text="村総草高は２５１石で、年貢は４割４歩。",
        expected_kana="ムラソークサダカワニヒャクゴジューイッコクデ、ネングワヨンワリヨンポ。",
        targets=(
            # TODO: 本来は「ブ」だが現状「ホ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="歩",
            #     expected_pronunciation="ブ",
            # ),
        ),
    ),
    _ReadingCase(
        text="失敗を恐れずに、夢に向かって着実に歩を進めていきたい。",
        expected_kana="シッパイヲオソレズニ、ユメニムカッテチャクジツニホヲススメテイキタイ。",
        targets=(
            _TargetExpectation(
                surface="歩",
                expected_pronunciation="ホ",
            ),
        ),
    ),
    _ReadingCase(
        text="まずは一歩を踏み出してみましょう。",
        expected_kana="マズワイッポヲフミダシテミマショー。",
        targets=(
            _TargetExpectation(
                surface="歩",
                expected_pronunciation="ポ",
            ),
        ),
    ),
    _ReadingCase(
        text="ドライブモードにより、古き良きアナログ機器の歪み感を最新のドーシステムで得ることができます。",
        expected_kana="ドライブモードニヨリ、フルキヨキアナログキキノヒズミカンヲサイシンノドーシステムデエルコトガデキマス。",
        targets=(
            _TargetExpectation(
                surface="歪み",
                expected_pronunciation="ヒズミ",
            ),
        ),
    ),
    _ReadingCase(
        text="間近で覗き込む彼の顔が歪んだ。",
        expected_kana="マジカデノゾキコムカレノカオガユガンダ。",
        targets=(
            _TargetExpectation(
                surface="歪ん",
                expected_pronunciation="ユガン",
            ),
        ),
    ),
    _ReadingCase(
        text="日本人の場合、名字のほかに「氏」や「姓」（かばね）と呼ばれるものがあります。",
        expected_kana="ニホンジンノバアイ、ミョージノホカニ「ウジ」ヤ「セー」（カバネ）トヨバレルモノガアリマス。",
        targets=(
            _TargetExpectation(
                surface="氏",
                expected_pronunciation="ウジ",
            ),
        ),
    ),
    _ReadingCase(
        text="氏は眼鏡をかけていない。",
        expected_kana="シワメガネヲカケテイナイ。",
        targets=(
            _TargetExpectation(
                surface="氏",
                expected_pronunciation="シ",
            ),
        ),
    ),
    _ReadingCase(
        text="気色悪くてしょうがねぇ！",
        expected_kana="キショクワルクテショーガネー！",
        targets=(
            _TargetExpectation(
                surface="気色",
                expected_pronunciation="キショク",
            ),
        ),
    ),
    _ReadingCase(
        text="生まれ持ったＨＳＰ気質。",
        expected_kana="ウマレモッタＨエスピーキシツ。",
        targets=(
            _TargetExpectation(
                surface="気質",
                expected_pronunciation="キシツ",
            ),
        ),
    ),
    _ReadingCase(
        text="なかなか気骨のある男だな",
        expected_kana="ナカナカキコツノアルオトコダナ",
        targets=(
            _TargetExpectation(
                surface="気骨",
                expected_pronunciation="キコツ",
            ),
        ),
    ),
    _ReadingCase(
        text="私は農業経営の指導を現地で行ないますことはなかなかむずかしい、また気骨の折れる、また責任も重い仕事であろうと思います。",
        expected_kana="ワタシワノーギョーケーエーノシドーヲゲンチデオコナイマスコトワナカナカムズカシイ、マタキボネノオレル、マタセキニンモオモイシゴトデアロートオモイマス。",
        targets=(
            _TargetExpectation(
                surface="気骨",
                expected_pronunciation="キボネ",
            ),
        ),
    ),
    _ReadingCase(
        text="数日いや数週間、吾輩は水と牛乳だけですごした。",
        expected_kana="スーニチイヤスーシューカン、ワガハイワミズトギューニューダケデスゴシタ。",
        targets=(
            _TargetExpectation(
                surface="水",
                expected_pronunciation="ミズ",
            ),
        ),
    ),
    _ReadingCase(
        text="水上での戦いに勇気を与える。",
        expected_kana="スイジョーデノタタカイニユーキヲアタエル。",
        targets=(
            _TargetExpectation(
                surface="水上",
                expected_pronunciation="スイジョー",
            ),
        ),
    ),
    _ReadingCase(
        text="水上さん……なんで泣いてるの？",
        expected_kana="ミナカミサン……ナンデナイテルノ？",
        targets=(
            _TargetExpectation(
                surface="水上",
                expected_pronunciation="ミナカミ",
            ),
        ),
    ),
    _ReadingCase(
        text="まさか……シャモーニの名を汚す気では……？",
        expected_kana="マサカ……シャモーニノナヲケガスキデワ……？",
        targets=(
            _TargetExpectation(
                surface="汚す",
                expected_pronunciation="ケガス",
            ),
        ),
    ),
    _ReadingCase(
        text="汚れた面は内側に折り込んでいけばよく、苦手な食べ物をそっと口から出しても巻き込めばこぼれません。",
        expected_kana="ヨゴレタメンワウチガワニオリコンデイケバヨク、ニガテナタベモノヲソットクチカラダシテモマキコメバコボレマセン。",
        targets=(
            _TargetExpectation(
                surface="汚れ",
                expected_pronunciation="ヨゴレ",
            ),
        ),
    ),
    _ReadingCase(
        text="ここも崖、つまり河岸段丘ではないのか。",
        expected_kana="ココモガケ、ツマリカガンダンキューデワナイノカ。",
        targets=(
            _TargetExpectation(
                surface="河岸",
                expected_pronunciation="カガン",
            ),
        ),
    ),
    _ReadingCase(
        text="荒川、綾瀬川が付近で交差していたことから水運（特に舟運）の中継地点として橋戸河岸（千住河岸）が置かれていた。",
        expected_kana="アラカワ、アヤセガワガフキンデコーサシテイタコトカラスイウン（トクニシューウン）ノチューケーチテントシテハシドカシ（センジュカシ）ガオカレテイタ。",
        targets=(
            _TargetExpectation(
                surface="河岸",
                expected_pronunciation="カシ",
                occurrence=1,
            ),
        ),
    ),
    _ReadingCase(
        text="乾杯の前に、彼は上司のグラスにビールを注いだ。",
        expected_kana="カンパイノマエニ、カレワジョーシノグラスニビールヲツイダ。",
        targets=(
            _TargetExpectation(
                surface="注い",
                expected_pronunciation="ツイ",
            ),
        ),
    ),
    _ReadingCase(
        text="山から流れ出た川の水は、やがて平野を抜けて海に注ぐ。",
        expected_kana="ヤマカラナガレデタカワノミズワ、ヤガテヘーヤヲヌケテウミニソソグ。",
        targets=(
            _TargetExpectation(
                surface="注ぐ",
                expected_pronunciation="ソソグ",
            ),
        ),
    ),
    _ReadingCase(
        text="来年の学会発表に向けて、今は研究に全力を注ぐつもりだ。",
        expected_kana="ライネンノガッカイハッピョーニムケテ、イマワケンキューニゼンリョクヲソソグツモリダ。",
        targets=(
            _TargetExpectation(
                surface="注ぐ",
                expected_pronunciation="ソソグ",
            ),
        ),
    ),
    _ReadingCase(
        text="はい、最寄りの清水五条駅からは徒歩で４分です。",
        expected_kana="ハイ、モヨリノキヨミズゴジョーエキカラワトホデヨンプンデス。",
        targets=(
            _TargetExpectation(
                surface="清水",
                expected_pronunciation="キヨミズ",
            ),
        ),
    ),
    _ReadingCase(
        text="東京、浅草寺門前「やげん堀（中島商店）」、京都、清水寺門前「七味家」、長野、善光寺門前「八幡屋礒五郎」が老舗である。",
        expected_kana="トーキョー、センソージモンマエ「ヤゲンボリ（ナカジマショーテン）」、キョート、シミズテラカドマエ「シチミケ」、ナガノ、ゼンコージモンマエ「ヤハタヤイソゴロー」ガシニセデアル。",
        targets=(
            # TODO: 本来は「キヨミズ」だが現状「シミズ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="清水",
            #     expected_pronunciation="キヨミズ",
            # ),
        ),
    ),
    _ReadingCase(
        text="「先発」成瀬、小林宏、渡辺俊、大嶺、唐川、清水直",
        expected_kana="「センパツ」ナルセ、コバヤシヒロシ、ワタナベシュン、オーミネ、カラカワ、シミズチョク",
        targets=(
            _TargetExpectation(
                surface="清水",
                expected_pronunciation="シミズ",
            ),
        ),
    ),
    _ReadingCase(
        text="和書、漢書、洋書の70余種の文献を参考に記述されていて、それらの参考文献には、日本の書物では『采覧異言』や『華夷通商考』等がみられる。",
        expected_kana="ワショ、カンショ、ヨーショノナナジューヨシュノブンケンヲサンコーニキジュツサレテイテ、ソレラノサンコーブンケンニワ、ニホンノショモツデワ『サイランイゲン』ヤ『カイツーショーコー』ナドガミラレル。",
        targets=(
            _TargetExpectation(
                surface="漢書",
                expected_pronunciation="カンショ",
            ),
        ),
    ),
    _ReadingCase(
        text="最終回、第四回目は、史部より漢書をとりあげた。",
        expected_kana="サイシューカイ、ダイヨンカイメワ、フヒトベヨリカンショヲトリアゲタ。",
        targets=(
            # TODO: 本来は「カンジョ」だが現状「カンショ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="漢書",
            #     expected_pronunciation="カンジョ",
            # ),
        ),
    ),
    _ReadingCase(
        text="班固の漢書律暦志に記載されている。",
        expected_kana="ハンコノカンジョリツレキココロザシニキサイサレテイル。",
        targets=(
            _TargetExpectation(
                surface="漢書",
                expected_pronunciation="カンジョ",
            ),
        ),
    ),
    _ReadingCase(
        text="ここにするか、とフラッと暖簾を潜った。",
        expected_kana="ココニスルカ、トフラットノレンヲモグッタ。",
        targets=(
            # TODO: 本来は「クグッ」だが現状「モグッ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="潜っ",
            #     expected_pronunciation="クグッ",
            # ),
        ),
    ),
    _ReadingCase(
        text="あとは海中を潜っていくか、途中崩落していた廊下をなんらかの手段を使って渡るか……",
        expected_kana="アトワカイチューヲクグッテイクカ、トチューホーラクシテイタローカヲナンラカノシュダンヲツカッテワタルカ……",
        targets=(
            # TODO: 本来は「モグッ」だが現状「クグッ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="潜っ",
            #     expected_pronunciation="モグッ",
            # ),
        ),
    ),
    _ReadingCase(
        text="鳥居を潜り、社へ行けと……休む暇もないか。",
        expected_kana="トリイヲクグリ、ヤシロエイケト……ヤスムヒマモナイカ。",
        targets=(
            _TargetExpectation(
                surface="潜り",
                expected_pronunciation="クグリ",
            ),
        ),
    ),
    _ReadingCase(
        text="それから、水に潜ります。",
        expected_kana="ソレカラ、ミズニモグリマス。",
        targets=(
            _TargetExpectation(
                surface="潜り",
                expected_pronunciation="モグリ",
            ),
        ),
    ),
    _ReadingCase(
        text="火をつけて赤ちゃんの頭の上で３回まわす。",
        expected_kana="ヒヲツケテアカチャンノアタマノウエデサンカイマワス。",
        targets=(
            _TargetExpectation(
                surface="火",
                expected_pronunciation="ヒ",
            ),
        ),
    ),
    _ReadingCase(
        text="停電した夜、母は暗い部屋にろうそくの明かりを点した。",
        expected_kana="テーデンシタヨル、ハハワクライヘヤニローソクノアカリヲトモシタ。",
        targets=(
            _TargetExpectation(
                surface="点し",
                expected_pronunciation="トモシ",
            ),
        ),
    ),
    _ReadingCase(
        text="パソコン作業で目が疲れたので、寝る前に目薬を点すようにしている。",
        expected_kana="パソコンサギョーデメガツカレタノデ、ネルマエニメグスリヲサスヨーニシテイル。",
        targets=(
            _TargetExpectation(
                surface="点す",
                expected_pronunciation="サス",
            ),
        ),
    ),
    _ReadingCase(
        text="はっはっはっ、然もありなん。",
        expected_kana="ハッハッハッ、サモアリナン。",
        targets=(
            _TargetExpectation(
                surface="然も",
                expected_pronunciation="サモ",
            ),
        ),
    ),
    _ReadingCase(
        text="私は、「サテンの夜」の様な楽曲と同時に、この様なアップテンポで然もメロディもいい曲が好みです（至極当然の事ではありますが）。",
        expected_kana="ワタシワ、「サテンノヨル」ノヨーナガッキョクトドージニ、コノヨーナアップテンポデシカモメロディモイイキョクガコノミデス（シゴクトーゼンノコトデワアリマスガ）。",
        targets=(
            _TargetExpectation(
                surface="然も",
                expected_pronunciation="シカモ",
            ),
        ),
    ),
    _ReadingCase(
        text="同窓会の皆様方には物心両面にてご支援いただき、大変感謝しております。",
        expected_kana="ドーソーカイノミナサマカタニワブッシンリョーメンニテゴシエンイタダキ、タイヘンカンシャシテオリマス。",
        targets=(
            _TargetExpectation(
                surface="物心",
                expected_pronunciation="ブッシン",
            ),
        ),
    ),
    _ReadingCase(
        text="物心ついた頃からあったからな。",
        expected_kana="モノゴコロツイタコロカラアッタカラナ。",
        targets=(
            _TargetExpectation(
                surface="物心",
                expected_pronunciation="モノゴコロ",
            ),
        ),
    ),
    _ReadingCase(
        text="だったら玉が膨らむのかじゃと？",
        expected_kana="ダッタラタマガフクラムノカジャト？",
        targets=(
            _TargetExpectation(
                surface="玉",
                expected_pronunciation="タマ",
            ),
        ),
    ),
    _ReadingCase(
        text="この貯金箱の中身が全部５００円玉ですか？",
        expected_kana="コノチョキンバコノナカミガゼンブゴヒャクエンタマデスカ？",
        targets=(
            # TODO: 本来は「ダマ」だが現状「タマ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="玉",
            #     expected_pronunciation="ダマ",
            # ),
        ),
    ),
    _ReadingCase(
        text="５００円玉だけ貯めていったのです。",
        expected_kana="ゴヒャクエンダマダケタメテイッタノデス。",
        targets=(
            _TargetExpectation(
                surface="玉",
                expected_pronunciation="ダマ",
            ),
        ),
    ),
    _ReadingCase(
        text="先頭。ウッズが２−２からの７球目。",
        expected_kana="セントー。ウッズガニ−ニカラノナナキューメ。",
        targets=(
            _TargetExpectation(
                surface="球",
                expected_pronunciation="キュー",
            ),
        ),
    ),
    _ReadingCase(
        text="その生は美しいのか。",
        expected_kana="ソノセーワウツクシイノカ。",
        targets=(
            _TargetExpectation(
                surface="生",
                expected_pronunciation="セー",
            ),
        ),
    ),
    _ReadingCase(
        text="２。生ハムは半分に切る。",
        expected_kana="ニ。ナマハムワハンブンニキル。",
        targets=(
            _TargetExpectation(
                surface="生",
                expected_pronunciation="ナマ",
            ),
        ),
    ),
    _ReadingCase(
        text="ピザを作るときは、生地をよくこねてから一時間ほど寝かせる。",
        expected_kana="ピザヲツクルトキワ、キジヲヨクコネテカライチジカンホドネカセル。",
        targets=(
            _TargetExpectation(
                surface="生地",
                expected_pronunciation="キジ",
            ),
        ),
    ),
    _ReadingCase(
        text="キエフでミコラ、リセンコ記念音楽学校に進学し、３つのコースを修了した後、生地に戻りレニングラード音楽院に進学する。",
        expected_kana="キエフデミコラ、リセンコキネンオンガクガッコーニシンガクシ、ミッツノコースヲシューリョーシタアト、キジニモドリレニングラードオンガクインニシンガクスル。",
        targets=(
            # TODO: 本来は「セーチ」だが現状「キジ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="生地",
            #     expected_pronunciation="セーチ",
            # ),
        ),
    ),
    _ReadingCase(
        text="作家の生地である青森県の金木町を訪ね、生家を改装した記念館を見学した。",
        expected_kana="サッカノキジデアルアオモリケンノカナギマチヲタズネ、セーカヲカイソーシタキネンカンヲケンガクシタ。",
        targets=(
            # TODO: 本来は「セーチ」だが現状「キジ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="生地",
            #     expected_pronunciation="セーチ",
            # ),
        ),
    ),
    _ReadingCase(
        text="尤も、医学的生物学的知識があるとも思えない主人の見立てであるから、信ずるにたる診断でないことは言うをまたないが、激しい下痢と腹痛に悩まされ、その後ずっと食欲不振が続いたものである。",
        expected_kana="モットモ、イガクテキセーブツガクテキチシキガアルトモオモエナイシュジンノミタテデアルカラ、シンズルニタルシンダンデナイコトワイウヲマタナイガ、ハゲシイゲリトフクツーニナヤマサレ、ソノゴズットショクヨクフシンガツズイタモノデアル。",
        targets=(
            _TargetExpectation(
                surface="生物",
                expected_pronunciation="セーブツ",
            ),
        ),
    ),
    _ReadingCase(
        text="支払方法代金引換、クレジットカード支払期限代金引換、商品到着時クレジットカード、ご注文時返品、キャンセル生物につきお客様のご都合による返品は受け付けておりません。",
        expected_kana="シハライホーホーダイキンヒキカエ、クレジットカードシハライキゲンダイキンヒキカエ、ショーヒントーチャクジクレジットカード、ゴチューモンジヘンピン、キャンセルナマモノニツキオキャクサマノゴツゴーニヨルヘンピンワウケツケテオリマセン。",
        targets=(
            _TargetExpectation(
                surface="生物",
                expected_pronunciation="ナマモノ",
            ),
        ),
    ),
    _ReadingCase(
        text="生花を習い、花を手に取るようになってから感じはじめたことです。",
        expected_kana="セーカヲナライ、ハナヲテニトルヨーニナッテカラカンジハジメタコトデス。",
        targets=(
            # TODO: 本来は「イケバナ」だが現状「セーカ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="生花",
            #     expected_pronunciation="イケバナ",
            # ),
        ),
    ),
    _ReadingCase(
        text="生花草月流、小原流を習う。",
        expected_kana="イケバナソーゲツリュー、オハラリューヲナラウ。",
        targets=(
            _TargetExpectation(
                surface="生花",
                expected_pronunciation="イケバナ",
            ),
        ),
    ),
    _ReadingCase(
        text="はじめは、生花部（花屋）からのスタートです、直接、お客様と接する事はありませんでした。",
        expected_kana="ハジメワ、イケバナブ（ハナヤ）カラノスタートデス、チョクセツ、オキャクサマトセッスルコトワアリマセンデシタ。",
        targets=(
            # TODO: 本来は「セーカ」だが現状「イケバナ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="生花",
            #     expected_pronunciation="セーカ",
            # ),
        ),
    ),
    _ReadingCase(
        text="新鮮な生花が揃っています。",
        expected_kana="シンセンナセーカガソロッテイマス。",
        targets=(
            _TargetExpectation(
                surface="生花",
                expected_pronunciation="セーカ",
            ),
        ),
    ),
    _ReadingCase(
        text="もうすぐ町中が桜に彩られる季節がやってきますよ。",
        expected_kana="モースグマチジューガサクラニイロドラレルキセツガヤッテキマスヨ。",
        targets=(
            _TargetExpectation(
                surface="町中",
                expected_pronunciation="マチジュー",
            ),
        ),
    ),
    _ReadingCase(
        text="したがいまして、町中を通すような場合には、思い切った事前にそういう工作の上に敷設をしてまいらなきゃならぬ。",
        expected_kana="シタガイマシテ、マチジューヲトースヨーナバアイニワ、オモイキッタジゼンニソーユウコーサクノウエニフセツヲシテマイラナキャナラヌ。",
        targets=(
            # TODO: 本来は「マチナカ」だが現状「マチジュー」が選ばれてしまう
            # _TargetExpectation(
            #     surface="町中",
            #     expected_pronunciation="マチナカ",
            # ),
        ),
    ),
    _ReadingCase(
        text="数江は町中から逃げてきた一人の庶民に声をかけた。",
        expected_kana="カズエワマチナカカラニゲテキタヒトリノショミンニコエヲカケタ。",
        targets=(
            _TargetExpectation(
                surface="町中",
                expected_pronunciation="マチナカ",
            ),
        ),
    ),
    _ReadingCase(
        text="庭の柿の木の枝に、小鳥が一羽留まって鳴いている。",
        expected_kana="ニワノカキノキノエダニ、コトリガイチワトマッテナイテイル。",
        targets=(
            _TargetExpectation(
                surface="留まっ",
                expected_pronunciation="トマッ",
            ),
        ),
    ),
    _ReadingCase(
        text="多くの同僚が転職していく中、彼は会社に留まることを選んだ。",
        expected_kana="オークノドーリョーガテンショクシテイクナカ、カレワカイシャニトドマルコトヲエランダ。",
        targets=(
            _TargetExpectation(
                surface="留まる",
                expected_pronunciation="トドマル",
            ),
        ),
    ),
    _ReadingCase(
        text="早めに避難を呼びかけたことで、台風の被害を最小限に留めることができた。",
        expected_kana="ハヤメニヒナンヲヨビカケタコトデ、タイフーノヒガイヲサイショーゲンニトドメルコトガデキタ。",
        targets=(
            _TargetExpectation(
                surface="留める",
                expected_pronunciation="トドメル",
            ),
        ),
    ),
    _ReadingCase(
        text="出かける前に、シャツの一番上のボタンを留めるのを忘れないように。",
        expected_kana="デカケルマエニ、シャツノイチバンウエノボタンヲトメルノヲワスレナイヨーニ。",
        targets=(
            _TargetExpectation(
                surface="留める",
                expected_pronunciation="トメル",
            ),
        ),
    ),
    _ReadingCase(
        text="白金の和食屋のオーナーシェフが劇団を立ち上げ！",
        expected_kana="シロカネノワショクヤノオーナーシェフガゲキダンヲタチアゲ！",
        targets=(
            _TargetExpectation(
                surface="白金",
                expected_pronunciation="シロカネ",
            ),
        ),
    ),
    _ReadingCase(
        text="白金も高値５２０８円に迫る勢い。",
        expected_kana="ハッキンモタカネゴセンニヒャクハチエンニセマルイキオイ。",
        targets=(
            _TargetExpectation(
                surface="白金",
                expected_pronunciation="ハッキン",
            ),
        ),
    ),
    _ReadingCase(
        text="そこに型破りな役人、白鳥が登場することで、「静」から「動」へと転換。",
        expected_kana="ソコニカタヤブリナヤクニン、シラトリガトージョースルコトデ、「セー」カラ「ドー」エトテンカン。",
        targets=(
            _TargetExpectation(
                surface="白鳥",
                expected_pronunciation="シラトリ",
            ),
        ),
    ),
    _ReadingCase(
        text="橋の名称は、室蘭港の別名「白鳥湾」から名づけられた。",
        expected_kana="ハシノメーショーワ、ムロランコーノベツメー「シラトリワン」カラナズケラレタ。",
        targets=(
            # TODO: 本来は「ハクチョー」だが現状「シラトリ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="白鳥",
            #     expected_pronunciation="ハクチョー",
            # ),
        ),
    ),
    _ReadingCase(
        text="支度金は通常、目下の者に与える金銭だからである。",
        expected_kana="シタクキンワツージョー、メシタノモノニアタエルキンセンダカラデアル。",
        targets=(
            _TargetExpectation(
                surface="目下",
                expected_pronunciation="メシタ",
            ),
        ),
    ),
    _ReadingCase(
        text="これは目下大阪の渋沢倉庫にある。",
        expected_kana="コレワモッカオーサカノシブサワソーコニアル。",
        targets=(
            _TargetExpectation(
                surface="目下",
                expected_pronunciation="モッカ",
            ),
        ),
    ),
    _ReadingCase(
        text="心配だったから、直で来ちゃった",
        expected_kana="シンパイダッタカラ、ジカデキチャッタ",
        targets=(
            _TargetExpectation(
                surface="直",
                expected_pronunciation="ジカ",
            ),
        ),
    ),
    _ReadingCase(
        text="我らも直、国に戻りますゆえ、呉での思い出……というわけでもございませぬが。",
        expected_kana="ワレラモジキ、クニニモドリマスユエ、クレデノオモイデ……トイウワケデモゴザイマセヌガ。",
        targets=(
            _TargetExpectation(
                surface="直",
                expected_pronunciation="ジキ",
            ),
        ),
    ),
    _ReadingCase(
        text="４０歳未満の農業直払制度の導入はムン、ジェイン（文在寅）大統領が、大統領選挙でも公約している。",
        expected_kana="ヨンジュッサイミマンノノーギョーチョクハライセードノドーニューワムン、ジェイン（ブンザイトラ）ダイトーリョーガ、ダイトーリョーセンキョデモコーヤクシテイル。",
        targets=(
            _TargetExpectation(
                surface="直",
                expected_pronunciation="チョク",
            ),
        ),
    ),
    _ReadingCase(
        text="非米価格での石油取引は国家間の相対取引で、統計に全く出てこない。",
        expected_kana="ヒベーカカクデノセキユトリヒキワコッカカンノアイタイトリヒキデ、トーケーニマッタクデテコナイ。",
        targets=(
            _TargetExpectation(
                surface="相対",
                expected_pronunciation="アイタイ",
            ),
        ),
    ),
    _ReadingCase(
        text="現在関割外でピーシー原料用ＮＣが輸入されている理由として、国産と輸入の相対価格の問題もあろうが、何よりも国産ＮＣ供給量不足が考えられる。",
        expected_kana="ゲンザイセキワリガイデピーシーゲンリョーヨーエヌシーガユニューサレテイルリユートシテ、コクサントユニューノアイタイカカクノモンダイモアローガ、ナニヨリモコクサンエヌシーキョーキューリョーブソクガカンガエラレル。",
        targets=(
            # TODO: 本来は「ソータイ」だが現状「アイタイ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="相対",
            #     expected_pronunciation="ソータイ",
            # ),
        ),
    ),
    _ReadingCase(
        text="１人当たり食料支出が減少する中で進行している「食の外部化」（調理食品、外食支出の比率上昇）の趨勢が今後も維持されるとすれば、業務用市場が相対的に拡大する可能性は高い８）。",
        expected_kana="ヒトリアタリショクリョーシシュツガゲンショースルナカデシンコーシテイル「ショクノガイブカ」（チョーリショクヒン、ガイショクシシュツノヒリツジョーショー）ノスーセーガコンゴモイジサレルトスレバ、ギョームヨーシジョーガソータイテキニカクダイスルカノーセーワタカイハチ）。",
        targets=(
            _TargetExpectation(
                surface="相対",
                expected_pronunciation="ソータイ",
            ),
        ),
    ),
    _ReadingCase(
        text="私達以外の命が瞬いてる。",
        expected_kana="ワタシタチイガイノイノチガマタタイテル。",
        targets=(
            _TargetExpectation(
                surface="瞬い",
                expected_pronunciation="マタタイ",
            ),
        ),
    ),
    _ReadingCase(
        text="そう考えると、私たちは瞬きするたびに、ほんの一瞬ずつ、サブリミナルのように闇を体験しているわけだ",
        expected_kana="ソーカンガエルト、ワタシタチワマバタキスルタビニ、ホンノイッシュンズツ、サブリミナルノヨーニヤミヲタイケンシテイルワケダ",
        targets=(
            _TargetExpectation(
                surface="瞬き",
                expected_pronunciation="マバタキ",
            ),
        ),
    ),
    _ReadingCase(
        text="本日はわが社の新製品について、担当者よりご説明いたします。",
        expected_kana="ホンジツワワガシャノシンセーヒンニツイテ、タントーシャヨリゴセツメーイタシマス。",
        targets=(
            _TargetExpectation(
                surface="社",
                expected_pronunciation="シャ",
            ),
        ),
    ),
    _ReadingCase(
        text="村はずれの森の奥には、古くから土地の神を祀る小さな社がある。",
        expected_kana="ムラハズレノモリノオクニワ、フルクカラトチノカミヲマツルチーサナヤシロガアル。",
        targets=(
            _TargetExpectation(
                surface="社",
                expected_pronunciation="ヤシロ",
            ),
        ),
    ),
    _ReadingCase(
        text="北伊勢の豪族神戸具盛の養嗣子となり、神戸家当主となる。",
        expected_kana="キタイセノゴーゾクカンベグモリノヨーシシトナリ、カンベケトーシュトナル。",
        targets=(
            _TargetExpectation(
                surface="神戸",
                expected_pronunciation="カンベ",
                occurrence=1,
            ),
        ),
    ),
    _ReadingCase(
        text="また神戸へ行って買った時には写真付きで紹介します！",
        expected_kana="マタコーベエイッテカッタトキニワシャシンツキデショーカイシマス！",
        targets=(
            _TargetExpectation(
                surface="神戸",
                expected_pronunciation="コーベ",
            ),
        ),
    ),
    _ReadingCase(
        text="というある種、諦念の境地を切り開ける向きにはこれも快適な生活かもしれない。",
        expected_kana="トイウアルシュ、テーネンノキョーチヲキリヒラケルムキニワコレモカイテキナセーカツカモシレナイ。",
        targets=(
            _TargetExpectation(
                surface="種",
                expected_pronunciation="シュ",
            ),
        ),
    ),
    _ReadingCase(
        text="黄ピーマンはへたと種を取り、縦に細切りにする。",
        expected_kana="キピーマンワヘタトタネヲトリ、タテニコマギリニスル。",
        targets=(
            _TargetExpectation(
                surface="種",
                expected_pronunciation="タネ",
            ),
        ),
    ),
    _ReadingCase(
        text="レストランみたいに、皿の左右にスプーンとナイフとフォークが並び、皿の斜め上に空のグラスが置いてある。",
        expected_kana="レストランミタイニ、サラノサユーニスプーントナイフトフォークガナラビ、サラノナナメウエニカラノグラスガオイテアル。",
        targets=(
            _TargetExpectation(
                surface="空",
                expected_pronunciation="カラ",
            ),
        ),
    ),
    _ReadingCase(
        text="１９８０年代後半のバブル期、就職戦線は空前の売り手市場でした。",
        expected_kana="センキューヒャクハチジューネンダイコーハンノバブルキ、シューショクセンセンワクーゼンノウリテシジョーデシタ。",
        targets=(
            # 辞書の「空前」の1語が選ばれ、「空」だけの形態素範囲を持たないため、全文読みと診断結果を固定
            _TargetExpectation(
                surface="空",
                expected_outcome="no_exact_morph_range",
            ),
        ),
    ),
    _ReadingCase(
        text="晴れてたのに着替えてる間に空一面雲になってしまったなんてことも。",
        expected_kana="ハレテタノニキガエテルアイダニクーイチメンクモニナッテシマッタナンテコトモ。",
        targets=(
            # TODO: 本来は「ソラ」だが現状「クー」が選ばれてしまう
            # _TargetExpectation(
            #     surface="空",
            #     expected_pronunciation="ソラ",
            # ),
        ),
    ),
    _ReadingCase(
        text="青さゆえ空もポプラの場所忘れ",
        expected_kana="アオサユエソラモポプラノバショワスレ",
        targets=(
            _TargetExpectation(
                surface="空",
                expected_pronunciation="ソラ",
            ),
        ),
    ),
    _ReadingCase(
        text="空いている助手席に身を乗り出した志津をそれかと確かめるように、正樹が上体を屈めて車をのぞき見た。",
        expected_kana="アイテイルジョシュセキニミヲノリダシタシズヲソレカトタシカメルヨーニ、マサキガジョータイヲカガメテクルマヲノゾキミタ。",
        targets=(
            _TargetExpectation(
                surface="空い",
                expected_pronunciation="アイ",
            ),
        ),
    ),
    _ReadingCase(
        text="お腹もあまり空かない…。",
        expected_kana="オナカモアマリスカナイ…。",
        targets=(
            _TargetExpectation(
                surface="空か",
                expected_pronunciation="スカ",
            ),
        ),
    ),
    _ReadingCase(
        text="竹田宮−「恒」を用いる。",
        expected_kana="タケダミヤ−「ツネ」ヲモチイル。",
        targets=(
            _TargetExpectation(
                surface="竹田",
                expected_pronunciation="タケダ",
            ),
        ),
    ),
    _ReadingCase(
        text="今日は、その当時とまったく同じような竹馬を作りよります。",
        expected_kana="コンニチワ、ソノトージトマッタクオナジヨーナタケウマヲツクリヨリマス。",
        targets=(
            _TargetExpectation(
                surface="竹馬",
                expected_pronunciation="タケウマ",
            ),
        ),
    ),
    _ReadingCase(
        text="マツボックリがマツフグリと呼ばれていたことを示す古い文献例は、俳諧竹馬狂吟集（１４９９）の中に見ることができます（日国（第二版）「まつふぐり」の項による）。",
        expected_kana="マツボックリガマツフグリトヨバレテイタコトヲシメスフルイブンケンレーワ、ハイカイタケウマキョーギンシュー（センヨンヒャクキュージューキュー）ノナカニミルコトガデキマス（ニッコク（ダイニハン）「マツフグリ」ノコーニヨル）。",
        targets=(
            # TODO: 本来は「チクバ」だが現状「タケウマ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="竹馬",
            #     expected_pronunciation="チクバ",
            # ),
        ),
    ),
    _ReadingCase(
        text="この定義の中に実在型看護診断およびウエルネス型看護診断という新しい用語が出てきたが、これらはナンダ看護診断のタイプであるので次の節で解説する。",
        expected_kana="コノテーギノナカニジツザイガタカンゴシンダンオヨビウエルネスガタカンゴシンダントイウアタラシイヨーゴガデテキタガ、コレラワナンダカンゴシンダンノタイプデアルノデツギノセツデカイセツスル。",
        targets=(
            _TargetExpectation(
                surface="節",
                expected_pronunciation="セツ",
            ),
        ),
    ),
    _ReadingCase(
        text="思い当たる節はあるようね",
        expected_kana="オモイアタルフシワアルヨーネ",
        targets=(
            _TargetExpectation(
                surface="節",
                expected_pronunciation="フシ",
            ),
        ),
    ),
    _ReadingCase(
        text="お米の主成分は炭水化物。",
        expected_kana="オコメノシュセーブンワタンスイカブツ。",
        targets=(
            _TargetExpectation(
                surface="米",
                expected_pronunciation="コメ",
            ),
        ),
    ),
    _ReadingCase(
        text="米テレビ番組がアイフォン工場に潜入！",
        expected_kana="ベーテレビバングミガアイフォンコージョーニセンニュー！",
        targets=(
            _TargetExpectation(
                surface="米",
                expected_pronunciation="ベー",
            ),
        ),
    ),
    _ReadingCase(
        text="むう、なんという粋な帯。",
        expected_kana="ムウ、ナントイウイキナオビ。",
        targets=(
            _TargetExpectation(
                surface="粋",
                expected_pronunciation="イキ",
            ),
        ),
    ),
    _ReadingCase(
        text="古より続く技術の粋を集めて鍛刀した傑作だ。",
        expected_kana="イニシエヨリツズクギジュツノスイヲアツメテキタエトーシタケッサクダ。",
        targets=(
            _TargetExpectation(
                surface="粋",
                expected_pronunciation="スイ",
            ),
        ),
    ),
    _ReadingCase(
        text="緊張するとつい格好をつけてしまうが、本当は素の自分を見てほしい。",
        expected_kana="キンチョースルトツイカッコーヲツケテシマウガ、ホントーワスノジブンヲミテホシイ。",
        targets=(
            _TargetExpectation(
                surface="素",
                expected_pronunciation="ス",
            ),
        ),
    ),
    _ReadingCase(
        text="これだけ色とりどりだと、どんな素材で色をつけているのか気になるところですが、着色料は基本的にはチョウマメなどの天然色素を使っているとのこと。",
        expected_kana="コレダケイロトリドリダト、ドンナソザイデイロヲツケテイルノカキニナルトコロデスガ、チャクショクリョーワキホンテキニワチョーマメナドノテンネンショクソヲツカッテイルトノコト。",
        targets=(
            _TargetExpectation(
                surface="素",
                expected_pronunciation="ソ",
                occurrence=1,
            ),
        ),
    ),
    _ReadingCase(
        text="鍋にお湯を沸かして、粉末のスープの素を溶かす。",
        expected_kana="ナベニオユヲワカシテ、フンマツノスープノモトヲトカス。",
        targets=(
            _TargetExpectation(
                surface="素",
                expected_pronunciation="モト",
            ),
        ),
    ),
    _ReadingCase(
        text="女将の細やかな気配りのおかげで、旅館での滞在はとても快適だった。",
        expected_kana="オカミノコマヤカナキクバリノオカゲデ、リョカンデノタイザイワトテモカイテキダッタ。",
        targets=(
            _TargetExpectation(
                surface="細やか",
                expected_pronunciation="コマヤカ",
            ),
        ),
    ),
    _ReadingCase(
        text="日頃の感謝を込めて、細やかながらお祝いの品をお贈りいたします。",
        expected_kana="ヒゴロノカンシャヲコメテ、コマヤカナガラオイワイノシナヲオオクリイタシマス。",
        targets=(
            # TODO: 本来は「ササヤカ」だが現状「コマヤカ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="細やか",
            #     expected_pronunciation="ササヤカ",
            # ),
        ),
    ),
    _ReadingCase(
        text="日頃の感謝を込めて細やかですが『プレゼント』もご用意しました",
        expected_kana="ヒゴロノカンシャヲコメテコマヤカデスガ『プレゼント』モゴヨーイシマシタ",
        targets=(
            # TODO: 本来は「ササヤカ」だが現状「コマヤカ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="細やか",
            #     expected_pronunciation="ササヤカ",
            # ),
        ),
    ),
    _ReadingCase(
        text="政府は細目に基づき、準備を加速させる。",
        expected_kana="セーフワサイモクニモトズキ、ジュンビヲカソクサセル。",
        targets=(
            _TargetExpectation(
                surface="細目",
                expected_pronunciation="サイモク",
            ),
        ),
    ),
    _ReadingCase(
        text="普段は前髪で目が隠れているが、設定画によれば細目。",
        expected_kana="フダンワマエガミデメガカクレテイルガ、セッテーガニヨレバホソメ。",
        targets=(
            _TargetExpectation(
                surface="細目",
                expected_pronunciation="ホソメ",
            ),
        ),
    ),
    _ReadingCase(
        text="現場の事情をよく理解していなければ、解決の緒をつかむことも難しかったと思います」と話す。",
        expected_kana="ゲンバノジジョーヲヨクリカイシテイナケレバ、カイケツノイトグチヲツカムコトモムズカシカッタトオモイマス」トハナス。",
        targets=(
            _TargetExpectation(
                surface="緒",
                expected_pronunciation="イトグチ",
            ),
        ),
    ),
    _ReadingCase(
        text="今回は、虫師続章の第１５話「光の緒」の感想を書きたいと思います１５話は虫の話ではなく、人が虫の性質を帯びてしまった話ですかね。",
        expected_kana="コンカイワ、ムシシゾクショーノダイジューゴワ「ヒカリノオ」ノカンソーヲカキタイトオモイマスジューゴワワムシノハナシデワナク、ヒトガムシノセーシツヲオビテシマッタハナシデスカネ。",
        targets=(
            _TargetExpectation(
                surface="緒",
                expected_pronunciation="オ",
            ),
        ),
    ),
    _ReadingCase(
        text="そこで会うことの、ご縁。",
        expected_kana="ソコデアウコトノ、ゴエン。",
        targets=(
            _TargetExpectation(
                surface="縁",
                expected_pronunciation="エン",
            ),
        ),
    ),
    _ReadingCase(
        text="それぞれが丸く縁取られていて、まるでタピオカのよう。",
        expected_kana="ソレゾレガマルクフチトラレテイテ、マルデタピオカノヨー。",
        targets=(
            _TargetExpectation(
                surface="縁",
                expected_pronunciation="フチ",
            ),
        ),
    ),
    _ReadingCase(
        text="受洗後まもなく高吉が急死し、キリシタンになったことが神仏の罰を招いたと恐れられたためである。",
        expected_kana="ジュセンゴマモナクコーキチガキューシシ、キリシタンニナッタコトガシンブツノバチヲマネイタトオソレラレタタメデアル。",
        targets=(
            _TargetExpectation(
                surface="罰",
                expected_pronunciation="バチ",
            ),
        ),
    ),
    _ReadingCase(
        text="罪と罰とかが結構好きですね",
        expected_kana="ツミトバツトカガケッコースキデスネ",
        targets=(
            _TargetExpectation(
                surface="罰",
                expected_pronunciation="バツ",
            ),
        ),
    ),
    _ReadingCase(
        text="羽生結弦、６６年ぶりの五輪連覇！",
        expected_kana="ハニューユズル、ロクジューロクネンブリノゴリンレンパ！",
        targets=(
            _TargetExpectation(
                surface="羽生",
                expected_pronunciation="ハニュー",
            ),
        ),
    ),
    _ReadingCase(
        text="舞岡公園の池にも翡翠が飛来するが行動範囲内かも知れない。",
        expected_kana="マイオカコーエンノイケニモカワセミガヒライスルガコードーハンイナイカモシレナイ。",
        targets=(
            _TargetExpectation(
                surface="翡翠",
                expected_pronunciation="カワセミ",
            ),
        ),
    ),
    _ReadingCase(
        text="翡翠「…あ、気付かず申し訳ございません。",
        expected_kana="ヒスイ「…ア、キズカズモーシワケゴザイマセン。",
        targets=(
            _TargetExpectation(
                surface="翡翠",
                expected_pronunciation="ヒスイ",
            ),
        ),
    ),
    _ReadingCase(
        text="その言葉を聞いて元八郎は背筋に寒いものを覚えた。",
        expected_kana="ソノコトバヲキイテモトハチローワセスジニサムイモノヲオボエタ。",
        targets=(
            _TargetExpectation(
                surface="背筋",
                expected_pronunciation="セスジ",
            ),
        ),
    ),
    _ReadingCase(
        text="気が付いた時に肘をつかないようにして、腹筋背筋をつけるようにはしていますが、こちらはなかなか…。",
        expected_kana="キガツイタトキニヒジヲツカナイヨーニシテ、フッキンハイキンヲツケルヨーニワシテイマスガ、コチラワナカナカ…。",
        targets=(
            _TargetExpectation(
                surface="背筋",
                expected_pronunciation="ハイキン",
            ),
        ),
    ),
    _ReadingCase(
        text="いろんな人の人生が脅かされたようでした。",
        expected_kana="イロンナヒトノジンセーガオビヤカサレタヨーデシタ。",
        targets=(
            _TargetExpectation(
                surface="脅かさ",
                expected_pronunciation="オビヤカサ",
            ),
        ),
    ),
    _ReadingCase(
        text="あはっ、脅かしてゴメン。",
        expected_kana="アハッ、オドカシテゴメン。",
        targets=(
            _TargetExpectation(
                surface="脅かし",
                expected_pronunciation="オドカシ",
            ),
        ),
    ),
    _ReadingCase(
        text="俺を脅かしたのは先輩だったか。",
        expected_kana="オレヲオビヤカシタノワセンパイダッタカ。",
        targets=(
            # TODO: 本来は「オドカシ」だが現状「オビヤカシ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="脅かし",
            #     expected_pronunciation="オドカシ",
            # ),
        ),
    ),
    _ReadingCase(
        text="腸疝痛（激しい腹痛）",
        expected_kana="チョーセンツー（ハゲシイフクツー）",
        targets=(
            _TargetExpectation(
                surface="腸",
                expected_pronunciation="チョー",
            ),
        ),
    ),
    _ReadingCase(
        text="なのに、このような文を寄越すとは……腸が煮えくり返りそうだ。",
        expected_kana="ナノニ、コノヨーナフミヲヨコストワ……ハラワタガニエクリカエリソーダ。",
        targets=(
            _TargetExpectation(
                surface="腸",
                expected_pronunciation="ハラワタ",
            ),
        ),
    ),
    _ReadingCase(
        text="自重４２トン、起重機の動力はディーゼル機関で出力は５０馬力。",
        expected_kana="ジジューヨンジューニトン、キジューキノドーリョクワディーゼルキカンデシュツリョクワゴジューバリキ。",
        targets=(
            _TargetExpectation(
                surface="自重",
                expected_pronunciation="ジジュー",
            ),
        ),
    ),
    _ReadingCase(
        text="もう少し自重するようにする",
        expected_kana="モースコシジチョースルヨーニスル",
        targets=(
            _TargetExpectation(
                surface="自重",
                expected_pronunciation="ジチョー",
            ),
        ),
    ),
    _ReadingCase(
        text="悔しいやら忌々しいやら、恥ずかしいやら臭いやら、烏と聞くだけで、吾輩の心には実に複雑な恥辱の念と忌まわしい戦慄の念が走るのである。",
        expected_kana="クヤシイヤライマイマシイヤラ、ハズカシイヤラクサイヤラ、カラストキクダケデ、ワガハイノココロニワジツニフクザツナチジョクノネントイマワシイセンリツノネンガハシルノデアル。",
        targets=(
            _TargetExpectation(
                surface="臭い",
                expected_pronunciation="クサイ",
            ),
        ),
    ),
    _ReadingCase(
        text="イグサの臭いがここまで届きそう。",
        expected_kana="イグサノニオイガココマデトドキソー。",
        targets=(
            _TargetExpectation(
                surface="臭い",
                expected_pronunciation="ニオイ",
            ),
        ),
    ),
    _ReadingCase(
        text="一生懸命ダンスしてくれるのが、かわいくてー…色紙の花束もらってね、もう顔ぐしゃぐしゃになるぐらい泣いちゃったよ…",
        expected_kana="イッショーケンメーダンスシテクレルノガ、カワイクテー…シキシノハナタバモラッテネ、モーカオグシャグシャニナルグライナイチャッタヨ…",
        targets=(
            # TODO: 本来は「イロガミ」だが現状「シキシ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="色紙",
            #     expected_pronunciation="イロガミ",
            # ),
        ),
    ),
    _ReadingCase(
        text="画用紙や色紙で、あじさいを作っていきます。",
        expected_kana="ガヨーシヤイロガミデ、アジサイヲツクッテイキマス。",
        targets=(
            _TargetExpectation(
                surface="色紙",
                expected_pronunciation="イロガミ",
            ),
        ),
    ),
    _ReadingCase(
        text="こちらが、その完成色紙になります。",
        expected_kana="コチラガ、ソノカンセーシキシニナリマス。",
        targets=(
            _TargetExpectation(
                surface="色紙",
                expected_pronunciation="シキシ",
            ),
        ),
    ),
    _ReadingCase(
        text="２日目にして、ららぽーと豊洲は既に色紙なし。",
        expected_kana="フツカメニシテ、ララポートトヨスワスデニイロガミナシ。",
        targets=(
            # TODO: 本来は「シキシ」だが現状「イロガミ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="色紙",
            #     expected_pronunciation="シキシ",
            # ),
        ),
    ),
    _ReadingCase(
        text="城の中庭で、艶やかな衣装に身を包んだ女性たちが舞う。",
        expected_kana="シロノナカニワデ、アデヤカナイショーニミヲツツンダジョセータチガマウ。",
        targets=(
            _TargetExpectation(
                surface="艶やか",
                expected_pronunciation="アデヤカ",
            ),
        ),
    ),
    _ReadingCase(
        text="マット仕上げの外壁には、艶やかな鏡面ガラスの玄関ドアを。",
        expected_kana="マットシアゲノガイヘキニワ、ツヤヤカナキョーメンガラスノゲンカンドアヲ。",
        targets=(
            _TargetExpectation(
                surface="艶やか",
                expected_pronunciation="ツヤヤカ",
            ),
        ),
    ),
    _ReadingCase(
        text="また、奄美空港行のバス停まで、徒歩０分。",
        expected_kana="マタ、アマミクーコーイキノバステーマデ、トホゼロフン。",
        targets=(
            _TargetExpectation(
                surface="行",
                expected_pronunciation="イキ",
            ),
        ),
    ),
    _ReadingCase(
        text="作文では、話題が変わるところで行を改めて段落を分けましょう。",
        expected_kana="サクブンデワ、ワダイガカワルトコロデギョーヲアラタメテダンラクヲワケマショー。",
        targets=(
            _TargetExpectation(
                surface="行",
                expected_pronunciation="ギョー",
            ),
        ),
    ),
    _ReadingCase(
        text="ただし自転車はあきらめ、久々に車での釣行です。",
        expected_kana="タダシジテンシャワアキラメ、ヒサビサニクルマデノツリコーデス。",
        targets=(
            _TargetExpectation(
                surface="行",
                expected_pronunciation="コー",
            ),
        ),
    ),
    _ReadingCase(
        text="仁義八行を尊び、過酷な運命を生きる者達の物語です。",
        expected_kana="ジンギハチギョーヲタットビ、カコクナウンメーヲイキルモノタチノモノガタリデス。",
        targets=(
            # TODO: 本来は「コー」だが現状「ギョー」が選ばれてしまう
            # _TargetExpectation(
            #     surface="行",
            #     expected_pronunciation="コー",
            # ),
        ),
    ),
    _ReadingCase(
        text="彼は創業者と苦楽を共にし、最後まで行を共にした数少ない仲間の一人だった。",
        expected_kana="カレワソーギョーシャトクラクヲトモニシ、サイゴマデイキヲトモニシタカズスクナイナカマノヒトリダッタ。",
        targets=(
            # TODO: 本来は「コー」だが現状「イキ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="行",
            #     expected_pronunciation="コー",
            # ),
        ),
    ),
    _ReadingCase(
        text="物語の中で、悪い魔法使いは王女に眠りの術をかけた。",
        expected_kana="モノガタリノナカデ、ワルイマホーツカイワオージョニネムリノジュツヲカケタ。",
        targets=(
            _TargetExpectation(
                surface="術",
                expected_pronunciation="ジュツ",
            ),
        ),
    ),
    _ReadingCase(
        text="濁流に流されていく家々を前に、村人たちはなす術もなく立ち尽くした。",
        expected_kana="ダクリューニナガサレテイクイエイエヲマエニ、ムラビトタチワナススベモナクタチツクシタ。",
        targets=(
            _TargetExpectation(
                surface="術",
                expected_pronunciation="スベ",
            ),
        ),
    ),
    _ReadingCase(
        text="総格は晩年の生活を表します",
        expected_kana="ソーカクワバンネンノセーカツヲアラワシマス",
        targets=(
            _TargetExpectation(
                surface="表し",
                expected_pronunciation="アラワシ",
            ),
        ),
    ),
    _ReadingCase(
        text="発情期にも平静を装うのが得意。",
        expected_kana="ハツジョーキニモヘーセーヲヨソオウノガトクイ。",
        targets=(
            _TargetExpectation(
                surface="装う",
                expected_pronunciation="ヨソオウ",
            ),
        ),
    ),
    _ReadingCase(
        text="水門は水害対策の要です。",
        expected_kana="スイモンワスイガイタイサクノカナメデス。",
        targets=(
            _TargetExpectation(
                surface="要",
                expected_pronunciation="カナメ",
            ),
        ),
    ),
    _ReadingCase(
        text="フランス人に招かれたときは要チェックですね。",
        expected_kana="フランスジンニマネカレタトキワヨーチェックデスネ。",
        targets=(
            _TargetExpectation(
                surface="要",
                expected_pronunciation="ヨー",
            ),
        ),
    ),
    _ReadingCase(
        text="のちほどわたくしが訊いてまいりますゆえ、三田村さまは釜山の町中でもご見物なされてはいかがで。",
        expected_kana="ノチホドワタクシガキイテマイリマスユエ、ミタムラサマワプサンノマチナカデモゴケンブツナサレテワイカガデ。",
        targets=(
            _TargetExpectation(
                surface="見物",
                expected_pronunciation="ケンブツ",
            ),
        ),
    ),
    _ReadingCase(
        text="あでやかなつるし飾りも見物です。",
        expected_kana="アデヤカナツルシカザリモミモノデス。",
        targets=(
            _TargetExpectation(
                surface="見物",
                expected_pronunciation="ミモノ",
            ),
        ),
    ),
    _ReadingCase(
        text="「あの角のがそうです」",
        expected_kana="「アノツノノガソーデス」",
        targets=(
            # TODO: 本来は「カド」だが現状「ツノ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="角",
            #     expected_pronunciation="カド",
            # ),
        ),
    ),
    _ReadingCase(
        text="いや、そこの角を左に曲がってくれ",
        expected_kana="イヤ、ソコノカドヲヒダリニマガッテクレ",
        targets=(
            _TargetExpectation(
                surface="角",
                expected_pronunciation="カド",
            ),
        ),
    ),
    _ReadingCase(
        text="春になって気温が上がると、山の雪が解けて川の水かさが増える。",
        expected_kana="ハルニナッテキオンガアガルト、ヤマノユキガトケテカワノミズカサガフエル。",
        targets=(
            _TargetExpectation(
                surface="解け",
                expected_pronunciation="トケ",
            ),
        ),
    ),
    _ReadingCase(
        text="歩いているうちに靴紐が解けたので、道の端にしゃがんで結び直した。",
        expected_kana="アルイテイルウチニクツヒモガホドケタノデ、ミチノハシニシャガンデムスビナオシタ。",
        targets=(
            _TargetExpectation(
                surface="解け",
                expected_pronunciation="ホドケ",
            ),
        ),
    ),
    _ReadingCase(
        text="公式さえ覚えておけば、この程度の問題は誰でも簡単に解ける。",
        expected_kana="コーシキサエオボエテオケバ、コノテードノモンダイワダレデモカンタンニトケル。",
        targets=(
            _TargetExpectation(
                surface="解ける",
                expected_pronunciation="トケル",
            ),
        ),
    ),
    _ReadingCase(
        text="彼は真面目すぎて冗談を解さないので、会話が少し堅苦しくなる。",
        expected_kana="カレワマジメスギテジョーダンヲカイサナイノデ、カイワガスコシカタクルシクナル。",
        targets=(
            _TargetExpectation(
                surface="解さ",
                expected_pronunciation="カイサ",
            ),
        ),
    ),
    _ReadingCase(
        text="長時間のデスクワークで固まった肩の凝りを、ストレッチで解す。",
        expected_kana="チョージカンノデスクワークデカタマッタカタノコリヲ、ストレッチデホグス。",
        targets=(
            _TargetExpectation(
                surface="解す",
                expected_pronunciation="ホグス",
            ),
        ),
    ),
    _ReadingCase(
        text="……あっ、じゃあもっと触れば、もっとくすぐったいってことだな！",
        expected_kana="……アッ、ジャーモットフレバ、モットクスグッタイッテコトダナ！",
        targets=(
            # TODO: 本来は「サワレ」だが現状「フレ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="触れ",
            #     expected_pronunciation="サワレ",
            # ),
        ),
    ),
    _ReadingCase(
        text="マナーの真髄に触れさせていただいたようです。",
        expected_kana="マナーノシンズイニフレサセテイタダイタヨーデス。",
        targets=(
            _TargetExpectation(
                surface="触れ",
                expected_pronunciation="フレ",
            ),
        ),
    ),
    _ReadingCase(
        text="「サレンダー」は「自発的に明け渡す、返却する」という意味があるので「自主的に運転免許を返納し」という訳にしています。",
        expected_kana="「サレンダー」ワ「ジハツテキニアケワタス、ヘンキャクスル」トイウイミガアルノデ「ジシュテキニウンテンメンキョヲヘンノーシ」トイウワケニシテイマス。",
        targets=(
            # TODO: 本来は「ヤク」だが現状「ワケ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="訳",
            #     expected_pronunciation="ヤク",
            # ),
        ),
    ),
    _ReadingCase(
        text="巻末に第二部「詩語法」の抜粋訳。",
        expected_kana="カンマツニダイニブ「シゴホー」ノバッスイヤク。",
        targets=(
            _TargetExpectation(
                surface="訳",
                expected_pronunciation="ヤク",
            ),
        ),
    ),
    _ReadingCase(
        text="訳が分かりません。",
        expected_kana="ワケガワカリマセン。",
        targets=(
            _TargetExpectation(
                surface="訳",
                expected_pronunciation="ワケ",
            ),
        ),
    ),
    _ReadingCase(
        text="それでは評定をはじめましょうか",
        expected_kana="ソレデワヒョージョーヲハジメマショーカ",
        targets=(
            _TargetExpectation(
                surface="評定",
                expected_pronunciation="ヒョージョー",
            ),
        ),
    ),
    _ReadingCase(
        text="そうなんですよね、どれだけ楽して評定３を勝ち取るかの戦いですよね",
        expected_kana="ソーナンデスヨネ、ドレダケタノシテヒョージョーサンヲカチトルカノタタカイデスヨネ",
        targets=(
            # TODO: 本来は「ヒョーテー」だが現状「ヒョージョー」が選ばれてしまう
            # _TargetExpectation(
            #     surface="評定",
            #     expected_pronunciation="ヒョーテー",
            # ),
        ),
    ),
    _ReadingCase(
        text="パソコンによる作成課題の提出や達成状況を中心にして、出席点なども合わせて評定を出す。",
        expected_kana="パソコンニヨルサクセーカダイノテーシュツヤタッセージョーキョーヲチューシンニシテ、シュッセキテンナドモアワセテヒョーテーヲダス。",
        targets=(
            _TargetExpectation(
                surface="評定",
                expected_pronunciation="ヒョーテー",
            ),
        ),
    ),
    _ReadingCase(
        text="何をせっぱ詰った表情をしておる？",
        expected_kana="ナニヲセッパナジッタヒョージョーヲシテオル？",
        targets=(
            # TODO: 本来は「ツマッ」だが現状「ナジッ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="詰っ",
            #     expected_pronunciation="ツマッ",
            # ),
        ),
    ),
    _ReadingCase(
        text="会議の席で、部長は部下の失敗を皆の前で厳しく詰った。",
        expected_kana="カイギノセキデ、ブチョーワブカノシッパイヲミナノマエデキビシクナジッタ。",
        targets=(
            _TargetExpectation(
                surface="詰っ",
                expected_pronunciation="ナジッ",
            ),
        ),
    ),
    _ReadingCase(
        text="チュートリアルが詰らない。",
        expected_kana="チュートリアルガツマラナイ。",
        targets=(
            _TargetExpectation(
                surface="詰ら",
                expected_pronunciation="ツマラ",
            ),
        ),
    ),
    _ReadingCase(
        text="星宿庁に戻ったウォルは、フォンへの謝罪の手紙を認めます。",
        expected_kana="セーシュクチョーニモドッタウォルワ、フォンエノシャザイノテガミヲシタタメマス。",
        targets=(
            _TargetExpectation(
                surface="認め",
                expected_pronunciation="シタタメ",
            ),
        ),
    ),
    _ReadingCase(
        text="原則は認められません。",
        expected_kana="ゲンソクワミトメラレマセン。",
        targets=(
            _TargetExpectation(
                surface="認め",
                expected_pronunciation="ミトメ",
            ),
        ),
    ),
    _ReadingCase(
        text="文を認めることができるのならば、私の従者に相応しい。",
        expected_kana="フミヲミトメルコトガデキルノナラバ、ワタシノジューシャニフサワシイ。",
        targets=(
            # TODO: 本来は「シタタメル」だが現状「ミトメル」が選ばれてしまう
            # _TargetExpectation(
            #     surface="認める",
            #     expected_pronunciation="シタタメル",
            # ),
        ),
    ),
    _ReadingCase(
        text="総一、そなた謀ったなっ？",
        expected_kana="ソーイチ、ソナタタバカッタナッ？",
        targets=(
            _TargetExpectation(
                surface="謀っ",
                expected_pronunciation="タバカッ",
            ),
        ),
    ),
    _ReadingCase(
        text="劉敞は、別に私党を結成しようと謀り、高陵侯翟宣の娘を劉祉に娶わせた。",
        expected_kana="リュータカシワ、ベツニシトーヲケッセーシヨートハカリ、コーリョーコーテキセンノムスメヲリュー祉ニメアワセタ。",
        targets=(
            _TargetExpectation(
                surface="謀り",
                expected_pronunciation="ハカリ",
            ),
        ),
    ),
    _ReadingCase(
        text="そのため、年賀状を郵便局に持参したり賞品を運ぶ時はリヤカーを使い、電化製品を質に入れたりしている。",
        expected_kana="ソノタメ、ネンガジョーヲユービンキョクニジサンシタリショーヒンヲハコブトキワリヤカーヲツカイ、デンカセーヒンヲシチニイレタリシテイル。",
        targets=(
            _TargetExpectation(
                surface="質",
                expected_pronunciation="シチ",
            ),
        ),
    ),
    _ReadingCase(
        text="べ、別に質が高いとかっていう、後輩さんみたいなくろうと目線の話じゃないんですよ。",
        expected_kana="ベ、ベツニシツガタカイトカッテイウ、コーハイサンミタイナクロートメセンノハナシジャナインデスヨ。",
        targets=(
            _TargetExpectation(
                surface="質",
                expected_pronunciation="シツ",
            ),
        ),
    ),
    _ReadingCase(
        text="わたしは完全に寒い方が我慢できる質なので、夏は参っちゃいますね。",
        expected_kana="ワタシワカンゼンニサムイホーガガマンデキルシツナノデ、ナツワマイッチャイマスネ。",
        targets=(
            # TODO: 本来は「タチ」だが現状「シツ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="質",
            #     expected_pronunciation="タチ",
            # ),
        ),
    ),
    _ReadingCase(
        text="勉強も、優しく……手取足取り教えてくれて……",
        expected_kana="ベンキョーモ、ヤサシク……テドリアシトリオシエテクレテ……",
        targets=(
            _TargetExpectation(
                surface="足取り",
                expected_pronunciation="アシトリ",
            ),
        ),
    ),
    _ReadingCase(
        text="そのまま広い窓の方にゆっくりとした足取りで歩いていき、外の景色に目を向ける。",
        expected_kana="ソノママヒロイマドノホーニユックリトシタアシドリデアルイテイキ、ソトノケシキニメヲムケル。",
        targets=(
            _TargetExpectation(
                surface="足取り",
                expected_pronunciation="アシドリ",
            ),
        ),
    ),
    _ReadingCase(
        text="思い足取りで立ち寄ってみると…ありました。",
        expected_kana="オモイアシトリデタチヨッテミルト…アリマシタ。",
        targets=(
            # TODO: 本来は「アシドリ」だが現状「アシトリ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="足取り",
            #     expected_pronunciation="アシドリ",
            # ),
        ),
    ),
    _ReadingCase(
        text="本当のカラスの足跡です。",
        expected_kana="ホントーノカラスノアシアトデス。",
        targets=(
            _TargetExpectation(
                surface="足跡",
                expected_pronunciation="アシアト",
            ),
        ),
    ),
    _ReadingCase(
        text="一体どのような経営哲学で、本業を深堀しながら一度たりとも売り上げを落とすことなく企業を進化させてきたのか、その足跡を辿ってみたい。",
        expected_kana="イッタイドノヨーナケーエーテツガクデ、ホンギョーヲフカホリシナガライチドタリトモウリアゲヲオトスコトナクキギョーヲシンカサセテキタノカ、ソノソクセキヲタドッテミタイ。",
        targets=(
            _TargetExpectation(
                surface="足跡",
                expected_pronunciation="ソクセキ",
            ),
        ),
    ),
    _ReadingCase(
        text="父は博打にのめり込んで、先祖代々の身上を潰してしまった。",
        expected_kana="チチワバクチニノメリコンデ、センゾダイダイノシンショーヲツブシテシマッタ。",
        targets=(
            _TargetExpectation(
                surface="身上",
                expected_pronunciation="シンショー",
            ),
        ),
    ),
    _ReadingCase(
        text="彼は口数こそ少ないが、何事にも正直なところが身上だ。",
        expected_kana="カレワクチカズコソスクナイガ、ナニゴトニモショージキナトコロガシンジョーダ。",
        targets=(
            _TargetExpectation(
                surface="身上",
                expected_pronunciation="シンジョー",
            ),
        ),
    ),
    _ReadingCase(
        text="うまいともまずいとも言えず、甘いとも辛いとも苦いともなんとも言えない、衝撃的な味なのだ。",
        expected_kana="ウマイトモマズイトモイエズ、アマイトモカライトモニガイトモナントモイエナイ、ショーゲキテキナアジナノダ。",
        targets=(
            _TargetExpectation(
                surface="辛い",
                expected_pronunciation="カライ",
            ),
        ),
    ),
    _ReadingCase(
        text="ただお追従した結果がこれだよ。",
        expected_kana="タダオツイショーシタケッカガコレダヨ。",
        targets=(
            _TargetExpectation(
                surface="追従",
                expected_pronunciation="ツイショー",
            ),
        ),
    ),
    _ReadingCase(
        text="体の回転に合わせて髪の毛が追従します。",
        expected_kana="カラダノカイテンニアワセテカミノケガツイジューシマス。",
        targets=(
            _TargetExpectation(
                surface="追従",
                expected_pronunciation="ツイジュー",
            ),
        ),
    ),
    _ReadingCase(
        text="長年チームを率いた監督は、今季限りで第一線を退いた。",
        expected_kana="ナガネンチームヲヒキイタカントクワ、コンキカギリデダイイッセンヲシリゾイタ。",
        targets=(
            _TargetExpectation(
                surface="退い",
                expected_pronunciation="シリゾイ",
            ),
        ),
    ),
    _ReadingCase(
        text="通れないから、その荷物を持ってそこを退いてくれないか。",
        expected_kana="トーレナイカラ、ソノニモツヲモッテソコヲドイテクレナイカ。",
        targets=(
            _TargetExpectation(
                surface="退い",
                expected_pronunciation="ドイ",
            ),
        ),
    ),
    _ReadingCase(
        text="それどころか、言外の言を逆手にとっていろんな奇論が横行する危険があるわけでございます。",
        expected_kana="ソレドコロカ、ゲンガイノゲンヲギャクテニトッテイロンナキロンガオーコースルキケンガアルワケデゴザイマス。",
        targets=(
            _TargetExpectation(
                surface="逆手",
                expected_pronunciation="ギャクテ",
            ),
        ),
    ),
    _ReadingCase(
        text="私にかかれば造作もありませんわね",
        expected_kana="ワタシニカカレバゾーサモアリマセンワネ",
        targets=(
            _TargetExpectation(
                surface="造作",
                expected_pronunciation="ゾーサ",
            ),
        ),
    ),
    _ReadingCase(
        text="施主様がデザインした造作洗面台。",
        expected_kana="セシュサマガデザインシタゾーサクセンメンダイ。",
        targets=(
            _TargetExpectation(
                surface="造作",
                expected_pronunciation="ゾーサク",
            ),
        ),
    ),
    _ReadingCase(
        text="話が少し逸れてしまいました。",
        expected_kana="ハナシガスコシソレテシマイマシタ。",
        targets=(
            _TargetExpectation(
                surface="逸れ",
                expected_pronunciation="ソレ",
            ),
        ),
    ),
    _ReadingCase(
        text="のどかと逸れてしまったが、城へ向かう列車で合流を果たす。",
        expected_kana="ノドカトハグレテシマッタガ、シロエムカウレッシャデゴーリューヲハタス。",
        targets=(
            _TargetExpectation(
                surface="逸れ",
                expected_pronunciation="ハグレ",
            ),
        ),
    ),
    _ReadingCase(
        text="朋也は渚を連れて学園祭を回るが途中、逸れてしまう。",
        expected_kana="トモヤワナギサヲツレテガクエンサイヲマワルガトチュー、ソレテシマウ。",
        targets=(
            # TODO: 本来は「ハグレ」だが現状「ソレ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="逸れ",
            #     expected_pronunciation="ハグレ",
            # ),
        ),
    ),
    _ReadingCase(
        text="ただいま申し上げ違えました。",
        expected_kana="タダイマモーシアゲチガエマシタ。",
        targets=(
            _TargetExpectation(
                surface="違え",
                expected_pronunciation="チガエ",
            ),
        ),
    ),
    _ReadingCase(
        text="百鬼丸も自分の体を取り戻す目的とどろろとは違えど、力強く生きていく。",
        expected_kana="ヒャクオニマルモジブンノカラダヲトリモドスモクテキトドロロトワタガエド、チカラズヨクイキテイク。",
        targets=(
            # TODO: 本来は「チガエ」だが現状「タガエ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="違え",
            #     expected_pronunciation="チガエ",
            # ),
        ),
    ),
    _ReadingCase(
        text="約束、いずれも違えるでないぞ",
        expected_kana="ヤクソク、イズレモタガエルデナイゾ",
        targets=(
            _TargetExpectation(
                surface="違える",
                expected_pronunciation="タガエル",
            ),
        ),
    ),
    _ReadingCase(
        text="病原因子との接触を避けるための知識不足",
        expected_kana="ビョーゲンインシトノセッショクヲサケルタメノチシキブソク",
        targets=(
            _TargetExpectation(
                surface="避ける",
                expected_pronunciation="サケル",
            ),
        ),
    ),
    _ReadingCase(
        text="都の職員として、彼は二十年にわたって水道事業に携わってきた。",
        expected_kana="トノショクイントシテ、カレワニジューネンニワタッテスイドージギョーニタズサワッテキタ。",
        targets=(
            _TargetExpectation(
                surface="都",
                expected_pronunciation="ト",
            ),
        ),
    ),
    _ReadingCase(
        text="かつて花の都と呼ばれたパリには、今も世界中から観光客が訪れる。",
        expected_kana="カツテハナノミヤコトヨバレタパリニワ、イマモセカイジューカラカンコーキャクガオトズレル。",
        targets=(
            _TargetExpectation(
                surface="都",
                expected_pronunciation="ミヤコ",
            ),
        ),
    ),
    _ReadingCase(
        text="その寺に伝わる重宝の仏像が、百年ぶりに一般公開された。",
        expected_kana="ソノテラニツタワルチョーホーノブツゾーガ、ヒャクネンブリニイッパンコーカイサレタ。",
        targets=(
            # TODO: 本来は「ジューホー」だが現状「チョーホー」が選ばれてしまう
            # _TargetExpectation(
            #     surface="重宝",
            #     expected_pronunciation="ジューホー",
            # ),
        ),
    ),
    _ReadingCase(
        text="県重宝として「円覚寺真言・修験聖教類及び文書」及び「寺下遺跡出土骨角器類」を指定するものです。",
        expected_kana="ケンチョーホートシテ「エンカクジシンゴン・シュゲンセーキョールイオヨビブンショ」オヨビ「テラシタイセキシュツドコッカクキルイ」ヲシテースルモノデス。",
        targets=(
            # TODO: 本来は「ジューホー」だが現状「チョーホー」が選ばれてしまう
            # _TargetExpectation(
            #     surface="重宝",
            #     expected_pronunciation="ジューホー",
            # ),
        ),
    ),
    _ReadingCase(
        text="この折りたたみ式のナイフは軽くて丈夫なので、キャンプでとても重宝する。",
        expected_kana="コノオリタタミシキノナイフワカルクテジョーブナノデ、キャンプデトテモチョーホースル。",
        targets=(
            _TargetExpectation(
                surface="重宝",
                expected_pronunciation="チョーホー",
            ),
        ),
    ),
    _ReadingCase(
        text="これを月と金星に置き換えてみましょうか。",
        expected_kana="コレヲツキトキンセーニオキカエテミマショーカ。",
        targets=(
            _TargetExpectation(
                surface="金星",
                expected_pronunciation="キンセー",
            ),
        ),
    ),
    _ReadingCase(
        text="１９８５年７月場所では２日目に横綱、千代の富士をうっちゃりで破り自身初となる金星を挙げた。",
        expected_kana="センキューヒャクハチジューゴネンシチガツバショデワフツカメニヨコズナ、チヨノフジヲウッチャリデヤブリジシンハツトナルキンボシヲアゲタ。",
        targets=(
            _TargetExpectation(
                surface="金星",
                expected_pronunciation="キンボシ",
            ),
        ),
    ),
    _ReadingCase(
        text="今日の島田市はは鈍よりした曇り空です。",
        expected_kana="キョーノシマダシワワドンヨリシタクモリゾラデス。",
        targets=(
            _TargetExpectation(
                surface="鈍",
                expected_pronunciation="ドン",
            ),
        ),
    ),
    _ReadingCase(
        text="親の気持ちに気づいてない鈍い人だとは思いますが。",
        expected_kana="オヤノキモチニキズイテナイニブイヒトダトワオモイマスガ。",
        targets=(
            _TargetExpectation(
                surface="鈍い",
                expected_pronunciation="ニブイ",
            ),
        ),
    ),
    _ReadingCase(
        text="どうも最近身体が鈍ってな。",
        expected_kana="ドーモサイキンシンタイガナマッテナ。",
        targets=(
            _TargetExpectation(
                surface="鈍っ",
                expected_pronunciation="ナマッ",
            ),
        ),
    ),
    _ReadingCase(
        text="銀杏の黄色もきれいですよねー。",
        expected_kana="イチョーノキイロモキレイデスヨネー。",
        targets=(
            _TargetExpectation(
                surface="銀杏",
                expected_pronunciation="イチョー",
            ),
        ),
    ),
    _ReadingCase(
        text="ここで、友人が銀杏拾いの穴場を知っている？",
        expected_kana="ココデ、ユージンガギンナンヒロイノアナバヲシッテイル？",
        targets=(
            _TargetExpectation(
                surface="銀杏",
                expected_pronunciation="ギンナン",
            ),
        ),
    ),
    _ReadingCase(
        text="藤九郎という名前の銀杏粒が大きく、匂いも臭くない。",
        expected_kana="トークロートイウナマエノイチョーツブガオーキク、ニオイモクサクナイ。",
        targets=(
            # TODO: 本来は「ギンナン」だが現状「イチョー」が選ばれてしまう
            # _TargetExpectation(
            #     surface="銀杏",
            #     expected_pronunciation="ギンナン",
            # ),
        ),
    ),
    _ReadingCase(
        text="しかし、瞼の闇は、目を開けたらすぐに終わる。",
        expected_kana="シカシ、マブタノヤミワ、メヲアケタラスグニオワル。",
        targets=(
            _TargetExpectation(
                surface="開け",
                expected_pronunciation="アケ",
            ),
        ),
    ),
    _ReadingCase(
        text="大学卒業後、ヨーロッパで本場のワインに開眼。",
        expected_kana="ダイガクソツギョーゴ、ヨーロッパデホンバノワインニカイガン。",
        targets=(
            _TargetExpectation(
                surface="開眼",
                expected_pronunciation="カイガン",
            ),
        ),
    ),
    _ReadingCase(
        text="また幅四点五メートル、高さ三点六メートルのキャンバスで期間中３ヶ月間をかけ制作された曼荼羅画は完成後開眼式を行い閉幕後福岡市美術館に寄贈された。",
        expected_kana="マタハバヨンテンゴメートル、タカササンテンロクメートルノキャンバスデキカンチューサンカゲツカンヲカケセーサクサレタマンダラガワカンセーゴカイガンシキヲオコナイヘーマクゴフクオカシビジュツカンニキゾーサレタ。",
        targets=(
            # TODO: 本来は「カイゲン」だが現状「カイガン」が選ばれてしまう
            # _TargetExpectation(
            #     surface="開眼",
            #     expected_pronunciation="カイゲン",
            # ),
        ),
    ),
    _ReadingCase(
        text="開眼会で献納された宝物は当代一の名品だ。",
        expected_kana="カイゲンカイデケンノーサレタタカラモノワトーダイイチノメーヒンダ。",
        targets=(
            _TargetExpectation(
                surface="開眼",
                expected_pronunciation="カイゲン",
            ),
        ),
    ),
    _ReadingCase(
        text="駆真は眉の間に少しだけしわを作って答えた。",
        expected_kana="カシンワマユノアイダニスコシダケシワヲツクッテコタエタ。",
        targets=(
            _TargetExpectation(
                surface="間",
                expected_pronunciation="アイダ",
            ),
        ),
    ),
    _ReadingCase(
        text="私は４時間半くらいかかりますよ。",
        expected_kana="ワタシワヨジカンハンクライカカリマスヨ。",
        targets=(
            _TargetExpectation(
                surface="間",
                expected_pronunciation="カン",
                expected_outcome="dictionary_default_protected",
                was_preserved=True,
            ),
        ),
    ),
    _ReadingCase(
        text="腹黒き輩はまさしく吾輩と指呼の間にいるのである。",
        expected_kana="ハラグロキトモガラワマサシクワガハイトシコノアイダニイルノデアル。",
        targets=(
            # TODO: 本来は「カン」だが現状「アイダ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="間",
            #     expected_pronunciation="カン",
            # ),
        ),
    ),
    _ReadingCase(
        text="六畳一間ながら一戸建てのあの借家は最初から今度こそは期待のもてそうな話だったが、その時も三村だけが紹介者の知人に連れられて見に行ったのだった。",
        expected_kana="ロクジョーイッケンナガライッコダテノアノシャクヤワサイショカラコンドコソワキタイノモテソーナハナシダッタガ、ソノトキモミツムラダケガショーカイシャノチジンニツレラレテミニイッタノダッタ。",
        targets=(
            # TODO: 本来は「マ」だが現状「ケン」が選ばれてしまう
            # _TargetExpectation(
            #     surface="間",
            #     expected_pronunciation="マ",
            # ),
        ),
    ),
    _ReadingCase(
        text="田中は「三振はいつも狙っています」と断言し、３球三振が多いことには「（ボールを投げて）遊ぶと打者に考える間を与えてしまうので」ときっぱり。",
        expected_kana="タナカワ「サンシンワイツモネラッテイマス」トダンゲンシ、サンキューサンシンガオーイコトニワ「（ボールヲナゲテ）アソブトダシャニカンガエルマヲアタエテシマウノデ」トキッパリ。",
        targets=(
            _TargetExpectation(
                surface="間",
                expected_pronunciation="マ",
            ),
        ),
    ),
    _ReadingCase(
        text="つい先程その先で車から降りた」",
        expected_kana="ツイサキホドソノサキデクルマカラオリタ」",
        targets=(
            _TargetExpectation(
                surface="降り",
                # 「た」へ接続できる読みが1候補だけになるため、モデル推論を省略する
                expected_outcome="lattice_reachable_lt2",
            ),
        ),
    ),
    _ReadingCase(
        text="今日は雨が降りそうです。",
        expected_kana="キョーワアメガフリソーデス。",
        targets=(
            _TargetExpectation(
                surface="降り",
                expected_pronunciation="フリ",
            ),
        ),
    ),
    _ReadingCase(
        text="りっくん（小柴陸）かわいい。",
        expected_kana="リックン（コシバリク）カワイイ。",
        targets=(
            _TargetExpectation(
                surface="陸",
                expected_pronunciation="リク",
            ),
        ),
    ),
    _ReadingCase(
        text="愛語を聞くは 面を喜ばしめ、心を楽しくす",
        expected_kana="アイゴヲキクワメンヲヨロコバシメ、ココロヲタノシクス",
        targets=(
            # TODO: 本来は「オモテ」だが現状「メン」が選ばれてしまう
            # _TargetExpectation(
            #     surface="面",
            #     expected_pronunciation="オモテ",
            # ),
        ),
    ),
    _ReadingCase(
        text="きれいな顔してるくせに、面の皮が厚いってのは、あんたみたいな奴のことを言うんだな」",
        expected_kana="キレイナカオシテルクセニ、ツラノカワガアツイッテノワ、アンタミタイナヤツノコトヲイウンダナ」",
        targets=(
            _TargetExpectation(
                surface="面",
                expected_pronunciation="ツラ",
            ),
        ),
    ),
    _ReadingCase(
        text="アストロッドは両手で面を覆う。",
        expected_kana="アストロッドワリョーテデメンヲオオウ。",
        targets=(
            _TargetExpectation(
                surface="面",
                expected_pronunciation="メン",
            ),
        ),
    ),
    _ReadingCase(
        text="「呼ばれて頭の中真っ白」",
        expected_kana="「ヨバレテアタマノナカマッシロ」",
        targets=(
            _TargetExpectation(
                surface="頭",
                expected_pronunciation="アタマ",
            ),
        ),
    ),
    _ReadingCase(
        text="数江が兵たちの頭らしい男の言葉を訳した。",
        expected_kana="カズコーガヘータチノアタマラシイオトコノコトバヲヤクシタ。",
        targets=(
            # TODO: 本来は「カシラ」だが現状「アタマ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="頭",
            #     expected_pronunciation="カシラ",
            # ),
        ),
    ),
    _ReadingCase(
        text="布良も監視役の頭数に入っている",
        expected_kana="メラモカンシヤクノアタマカズニハイッテイル",
        targets=(
            _TargetExpectation(
                surface="頭数",
                expected_pronunciation="アタマカズ",
            ),
        ),
    ),
    _ReadingCase(
        text="北海道和種以外は非常に飼育頭数が少ない。",
        expected_kana="ホッカイドーワシュイガイワヒジョーニシイクトースーガスクナイ。",
        targets=(
            _TargetExpectation(
                surface="頭数",
                expected_pronunciation="トースー",
            ),
        ),
    ),
    _ReadingCase(
        text="今月の請求書を見ると、思っていたよりも額が大きくて驚いた。",
        expected_kana="コンゲツノセーキューショヲミルト、オモッテイタヨリモガクガオーキクテオドロイタ。",
        targets=(
            _TargetExpectation(
                surface="額",
                expected_pronunciation="ガク",
            ),
        ),
    ),
    _ReadingCase(
        text="表彰式でもらった賞状を額に入れて、居間の壁に飾った。",
        expected_kana="ヒョーショーシキデモラッタショージョーヲガクニイレテ、イマノカベニカザッタ。",
        targets=(
            _TargetExpectation(
                surface="額",
                expected_pronunciation="ガク",
            ),
        ),
    ),
    _ReadingCase(
        text="炎天下での作業で、額から汗が流れ落ちた。",
        expected_kana="エンテンカデノサギョーデ、ヒタイカラアセガナガレオチタ。",
        targets=(
            _TargetExpectation(
                surface="額",
                expected_pronunciation="ヒタイ",
            ),
        ),
    ),
    _ReadingCase(
        text="風と波をさえぎる大きな島をもち、湾は深い。",
        expected_kana="カゼトナミヲサエギルオーキナシマヲモチ、ワンワフカイ。",
        targets=(
            _TargetExpectation(
                surface="風",
                expected_pronunciation="カゼ",
            ),
        ),
    ),
    _ReadingCase(
        text="「本物風」の名称",
        expected_kana="「ホンモノフー」ノメーショー",
        targets=(
            _TargetExpectation(
                surface="風",
                expected_pronunciation="フー",
            ),
        ),
    ),
    _ReadingCase(
        text="件の二羽は、吾輩がその自慢の肉球で、真下に忍び寄ってもまだ気付かぬ風で、のんびりと嘴を研いでいる。",
        expected_kana="ケンノニワワ、ワガハイガソノジマンノニクキューデ、マシタニシノビヨッテモマダキズカヌカゼデ、ノンビリトクチバシヲトイデイル。",
        targets=(
            # TODO: 本来は「フー」だが現状「カゼ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="風",
            #     expected_pronunciation="フー",
            # ),
        ),
    ),
    _ReadingCase(
        text="おめーなぁ……脳天に風穴あけんぞ！",
        expected_kana="オメーナー……ノーテンニカザアナアケンゾ！",
        targets=(
            _TargetExpectation(
                surface="風穴",
                expected_pronunciation="カザアナ",
            ),
        ),
    ),
    _ReadingCase(
        text="の風車が凍っていて面白！",
        expected_kana="ノフーシャガコーッテイテオモシロ！",
        targets=(
            _TargetExpectation(
                surface="風車",
                expected_pronunciation="フーシャ",
            ),
        ),
    ),
    _ReadingCase(
        text="奇抜なめがねやら、黒子のようなマスクなどは実は顔を隠すためのもの。",
        expected_kana="キバツナメガネヤラ、ホクロノヨーナマスクナドワジツワカオヲカクスタメノモノ。",
        targets=(
            # TODO: 本来は「クロコ」だが現状「ホクロ」が選ばれてしまう
            # _TargetExpectation(
            #     surface="黒子",
            #     expected_pronunciation="クロコ",
            # ),
        ),
    ),
    _ReadingCase(
        text="黒子のバスケが最近やばい。",
        expected_kana="クロコノバスケガサイキンヤバイ。",
        targets=(
            _TargetExpectation(
                surface="黒子",
                expected_pronunciation="クロコ",
            ),
        ),
    ),
    _ReadingCase(
        text="左目付近に泣き黒子がある。",
        expected_kana="ヒダリメフキンニナキホクロガアル。",
        targets=(
            _TargetExpectation(
                surface="黒子",
                expected_pronunciation="ホクロ",
            ),
        ),
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


@pytest.mark.parametrize(
    ("text", "expected_kana"),
    (
        ("何色が好きですか。", "ナニイロガスキデスカ。"),
        # TODO: 本来は「ナンショク」だが現状「ナニイロ」が選ばれてしまう
        # ("何色ありますか。", "ナンショクアリマスカ。"),
        ("何色ありますか。", "ナニイロアリマスカ。"),
        ("何人いますか。", "ナンニンイマスカ。"),
        ("何人ですか。", "ナンニンデスカ。"),
    ),
)
def test_ambiguous_nan_counter_expressions_remain_available_to_tsqyomi(
    tsqyomi_default_model: None,
    text: str,
    expected_kana: str,
) -> None:
    """「何 + 助数詞」の読みを文脈に応じて選択する。"""

    with_tsqyomi, _diagnostics = _run_with_diagnostics(text)

    assert with_tsqyomi == expected_kana


@pytest.mark.parametrize(
    ("text", "expected_kana"),
    (
        ("一月前です。", "ヒトツキマエデス。"),
        ("一月程度かかります。", "ヒトツキテードカカリマス。"),
        ("一月号を読みます。", "イチガツゴーヲヨミマス。"),
        ("来年の一月分です。", "ライネンノイチガツブンデス。"),
    ),
)
def test_ambiguous_ichigatsu_expressions_use_contextual_reading(
    tsqyomi_default_model: None,
    text: str,
    expected_kana: str,
) -> None:
    """「一月」の暦月と期間の読みを文脈に応じて選択する。"""

    with_tsqyomi, _diagnostics = _run_with_diagnostics(text)

    assert with_tsqyomi == expected_kana


_NAN_CLOCK_SENTENCE_CASES: tuple[_EmbeddedSentenceCase, ...] = (
    _EmbeddedSentenceCase("何時何分です。", "何時何分"),
    _EmbeddedSentenceCase("何時間何分かかります。", "何時間"),
    _EmbeddedSentenceCase("何時後に届きます。", "何時"),
    _EmbeddedSentenceCase("何時まで営業しますか。", "何時"),
    _EmbeddedSentenceCase("あと十時間後です。", "十時間"),
)
_JANUARY_DURATION_SENTENCE_CASES: tuple[_EmbeddedSentenceCase, ...] = (
    _EmbeddedSentenceCase("一月前かかります。", "一月"),
    _EmbeddedSentenceCase("あと一月前です。", "一月"),
    _EmbeddedSentenceCase("一月程度かかります。", "一月"),
    _EmbeddedSentenceCase("一月前に申し込みました。", "一月"),
)


def _build_duration_embedded_sentence_cases() -> tuple[_EmbeddedSentenceCase, ...]:
    """
    時間量辞書表層を代表文型へ機械的に埋め込んだ症例列を返す。

    Returns:
        tuple[_EmbeddedSentenceCase, ...]: 時間量表層と代表文型を組み合わせた症例列
    """

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
    """
    MeCab 既定読みと tsqyomi 有効時のカタカナ出力が一致することを検証する。

    Args:
        text (str): 比較する日本語文

    Returns:
        str: tsqyomi 有効時のカタカナ出力
    """

    baseline = pyopenjtalk.g2p(text, kana=True, use_tsqyomi=False, use_vanilla=True)
    with_tsqyomi = pyopenjtalk.g2p(text, kana=True, use_tsqyomi=True, use_vanilla=True)
    assert isinstance(baseline, str)
    assert isinstance(with_tsqyomi, str)
    assert baseline == with_tsqyomi
    return with_tsqyomi


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


@pytest.mark.parametrize("case", _READING_CASES, ids=lambda case: case.text)
def test_reading_regression(case: _ReadingCase, tsqyomi_default_model: None) -> None:
    """既定モデルの読み選択とカタカナ出力が固定した期待値と一致する。"""

    kana, diagnostics = _run_with_diagnostics(case.text)

    assert kana == case.expected_kana

    # 辞書所有の語は tsqyomi 無効時の最良形態素で読みを確定し、対象範囲へモデルが介入しないことを確認する
    if case.dictionary_targets:
        _baseline_features, baseline_morphs = pyopenjtalk.run_frontend_detailed(
            case.text,
            use_tsqyomi=False,
            use_vanilla=True,
        )
        for expectation in case.dictionary_targets:
            char_span = _resolve_char_span(case.text, expectation)
            matched_morphs = [
                morph
                for morph in baseline_morphs
                if morph["char_span"] == char_span and morph["surface"] == expectation.surface
            ]
            assert len(matched_morphs) == 1
            assert matched_morphs[0]["features"][9] == expectation.expected_pronunciation
            assert all(
                diagnostic.char_span[1] <= char_span[0] or diagnostic.char_span[0] >= char_span[1]
                for diagnostic in diagnostics
            )

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


def test_load_model_is_idempotent(tsqyomi_default_model: None) -> None:
    """ロード済みのモデルを繰り返し取得しない。"""

    loaded_model = tsqyomi.get_loaded_model()
    tsqyomi.load_model(["CPUExecutionProvider"])
    assert tsqyomi.get_loaded_model() is loaded_model


def test_unload_model_can_reload(tsqyomi_default_model: None) -> None:
    """unload_model() 後に再ロードできる。"""

    assert tsqyomi.is_model_loaded() is True
    tsqyomi.unload_model()
    assert tsqyomi.is_model_loaded() is False
    _load_tsqyomi_default_model()
    assert tsqyomi.is_model_loaded() is True


def test_long_text_passes_only_target_sentence_to_model(tsqyomi_default_model: None) -> None:
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


def test_adjacent_targets_use_candidate_connection_cost(tsqyomi_default_model: None) -> None:
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
    tsqyomi_default_model: None,
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


def test_include_morphs_false_skips_morph_rebuild(tsqyomi_default_model: None) -> None:
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


def test_onnx_contract_matches_loaded_model(tsqyomi_default_model: None) -> None:
    """ロード済みの既定モデルのセッションがメタデータ契約を満たす。"""

    model = tsqyomi.get_loaded_model()
    tsqyomi.TsqyomiModel.validate_onnx_contract(model.session, model.metadata)


def test_model_revision_is_pinned() -> None:
    """テストが参照するモデル revision が実装側の固定値と一致する。"""

    assert tsqyomi_model._MODEL_REVISION == "81add61ddba9669d328e307c883d05d77d60f5f4"
    assert tsqyomi_model._MODEL_FILES["model"] == "v5/model.onnx"


def test_g2p_mapping_aligns_tsqyomi_reading_to_morph_char_span(tsqyomi_default_model: None) -> None:
    """g2p_mapping() の char_span と phoneme 列が、tsqyomi の選択結果と一致する。"""

    text = "深夜の路地は人気が無くて怖い。"
    mapping = pyopenjtalk.g2p_mapping(text, use_tsqyomi=True, use_vanilla=True)
    ninki = next(entry for entry in mapping if entry["surface"] == "人気")
    assert ninki["char_span"] == (6, 8)
    assert ninki["phonemes"] == ["h", "I", "t", "o", "k", "e"]


def test_run_frontend_detailed_reflects_tsqyomi_pronunciation(tsqyomi_default_model: None) -> None:
    """run_frontend_detailed() の NJD feature が tsqyomi による選択発音を反映する。"""

    text = "大分県にもう大分長いこと住んでいるな。"
    _features, morphs = pyopenjtalk.run_frontend_detailed(
        text,
        use_tsqyomi=True,
        use_vanilla=True,
    )
    oita_first = next(morph for morph in morphs if morph["char_span"] == (0, 2))
    oita_second = next(morph for morph in morphs if morph["char_span"] == (6, 8))
    assert oita_first["features"][9] == "オーイタ"
    assert oita_second["features"][9] == "ダイブ"


def test_extract_fullcontext_succeeds_with_tsqyomi(tsqyomi_default_model: None) -> None:
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
    tsqyomi_default_model: None,
    text: str,
    expected_surfaces: tuple[str, ...],
) -> None:
    """辞書形態素境界と target span が一致しない複合表層は差し替えを行わない。"""

    _kana, diagnostics = _run_with_diagnostics(text)
    assert tuple(diagnostic.surface for diagnostic in diagnostics) == expected_surfaces
    assert all(diagnostic.outcome == "no_exact_morph_range" for diagnostic in diagnostics)
    assert all(diagnostic.selected_pronunciation is None for diagnostic in diagnostics)


def test_enabled_tsqyomi_changes_g2p_output_from_baseline(tsqyomi_default_model: None) -> None:
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
        ("二十四時間営業です。", "ニジューヨジカンエーギョーデス。"),
        ("二時間かかります。", "ニジカンカカリマス。"),
        ("三時間かかります。", "サンジカンカカリマス。"),
        ("四時間かかります。", "ヨジカンカカリマス。"),
        ("五時間かかります。", "ゴジカンカカリマス。"),
        ("六時間かかります。", "ロクジカンカカリマス。"),
        ("七時間かかります。", "ナナジカンカカリマス。"),
        ("八時間かかります。", "ハチジカンカカリマス。"),
        ("九時間かかります。", "クジカンカカリマス。"),
        ("十時間かかります。", "ジュージカンカカリマス。"),
        ("あと二時間です。", "アトニジカンデス。"),
        ("あと十時間後です。", "アトジュージカンゴデス。"),
        ("四分後に届きます。", "ヨンプンゴニトドキマス。"),
        ("四分かかります。", "ヨンプンカカリマス。"),
        ("何時まで営業しますか。", "ナンジマデエーギョーシマスカ。"),
        ("いつまで営業しますか。", "イツマデエーギョーシマスカ。"),
        ("門を通った時に止める間もなく進んだ。", "モンヲトーッタトキニトメルマモナクススンダ。"),
    ),
)
def test_hour_duration_expressions_keep_dictionary_owned_readings(
    tsqyomi_default_model: None,
    text: str,
    expected_kana: str,
) -> None:
    """時間量表現に対する tsqyomi の不要な介入を防ぎつつ、文脈選択が必要な前後の語が正しく読まれることを確認する。"""

    baseline = pyopenjtalk.g2p(text, kana=True, use_tsqyomi=False, use_vanilla=True)
    with_tsqyomi, _diagnostics = _run_with_diagnostics(text)

    assert baseline == expected_kana
    assert with_tsqyomi == expected_kana


@pytest.mark.parametrize(
    ("text", "expected_baseline", "expected_with_tsqyomi"),
    (("何分かかりますか。", "ナンフンカカリマスカ。", "ナンフンカカリマスカ。"),),
)
def test_non_hour_expressions_remain_available_to_tsqyomi(
    tsqyomi_default_model: None,
    text: str,
    expected_baseline: str,
    expected_with_tsqyomi: str,
) -> None:
    """時間量保護の対象外である「何分かかりますか」のような文脈において、tsqyomi による文脈に応じた読み選択が正常に機能することを確認する。"""

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
    ),
)
def test_minute_duration_expressions_keep_dictionary_owned_readings(
    tsqyomi_default_model: None,
    text: str,
    expected_kana: str,
) -> None:
    """「百二十分」「数分後」のように辞書のデフォルトの読みを維持すべき分単位の時間量表現において、tsqyomi による誤った読みの上書きが行われないことを確認する。"""

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
    tsqyomi_default_model: None,
    case: _EmbeddedSentenceCase,
) -> None:
    """時間量表現を様々な代表的な文型へ埋め込んだ場合でも、tsqyomi を有効にした際に MeCab 本来の正しい読みが維持されることを確認する。"""

    with_tsqyomi_kana = _assert_tsqyomi_kana_matches_mecab_baseline(case.text)
    assert case.embedded_surface in case.text
    assert len(with_tsqyomi_kana) > 0


@pytest.mark.parametrize(
    ("text", "surface", "char_span", "expected_outcome", "expected_was_preserved"),
    (
        (
            "百二十分です。",
            "十分",
            (2, 4),
            "dictionary_default_protected",
            True,
        ),
        (
            "あと三十分。",
            "十分",
            (3, 5),
            "dictionary_default_protected",
            True,
        ),
        (
            "数分後に届きます。",
            "後",
            (2, 3),
            "dictionary_default_protected",
            True,
        ),
        (
            "何分",
            "何分",
            (0, 2),
            "dictionary_default_protected",
            True,
        ),
        (
            "何軒か見学できますか。",
            "何",
            (0, 1),
            "dictionary_default_protected",
            True,
        ),
        (
            "あと一月前です。",
            "一月",
            (2, 4),
            "dictionary_default_protected",
            True,
        ),
        (
            "何時後に届きます。",
            "何時",
            (0, 2),
            "dictionary_default_protected",
            True,
        ),
        (
            "何時まで後。",
            "何時",
            (0, 2),
            "applied",
            False,
        ),
        (
            "四十分後に戻ります。",
            "後",
            (3, 4),
            "dictionary_default_protected",
            True,
        ),
    ),
)
def test_minute_duration_protection_records_dictionary_default_outcome(
    tsqyomi_default_model: None,
    text: str,
    surface: str,
    char_span: tuple[int, int],
    expected_outcome: TargetDiagnosticOutcome,
    expected_was_preserved: bool,
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
    assert matched[0].was_preserved is expected_was_preserved


@pytest.mark.parametrize(
    (
        "text",
        "surface",
        "char_span",
        "expected_baseline",
        "expected_with_tsqyomi",
        "expected_baseline_reading",
        "expected_baseline_pronunciation",
        "expected_tsqyomi_reading",
        "expected_tsqyomi_pronunciation",
    ),
    (
        (
            "その後どうする。",
            "その後",
            (0, 3),
            "ソノゴドースル。",
            "ソノアトドースル。",
            "ソノゴ",
            "ソノゴ",
            "ソノアト",
            "ソノアト",
        ),
        (
            "作業の後で。",
            "後",
            (3, 4),
            "サギョーノアトデ。",
            "サギョーノアトデ。",
            "アト",
            "アト",
            "アト",
            "アト",
        ),
        (
            "晴れた後。",
            "後",
            (3, 4),
            "ハレタアト。",
            "ハレタアト。",
            "アト",
            "アト",
            "アト",
            "アト",
        ),
    ),
)
def test_non_quantity_go_remains_available_to_tsqyomi(
    tsqyomi_default_model: None,
    text: str,
    surface: str,
    char_span: tuple[int, int],
    expected_baseline: str,
    expected_with_tsqyomi: str,
    expected_baseline_reading: str,
    expected_baseline_pronunciation: str,
    expected_tsqyomi_reading: str,
    expected_tsqyomi_pronunciation: str,
) -> None:
    """数量・順序表現の直後に続かない「後」は辞書保護せず、文脈に応じた読み分けの対象とする。"""

    baseline = pyopenjtalk.g2p(text, kana=True, use_tsqyomi=False, use_vanilla=True)
    with_tsqyomi, diagnostics = _run_with_diagnostics(text)
    baseline_features = pyopenjtalk.run_frontend(text, use_tsqyomi=False, use_vanilla=True)
    tsqyomi_features = pyopenjtalk.run_frontend(text, use_tsqyomi=True, use_vanilla=True)
    baseline_feature = next(
        (feature for feature in baseline_features if feature["string"] == surface),
        None,
    )
    tsqyomi_feature = next(
        (feature for feature in tsqyomi_features if feature["string"] == surface),
        None,
    )
    assert baseline_feature is not None, (
        f"baseline feature for {surface!r} not found: "
        f"{[feature['string'] for feature in baseline_features]}"
    )
    assert tsqyomi_feature is not None, (
        f"tsqyomi feature for {surface!r} not found: "
        f"{[feature['string'] for feature in tsqyomi_features]}"
    )
    matched = [
        diagnostic
        for diagnostic in diagnostics
        if diagnostic.surface == surface and diagnostic.char_span == char_span
    ]

    assert baseline == expected_baseline
    assert with_tsqyomi == expected_with_tsqyomi
    assert baseline_feature["read"] == expected_baseline_reading
    assert baseline_feature["pron"] == expected_baseline_pronunciation
    assert tsqyomi_feature["read"] == expected_tsqyomi_reading
    assert tsqyomi_feature["pron"] == expected_tsqyomi_pronunciation
    assert len(matched) == 1
    assert matched[0].outcome == "applied"
    assert matched[0].was_preserved is False


@pytest.mark.parametrize(
    ("text", "expected_kana"),
    (
        ("三人後に並びます。", "サンニンゴニナラビマス。"),
        ("三秒後です。", "サンビョーゴデス。"),
        ("三日後です。", "ミッカゴデス。"),
        ("三週間後です。", "サンシューカンゴデス。"),
        ("三か月後です。", "サンカゲツゴデス。"),
        ("三年後です。", "サンネンゴデス。"),
        ("三軒後です。", "サンケンゴデス。"),
        ("三個後です。", "サンコゴデス。"),
        ("三回後です。", "サンカイゴデス。"),
        ("三枚後です。", "サンマイゴデス。"),
    ),
)
def test_quantity_counter_go_keeps_dictionary_pronunciation(
    tsqyomi_default_model: None,
    text: str,
    expected_kana: str,
) -> None:
    """「三個後」「三回後」のように数量や順序を表す表現に続く「後」について、tsqyomi によって誤判定されず、辞書通りの「ゴ」という読みが維持されることを確認する。"""

    baseline = pyopenjtalk.g2p(text, kana=True, use_tsqyomi=False, use_vanilla=True)
    with_tsqyomi, _diagnostics = _run_with_diagnostics(text)
    baseline_features = pyopenjtalk.run_frontend(text, use_tsqyomi=False, use_vanilla=True)
    tsqyomi_features = pyopenjtalk.run_frontend(text, use_tsqyomi=True, use_vanilla=True)
    baseline_go_feature = next(
        (feature for feature in baseline_features if feature["string"] == "後"),
        None,
    )
    tsqyomi_go_feature = next(
        (feature for feature in tsqyomi_features if feature["string"] == "後"),
        None,
    )
    assert baseline_go_feature is not None, (
        f"baseline feature for '後' not found: "
        f"{[feature['string'] for feature in baseline_features]}"
    )
    assert tsqyomi_go_feature is not None, (
        f"tsqyomi feature for '後' not found: {[feature['string'] for feature in tsqyomi_features]}"
    )

    assert baseline == expected_kana
    assert with_tsqyomi == expected_kana
    assert (baseline_go_feature["read"], baseline_go_feature["pron"]) == ("ゴ", "ゴ")
    assert (tsqyomi_go_feature["read"], tsqyomi_go_feature["pron"]) == ("ゴ", "ゴ")


@pytest.mark.parametrize(
    ("text", "expected_baseline", "expected_with_tsqyomi"),
    (
        ("五分かかります。", "ゴブカカリマス。", "ゴフンカカリマス。"),
        ("十分かかります。", "ジューブンカカリマス。", "ジュップンカカリマス。"),
        ("五分後に始めます。", "ゴブゴニハジメマス。", "ゴフンゴニハジメマス。"),
        ("十分後に始めます。", "ジュップンゴニハジメマス。", "ジュップンゴニハジメマス。"),
        ("体中が痛い。", "カラダチューガイタイ。", "カラダジューガイタイ。"),
    ),
)
def test_minute_heteronyms_and_taichu_remain_available_to_tsqyomi(
    tsqyomi_default_model: None,
    text: str,
    expected_baseline: str,
    expected_with_tsqyomi: str,
) -> None:
    """「五分」「十分」「体中」は tsqyomi の文脈選択を残す。"""

    baseline = pyopenjtalk.g2p(text, kana=True, use_tsqyomi=False, use_vanilla=True)
    with_tsqyomi, _diagnostics = _run_with_diagnostics(text)

    assert baseline == expected_baseline
    assert with_tsqyomi == expected_with_tsqyomi


@pytest.mark.parametrize(
    ("text", "expected_selected_kana", "expected_postprocessed_kana"),
    (
        ("駆け込み寺", "カケコミテラ", "カケコミデラ"),
        ("田舎寺", "イナカテラ", "イナカデラ"),
        ("先生方", "センセーカタ", "センセーガタ"),
        ("石見国", "イワミコク", "イワミノクニ"),
        ("霊山寺", "リョーゼンテラ", "リョーゼンジ"),
    ),
)
def test_deterministic_reading_postprocessing_corrects_tsqyomi_selection(
    tsqyomi_default_model: None,
    text: str,
    expected_selected_kana: str,
    expected_postprocessed_kana: str,
) -> None:
    """「駆け込み寺」「先生方」「石見国」のように特定の語形や連語によって確定する読みについて、tsqyomi による候補選択が行われた後でも文脈読み補正によって正しい読みに補正されることを確認する。"""

    selected_kana = pyopenjtalk.g2p(text, kana=True, use_tsqyomi=True, use_vanilla=True)
    postprocessed_kana = pyopenjtalk.g2p(text, kana=True, use_tsqyomi=True)
    normal_kana = pyopenjtalk.g2p(text, kana=True, use_tsqyomi=False)
    detailed_features, _morphs = pyopenjtalk.run_frontend_detailed(text, use_tsqyomi=True)

    assert selected_kana == expected_selected_kana
    assert postprocessed_kana == expected_postprocessed_kana
    assert normal_kana == expected_postprocessed_kana
    assert "".join(feature["pron"].replace("’", "") for feature in detailed_features) == (
        expected_postprocessed_kana
    )


def test_deterministic_reading_postprocessing_keeps_person_name_after_tsqyomi(
    tsqyomi_default_model: None,
) -> None:
    """「記念章」のような特定の複合語以外の文脈において、人名の「章」（「田中章さん」など）が文脈読み補正によって誤って書き換えられず、「アキラ」という読みのまま維持されることを確認する。"""

    text = "田中章さん"

    assert pyopenjtalk.g2p(text, kana=True, use_tsqyomi=True, use_vanilla=True) == (
        "タナカアキラサン"
    )
    assert pyopenjtalk.g2p(text, kana=True, use_tsqyomi=True) == "タナカアキラサン"
    assert pyopenjtalk.g2p(text, kana=True, use_tsqyomi=False) == "タナカアキラサン"


@pytest.mark.parametrize(
    ("text", "expected_kana"),
    (
        ("一見して分かる", "イッケンシテワカル"),
        ("一声も出ない", "ヒトコエモデナイ"),
        ("兵の数", "ヘーノカズ"),
        ("金を払う", "カネヲハラウ"),
        ("この方", "コノカタ"),
    ),
)
def test_deterministic_reading_postprocessing_keeps_tsqyomi_outside_closed_conditions(
    tsqyomi_default_model: None,
    text: str,
    expected_kana: str,
) -> None:
    """特定の決定的な補正規則に合致しない一般的な用法（「一見して分かる」「一声も出ない」など）において、tsqyomi が推定した文脈読みがそのまま維持されることを確認する。"""

    selected_kana = pyopenjtalk.g2p(text, kana=True, use_tsqyomi=True, use_vanilla=True)
    postprocessed_kana = pyopenjtalk.g2p(text, kana=True, use_tsqyomi=True)

    assert selected_kana == expected_kana
    assert postprocessed_kana == expected_kana


def test_deterministic_reading_postprocessing_can_be_disabled_with_use_vanilla(
    tsqyomi_default_model: None,
) -> None:
    """tsqyomi を有効にした場合（use_tsqyomi=True）でも、use_vanilla=True を指定すれば文脈読み補正などの pyopenjtalk-plus 独自の後処理が無効化されることを確認する。"""

    with_postprocessing = pyopenjtalk.g2p("先生方", kana=True, use_tsqyomi=True)
    without_postprocessing = pyopenjtalk.g2p(
        "先生方",
        kana=True,
        use_tsqyomi=True,
        use_vanilla=True,
    )

    assert with_postprocessing == "センセーガタ"
    assert without_postprocessing == "センセーカタ"


def test_deterministic_reading_postprocessing_preserves_protected_user_dictionary(
    tsqyomi_default_model: None,
    tmp_path: Path,
) -> None:
    """読み保護を有効にしたユーザー辞書のエントリ（「方」の「ホウ」など）は、tsqyomi や文脈読み補正のルールに優先してユーザー辞書の読みが反映されることを確認する。"""

    user_csv = tmp_path / "protected_tsqyomi.csv"
    user_dic = tmp_path / "protected_tsqyomi.dic"
    user_csv.write_text(
        "方,1358,1358,1,名詞,接尾,一般,*,*,*,方,ホウ,ホウ,1/2,C3\n",
        encoding="utf-8",
    )

    try:
        pyopenjtalk.mecab_dict_index(str(user_csv), str(user_dic))
        pyopenjtalk.update_global_jtalk_with_user_dict(
            [
                UserDictionaryEntry(
                    dic_path=str(user_dic),
                    is_reading_protected=True,
                )
            ]
        )

        # tsqyomi と決定規則が別の候補を持っていても、明示的に保護した読みを marine へ渡す
        assert (
            pyopenjtalk.g2p(
                "先生方",
                kana=True,
                use_tsqyomi=True,
                run_marine=True,
            )
            == "センセーホウ"
        )
    finally:
        pyopenjtalk.unset_user_dict()


def test_deterministic_reading_postprocessing_runs_before_marine(
    tsqyomi_default_model: None,
) -> None:
    """tsqyomi と文脈読み補正によって確定した読み（「石見国」の「ノクニ」など）が marine に渡され、補正後の発音単位に基づいてアクセント推定が行われることを確認する。"""

    features = pyopenjtalk.run_frontend("石見国", use_tsqyomi=True, run_marine=True)
    country = next(feature for feature in features if feature["string"] == "国")

    assert country["read"] == "ノクニ"
    assert country["pron"] == "ノクニ"
    assert pyopenjtalk.g2p("石見国", kana=True, use_tsqyomi=True, run_marine=True) == (
        "イワミノクニ"
    )
