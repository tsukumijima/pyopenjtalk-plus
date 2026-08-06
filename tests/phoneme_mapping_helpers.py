"""音素マッピング系テストで共有するコーパスとヘルパ。"""

PHONEME_MAPPING_CORPUS = [
    "こんにちは",
    "おはようございます",
    "東京は日本の首都です",
    "東京都知事が記者会見を行った。",
    "大阪",
    "外国人参政権",
    "学生生活",
    "学生々活は楽しい",
    "部分々々",
    "東京、大阪",
    "東京　大阪",
    "（テスト・ケース）",
    "今日は2112年9月3日です",
    "電話番号は090-1234-5678です",
    "明日は雨が降るでしょう",
    "ご遠慮ください",
    "お入りください",
    "食べよう",
    "見よう",
    "読もう",
    "書こう",
    "遊ぼう",
    "起きよう",
    "考えよう",
    "見せよう",
    "行こう",
    "入ろう",
    "来よう",
    "しよう",
    "食べている",
    "読んでいる",
    "書いている",
    "走っている",
    "見ている",
    "起きている",
    "つまみ出されようとした",
]


def extract_label_phonemes(labels: list[str], keep_pause: bool = False) -> list[str]:
    """
    フルコンテキストラベル列から音素列を抽出する。

    Args:
        labels (list[str]): JPCommon ラベル列
        keep_pause (bool): pau を残すか

    Returns:
        list[str]: 抽出した音素列
    """

    phonemes = [label.split("-")[1].split("+")[0] for label in labels[1:-1]]
    if keep_pause is False:
        phonemes = [phoneme for phoneme in phonemes if phoneme != "pau"]
    return phonemes
