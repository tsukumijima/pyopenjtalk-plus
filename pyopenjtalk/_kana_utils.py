"""
文字列が仮名だけでできているかなど、仮名の文字種を判定する関数群。
コードベースのどこからでも読み込めるよう、pyopenjtalk の他のモジュールに依存しない設計としている。
"""


def is_katakana_word(text: str) -> bool:
    """
    文字列がカタカナ (長音符と踊り字「ヽ」「ヾ」を含む) だけでできているかを判定する。

    Args:
        text (str): 判定する文字列

    Returns:
        bool: 1文字以上あり、すべてカタカナの場合は True
    """

    return text != "" and all(
        "ァ" <= character <= "ヴ" or character in "ーヽヾ" for character in text
    )


def is_hiragana_word(text: str) -> bool:
    """
    文字列がひらがなだけでできているかを判定する。

    Args:
        text (str): 判定する文字列

    Returns:
        bool: 1文字以上あり、すべてひらがなの場合は True
    """

    return text != "" and all("ぁ" <= character <= "ゖ" for character in text)


def contains_hiragana(text: str) -> bool:
    """
    文字列にひらがなが1文字でも含まれるかを判定する。

    Args:
        text (str): 判定する文字列

    Returns:
        bool: ひらがなを含む場合は True
    """

    return any("ぁ" <= character <= "ゖ" for character in text)
