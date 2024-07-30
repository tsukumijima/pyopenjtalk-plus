import sys
from collections.abc import Iterable

if sys.version_info >= (3, 8):
    from typing import TypedDict
    class _NJDFeature(TypedDict):  # NOQA
        string: str
        pos: str
        pos_group1: str
        pos_group2: str
        pos_group3: str
        ctype: str
        cform: str
        orig: str
        read: str
        pron: str
        acc: int
        mora_size: int
        chain_rule: str
        chain_flag: int

else:
    _NJDFeature = dict[str, str | int]

class OpenJTalk:  # NOQA
    def __init__(
        self, dn_mecab: bytes = b"/usr/local/dic", userdic: bytes = b""
    ) -> None:
        """OpenJTalk

        Args:
            dn_mecab (bytes): Dictionaly path for MeCab.
            userdic (bytes): Dictionary path for MeCab userdic.
                This option is ignored when empty bytestring is given.
                Default is empty.
        """
        pass
    def run_frontend(self, text: str | bytes | bytearray) -> list[_NJDFeature]:  # NOQA
        """Run OpenJTalk's text processing frontend"""
        pass
    def make_label(self, features: Iterable[_NJDFeature]) -> list[str]:  # NOQA
        """Make full-context label"""
        pass
    def g2p(  # NOQA
        self, text: str | bytes | bytearray, kana: bool = False, join: bool = True
    ) -> list[str] | str:
        """Grapheme-to-phoeneme (G2P) conversion"""
        pass

def mecab_dict_index(dn_mecab: bytes, path: bytes, out_path: bytes) -> int: ...  # NOQA
