# flake8: noqa

from typing import Iterable, List

from .types import NJDFeature

class OpenJTalk:
    def __init__(self, dn_mecab: bytes = b"/usr/local/dic", userdic: bytes = b"") -> None:
        """OpenJTalk

        Args:
            dn_mecab (bytes): Dictionary path for MeCab.
            userdic (bytes): Dictionary path for MeCab userdic.
                This option is ignored when empty bytestring is given.
                Default is empty.
        """
        pass

    def run_frontend(self, text: str | bytes | bytearray) -> List[NJDFeature]:
        """Run OpenJTalk's text processing frontend"""
        pass

    def make_label(self, features: Iterable[NJDFeature]) -> List[str]:
        """Make full-context label"""
        pass

    def g2p(
        self, text: str | bytes | bytearray, kana: bool = False, join: bool = True
    ) -> List[str] | str:
        """Grapheme-to-phoeneme (G2P) conversion (Cython implementation)"""
        pass

def mecab_dict_index(dn_mecab: bytes, path: bytes, out_path: bytes) -> int: ...
def build_mecab_dictionary(dn_mecab: bytes) -> int: ...
