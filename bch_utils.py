# bch_utils.py   –  very thin wrapper around reedsolo (BCH(15,11) ~= RS(15,11))
import binascii, itertools, reedsolo

_rs = reedsolo.RSCodec(nsym=4)          # 11 data + 4 parity  = 15

def bch_encode(txt: str) -> bytes:
    """→ bytes with 4 Reed-Solomon parity symbols appended"""
    return _rs.encode(txt.encode("latin1"))

def bch_decode(raw: bytes) -> str | None:
    """returns the decoded text or None if RS correction failed"""
    try:
        return _rs.decode(raw)[0].decode("latin1")
    except reedsolo.ReedSolomonError:
        return None
