# ---------- crc_utils.py  -------------------------------------------
import binascii

def add_crc(msg: str) -> str:
    """Return ASCII payload + 2-byte CRC16 (latin-1 encoded)."""
    data = msg.encode('ascii')
    crc  = binascii.crc_hqx(data, 0xFFFF).to_bytes(2, 'big')
    return (data + crc).decode('latin1')

def check_crc(payload: str) -> str | None:
    """Verify CRC.  Return original ASCII if ok, else None."""
    blob = payload.encode('latin1')
    if len(blob) < 3:
        return None
    data, crc = blob[:-2], blob[-2:]
    good      = binascii.crc_hqx(data, 0xFFFF).to_bytes(2, 'big')
    return data.decode('ascii') if crc == good else None
# --------------------------------------------------------------------
