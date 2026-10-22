from collections.abc import Iterator

from ..model import Transaction, TxInput


def script_bytes(value: str) -> bytes:
    try:
        return bytes.fromhex(value)
    except (TypeError, ValueError):
        return b""


def is_multisig_script(script: bytes) -> bool:
    return bool(script) and script[-1] in {0xAE, 0xAF}


def last_push(script: bytes) -> bytes:
    pushes = script_pushes(script)
    return pushes[-1] if pushes else b""


def script_pushes(script: bytes) -> list[bytes]:
    pushes: list[bytes] = []
    index = 0

    while index < len(script):
        opcode = script[index]
        index += 1

        if opcode == 0:
            pushes.append(b"")
            continue

        if opcode <= 75:
            length = opcode
        elif opcode == 0x4C and index < len(script):
            length = script[index]
            index += 1
        elif opcode == 0x4D and index + 2 <= len(script):
            length = int.from_bytes(script[index : index + 2], "little")
            index += 2
        elif opcode == 0x4E and index + 4 <= len(script):
            length = int.from_bytes(script[index : index + 4], "little")
            index += 4
        else:
            break

        if index + length > len(script):
            break

        pushes.append(script[index : index + length])
        index += length

    return pushes


def is_ecdsa_signature(raw: bytes) -> bool:
    if len(raw) < 9 or raw[0] != 0x30 or raw[1] != len(raw) - 3:
        return False

    r_len = raw[3]
    s_tag_index = 4 + r_len
    if r_len == 0 or s_tag_index + 2 >= len(raw) or raw[2] != 0x02:
        return False

    s_len = raw[s_tag_index + 1]
    return (
        raw[s_tag_index] == 0x02
        and s_len > 0
        and s_tag_index + 2 + s_len == len(raw) - 1
    )


def input_type(inp: TxInput) -> str:
    if not inp.prevout:
        return "unknown"

    script_type = inp.prevout.scriptpubkey_type.lower()
    script = script_bytes(inp.prevout.scriptpubkey)

    if script_type in {"p2pk", "pubkey"} or (
        len(script) in {35, 67}
        and script[0] in {33, 65}
        and script[-1] == 0xAC
    ):
        return "p2pk"
    if script_type in {"p2pkh", "pubkeyhash"} or (
        len(script) == 25
        and script.startswith(b"\x76\xa9\x14")
        and script.endswith(b"\x88\xac")
    ):
        return "p2pkh"
    if script_type in {"p2tr", "witness_v1_taproot"} or script.startswith(b"\x51\x20"):
        return "p2tr"
    if script_type in {"p2wpkh", "witness_v0_keyhash"} or script.startswith(b"\x00\x14"):
        return "p2wpkh"
    if script_type in {"p2wsh", "witness_v0_scripthash"} or script.startswith(b"\x00\x20"):
        if not inp.witness:
            return "p2wsh_unknown"
        witness_script = script_bytes(inp.witness[-1])
        return "p2wsh_multisig" if is_multisig_script(witness_script) else "p2wsh_all"
    if script_type in {"p2sh", "scripthash"} or (
        len(script) == 23 and script.startswith(b"\xa9\x14") and script.endswith(b"\x87")
    ):
        redeem = last_push(script_bytes(inp.scriptsig))
        if not redeem:
            return "p2sh_unknown"
        if redeem.startswith(b"\x00\x14"):
            return "p2sh_p2wpkh"
        if redeem.startswith(b"\x00\x20"):
            return "p2sh_p2wsh"
        return "p2sh_multisig" if is_multisig_script(redeem) else "p2sh_all"
    return script_type or "unknown"


def output_type(output) -> str:
    script_type = output.scriptpubkey_type.lower()
    script = script_bytes(output.scriptpubkey)

    if script_type in {"op_return", "nulldata"} or script.startswith(b"\x6a"):
        return "op_return"
    if script_type in {"p2tr", "witness_v1_taproot"} or script.startswith(b"\x51\x20"):
        return "bech32m"
    if script_type.startswith("p2w") or script_type == "witness_v0_keyhash" or script.startswith(b"\x00"):
        return "bech32"
    if script_type in {"p2pkh", "p2sh", "pubkeyhash", "scripthash"}:
        return "base58"

    address = output.scriptpubkey_address.lower()
    if address.startswith(("bc1", "tb1", "bcrt1")):
        return "bech32m" if address.startswith(("bc1p", "tb1p", "bcrt1p")) else "bech32"
    return script_type or "unknown"


def signature_candidates(tx: Transaction) -> Iterator[bytes]:
    for inp in tx.inputs:
        yield from (script_bytes(item) for item in (inp.witness or []))
        if inp.scriptsig:
            yield from script_pushes(script_bytes(inp.scriptsig))


def public_keys(tx: Transaction) -> Iterator[bytes]:
    for candidate in signature_candidates(tx):
        if len(candidate) in {33, 65} and candidate[0] in {2, 3, 4}:
            yield candidate

    for inp in tx.inputs:
        prevout_script = script_bytes(inp.prevout.scriptpubkey) if inp.prevout else b""
        prevout_type = inp.prevout.scriptpubkey_type.lower() if inp.prevout else ""
        prevout_pushes = script_pushes(prevout_script)
        if (
            (prevout_type in {"p2pk", "pubkey"} or prevout_script.endswith(b"\xac"))
            and len(prevout_pushes) == 1
        ):
            pubkey = prevout_pushes[0]
            if len(pubkey) in {33, 65} and pubkey[0] in {2, 3, 4}:
                yield pubkey
