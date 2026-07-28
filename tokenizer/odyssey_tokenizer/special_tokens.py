"""Reserved special tokens for OdysseyTokenizer."""

from __future__ import annotations

from dataclasses import dataclass

from odyssey_tokenizer.config import DEFAULT_SPECIAL_TOKENS, BPEConfig


@dataclass(frozen=True, slots=True)
class SpecialToken:
    name: str
    surface: str
    token_id: int
    purpose: str


_PURPOSES: dict[str, str] = {
    "<pad>": "Padding sequences in batched training.",
    "<bos>": "Beginning-of-sequence marker.",
    "<eos>": "End-of-sequence / generation stop marker.",
    "<unk>": "Fallback for bytes outside the trained vocabulary (rare with byte-level BPE).",
    "<mask>": "Reserved for masking / span corruption experiments.",
    "<system>": "Chat system / policy turn delimiter.",
    "<user>": "Chat user turn delimiter.",
    "<assistant>": "Chat assistant turn delimiter.",
    "<tool>": "Tool / function-call turn delimiter.",
    "<think>": "Explicit reasoning / scratchpad segment delimiter.",
}


def build_special_tokens(config: BPEConfig) -> list[SpecialToken]:
    """Build the reserved-token table with stable IDs 0..N-1."""
    tokens: list[SpecialToken] = []
    for index, surface in enumerate(config.special_tokens):
        name = surface.strip("<>") or surface
        tokens.append(
            SpecialToken(
                name=name,
                surface=surface,
                token_id=index,
                purpose=_PURPOSES.get(surface, "Reserved control token."),
            )
        )
    return tokens


def core_special_id_map(config: BPEConfig) -> dict[str, int]:
    """Map canonical names to IDs for pad/bos/eos/unk."""
    surfaces = {token: index for index, token in enumerate(config.special_tokens)}
    return {
        "pad": surfaces.get("<pad>", config.pad_id),
        "bos": surfaces.get("<bos>", config.bos_id),
        "eos": surfaces.get("<eos>", config.eos_id),
        "unk": surfaces.get("<unk>", config.unk_id),
    }


def describe_special_tokens(config: BPEConfig) -> str:
    lines = [
        "Special Tokens",
        "==============",
        "",
        f"{'ID':<4} {'Surface':<14} Purpose",
        "-" * 72,
    ]
    for token in build_special_tokens(config):
        lines.append(f"{token.token_id:<4} {token.surface:<14} {token.purpose}")
    # Mention defaults for documentation completeness.
    _ = DEFAULT_SPECIAL_TOKENS
    return "\n".join(lines)
