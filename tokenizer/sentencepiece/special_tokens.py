"""Reserved special tokens for Odyssey.

Why this exists:
    Control and chat-role tokens must keep stable IDs across training and
    inference. Documenting them in one module prevents silent vocabulary drift.
"""

from __future__ import annotations

from dataclasses import dataclass

from tokenizer.sentencepiece.config import TokenizerConfig


@dataclass(frozen=True, slots=True)
class SpecialToken:
    """One reserved vocabulary entry."""

    name: str
    surface: str
    token_id: int
    purpose: str
    usage: str


def build_special_tokens(config: TokenizerConfig) -> list[SpecialToken]:
    """Construct the documented reserved-token table from config."""
    core = [
        SpecialToken(
            name="pad",
            surface=config.special_tokens["pad"],
            token_id=config.pad_id,
            purpose="Padding sequences to a common length in batched training.",
            usage="Inserted by collators; usually masked in the loss.",
        ),
        SpecialToken(
            name="bos",
            surface=config.special_tokens["bos"],
            token_id=config.bos_id,
            purpose="Marks the beginning of a sequence.",
            usage="Optionally prepended before model inputs.",
        ),
        SpecialToken(
            name="eos",
            surface=config.special_tokens["eos"],
            token_id=config.eos_id,
            purpose="Marks the end of a sequence / generation stop.",
            usage="Appended at the end of targets; used as a stop token.",
        ),
        SpecialToken(
            name="unk",
            surface=config.special_tokens["unk"],
            token_id=config.unk_id,
            purpose="Fallback for pieces outside the trained vocabulary.",
            usage="Emitted by the encoder when no better piece exists.",
        ),
    ]

    # User-defined symbols receive IDs assigned by SentencePiece after the core
    # control tokens. Exact numeric IDs are resolved from a loaded model.
    extras = [
        SpecialToken(
            name="mask",
            surface=config.special_tokens["mask"],
            token_id=-1,
            purpose="Reserved for masked-language or span corruption experiments.",
            usage="Inserted explicitly by dataset pipelines when needed.",
        ),
        SpecialToken(
            name="system",
            surface=config.special_tokens["system"],
            token_id=-1,
            purpose="Delimits system / policy instructions in chat templates.",
            usage="Wrapped around system prompts during formatting.",
        ),
        SpecialToken(
            name="user",
            surface=config.special_tokens["user"],
            token_id=-1,
            purpose="Delimits end-user turns in chat templates.",
            usage="Wrapped around user messages during formatting.",
        ),
        SpecialToken(
            name="assistant",
            surface=config.special_tokens["assistant"],
            token_id=-1,
            purpose="Delimits model / assistant turns in chat templates.",
            usage="Wrapped around assistant responses during formatting.",
        ),
    ]
    return [*core, *extras]


def special_token_surfaces(config: TokenizerConfig) -> list[str]:
    """Return every reserved surface form that must exist in the vocabulary."""
    return [
        config.special_tokens["pad"],
        config.special_tokens["bos"],
        config.special_tokens["eos"],
        config.special_tokens["unk"],
        *config.user_defined_symbols,
    ]


def describe_special_tokens(config: TokenizerConfig) -> str:
    """Human-readable documentation block for README / inspector output."""
    lines = [
        "Special Tokens",
        "==============",
        "",
        f"{'Name':<12} {'Surface':<14} {'ID':<6} Purpose",
        "-" * 72,
    ]
    for token in build_special_tokens(config):
        token_id = str(token.token_id) if token.token_id >= 0 else "dyn"
        lines.append(
            f"{token.name:<12} {token.surface:<14} {token_id:<6} {token.purpose}"
        )
    return "\n".join(lines)
