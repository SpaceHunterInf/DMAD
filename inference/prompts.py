"""
Prompt templates for multi-agent debate.

Contains all prompt templates used in the DMAD framework:
- Initial round prompts (with and without confidence)
- Multi-agent debate prompts (with and without confidence)
- Agent response templates

Reference: Section 4 & Appendix A of the paper.
"""

from typing import Dict
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MCQA_INSTRUCTION = ", where YOUR FINAL ANSWER is one of (A), (B), (C) or (D)."
CSQA_INSTRUCTION = (
    ", where YOUR FINAL ANSWER is one of (A), (B), (C), (D) or (E)."
)

# ---------------------------------------------------------------------------
# Initial round templates
# ---------------------------------------------------------------------------

INITIAL_PROMPT_WITH_CONFIDENCE = """\
Answer the question. Think step by step first, at the end of your reasoning, \
provide your final answer and confidence level, ranging from 0 to 10, where \
0 means no confidence at all and 10 means complete confidence.

Output format:
<reasoning>YOUR DETAILED REASONING HERE</reasoning>
<answer>YOUR FINAL ANSWER</answer>{extra_info}
<confidence>INTEGER</confidence>

{question}"""

INITIAL_PROMPT = """\
Answer the question. Think step by step first, at the end of your reasoning, \
provide your final answer.

Output format:
<reasoning>YOUR DETAILED REASONING HERE</reasoning>
<answer>YOUR FINAL ANSWER</answer>{extra_info}
{question}"""

# ---------------------------------------------------------------------------
# Multi-agent debate templates
# ---------------------------------------------------------------------------

DEBATE_PROMPT_WITH_CONFIDENCE = """\
You are revising your answer after reviewing other agents' reasoning.

Question: {question}

Other agents' responses:
{other_agents}

Your previous reasoning and answer:
{reasoning}

Instructions:
- Reflect on how others reasoned.
- You may revise your answer if someone's reasoning provides stronger evidence.
- However, if you believe all of them missed something important, propose a \
better or alternative answer — clearly explain why.
- Be concise and clear.
- Update your confidence level to reflect how certain you are now. Confidence \
level is ranging from 0 to 10, where 0 means no confidence at all and 10 \
means complete confidence

Think step by step first, at the end of your reasoning, provide your final \
answer and confidence level in the following format:

<reasoning>YOUR DETAILED REASONING HERE</reasoning>
<answer>YOUR FINAL ANSWER</answer>{extra_info}
<confidence>INTEGER</confidence>"""

DEBATE_PROMPT = """\
You are revising your answer after reviewing other agents' reasoning.

Question: {question}
Other agents' responses:
{other_agents}

Your previous reasoning and answer:
{reasoning}

Instructions:
- Reflect on how others reasoned.
- You may revise your answer if someone's reasoning provides stronger evidence.
- However, if you believe all of them missed something important, propose a \
better or alternative answer — clearly explain why.
- Be concise and clear.
Think step by step first, at the end of your reasoning, provide your final \
answer in the following format:
<answer>YOUR FINAL ANSWER</answer>{extra_info}"""

# ---------------------------------------------------------------------------
# Agent response summary templates (used inside debate prompts)
# ---------------------------------------------------------------------------

AGENT_RESPONSE_WITH_CONFIDENCE = """\
Agent {agent_id} has provided the following reasoning and final answer \
with confidence level:
<reasoning>{reasoning}</reasoning>
<answer>{final_answer}</answer>
<confidence>{confidence_level}</confidence>

"""

AGENT_RESPONSE = """\
Agent {agent_id} has provided the following reasoning and final answer:
<reasoning>{reasoning}</reasoning>
<answer>{final_answer}</answer>

"""


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _is_mcqa(dataset_name: str) -> bool:
    """Check whether a dataset uses multiple-choice answer format."""
    return any(
        tag in dataset_name.lower()
        for tag in ["mmlu", "csqa", "hellaswag", "gpqa", "arc"]
    )


def _extra_info(dataset_name: str) -> str:
    """Return extra answer-format instructions for MCQA datasets."""
    if not _is_mcqa(dataset_name):
        return ""
    if "csqa" in dataset_name.lower():
        return CSQA_INSTRUCTION
    return MCQA_INSTRUCTION


def format_initial_round_prompt(
    example: Dict,
    use_confidence: bool = False,
    tokenizer: AutoTokenizer = None,
    chat_mode: bool = False,
) -> Dict:
    """
    Format a single example with the initial-round prompt template.

    Args:
        example: Dictionary with ``question``, ``label``, ``dataset`` keys.
        use_confidence: Whether to include confidence in the prompt.
        tokenizer: Tokenizer for chat-template formatting.
        chat_mode: If *True*, wrap as a chat message and apply the
                   tokenizer's chat template.

    Returns:
        The input *example* dict with an added ``prompt`` key.
    """
    template = (
        INITIAL_PROMPT_WITH_CONFIDENCE if use_confidence else INITIAL_PROMPT
    )
    prompt = template.format(
        question=example["question"],
        extra_info=_extra_info(example["dataset"]),
    )

    if chat_mode:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)

    example["prompt"] = prompt
    example["label"] = str(example["label"])
    return example


def format_multi_agent_prompt(
    example: Dict,
    other_agent_responses: list,
    own_response: Dict,
    use_confidence: bool = False,
    tokenizer: AutoTokenizer = None,
    chat_mode: bool = False,
) -> Dict:
    """
    Format a debate-round prompt that includes other agents' responses.

    Args:
        example: Dictionary with ``question``, ``label``, ``dataset`` keys.
        other_agent_responses: List of dicts, each with ``agent_id``,
            ``reasoning``, ``final_answer``, and optionally
            ``confidence_level``.
        own_response: Dict with ``reasoning``, ``final_answer``, and
            optionally ``confidence_level``.
        use_confidence: Whether to include confidence.
        tokenizer: Tokenizer for chat-template formatting.
        chat_mode: If *True*, apply chat template.

    Returns:
        The input *example* dict with an added ``prompt`` key.
    """
    # Select templates
    prompt_tmpl = (
        DEBATE_PROMPT_WITH_CONFIDENCE if use_confidence else DEBATE_PROMPT
    )
    agent_tmpl = (
        AGENT_RESPONSE_WITH_CONFIDENCE if use_confidence else AGENT_RESPONSE
    )

    # Build other-agents block
    other_text = ""
    for resp in other_agent_responses:
        other_text += agent_tmpl.format(
            agent_id=resp["agent_id"],
            reasoning=resp["reasoning"],
            final_answer=resp["final_answer"],
            confidence_level=resp.get("confidence_level", "N/A"),
        )

    # Build own-response block
    own_text = own_response["reasoning"]
    if use_confidence:
        own_text += (
            f"\n<answer>{own_response['final_answer']}</answer>"
            f"\n<confidence>{own_response.get('confidence_level', 'N/A')}"
            f"</confidence>"
        )
    else:
        own_text += f"\n<answer>{own_response['final_answer']}</answer>"

    prompt = prompt_tmpl.format(
        question=example["question"],
        other_agents=other_text.strip(),
        reasoning=own_text,
        extra_info=_extra_info(example["dataset"]),
    )

    if chat_mode:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)

    example["prompt"] = prompt
    example["label"] = str(example["label"])
    return example
