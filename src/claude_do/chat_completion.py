from anthropic import Anthropic
from openai import OpenAI
from typing import Optional


def chat_completion(
    prompt: str, model: str, max_tokens: int, temperature: float, base_url: Optional[str]
) -> str:
    chat_completion_function = (
        chat_completion_anthropic
        if model.lower().startswith("claude")
        else chat_completion_openai
    )

    return chat_completion_function(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        base_url=base_url,
    )


def chat_completion_anthropic(
    prompt: str, model: str, max_tokens: int, temperature: float, base_url: Optional[str]
) -> str:
    client = Anthropic(base_url=base_url)

    message = client.messages.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    response_text = message.content[0].text  # type: ignore
    assert message.stop_reason == "end_turn", (
        f"Claude finished its message for the wrong reason: the reason is `{message.stop_reason}`."
    )
    assert isinstance(response_text, str)
    return response_text


def chat_completion_openai(
    prompt: str, model: str, max_tokens: int, temperature: float, base_url: Optional[str]
) -> str:
    client = OpenAI(base_url=base_url)

    message = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_completion_tokens=max_tokens,
        temperature=temperature,
    )

    response_text = message.choices[0].message.content
    assert message.choices[0].finish_reason == "stop", (
        f"The OpenAI model finished its message for the wrong reason: the reason is `{message.choices[0].finish_reason}`."
    )
    assert isinstance(response_text, str)
    return response_text
