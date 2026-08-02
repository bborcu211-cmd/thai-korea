import os
import re
from collections import defaultdict, deque
from typing import Deque, Dict, Literal, TypedDict

from openai import OpenAI

from translation_prompt import TRANSLATION_INSTRUCTIONS


# Render에 등록된 OPENAI_API_KEY를 자동으로 읽는다.
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


class RawMessage(TypedDict):
    speaker: str
    text: str


# 채팅방별 최근 원문 메시지 8개만 보관한다.
# 번역봇이 만든 번역문은 여기에 넣으면 안 된다.
conversation_history: Dict[str, Deque[RawMessage]] = defaultdict(
    lambda: deque(maxlen=8)
)


def detect_source_language(
    text: str,
    speaker: Literal["J", "LOVE"],
) -> Literal["ko", "th"]:
    """
    문자 종류로 언어를 판단한다.

    한글과 태국 문자가 없는 경우:
    - J의 메시지는 한국어 원문으로 취급
    - LOVE의 메시지는 태국어 원문으로 취급
    """

    has_korean = bool(re.search(r"[가-힣]", text))
    has_thai = bool(re.search(r"[\u0E00-\u0E7F]", text))

    if has_korean and not has_thai:
        return "ko"

    if has_thai and not has_korean:
        return "th"

    return "ko" if speaker == "J" else "th"


def make_context(chat_id: str) -> str:
    """
    최근 원문 메시지를 모델이 이해할 수 있는 형태로 만든다.
    """

    messages = conversation_history[chat_id]

    if not messages:
        return "(이전 대화 없음)"

    return "\n".join(
        f"{message['speaker']}: {message['text']}"
        for message in messages
    )


def translate_message(
    chat_id: str,
    speaker: Literal["J", "LOVE"],
    text: str,
) -> str:
    """
    최신 메시지 하나를 번역한다.
    """

    cleaned_text = text.strip()

    if not cleaned_text:
        return ""

    source_language = detect_source_language(cleaned_text, speaker)
    target_language = "th" if source_language == "ko" else "ko"

    recent_context = make_context(chat_id)

    input_text = f"""
<translation_information>
화자: {speaker}
원문 언어: {source_language}
번역할 언어: {target_language}
</translation_information>

<recent_conversation>
{recent_context}
</recent_conversation>

<message_to_translate>
{cleaned_text}
</message_to_translate>
""".strip()

    try:
        response = client.responses.create(
            model="gpt-5.6",
            reasoning={
                "effort": "medium",
            },
            instructions=TRANSLATION_INSTRUCTIONS,
            input=input_text,
            store=False,
        )

        translated_text = response.output_text.strip()

        if not translated_text:
            raise RuntimeError("OpenAI가 빈 번역문을 반환했습니다.")

    except Exception as error:
        print(f"OpenAI translation error: {error}")
        raise

    # OpenAI 호출이 끝난 후 원문만 대화 기록에 추가한다.
    conversation_history[chat_id].append(
        {
            "speaker": speaker,
            "text": cleaned_text,
        }
    )

    return translated_text


def clear_conversation(chat_id: str) -> None:
    """
    해당 채팅방의 문맥을 초기화한다.
    """

    conversation_history.pop(chat_id, None)
