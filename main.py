from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import threading
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Literal, Optional, TypedDict

import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from openai import OpenAI


# -----------------------------------------------------------------------------
# 기본 설정
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("thai-korea-translator")

app = FastAPI(title="LINE Korean-Thai Translator")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6").strip()

OPENAI_REASONING_EFFORT = os.getenv(
    "OPENAI_REASONING_EFFORT",
    "medium",
).strip().lower()

if OPENAI_REASONING_EFFORT not in {
    "none",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
}:
    OPENAI_REASONING_EFFORT = "medium"


LINE_CHANNEL_SECRET = os.getenv(
    "LINE_CHANNEL_SECRET",
    "",
).strip()

LINE_CHANNEL_ACCESS_TOKEN = os.getenv(
    "LINE_CHANNEL_ACCESS_TOKEN",
    "",
).strip()


# 선택 사항:
# 두 사람의 LINE userId를 Render 환경변수에 넣으면
# 영어·숫자·이모지만 있는 메시지도 화자를 기준으로
# 번역 방향을 판단할 수 있다.
J_LINE_USER_ID = os.getenv(
    "J_LINE_USER_ID",
    "",
).strip()

LOVE_LINE_USER_ID = os.getenv(
    "LOVE_LINE_USER_ID",
    "",
).strip()


MAX_HISTORY_ITEMS = int(
    os.getenv("MAX_HISTORY_ITEMS", "10")
)

MAX_HISTORY_TEXT_CHARS = int(
    os.getenv("MAX_HISTORY_TEXT_CHARS", "800")
)

MAX_CONTEXT_CHARS = int(
    os.getenv("MAX_CONTEXT_CHARS", "6000")
)

MAX_LINE_TEXT_CHARS = 5000


RESET_COMMANDS = {
    "/reset",
    "/clear",
    "/forget",
    "reset",
    "clear",
    "forget",
    "초기화",
    "문맥초기화",
    "기억초기화",
    "รีเซ็ต",
    "ล้าง",
}


Direction = Literal[
    "ko_to_th",
    "th_to_ko",
]

Speaker = Literal[
    "J",
    "LOVE",
]


class HistoryMessage(TypedDict):
    speaker: Speaker
    language: Literal["Korean", "Thai"]
    text: str


# 채팅방별로 최근 원문 메시지를 저장한다.
# 번역봇이 만든 번역문은 저장하지 않는다.
CHAT_HISTORY: Dict[
    str,
    Deque[HistoryMessage],
] = defaultdict(
    lambda: deque(maxlen=MAX_HISTORY_ITEMS)
)


# 같은 채팅방에서 메시지가 동시에 들어왔을 때
# 문맥 순서가 뒤섞이지 않게 잠금을 건다.
CHAT_LOCKS: Dict[
    str,
    threading.Lock,
] = {}


STATE_LOCK = threading.RLock()


# LINE이 같은 webhook 이벤트를 다시 보내는 경우
# 중복 번역을 방지하기 위한 메모리 캐시다.
SEEN_EVENT_IDS: Deque[str] = deque(
    maxlen=1000
)

SEEN_EVENT_ID_SET: set[str] = set()


LINE_REPLY_URL = (
    "https://api.line.me/v2/bot/message/reply"
)


openai_client: Optional[OpenAI]

if OPENAI_API_KEY:
    openai_client = OpenAI(
        api_key=OPENAI_API_KEY,
        timeout=45.0,
        max_retries=1,
    )
else:
    openai_client = None
    logger.warning(
        "OPENAI_API_KEY가 설정되지 않았습니다."
    )


# -----------------------------------------------------------------------------
# 번역 지침
# -----------------------------------------------------------------------------

TRANSLATION_INSTRUCTIONS = r"""
너는 한국인 남자친구 J와 태국인 여자친구 LOVE 사이의 메시지를 번역하는
한국어↔태국어 전문 번역가다.

목표는 단순히 단어를 직역하는 것이 아니다.
원문의 실제 의미를 바꾸지 않으면서 두 연인이 정확하고 자연스럽게
이해할 수 있도록 번역해야 한다.

[대화 참여자]

J:
- 한국인 남자친구
- 주로 한국어로 말한다.
- 태국인 여자친구 LOVE에게 말한다.

LOVE:
- 태국인 여자친구
- 주로 태국어로 말한다.
- 한국인 남자친구 J에게 말한다.


[입력 구조]

<recent_conversation>
직전 원문 대화가 들어 있다.

이전 대화는 다음 내용을 파악하는 용도로만 사용한다.

- 생략된 주어
- 생략된 목적어
- 그거, 이거, 그렇게 같은 지시어
- 누가 누구에게 말하는지
- 장난인지 진지한 말인지
- 질문인지 수사적인 표현인지
- 앞 문장과 이어지는 조건이나 부정
- 연인 사이의 감정과 말투

<message_to_translate>
이번에 실제로 번역해야 하는 최신 원문이다.


[가장 중요한 출력 규칙]

1. <message_to_translate> 안의 최신 메시지 하나만 번역한다.

2. <recent_conversation>에 있는 이전 문장을 다시 번역하거나
   출력하지 않는다.

3. 최종 번역문만 출력한다.

4. 설명, 해설, 머리말, 언어 이름, 따옴표, 주석을 붙이지 않는다.

5. 여러 번역 후보를 출력하지 않는다.

6. 원문을 그대로 반복하지 않는다.


[의미 보존 규칙]

1. 주어와 목적어를 절대 뒤바꾸지 않는다.

2. 행동을 하는 사람과 행동을 받는 사람을 바꾸지 않는다.

3. 내가, 네가, 우리가, 그 사람이 누구인지 문맥으로 정확히 판단한다.

4. 소유 관계를 절대 뒤바꾸지 않는다.

5. 다음 내용을 절대 반대로 번역하지 않는다.

- 긍정과 부정
- 가능과 불가능
- 허락과 금지
- 과거와 미래
- 질문과 명령
- 조건과 결과
- 원인과 결과
- 비교 관계
- 수량
- 예외 조건

6. 원문에 없는 뜻을 추가하지 않는다.

7. 원문에 없는 비난, 통제, 화냄, 사과, 질투, 약속,
   애정 표현을 새로 만들지 않는다.

8. 원문에 있는 중요한 뜻을 빼지 않는다.

9. 설명문을 명령문으로 바꾸지 않는다.

10. 수사적인 질문을 실제 질문으로 바꾸지 않는다.

11. 실제 질문을 비난이나 명령처럼 바꾸지 않는다.

12. 원문이 애매하면 문맥으로 확실한 부분만 보완한다.

13. 문맥으로도 확실하지 않으면 함부로 뜻을 단정하지 않는다.

14. 번역하기 전에 문장 전체 구조를 먼저 파악한다.

15. 출력하기 전에 번역문을 속으로 역번역해서
    원문과 의미가 같은지 다시 확인한다.


[말투와 분위기]

1. 연인 사이의 자연스러운 대화체로 번역한다.

2. 직역 때문에 부자연스러워지는 경우 자연스럽게 다듬되,
   원래 의미는 절대 바꾸지 않는다.

3. 원문의 감정 강도를 그대로 유지한다.

4. 장난, 놀림, 삐침, 서운함, 애교, 농담을 유지한다.

5. 욕설, 은어, 성적인 표현이 있어도 검열하거나
   임의로 순화하지 않는다.

6. 555, 이모지, 반복되는 물음표와 느낌표를 가능한 한 유지한다.

7. 원문이 짧으면 번역도 불필요하게 길게 만들지 않는다.

8. 원문이 무뚝뚝하면 지나치게 다정하게 바꾸지 않는다.

9. 원문이 다정하면 딱딱하게 바꾸지 않는다.


[한국어에서 태국어로 번역할 때]

1. 한국인 남자친구가 태국인 여자친구에게 말하는
   자연스러운 태국어로 번역한다.

2. '여보'는 문맥에 맞으면 'ที่รัก'으로 번역한다.

3. 남성 화자라는 이유만으로 모든 문장 끝에
   ครับ을 습관적으로 붙이지 않는다.

4. 원문의 말투가 정중하거나 부드러울 때만
   자연스럽게 남성 말투를 반영한다.

5. 한국어에서 생략된 주어와 목적어는
   직전 대화로 판단한다.


[태국어에서 한국어로 번역할 때]

1. 태국인 여자친구가 한국인 남자친구에게 말하는
   자연스러운 한국어로 번역한다.

2. 'ที่รัก'은 문맥에 맞으면 '여보'로 번역한다.

3. ค่ะ, คะ, นะคะ가 가진 부드러운 느낌을
   자연스러운 한국어 말투로 살린다.

4. 태국어에서 생략된 주어와 목적어는
   직전 대화로 판단한다.

5. 태국어 어순을 그대로 옮겨 어색한 명령문이나
   반대 의미를 만들지 않는다.


[특히 자주 확인해야 하는 한국어 표현]

- 안
- 못
- 아니다
- 없다
- 하지 마
- 한 적 없다
- 아니면
- 하지만
- 그런데
- -면
- -지만
- -는 것만 아니면
- -기만 하면
- 사놓고 안 쓰다
- 필요 없는 것을 사다
- 쇼핑하지 말다


[특히 자주 확인해야 하는 태국어 표현]

- ไม่
- ไม่ได้
- ไม่เคย
- อย่า
- ถ้า
- ถ้าไม่
- แค่
- เท่านั้น
- ก็
- แล้ว
- อยู่แล้ว
- สักหน่อย
- นี่นา
- หรอก
- ทำไม


[반드시 구분해야 하는 의미]

- 쇼핑 자체를 하지 말라는 뜻
- 필요 없는 물건만 사지 말라는 뜻
- 사놓고 사용하지 않는 물건만 사지 말라는 뜻

이 세 가지는 서로 다른 뜻이다.
절대로 같은 뜻으로 번역하지 않는다.


[올바른 의미 구분 예시 1]

한국어 원문:
안 쓸 건데 사는 것만 아니면 됐지 뭐

올바른 태국어:
แค่ไม่ซื้อของที่ซื้อมาแล้วไม่ใช้ก็พอ

잘못된 태국어:
ไม่ใช้ก็ได้ แค่ไม่ซื้อก็พอแล้ว

잘못된 번역은
'안 써도 되고 그냥 사지만 않으면 된다'는 뜻으로 바뀌므로
절대 사용하지 않는다.


[올바른 의미 구분 예시 2]

태국어 원문:
ซื้อของที่ใช้ปกติอยู่แล้ว

올바른 한국어:
평소에도 원래 쓰는 물건을 산 거예요.

잘못된 한국어:
평소에 사용하던 물건을 사라.

원문은 설명문이지 명령문이 아니다.


[올바른 의미 구분 예시 3]

한국어 원문:
내가 옷 샀다고 쓸데없는 거 샀다고 뭐라고 했냐?

올바른 태국어:
ฉันเคยว่าอะไรที่รักไหมว่าซื้อเสื้อผ้าเป็นการซื้อของที่ไม่จำเป็น?

원문은 상대방에게 옷을 사지 말라고 명령하는 문장이 아니다.


[올바른 의미 구분 예시 4]

태국어 원문:
ถ้าซื้อมาแล้วไม่ใช้จะซื้อมาทำไม

올바른 한국어:
사놓고 안 쓸 거면 왜 사겠어요?

원문은 실제로 구매 이유를 묻는 단순한 질문이라기보다,
안 쓸 물건을 자신이 살 이유가 없다는 뜻이다.


[최종 검수]

최종 번역을 출력하기 전에 반드시 다음을 확인한다.

- 부정이 반대로 바뀌지 않았는가?
- 조건이 반대로 바뀌지 않았는가?
- 주어와 목적어가 바뀌지 않았는가?
- 명령문으로 잘못 바뀌지 않았는가?
- 원문에 없는 비난이 추가되지 않았는가?
- 원문에 없는 사과가 추가되지 않았는가?
- 이전 대화의 잘못된 내용을 최신 문장에 섞지 않았는가?
- 최신 메시지 하나만 번역했는가?
- 번역문 외의 설명을 출력하지 않았는가?
""".strip()


# -----------------------------------------------------------------------------
# 언어·화자·문맥 처리
# -----------------------------------------------------------------------------

KOREAN_RE = re.compile(
    r"[가-힣]"
)

THAI_RE = re.compile(
    r"[\u0E00-\u0E7F]"
)

TRANSLATION_MARKER_RE = re.compile(
    r"\s*(ㅂㅂ|ㅂ)\s*$"
)


def normalize_text(text: str) -> str:
    """
    LINE 메시지에 섞일 수 있는
    보이지 않는 문자를 제거한다.
    """
    return text.replace(
        "\u200b",
        "",
    ).strip()


def parse_translation_marker(
    text: str,
) -> tuple[str, Optional[Direction]]:
    """
    선택 기능:

    문장 끝 ㅂ:
    한국어에서 태국어로 강제 번역

    문장 끝 ㅂㅂ:
    태국어에서 한국어로 강제 번역
    """

    match = TRANSLATION_MARKER_RE.search(
        text
    )

    if not match:
        return text, None

    marker = match.group(1)

    cleaned_text = text[
        :match.start()
    ].rstrip()

    if marker == "ㅂㅂ":
        forced_direction: Direction = "th_to_ko"
    else:
        forced_direction = "ko_to_th"

    return cleaned_text, forced_direction


def infer_speaker(
    event: dict[str, Any],
) -> Optional[Speaker]:
    """
    Render 환경변수에 LINE userId가 등록돼 있으면
    실제 메시지를 보낸 사람을 확인한다.
    """

    source = event.get(
        "source",
        {},
    )

    user_id = source.get(
        "userId",
        "",
    )

    if (
        J_LINE_USER_ID
        and user_id == J_LINE_USER_ID
    ):
        return "J"

    if (
        LOVE_LINE_USER_ID
        and user_id == LOVE_LINE_USER_ID
    ):
        return "LOVE"

    return None


def detect_direction(
    text: str,
    speaker_hint: Optional[Speaker] = None,
    forced_direction: Optional[Direction] = None,
) -> Optional[Direction]:
    """
    한글과 태국 문자의 수를 확인해
    번역 방향을 자동으로 판단한다.
    """

    if forced_direction is not None:
        return forced_direction

    korean_count = len(
        KOREAN_RE.findall(text)
    )

    thai_count = len(
        THAI_RE.findall(text)
    )

    if korean_count > thai_count:
        return "ko_to_th"

    if thai_count > korean_count:
        return "th_to_ko"

    # 한국어와 태국어가 같은 수로 섞여 있고
    # 한글이 하나라도 있으면 한국어로 판단한다.
    if korean_count > 0:
        return "ko_to_th"

    # 영어·숫자·이모지만 있는 경우에는
    # LINE userId가 등록돼 있을 때만 방향을 판단한다.
    if speaker_hint == "J":
        return "ko_to_th"

    if speaker_hint == "LOVE":
        return "th_to_ko"

    return None


def speaker_for_direction(
    direction: Direction,
    speaker_hint: Optional[Speaker] = None,
) -> Speaker:
    """
    메시지의 화자를 정한다.

    한국어 메시지:
    기본적으로 J

    태국어 메시지:
    기본적으로 LOVE
    """

    expected_speaker: Speaker

    if direction == "ko_to_th":
        expected_speaker = "J"
    else:
        expected_speaker = "LOVE"

    if speaker_hint == expected_speaker:
        return speaker_hint

    return expected_speaker


def get_chat_id(
    event: dict[str, Any],
) -> str:
    """
    그룹채팅이면 groupId,
    방이면 roomId,
    개인채팅이면 userId를 사용한다.
    """

    source = event.get(
        "source",
        {},
    )

    source_type = source.get(
        "type",
        "",
    )

    if source_type == "group":
        return (
            source.get("groupId")
            or "group_default"
        )

    if source_type == "room":
        return (
            source.get("roomId")
            or "room_default"
        )

    if source_type == "user":
        return (
            source.get("userId")
            or "user_default"
        )

    return "default"


def get_chat_lock(
    chat_id: str,
) -> threading.Lock:
    """
    채팅방별 잠금을 가져온다.
    """

    with STATE_LOCK:
        lock = CHAT_LOCKS.get(
            chat_id
        )

        if lock is None:
            lock = threading.Lock()
            CHAT_LOCKS[chat_id] = lock

        return lock


def shorten_history_text(
    text: str,
) -> str:
    """
    문맥에 저장되는 원문이 지나치게 길어지는 것을 막는다.
    """

    clean_text = " ".join(
        text.split()
    )

    if (
        len(clean_text)
        <= MAX_HISTORY_TEXT_CHARS
    ):
        return clean_text

    return (
        clean_text[
            :MAX_HISTORY_TEXT_CHARS
        ]
        + "…"
    )


def add_history(
    chat_id: str,
    speaker: Speaker,
    direction: Direction,
    text: str,
) -> None:
    """
    사용자가 보낸 원문만 저장한다.

    OpenAI가 만든 번역문은
    이 함수에 절대 전달하지 않는다.
    """

    language: Literal[
        "Korean",
        "Thai",
    ]

    if direction == "ko_to_th":
        language = "Korean"
    else:
        language = "Thai"

    history_message: HistoryMessage = {
        "speaker": speaker,
        "language": language,
        "text": shorten_history_text(text),
    }

    with STATE_LOCK:
        CHAT_HISTORY[
            chat_id
        ].append(
            history_message
        )


def clear_history(
    chat_id: str,
) -> None:
    """
    해당 채팅방의 문맥을 초기화한다.
    """

    with STATE_LOCK:
        CHAT_HISTORY.pop(
            chat_id,
            None,
        )


def get_recent_context(
    chat_id: str,
) -> str:
    """
    최근 원문 메시지를 OpenAI에 전달할
    문맥 형식으로 만든다.
    """

    with STATE_LOCK:
        history = list(
            CHAT_HISTORY.get(
                chat_id,
                (),
            )
        )

    if not history:
        return "(이전 원문 대화 없음)"

    context_lines = []

    for item in history:
        context_lines.append(
            f"{item['speaker']} "
            f"[{item['language']}]: "
            f"{item['text']}"
        )

    context = "\n".join(
        context_lines
    )

    if len(context) <= MAX_CONTEXT_CHARS:
        return context

    # 제한을 넘는 경우
    # 최신 문맥을 우선해서 보낸다.
    return context[
        -MAX_CONTEXT_CHARS:
    ]


# -----------------------------------------------------------------------------
# OpenAI 번역
# -----------------------------------------------------------------------------

def model_supports_reasoning(
    model: str,
) -> bool:
    """
    GPT-5 계열과 o 계열 모델에만
    reasoning 옵션을 전달한다.
    """

    normalized_model = model.lower()

    return (
        normalized_model.startswith("gpt-5")
        or normalized_model.startswith("o")
    )


def build_translation_input(
    *,
    text: str,
    direction: Direction,
    speaker: Speaker,
    recent_context: str,
) -> str:
    """
    최근 문맥과 현재 원문을
    명확하게 구분해서 전달한다.
    """

    if direction == "ko_to_th":
        source_language = "Korean"
        target_language = "Thai"
    else:
        source_language = "Thai"
        target_language = "Korean"

    return f"""
<translation_information>
speaker: {speaker}
source_language: {source_language}
target_language: {target_language}
</translation_information>

<recent_conversation>
{recent_context}
</recent_conversation>

<message_to_translate>
{text}
</message_to_translate>
""".strip()


def translate_text(
    *,
    text: str,
    chat_id: str,
    direction: Direction,
    speaker: Speaker,
) -> str:
    """
    OpenAI Responses API를 사용해
    최신 원문 하나를 번역한다.
    """

    if openai_client is None:
        raise RuntimeError(
            "OPENAI_API_KEY가 설정되지 않았습니다."
        )

    recent_context = get_recent_context(
        chat_id
    )

    user_input = build_translation_input(
        text=text,
        direction=direction,
        speaker=speaker,
        recent_context=recent_context,
    )

    request_args: dict[
        str,
        Any,
    ] = {
        "model": OPENAI_MODEL,
        "instructions": TRANSLATION_INSTRUCTIONS,
        "input": user_input,
        "max_output_tokens": 1500,
        "store": False,
    }

    if model_supports_reasoning(
        OPENAI_MODEL
    ):
        request_args["reasoning"] = {
            "effort": OPENAI_REASONING_EFFORT
        }

    response = openai_client.responses.create(
        **request_args
    )

    translated_text = (
        response.output_text
        or ""
    ).strip()

    if not translated_text:
        raise RuntimeError(
            "OpenAI가 빈 번역문을 반환했습니다."
        )

    return translated_text


# -----------------------------------------------------------------------------
# LINE 처리
# -----------------------------------------------------------------------------

def verify_line_signature(
    body: bytes,
    signature: str,
) -> bool:
    """
    webhook 요청이 실제 LINE에서 온 것인지 확인한다.
    """

    if (
        not LINE_CHANNEL_SECRET
        or not signature
    ):
        return False

    expected_signature = base64.b64encode(
        hmac.new(
            LINE_CHANNEL_SECRET.encode(
                "utf-8"
            ),
            body,
            hashlib.sha256,
        ).digest()
    ).decode(
        "utf-8"
    )

    return hmac.compare_digest(
        signature,
        expected_signature,
    )


def reply_line(
    reply_token: str,
    text: str,
) -> bool:
    """
    LINE Messaging API로 번역문을 답장한다.
    """

    if not reply_token:
        logger.error(
            "LINE replyToken이 없습니다."
        )
        return False

    if not LINE_CHANNEL_ACCESS_TOKEN:
        logger.error(
            "LINE_CHANNEL_ACCESS_TOKEN이 설정되지 않았습니다."
        )
        return False

    reply_text = text.strip()

    if not reply_text:
        return False

    if (
        len(reply_text)
        > MAX_LINE_TEXT_CHARS
    ):
        reply_text = reply_text[
            :MAX_LINE_TEXT_CHARS
        ]

    headers = {
        "Content-Type": "application/json",
        "Authorization": (
            f"Bearer "
            f"{LINE_CHANNEL_ACCESS_TOKEN}"
        ),
    }

    payload = {
        "replyToken": reply_token,
        "messages": [
            {
                "type": "text",
                "text": reply_text,
            }
        ],
    }

    try:
        response = requests.post(
            LINE_REPLY_URL,
            headers=headers,
            json=payload,
            timeout=(5, 15),
        )

        response.raise_for_status()
        return True

    except requests.RequestException as error:
        response_text = ""

        if error.response is not None:
            response_text = error.response.text

        logger.error(
            "LINE 답장 실패: %s %s",
            error,
            response_text,
        )

        return False


def is_duplicate_event(
    event: dict[str, Any],
) -> bool:
    """
    LINE에서 같은 webhook 이벤트가 재전송될 경우
    같은 번역을 두 번 답장하지 않도록 한다.
    """

    event_id = str(
        event.get(
            "webhookEventId",
            "",
        )
    ).strip()

    if not event_id:
        return False

    with STATE_LOCK:
        if event_id in SEEN_EVENT_ID_SET:
            return True

        if (
            len(SEEN_EVENT_IDS)
            == SEEN_EVENT_IDS.maxlen
        ):
            oldest_event_id = (
                SEEN_EVENT_IDS.popleft()
            )

            SEEN_EVENT_ID_SET.discard(
                oldest_event_id
            )

        SEEN_EVENT_IDS.append(
            event_id
        )

        SEEN_EVENT_ID_SET.add(
            event_id
        )

    return False


def process_text_event(
    event: dict[str, Any],
) -> None:
    """
    LINE의 텍스트 메시지 이벤트 하나를 처리한다.
    """

    if event.get("type") != "message":
        return

    message = event.get(
        "message",
        {},
    )

    if message.get("type") != "text":
        return

    if is_duplicate_event(event):
        logger.info(
            "중복 webhook 이벤트를 건너뜁니다."
        )
        return

    reply_token = str(
        event.get(
            "replyToken",
            "",
        )
    )

    original_text = normalize_text(
        str(
            message.get(
                "text",
                "",
            )
        )
    )

    if not original_text:
        return

    chat_id = get_chat_id(
        event
    )

    speaker_hint = infer_speaker(
        event
    )

    chat_lock = get_chat_lock(
        chat_id
    )

    # 같은 채팅방에서 메시지가 동시에 들어와도
    # 대화 문맥 순서가 유지되게 한다.
    with chat_lock:
        if (
            original_text.casefold()
            in RESET_COMMANDS
        ):
            clear_history(
                chat_id
            )

            reply_line(
                reply_token,
                (
                    "문맥 기억을 초기화했어요.\n"
                    "ล้างความจำบริบทแล้ว"
                ),
            )
            return

        (
            text_to_translate,
            forced_direction,
        ) = parse_translation_marker(
            original_text
        )

        if not text_to_translate:
            return

        direction = detect_direction(
            text_to_translate,
            speaker_hint=speaker_hint,
            forced_direction=forced_direction,
        )

        if direction is None:
            # 한국어·태국어가 없고
            # userId도 등록되지 않은 메시지는
            # 번역 방향을 알 수 없으므로 무시한다.
            return

        speaker = speaker_for_direction(
            direction,
            speaker_hint,
        )

        try:
            translated_text = translate_text(
                text=text_to_translate,
                chat_id=chat_id,
                direction=direction,
                speaker=speaker,
            )

            # 가장 중요:
            # 번역봇이 만든 번역문은 저장하지 않는다.
            # 사용자가 실제로 보낸 원문만 저장한다.
            add_history(
                chat_id=chat_id,
                speaker=speaker,
                direction=direction,
                text=text_to_translate,
            )

            reply_line(
                reply_token,
                translated_text,
            )

        except Exception:
            logger.exception(
                "번역 처리 중 오류가 발생했습니다."
            )

            reply_line(
                reply_token,
                (
                    "번역 오류가 발생했어요. "
                    "잠시 후 다시 보내주세요.\n"
                    "เกิดข้อผิดพลาดในการแปล "
                    "กรุณาส่งใหม่อีกครั้งสักครู่"
                ),
            )


def process_events(
    events: list[dict[str, Any]],
) -> None:
    """
    webhook에 들어온 여러 LINE 이벤트를 처리한다.
    """

    for event in events:
        process_text_event(
            event
        )


# -----------------------------------------------------------------------------
# FastAPI 경로
# -----------------------------------------------------------------------------

@app.get("/")
def home() -> dict[str, Any]:
    """
    Render에서 서버가 살아 있는지 확인하는 주소다.
    """

    return {
        "status": "ok",
        "service": (
            "LINE Korean-Thai translator"
        ),
        "model": OPENAI_MODEL,
        "openai_key_configured": bool(
            OPENAI_API_KEY
        ),
        "line_secret_configured": bool(
            LINE_CHANNEL_SECRET
        ),
        "line_token_configured": bool(
            LINE_CHANNEL_ACCESS_TOKEN
        ),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok"
    }


@app.post("/webhook")
async def webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    """
    LINE webhook 주소다.
    """

    body = await request.body()

    signature = request.headers.get(
        "x-line-signature",
        "",
    )

    if not verify_line_signature(
        body,
        signature,
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid signature",
        )

    try:
        payload = json.loads(
            body.decode("utf-8")
        )

    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON",
        ) from error

    events = payload.get(
        "events",
        [],
    )

    if not isinstance(
        events,
        list,
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid events",
        )

    # LINE에는 즉시 성공 응답을 보낸다.
    # OpenAI 번역과 LINE 답장은 백그라운드에서 처리한다.
    background_tasks.add_task(
        process_events,
        events,
    )

    return {
        "status": "ok"
    }
