import os
import re
import hmac
import json
import hashlib
import base64
import requests
from collections import defaultdict, deque

from fastapi import FastAPI, Request, HTTPException
from openai import OpenAI

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

MAX_HISTORY_ITEMS = 10
MAX_CONTEXT_CHARS_PER_MESSAGE = 500

# 채팅방별 최근 대화 기억
# Render 무료 서버가 잠들거나 재배포되면 초기화됩니다.
CHAT_HISTORY = defaultdict(lambda: deque(maxlen=MAX_HISTORY_ITEMS))

RESET_COMMANDS = {
    "/reset",
    "/clear",
    "/forget",
    "reset",
    "clear"
}


def count_korean(text):
    return len(re.findall(r"[가-힣]", text))


def count_thai(text):
    return len(re.findall(r"[\u0E00-\u0E7F]", text))


def detect_direction(text):
    korean_count = count_korean(text)
    thai_count = count_thai(text)

    if korean_count == 0 and thai_count == 0:
        return None

    if korean_count >= thai_count:
        return "ko_to_th"

    return "th_to_ko"


def get_chat_id(event):
    source = event.get("source", {})
    source_type = source.get("type", "")

    if source_type == "group":
        return source.get("groupId", "group_default")

    if source_type == "room":
        return source.get("roomId", "room_default")

    if source_type == "user":
        return source.get("userId", "user_default")

    return "default"


def get_sender_label(event):
    source = event.get("source", {})
    user_id = source.get("userId", "unknown")

    if user_id == "unknown":
        return "speaker_unknown"

    return "speaker_" + user_id[-6:]


def shorten_text(text, limit=MAX_CONTEXT_CHARS_PER_MESSAGE):
    clean = " ".join(text.split())

    if len(clean) <= limit:
        return clean

    return clean[:limit] + "..."


def add_history(chat_id, sender_label, language_label, text):
    if not text:
        return

    CHAT_HISTORY[chat_id].append({
        "sender": sender_label,
        "language": language_label,
        "text": shorten_text(text)
    })


def clear_history(chat_id):
    if chat_id in CHAT_HISTORY:
        CHAT_HISTORY[chat_id].clear()


def get_recent_context(chat_id):
    history = CHAT_HISTORY.get(chat_id)

    if not history:
        return "No previous context."

    lines = []

    for item in history:
        sender = item.get("sender", "speaker_unknown")
        language = item.get("language", "unknown")
        text = item.get("text", "")
        lines.append(f"{sender} [{language}]: {text}")

    return "\n".join(lines)


def get_language_label(direction):
    if direction == "ko_to_th":
        return "Korean"

    if direction == "th_to_ko":
        return "Thai"

    return "Unknown"


def get_korean_to_thai_prompt():
    return """
You are a careful Korean-to-Thai translator for private romantic LINE chat.

Translate only the CURRENT Korean message into Thai.
Use the recent conversation context only to resolve ambiguity.
Do not translate or summarize the context.

Accuracy comes first:
- Understand the whole sentence before translating.
- Preserve the original meaning, subject, object, tense, question form, and emotional intention.
- Do not simplify, summarize, exaggerate, or reinterpret the message.
- Do not add new information that is not in the original message.
- If the meaning is ambiguous, choose a neutral translation that keeps the ambiguity instead of guessing too much.

Naturalness comes second:
- Make the Thai sound natural for a real LINE conversation between lovers.
- The speaker is a Korean man talking to his Thai girlfriend.
- Use masculine polite Thai such as ครับ only when it sounds natural.
- Preserve joking, teasing, sulking, playful complaints, dialect-like endings, laughter, 555, emojis, and punctuation.
- Keep short casual messages short and casual.

Korean grammar and nuance:
- Preserve past tense, present tense, future intention, negation, and questions.
- Korean casual or idiomatic expressions should be translated by intended meaning, not word by word.
- Korean particles such as 은/는, 이/가, 을/를 often show focus. Preserve what the sentence is really about.
- If the Korean sentence misses a place, translate the place as the thing being missed. If it misses a person, translate the person as the thing being missed. Do not confuse the two.
- If the Korean sentence is a confirming question, keep it as a confirming question in Thai.

Relationship tone:
- Translate 여보 naturally as ที่รัก when appropriate.
- Translate 자기 naturally as ที่รัก, ตัวเอง, or another natural Thai expression depending on context.
- Do not make romantic chat sound like business Thai.

Output only the Thai translation.
"""


def get_thai_to_korean_prompt():
    return """
You are a careful Thai-to-Korean translator for private romantic LINE chat.

Translate only the CURRENT Thai message into Korean.
Use the recent conversation context only to resolve ambiguity.
Do not translate or summarize the context.

Accuracy comes first:
- Understand the whole sentence before translating.
- Preserve the original meaning, subject, object, tense, question form, and emotional intention.
- Do not simplify, summarize, exaggerate, or reinterpret the message.
- Do not add new information that is not in the original message.
- If the meaning is ambiguous, choose a neutral translation that keeps the ambiguity instead of guessing too much.

Naturalness comes second:
- Make the Korean sound natural for a real LINE conversation between lovers.
- The speaker is a Thai woman talking to her Korean boyfriend.
- Preserve soft feminine tone from ค่ะ and คะ naturally, but do not overdo it.
- Preserve joking, teasing, sulking, playful complaints, laughter, 555, emojis, and punctuation.
- Keep short casual messages short and casual.

Thai grammar and nuance:
- Thai relationship chat often uses flexible self-reference, lover-reference, kinship words, nicknames, and particles. Infer the role from context instead of defaulting to a literal family meaning.
- Particles such as นะ, ล่ะ, เนี่ย, นี่นา, สิ, อะ, อ่ะ carry emotion. Preserve the feeling naturally instead of translating them word by word.
- Preserve the exact target of verbs like คิดถึง, ชอบ, อยาก, เป็นห่วง, and ไม่อยาก. Do not confuse a person, place, action, or situation.
- For app slang, filters, trends, or unclear TikTok/social-media expressions, use a neutral translation instead of over-interpreting.

Relationship tone:
- Translate ที่รัก naturally as 여보 when appropriate.
- Do not make romantic chat sound stiff, dry, or formal.

Output only the Korean translation.
"""


def translate_text(text, chat_id):
    direction = detect_direction(text)

    if direction is None:
        return None

    if direction == "ko_to_th":
        system_prompt = get_korean_to_thai_prompt()
    else:
        system_prompt = get_thai_to_korean_prompt()

    recent_context = get_recent_context(chat_id)

    user_prompt = f"""
Recent conversation context:
{recent_context}

Current message to translate:
{text}
"""

    result = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt.strip()
            },
            {
                "role": "user",
                "content": user_prompt.strip()
            }
        ],
        temperature=0.2
    )

    return result.choices[0].message.content.strip()


def reply_line(reply_token, text):
    if not reply_token:
        return False

    if not LINE_CHANNEL_ACCESS_TOKEN:
        print("LINE_CHANNEL_ACCESS_TOKEN is missing.")
        return False

    url = "https://api.line.me/v2/bot/message/reply"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"
    }

    data = {
        "replyToken": reply_token,
        "messages": [
            {
                "type": "text",
                "text": text
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
    except Exception as error:
        print("LINE reply request failed:", str(error))
        return False

    if response.status_code >= 400:
        print("LINE reply error:", response.status_code, response.text)
        return False

    return True


def verify_line_signature(body, signature):
    if not LINE_CHANNEL_SECRET:
        print("LINE_CHANNEL_SECRET is missing.")
        return False

    expected_signature = base64.b64encode(
        hmac.new(
            LINE_CHANNEL_SECRET.encode("utf-8"),
            body,
            hashlib.sha256
        ).digest()
    ).decode("utf-8")

    return hmac.compare_digest(signature, expected_signature)


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "LINE Korean Thai translator bot is running"
    }


@app.post("/webhook")
async def webhook(request: Request):
    body = await request.body()
    signature = request.headers.get("x-line-signature", "")

    if not verify_line_signature(body, signature):
        raise HTTPException(status_code=400, detail="Invalid signature")

    try:
        data = json.loads(body.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    for event in data.get("events", []):
        if event.get("type") != "message":
            continue

        message = event.get("message", {})

        if message.get("type") != "text":
            continue

        user_text = message.get("text", "").strip()
        reply_token = event.get("replyToken", "")
        chat_id = get_chat_id(event)
        sender_label = get_sender_label(event)

        if not user_text:
            continue

        if user_text.lower() in RESET_COMMANDS:
            clear_history(chat_id)
            reply_line(reply_token, "문맥 기억을 초기화했어요.")
            continue

        direction = detect_direction(user_text)

        if direction is None:
            continue

        try:
            translated = translate_text(user_text, chat_id)

            if translated:
                reply_line(reply_token, translated)
                language_label = get_language_label(direction)
                add_history(chat_id, sender_label, language_label, user_text)

        except Exception as error:
            print("Translation error:", str(error))
            reply_line(reply_token, "번역 오류가 발생했어요. 잠시 후 다시 보내주세요.")

    return {
        "status": "ok"
    }
