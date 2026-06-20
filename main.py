import os
import re
import hmac
import json
import base64
import hashlib
import requests
from collections import defaultdict, deque

from fastapi import FastAPI, Request, HTTPException
from openai import OpenAI

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

MAX_HISTORY_ITEMS = 6
MAX_CONTEXT_CHARS = 350

CHAT_HISTORY = defaultdict(lambda: deque(maxlen=MAX_HISTORY_ITEMS))

RESET_COMMANDS = {"/reset", "/clear", "/forget", "reset", "clear"}


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


def get_language_label(direction):
    if direction == "ko_to_th":
        return "Korean"

    if direction == "th_to_ko":
        return "Thai"

    return "Unknown"


def get_role_label(direction):
    if direction == "ko_to_th":
        return "Korean boyfriend"

    if direction == "th_to_ko":
        return "Thai girlfriend"

    return "Unknown speaker"


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


def shorten_text(text, limit=MAX_CONTEXT_CHARS):
    clean = " ".join(text.split())

    if len(clean) <= limit:
        return clean

    return clean[:limit] + "..."


def add_history(chat_id, direction, text):
    if not text:
        return

    CHAT_HISTORY[chat_id].append({
        "role": get_role_label(direction),
        "language": get_language_label(direction),
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
        role = item.get("role", "Unknown speaker")
        language = item.get("language", "Unknown")
        text = item.get("text", "")
        lines.append(f"{role} [{language}]: {text}")

    return "\n".join(lines)


def get_translation_prompt(direction):
    if direction == "ko_to_th":
        target_language = "Thai"
        direction_rules = """
Source language: Korean
Target language: Thai

The speaker is usually a Korean man talking to his Thai girlfriend.
Use Thai male speech only when the source tone naturally needs it.
Use ครับ when the Korean is polite, soft, affectionate, or naturally polite.
Do not add ครับ if the Korean is very short, blunt, or casual unless it sounds necessary.
Translate 여보 as ที่รัก when it appears.
"""
    else:
        target_language = "Korean"
        direction_rules = """
Source language: Thai
Target language: Korean

The speaker is usually a Thai woman talking to her Korean boyfriend.
Reflect ค่ะ, คะ, นะคะ as a soft tone naturally, but do not over-explain them.
Translate ที่รัก as 여보 when it appears.
Keep 555 as 555 unless Korean laughter is clearly more natural.
"""

    return f"""
You are a strict Korean-Thai translator.

Translate the CURRENT message into {target_language} as faithfully and literally as possible.

Most important rule:
Translate only what is written in the source message.
Do not add hidden meaning.
Do not explain.
Do not summarize.
Do not make the message prettier.
Do not make the sentence longer than necessary.
Do not guess the speaker's intention beyond the words.

Meaning rules:
- Preserve the exact meaning.
- Preserve subject, object, tense, negation, question form, and cause-result relation.
- If the original is vague, keep it vague.
- If the original is short, the translation must also be short.
- If the original has no subject, do not add a subject unless the target language requires it.
- Do not turn a message/contact/chat into a phone call unless the source clearly says call/phone.
- Do not change "I", "you", "we", "that", "this", "there", "here" unless the source requires it.
- Do not replace the original with a more emotional or romantic sentence.
- Do not turn a simple sentence into an explanation.

Style rules:
- Preserve the original tone, mood, slang, teasing, joking, sulking, awkwardness, 555, emojis, and punctuation as much as possible.
- Naturalness is secondary to faithfulness.
- A slightly awkward but faithful translation is better than a smooth but changed translation.
- Output only the translation.

Direction rules:
{direction_rules}
"""


def get_review_prompt(direction):
    if direction == "ko_to_th":
        target_language = "Thai"
    else:
        target_language = "Korean"

    return f"""
You are a strict literal translation reviewer.

Compare the source message and draft translation.

Your job:
- If the draft translation accurately preserves the source, output it unchanged.
- If the draft adds meaning, removes meaning, explains too much, or rewrites too freely, fix it.
- The final result must be a faithful {target_language} translation of the source.

Check carefully:
- added details not in source
- missing details from source
- wrong tense
- wrong negation
- wrong subject or object
- wrong question form
- missing cause-result relation
- over-interpreting vague words
- making a short sentence too long
- making romantic chat too polished or too dramatic
- translating chat/message/contact as phone call without evidence
- changing the original emotional tone
- dropping 555, emoji, joke, teasing, or awkwardness

Important:
Faithfulness is more important than smoothness.
Do not explain your correction.
Output only the final translation.
"""

def call_openai(system_prompt, user_prompt):
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
        temperature=0.0
    )

    return result.choices[0].message.content.strip()


def translate_text(text, chat_id):
    direction = detect_direction(text)

    if direction is None:
        return None

    recent_context = get_recent_context(chat_id)

    translation_prompt = get_translation_prompt(direction)
    translation_user_prompt = f"""
Current message:
{text}
"""

Current message:
{text}
"""

    draft_translation = call_openai(translation_prompt, translation_user_prompt)

    review_prompt = get_review_prompt(direction)
    review_user_prompt = f"""
Source message:
{text}

Draft translation:
{draft_translation}
"""

Draft translation:
{draft_translation}
"""

    final_translation = call_openai(review_prompt, review_user_prompt)

    return final_translation


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
                add_history(chat_id, direction, user_text)

        except Exception as error:
            print("Translation error:", str(error))
            reply_line(reply_token, "번역 오류가 발생했어요. 잠시 후 다시 보내주세요.")

    return {
        "status": "ok"
    }
