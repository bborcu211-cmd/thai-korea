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
        relationship = "The source message is usually from a Korean boyfriend to his Thai girlfriend."
        style_rules = """
Write natural Thai used in LINE couple chat.
Use masculine polite Thai such as ครับ when natural.
Translate 여보 as ที่รัก when natural.
Translate 자기 as ที่รัก, ตัวเอง, or another natural Thai expression depending on context.
Do not make it sound like business Thai.
"""
    else:
        target_language = "Korean"
        relationship = "The source message is usually from a Thai girlfriend to her Korean boyfriend."
        style_rules = """
Write natural Korean used in LINE couple chat.
Reflect ค่ะ and คะ as a soft feminine tone when natural.
Translate ที่รัก as 여보 when natural.
Do not make it stiff, dry, or textbook-like.
"""

    return f"""
You are a careful translator for private romantic LINE chat.

Task:
Translate only the CURRENT message into {target_language}.
Use the recent conversation context only to resolve ambiguity.
Do not translate, summarize, or repeat the context.

Relationship:
{relationship}

Priority:
1. Preserve exact meaning.
2. Preserve subject, object, tense, negation, question form, cause-result relation, and emotional intention.
3. Make the result natural for couple chat.
4. Do not add details that are not in the source.

Translation rules:
- Understand the whole sentence before translating.
- Do not translate word by word when it changes the intended meaning.
- Do not simplify, summarize, exaggerate, or reinterpret.
- If the source is ambiguous, keep it neutral instead of guessing too much.
- Preserve 555, emojis, laughter, punctuation, teasing, joking, sulking, and playful complaints.
- Do not change message/contact into phone call unless the source clearly says phone/call.
- Do not turn vague words into specific actions unless the source clearly says them.
- Preserve cause-result structures such as "because A, so B", "-라서", "-니까", "เพราะ", and "เลย".

Korean caution:
- Korean chat spacing can be informal or wrong. Do not split a natural Korean ending into a negative meaning unless it is clearly negative.
- For example, "안답니다" can mean "알고 있어요 / 알아요" in context, not necessarily "안 답니다".
- Korean particles 은/는, 이/가, 을/를 often show focus. Preserve what is actually being talked about.
- If Korean says a place is missed, the place is the thing being missed. Do not turn it into missing a person.
- Preserve past tense such as -었어요, -였어요, 이었어요, 했어요.

Thai caution:
- Thai relationship chat often uses flexible self-reference, lover-reference, nicknames, kinship words, and particles.
- Do not translate kinship words literally unless the context clearly means family.
- Thai classifiers such as ตัว, อัน, เรื่อง, คน can refer to omitted nouns from previous context.
- ตัวเดียว can mean one item/piece if the context is clothes or things, not "alone".
- Particles like นะ, ล่ะ, เนี่ย, นี่นา, สิ, อะ, อ่ะ carry emotion. Reflect the feeling naturally.
- For social-media, TikTok, filter, or trend expressions, translate neutrally if unclear.

Style:
{style_rules}

Output only the translation.
"""


def get_review_prompt(direction):
    if direction == "ko_to_th":
        target_language = "Thai"
    else:
        target_language = "Korean"

    return f"""
You are a strict translation reviewer.

Your job:
Compare the source message and draft translation.
If the draft is accurate, output it unchanged.
If the draft changes meaning, rewrite it into accurate natural {target_language}.

Check these problems carefully:
- wrong tense
- wrong negation
- wrong subject or object
- missing question form
- missing cause-result relation
- confusing person/place/action/item
- translating message/contact as phone call without evidence
- translating family/kinship words literally when context does not mean family
- over-interpreting an ambiguous phrase
- adding details not in the source
- making romantic chat too stiff or formal
- dropping 555, emoji, joke, teasing, or emotional tone

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
Recent conversation context:
{recent_context}

Current message:
{text}
"""

    draft_translation = call_openai(translation_prompt, translation_user_prompt)

    review_prompt = get_review_prompt(direction)
    review_user_prompt = f"""
Recent conversation context:
{recent_context}

Source message:
{text}

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
