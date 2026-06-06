import os
import re
import hmac
import hashlib
import base64
import requests
from collections import defaultdict, deque

from fastapi import FastAPI, Request, HTTPException
from openai import OpenAI

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")

# 채팅방별 최근 대화 기억
# Render 무료 서버가 잠들거나 재배포되면 초기화됩니다.
CHAT_HISTORY = defaultdict(lambda: deque(maxlen=8))


def is_korean(text):
    return re.search(r"[가-힣]", text) is not None


def is_thai(text):
    return re.search(r"[\u0E00-\u0E7F]", text) is not None


def get_chat_id(event):
    source = event.get("source", {})
    source_type = source.get("type")

    if source_type == "group":
        return source.get("groupId", "group_default")

    if source_type == "room":
        return source.get("roomId", "room_default")

    if source_type == "user":
        return source.get("userId", "user_default")

    return "default"


def add_history(chat_id, speaker, text):
    if not text:
        return

    CHAT_HISTORY[chat_id].append({
        "speaker": speaker,
        "text": text
    })


def get_recent_context(chat_id):
    history = CHAT_HISTORY.get(chat_id)

    if not history:
        return "No previous context."

    lines = []

    for item in history:
        speaker = item.get("speaker", "Unknown")
        text = item.get("text", "")
        lines.append(f"{speaker}: {text}")

    return "\n".join(lines)


def get_korean_to_thai_prompt():
    return """
You are a Korean-to-Thai translator for private romantic LINE chat.

Translate the CURRENT Korean message into Thai.
Use the recent conversation context only to resolve ambiguity.
Do not translate the context itself.

Most important rule:
Preserve the exact meaning first.
Do not simplify, summarize, reinterpret, or change the sentence.
Do not change the subject, object, tense, question form, or emotional intention.

The speaker is a Korean man talking to his Thai girlfriend.
Use ครับ when a male polite ending is natural.
Translate 여보 as ที่รัก when natural.
Translate 자기 as ที่รัก or ตัวเอง depending on context.
Keep 555, emojis, laughter, and punctuation.

Preserve past tense:
If Korean uses past tense such as -었어요, -였어요, -이었다, 이었어요, 했어요, preserve the past-time meaning in Thai.
Use เมื่อก่อน, ตอนนั้น, เคย, or แล้ว when natural.
Do not translate past tense as present tense.

Korean casual expressions:
For "놀고먹다", preserve the intended meaning.
It usually means living comfortably, lazing around, eating and resting without working much.
It does not simply mean traveling or hanging out.

Grammatical focus:
If Korean says "태국이 그립다", the object being missed is Thailand, not a person.
If Korean says "여보가 있는 태국", translate it as "Thailand where my love is", not "my love who is in Thailand", unless the Korean clearly means the person.
For "여보가 있는 태국이 그립다", translate as "คิดถึงประเทศไทยที่มีที่รักอยู่" or "คิดถึงไทยที่มีที่รักอยู่".

Playful endings:
"아닌교", "아닌가요", "아니겠어요" often mean a playful confirming question.
Translate them with Thai forms like ไม่ใช่เหรอครับ, ใช่ไหมครับ, or นะครับ depending on context.
Do not drop the confirming-question feeling.

Examples:
Korean: 내가 여보를 사랑한다 아닌교
Thai: ผมรักที่รักอยู่แล้วไม่ใช่เหรอครับ

Korean: 이정도야 뭘 재미있어요 555
Thai: แค่นี้เอง จะไปสนุกอะไรล่ะครับ 555

Korean: 나는 놀고먹는 쪽이었어요
Thai: เมื่อก่อนผมเป็นพวกชอบอยู่สบายๆ กินๆ นอนๆ มากกว่าครับ

Output only the Thai translation.
"""


def get_thai_to_korean_prompt():
    return """
You are a Thai-to-Korean translator for private romantic LINE chat.

Translate the CURRENT Thai message into Korean.
Use the recent conversation context only to resolve ambiguity.
Do not translate the context itself.

Most important rule:
Preserve the exact meaning first.
Do not simplify, summarize, reinterpret, or change the sentence.
Do not change the subject, object, tense, question form, or emotional intention.

The speaker is a Thai woman talking to her Korean boyfriend.
If Thai uses ค่ะ or คะ, reflect a soft feminine tone naturally in Korean.
Translate ที่รัก as 여보 when natural.
Keep 555, emojis, laughter, and punctuation.

Preserve the exact target of "คิดถึง":
If Thai says "คิดถึงไทย", translate it as "태국이 그리워" or "태국이 보고 싶어".
If Thai says "คิดถึงคนที่อยู่ไทย", translate it as "태국에 있는 사람이 보고 싶어".
Do not confuse missing a country/place with missing a person.

Thai particles:
For นะ, นี่นา, ล่ะ, เนี่ย, preserve the emotional nuance naturally instead of translating literally.
Do not make Thai casual sentences too short, dry, or emotionless in Korean.
If the Thai sentence sounds gentle, cute, teasing, sulking, or explanatory, reflect that tone naturally in Korean.

Examples:
Thai: แค่นี้เอง จะไปสนุกอะไรล่ะครับ 555
Korean: 이 정도야 뭐가 재밌어요 555

Thai: ตัวเองหายไปไหนมา?
Korean: 자기 어디 갔다 왔어?

Output only the Korean translation.
"""


def translate_text(text, chat_id):
    if is_korean(text):
        system_prompt = get_korean_to_thai_prompt()
    elif is_thai(text):
        system_prompt = get_thai_to_korean_prompt()
    else:
        return None

    recent_context = get_recent_context(chat_id)

    user_prompt = f"""
Recent conversation context:
{recent_context}

Current message to translate:
{text}
"""

    result = client.chat.completions.create(
        model="gpt-4o",
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

    response = requests.post(url, headers=headers, json=data)

    if response.status_code >= 400:
        print("LINE reply error:", response.status_code, response.text)


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "LINE translator bot is running"
    }


@app.post("/webhook")
async def webhook(request: Request):
    body = await request.body()
    signature = request.headers.get("x-line-signature", "")

    expected_signature = base64.b64encode(
        hmac.new(
            LINE_CHANNEL_SECRET.encode("utf-8"),
            body,
            hashlib.sha256
        ).digest()
    ).decode("utf-8")

    if signature != expected_signature:
        raise HTTPException(status_code=400, detail="Invalid signature")

    data = await request.json()

    for event in data.get("events", []):
        if event.get("type") != "message":
            continue

        message = event.get("message", {})

        if message.get("type") != "text":
            continue

        user_text = message.get("text", "")
        reply_token = event.get("replyToken")
        chat_id = get_chat_id(event)

        try:
            translated = translate_text(user_text, chat_id)

            if translated and reply_token:
                reply_line(reply_token, translated)

                add_history(chat_id, "Original", user_text)
                add_history(chat_id, "Translation", translated)

        except Exception as error:
            print("Translation error:", str(error))

            if reply_token:
                reply_line(reply_token, "번역 오류가 발생했어요. 잠시 후 다시 보내주세요.")

    return {
        "status": "ok"
    }
