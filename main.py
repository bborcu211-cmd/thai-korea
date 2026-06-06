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
# Render 무료 서버가 잠들거나 재배포되면 이 기억은 초기화됩니다.
CHAT_HISTORY = defaultdict(lambda: deque(maxlen=8))


def get_chat_id(event):
    source = event.get("source", {})
    if source.get("type") == "group":
        return source.get("groupId")
    if source.get("type") == "room":
        return source.get("roomId")
    return source.get("userId", "default")


def add_history(chat_id, role, text):
    if text:
        CHAT_HISTORY[chat_id].append({
            "role": role,
            "text": text
        })


def format_history(chat_id):
    history = CHAT_HISTORY.get(chat_id, [])
    if not history:
        return "No previous context."

    lines = []
    for item in history:
        role = item["role"]
        text = item["text"]
        lines.append(f"{role}: {text}")

    return "\n".join(lines)


def translate_text(text, chat_id):
    recent_context = format_history(chat_id)

    if re.search(r"[가-힣]", text):
        system_prompt = """
You are a Korean-to-Thai translator for private romantic LINE chat.

Translate the user's Korean message into Thai.
Preserve the exact meaning first.
Do not simplify, summarize, or reinterpret the sentence.
Do not change the subject, object, tense, or emotional intention.
Make the Thai natural, but keep the same meaning and sentence intention.

The speaker is a Korean man talking to his Thai girlfriend.
Use ครับ when a male polite ending is natural.
Translate 여보 as ที่รัก when natural.
Translate 자기 as ที่รัก or ตัวเอง depending on context.
Keep 555, emojis, laughter, and punctuation.

Preserve Korean past tense.
When Korean uses past tense such as -었어요, -였어요, -이었다, preserve the past-time meaning in Thai using เมื่อก่อน, ตอนนั้น, เคย, or แล้ว when natural.
Do not translate past tense as present tense.

For Korean casual expressions like "놀고먹다", preserve the intended meaning.
"놀고먹다" usually means living comfortably, lazing around, eating and resting without working much, not simply traveling or hanging out.

Preserve the grammatical focus of the Korean sentence.
If the Korean sentence says "태국이 그립다", the object being missed is Thailand, not the person.
If the Korean sentence says "여보가 있는 태국", translate it as "Thailand where my love is", not "my love who is in Thailand", unless the Korean clearly means the person.
For sentences like "여보가 있는 태국이 그립다", translate as "คิดถึงประเทศไทยที่มีที่รักอยู่" or "คิดถึงไทยที่มีที่รักอยู่", preserving that Thailand is the thing being missed.

Important examples:
"내가 여보를 사랑한다 아닌교" means "I love you, don't I?" / "You know I love you, right?" Translate it as a playful confirming question, not just "I love you."
"아닌교", "아닌가요", "아니겠어요" should usually become a Thai confirming question such as ไม่ใช่เหรอครับ, ใช่ไหมครับ, or นะครับ depending on context.
"이정도야 뭘 재미있어요 555" should sound like "แค่นี้เอง จะไปสนุกอะไรล่ะครับ 555"

Use the recent conversation context only to resolve ambiguity.
Do not translate the context itself.
Only translate the current message.

Output only the Thai translation.
"""
    elif re.search(r"[\u0E00-\u0E7F]", text):
        system_prompt = """
You are a Thai-to-Korean translator for private romantic LINE chat.

Translate the user's Thai message into Korean.
Preserve the exact meaning first.
Do not simplify, summarize, or reinterpret the sentence.
Do not change the subject, object, tense, or emotional intention.
Make the Korean natural, but keep the same meaning and sentence intention.

The speaker is a Thai woman talking to her Korean boyfriend.
If the Thai uses ค่ะ or คะ, reflect a soft feminine tone naturally in Korean.
Translate ที่รัก as 여보 when natural.
Keep 555, emojis, laughter, and punctuation.

Preserve the exact target of "คิดถึง".
If the Thai sentence says "คิดถึงไทย", translate it as "태국이 그리워" or "태국이 보고 싶어".
If the Thai sentence says "คิดถึงคนที่อยู่ไทย", translate it as "태국에 있는 사람이 보고 싶어".
Do not confuse missing a country/place with missing a person.

For Thai particles like นะ, นี่นา, ล่ะ, เนี่ย, preserve the emotional nuance naturally instead of translating literally.
Do not make Thai casual sentences too short, dry, or emotionless in Korean.
If the Thai sentence sounds gentle, cute, teasing, sulking, or explanatory, reflect that tone naturally in Korean.

Important examples:
"แค่นี้เอง จะไปสนุกอะไรล่ะครับ 555" means "이 정도로 뭐가 재밌겠어요 555" or "이 정도야 뭐가 재밌어요 555", not "이게 뭐가 재밌어 ㅋㅋㅋ" if the tone is softer.
"ตัวเองหายไปไหนมา?" means "자기 어디 갔다 왔어?" or "자기 어디 갔다 온 거야?"

Use the recent conversation context only to resolve ambiguity.
Do not translate the context itself.
Only translate the current message.

Output only the Korean translation.
"""
    else:
        return None

    user_prompt = f"""
Recent conversation context:
{recent_context}

Current message to translate:
{text}
"""

    result = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
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
        "messages": [{"type": "text", "text": text}]
    }
    requests.post(url, headers=headers, json=data)


@app.get("/")
def home():
    return {"status": "ok"}


@app.post("/webhook")
async def webhook(request: Request):
    body = await request.body()
    signature = request.headers.get("x-line-signature")

    expected = base64.b64encode(
        hmac.new(
            LINE_CHANNEL_SECRET.encode("utf-8"),
            body,
            hashlib.sha256
        ).digest()
    ).decode("utf-8")

    if signature != expected:
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

                # 최근 대화 저장
                add_history(chat_id, "Original", user_text)
                add_history(chat_id, "Translation", translated)

        except Exception as e:
            print("Translation error:", str(e))

    return {"status": "ok"}        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": text}
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
        "messages": [{"type": "text", "text": text}]
    }
    requests.post(url, headers=headers, json=data)


@app.get("/")
def home():
    return {"status": "ok"}


@app.post("/webhook")
async def webhook(request: Request):
    body = await request.body()
    signature = request.headers.get("x-line-signature")

    expected = base64.b64encode(
        hmac.new(
            LINE_CHANNEL_SECRET.encode("utf-8"),
            body,
            hashlib.sha256
        ).digest()
    ).decode("utf-8")

    if signature != expected:
        raise HTTPException(status_code=400, detail="Invalid signature")

    data = await request.json()

    for event in data.get("events", []):
        if event.get("type") == "message":
            message = event.get("message", {})
            if message.get("type") == "text":
                translated = translate_text(message.get("text", ""))
                if translated:
                    reply_line(event["replyToken"], translated)

    return {"status": "ok"}
