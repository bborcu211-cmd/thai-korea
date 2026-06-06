import os
import re
import hmac
import hashlib
import base64
import requests

from fastapi import FastAPI, Request, HTTPException
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")


def translate_text(text):
    if re.search(r"[가-힣]", text):
        system_prompt = """
You are an expert Korean-to-Thai translator for private romantic LINE conversations.

Translate Korean into very natural Thai, not literal textbook Thai.
The speaker is a Korean man talking to his Thai girlfriend.
Use masculine polite ending ครับ only when it sounds natural.
Preserve the exact feeling of the original message: love, teasing, joking, jealousy, worry, sadness, desire, cuteness, and casual intimacy.
Do not over-polite the sentence.
Do not make it sound like business Thai.
Do not add new meaning.
Do not remove emojis, 555, laughter, or playful tone.
Translate 여보 as ที่รัก when natural.
Translate 자기 as ที่รัก or ตัวเอง depending on context.
If the Korean sentence is short and casual, make the Thai short and casual too.
When translating Korean to Thai, preserve Korean dialect, playful complaint, teasing, sulking, joking, and couple-like emotional tone naturally in Thai.
For playful Korean endings like "아닌교", "아니겠어요?", "뭐야 555", "아닌데요 555", use natural Thai particles such as ล่ะ, นะ, เนี่ย, สิครับ, เหรอครับ when appropriate.
Do not translate too literally.
Make the Thai sound like a real Thai girlfriend/boyfriend would read in LINE chat.
When Korean uses past tense such as -었어요, -였어요, -이었다, preserve the past-time meaning in Thai using เมื่อก่อน, ตอนนั้น, เคย, or แล้ว when natural. Do not translate past tense as present tense.
Output only the Thai translation.
"""
    elif re.search(r"[\u0E00-\u0E7F]", text):
        system_prompt = """
You are an expert Thai-to-Korean translator for private romantic LINE conversations.

Translate Thai into very natural Korean, not literal textbook Korean.
The speaker is a Thai woman talking to her Korean boyfriend.
Preserve the exact feeling of the original message: love, teasing, joking, jealousy, worry, sadness, desire, cuteness, and casual intimacy.
If the Thai uses ค่ะ or คะ, reflect a soft feminine tone naturally in Korean.
Do not make it sound stiff or formal.
Do not add new meaning.
Do not remove emojis, 555, laughter, or playful tone.
Translate ที่รัก as 여보 when natural.
Translate คิดถึง as 보고 싶어 / 그리워 depending on emotional strength.
If the Thai sentence is short and casual, make the Korean short and casual too.
When translating Thai to Korean, preserve soft feminine Thai tone from ค่ะ/คะ naturally.
Do not make Thai casual sentences too short, dry, or emotionless in Korean.
If the Thai sentence sounds gentle, cute, teasing, sulking, or explanatory, reflect that tone naturally in Korean.
For Thai particles like นะ, นี่นา, ล่ะ, เนี่ย, preserve the emotional nuance naturally instead of translating literally.
Output only the Korean translation.
"""
    else:
        return None

    result = client.chat.completions.create(
        model="gpt-4o",
        messages=[
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
