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
        system_prompt = "Translate Korean to natural Thai only. Speaker is male, use ครับ when needed. Preserve emotion and casual tone."
    elif re.search(r"[\u0E00-\u0E7F]", text):
        system_prompt = "Translate Thai to natural Korean only. Preserve emotion and casual tone."
    else:
        return None

    result = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0.3
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
