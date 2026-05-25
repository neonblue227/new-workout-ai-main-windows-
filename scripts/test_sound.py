import os
import wave

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.environ["google_ai_studio_api_key"])

script = """
ทางการจีนได้ยกระดับมาตรการรักษาความปลอดภัยโดยรอบจตุรัสเทียนอันเหมินอันเก่าแก่ของกรุงปักกิ่งมาหลายวันแล้ว พร้อมกับข่าวลือในโซเชียลมีเดียเกี่ยวกับการจัดขบวนพาเหรดครั้งพิเศษหรืองานใหญ่ที่จัดเตรียมไว้เป็นอย่างดี
การเตรียมการสำหรับงานสำคัญครั้งนี้เริ่มต้นขึ้นอย่างเงียบ ๆ แต่ดูเหมือนว่าจีนพร้อมแล้วที่จะจัดงานต้อนรับประธานาธิบดีโดนัลด์ ทรัมป์ แห่งสหรัฐอเมริกา
การเยือนครั้งนี้มีหมายกำหนดทั้งการเจรจา งานเลี้ยง และการเยี่ยมชมหอฟ้าเทียนถาน ซึ่งเป็นหนึ่งในวัดอันศักดิ์สิทธิ์ที่อดีตจักรพรรดิจีนทรงมาอธิษฐานเพื่อขอพรให้ได้ผลผลิตที่ดี
ทั้งทรัมป์และประธานาธิบดีสี จิ้นผิง ต่างหวังว่าการเยือนครั้งนี้จะประสบผลสำเร็จ การประชุมสุดยอดระหว่างผู้นำที่ทรงอำนาจที่สุดสองคนของโลกครั้งนี้ จะเป็นการพบปะที่สำคัญที่สุดครั้งหนึ่งในรอบหลายปี
"""

response = client.models.generate_content(
    model="gemini-2.5-flash-preview-tts",
    contents="ข่าววันนี้: " + script,
    config=types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Charon",
                )
            )
        ),
    ),
)

pcm = response.candidates[0].content.parts[0].inline_data.data
out_path = os.path.join(os.path.dirname(__file__), "test_sound.wav")
with wave.open(out_path, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(24000)
    wf.writeframes(pcm)

print(f"wrote {len(pcm)} bytes of PCM → {out_path}")
