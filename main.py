import os
import json
import random
import torch
import warnings  
import scipy.io.wavfile as wavfile
import pandas as pd
import fitz  
from PIL import Image
import io
import re
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Dict, Any
from pydantic import BaseModel
from auth import router as auth_router

class ReportData(BaseModel):
    patient_name: str
    doctor_name: str
    risk_level: str
    summary: str
    full_data: Dict[str, Any]  

warnings.filterwarnings("ignore")  
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"   
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  

from transformers import AutoProcessor, BarkModel, logging as hf_logging
hf_logging.set_verbosity_error()  

from google import genai 
from config import GEMINI_KEYS

from transformers import AutoProcessor, BarkModel

print("🔄 Bark Model Load ho raha hai (GPU Par)... Kripya download poora hone dein!")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f" Execution Engine: {device.upper()}")

processor = AutoProcessor.from_pretrained("suno/bark-small")
bark_model = BarkModel.from_pretrained("suno/bark-small", use_safetensors = True ).to(device)
print("Bark Activated!")

app = FastAPI(title="Medi-Decode AI", version="1.0")
app.include_router(auth_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

current_report_context = ""
def get_gemini_client():
    if not GEMINI_KEYS:
        raise Exception("keys missing")
    current_key = random.choice(GEMINI_KEYS)
    client = genai.Client(api_key=current_key)
    return client

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        return str(e)

def generate_audio_with_bark(text, lang_choice, filename):
    voice_preset = "v2/hi_speaker_0" if lang_choice == "Hindi" else "v2/en_speaker_6"
    filepath = f"static/{filename}.wav" 
    try:
        sentences = re.split(r'[.?!।]\s*', text.strip())
        sentences = [s for s in sentences if len(s.strip()) > 0]
        all_audio_pieces = []
        sample_rate = bark_model.generation_config.sample_rate
        print(f"🎙️ Bark is rendering {len(sentences)} Audio")
        for sentence in sentences:
            inputs = processor(sentence, voice_preset=voice_preset).to(device)
            audio_array = bark_model.generate(**inputs)
            audio_array = audio_array.cpu().numpy().squeeze()
            all_audio_pieces.append(audio_array)
            silence = np.zeros(int(0.2 * sample_rate)) 
            all_audio_pieces.append(silence)
        final_audio = np.concatenate(all_audio_pieces)
        final_audio = np.clip(final_audio,-1.0,1.0)
        final_audio_int16 = (final_audio*32767.0).astype(np.int16)
        wavfile.write(filepath, rate=sample_rate, data=final_audio_int16)
        return f"/static/{filename}.wav"
    except Exception as e:
        print(f"⚠️ Bark Audio Error: {e}")
        return None
@app.post("/analyze-report/")
async def analyze_medical_report(
    file: UploadFile = File(...), 
    language: str = Form("Hindi") 
):
    global current_report_context 
    try:
        file_ext = file.filename.split('.')[-1].lower()
        client = get_gemini_client() 
        base_prompt = f"""
You are an expert, empathetic medical AI assistant. Read this medical lab report or image carefully.
Return the analysis EXACTLY in this JSON format (Do not add any extra text or markdown):
{{
    "patient_name": "Extract the exact patient name from the top of the report (e.g., 'Rahul Sharma'). If not found, return 'Unknown Patient'",
    "metrics": [
        {{
            "name": "Hemoglobin", 
            "patient_value": 11.2, 
            "normal_min": 13.0, 
            "normal_max": 17.0, 
            "unit": "g/dL", 
            "status": "Low"
        }}
    ],
    "patient_summary": "Explain the report clearly in {language} like a caring doctor. Use simple terms, highlight red flags. STRICTLY provide only 1 or 2 extremely short sentences (maximum 20 words total). (This will be converted to audio).",
    "diet_chart": [
        "Morning: [Breakfast suggestion]",
        "Afternoon: [Lunch suggestion]",
        "Night: [Dinner suggestion]",
        "Avoid: [What to avoid]"
    ],
    "prescribed_tests": [
        "[Test Name] - [Reason]"
    ],
    "doctor_appointment": "[Specialist Name] within [Timeframe]",
    "urgency": "High/Medium/Low"
}}
Note: If a metric does not have a numerical value (e.g., 'Positive/Negative'), skip adding it to the metrics array or set normal_min/max to 0.
"""
        if file_ext in ['pdf']:
            file_path = f"temp_{file.filename}"
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
            raw_text = extract_text_from_pdf(file_path)
            if os.path.exists(file_path):
                os.remove(file_path)
            if not raw_text.strip():
                raise HTTPException(status_code=400, detail="PDF se text nahi nikal paya.")
            current_report_context = raw_text 
            full_prompt = base_prompt + f"\n\nReport Text:\n{raw_text[:2000]}"
            response = client.models.generate_content(
                model='gemini-2.5-flash', 
                contents=full_prompt,
            )

        elif file_ext in ['jpg', 'jpeg', 'png']:
            print("📸 Image detect hui! Direct Gemini Vision use kar rahe hain...")
            image_data = await file.read()
            img = Image.open(io.BytesIO(image_data))
            current_report_context = "Patient uploaded an image report. Extracted insights will be based on that image."
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[base_prompt, img],
            ) 
        else:
            raise HTTPException(status_code=400, detail="Bhai, sirf PDF, JPG ya PNG upload kar sakte ho.")

        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        report_data = json.loads(clean_json)

        df_metrics = pd.DataFrame(report_data.get("metrics", []))
        print("graph data")
        if not df_metrics.empty:
            print(df_metrics[['name', 'patient_value', 'normal_min', 'normal_max', 'status']].to_string())
        else:
            print("No numerical data found for graphs.")
        print("--------------------------------------\n")

        print("Bark working")
        audio_url = generate_audio_with_bark(report_data["patient_summary"], language, "patient_audio")

        dummy_doctors = {
            "Cardiologist": [
                {"name": "Dr. Sharma (Heart Specialist)", "distance": "1.2 km", "availability": "Today, 4 PM", "rating": "⭐ 4.8"},
                {"name": "City Heart Clinic", "distance": "3.5 km", "availability": "Tomorrow, 10 AM", "rating": "⭐ 4.5"}
            ],
            "Endocrinologist": [
                {"name": "Dr. Verma (Diabetes & Thyroid)", "distance": "2.0 km", "availability": "Today, 5 PM", "rating": "⭐ 4.9"}
            ],
            "General Physician": [
                {"name": "Dr. Kapoor (General Medicine)", "distance": "0.8 km", "availability": "Today, 2 PM", "rating": "⭐ 4.6"},
                {"name": "HealthFirst Clinic", "distance": "1.5 km", "availability": "Today, 6 PM", "rating": "⭐ 4.3"}
            ]
        }

        recommended_specialist = report_data.get("doctor_appointment", "").lower()
        
        nearby_docs = dummy_doctors["General Physician"] 
        if "cardio" in recommended_specialist:
            nearby_docs = dummy_doctors["Cardiologist"]
        elif "diabet" in recommended_specialist or "thyroid" in recommended_specialist:
            nearby_docs = dummy_doctors["Endocrinologist"]

        return {
            "status": "success",
            "patient_name" : report_data.get("patient_name","unknown Patient"),
            "urgency": report_data.get("urgency", "Medium"),
            "data_table": report_data.get("metrics", []), 
            "summary_text": report_data.get("patient_summary", ""),
            "diet_chart": report_data.get("diet_chart", []),
            "prescribed_tests": report_data.get("prescribed_tests", []),
            "doctor_appointment": report_data.get("doctor_appointment", ""),
            "nearby_doctors": nearby_docs, 
            "audio_link": audio_url
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/chat/")
async def chat_with_report(question: str = Form(...)):
    global current_report_context
    
    if not current_report_context:
        return {"answer": "Pehle koi report upload karo taaki main usse padh saku!"}
    
    try:
        prompt = f"""
        You are an AI Medical Assistant. 
        Here is the patient's medical report data: 
        {current_report_context[:3000]}
        
        Patient's Question: {question}
        
        Answer accurately based ONLY on the report data provided. 
        Keep the answer friendly, simple, and under 3 lines.
        """
        
        client = get_gemini_client()
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        
        return {"answer": response.text.strip()}
        
    except Exception as e:
        return {"answer": f"Backend Error: {str(e)}"}
    
DB_file  = "Database.json"
if not os.path.exists(DB_file):
        with open (DB_file ,'w') as f:
            json.dump([],f)

@app.post("/send-to-doctor/")
async def srfd (data : ReportData):
        with open (DB_file , "r") as f:
            inbox = json.load(f)

        inbox.append(data.dict())

        with open (DB_file , "w") as f:
            json.dump(inbox ,f)
        return {'status':'success','message':'saved permanently'}
@app.get("/get-inbox/{doctor_name}")
async def gdi (doctor_name :str ):
        with open (DB_file , 'r') as f:
            inbox = json.load(f)
        doctor_reports = [ r for r in inbox if r["doctor_name"] == doctor_name ]

        return {'status':'success','reports': doctor_reports}

    
    