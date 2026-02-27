import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("No API Key")
else:
    genai.configure(api_key=api_key)
    with open("models_available.txt", "w") as f:
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    f.write(f"{m.name}\n")
            print("Done writing models.")
        except Exception as e:
            f.write(f"Error: {e}")
            print(f"Error: {e}")
