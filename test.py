import google.generativeai as genai
import os 
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

try:
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content("Test prompt")
    print(response.text)
except Exception as e:
    print(f"Error: {str(e)}")
