import os
import requests
import json

# In-memory cache to avoid repeated API calls for the same disease
remedy_cache = {}

def generate_remedy(disease_name):
    # If the plant is healthy or unknown, no remedy needed
    if not disease_name or 'healthy' in disease_name.lower() or 'unknown' in disease_name.lower():
        return None

    # Check cache first
    if disease_name in remedy_cache:
        return remedy_cache[disease_name]

    # Fallback remedies if API fails
    fallback_remedy = {
        "immediate": "Isolate the affected plant to prevent the disease from spreading to healthy plants.",
        "organic": "Prune affected areas. Apply a neem oil or mild soap solution to the leaves.",
        "chemical": "Consult your local agricultural extension for specific, approved fungicides or pesticides for this crop."
    }

    # Fetch API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Warning: GROQ_API_KEY not found in environment variables. Using fallback remedies.")
        return fallback_remedy

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    prompt = (
        f"You are an expert plant pathologist. Provide highly specific, distinct, and unique farmer-friendly remedies "
        f"strictly tailored to this exact plant disease: {disease_name}. "
        "Do NOT give generic advice (like just 'neem oil'). Mention the exact chemical names, specific organic compounds, "
        "and precise immediate physical actions unique to this specific disease and crop. "
        "Return strictly in JSON format: "
        '{"immediate": "...", "organic": "...", "chemical": "..."}. '
        "Keep responses short but highly specific to the disease."
    )

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.3
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Parse the JSON string from the response content
        content = data['choices'][0]['message']['content']
        remedy = json.loads(content)
        
        # Validate the response has the required keys
        if "immediate" in remedy and "organic" in remedy and "chemical" in remedy:
            remedy_cache[disease_name] = remedy
            return remedy
        else:
            print("Warning: Groq API returned malformed JSON structure.")
            return fallback_remedy
            
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return fallback_remedy
