# genai_prescriptions.py
# Purpose:
# This script integrates multiple Generative AI providers (Google Gemini, OpenAI GPT, and xAI Grok)
# to automatically generate a prescriptive incident response plan for phishing alerts.
# It returns the plan in a strict JSON format for automation workflows (e.g., SOAR systems).

# -----------------------------
# Imports
# -----------------------------
import google.generativeai as genai   # Google Gemini API SDK
import openai                         # OpenAI API SDK
import requests                       # For HTTP requests (used for Grok API)
import streamlit as st                # For web UI and secrets management
import json                           # For handling JSON serialization/deserialization

# -----------------------------
# API Key Configuration
# -----------------------------
try:
    # Load API keys from Streamlit's secret storage (.streamlit/secrets.toml)
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])      # Google Gemini API
    openai.api_key = st.secrets["OPENAI_API_KEY"]              # OpenAI GPT API
    grok_api_key = st.secrets["GROK_API_KEY"]                  # xAI Grok API
except (KeyError, FileNotFoundError):
    # Fallback if keys are missing (some features will be unavailable)
    print("API keys not found in .streamlit/secrets.toml. Some features may be disabled.")
    grok_api_key = None

# -----------------------------
# Base Prompt Builder
# -----------------------------
def get_base_prompt(alert_details):
    """
    Constructs a standardized prompt for all AI providers.

    Args:
        alert_details (dict): Details of the phishing alert (features, metadata, etc.)

    Returns:
        str: A formatted multi-line string instructing the AI to output a JSON object.
    """
    return f"""
    You are an expert Security Orchestration, Automation, and Response (SOAR) system.
    A URL has been flagged as a potential phishing attack based on the following characteristics:
    {json.dumps(alert_details, indent=2)}

    Your task is to generate a prescriptive incident response plan.
    Provide your response in a structured JSON format with the following keys:
    - "summary": A brief, one-sentence summary of the threat.
    - "risk_level": A single-word risk level (e.g., "Critical", "High", "Medium").
    - "recommended_actions": A list of specific, technical, step-by-step actions for a security analyst to take.
    - "communication_draft": A brief, professional draft to communicate to the employee who reported the suspicious URL.

    Return ONLY the raw JSON object and nothing else.
    """

# -----------------------------
# Gemini Prescription Generator
# -----------------------------
def get_gemini_prescription(alert_details):
    """
    Generates a prescription using Google's Gemini model.

    Steps:
    1. Build the base prompt.
    2. Call Gemini API to get a structured JSON response.
    3. Remove any Markdown formatting (```json ... ```).
    4. Convert string output to Python dict.

    Returns:
        dict: JSON-parsed prescription plan.
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = get_base_prompt(alert_details)
    response_text = model.generate_content(prompt).text.strip() \
                       .lstrip("```json\n").rstrip("```")
    return json.loads(response_text)

# -----------------------------
# OpenAI Prescription Generator
# -----------------------------
def get_openai_prescription(alert_details):
    """
    Generates a prescription using OpenAI's GPT models.

    Uses:
        - gpt-4o model
        - response_format={"type": "json_object"} to enforce JSON output

    Returns:
        dict: JSON-parsed prescription plan.
    """
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    prompt = get_base_prompt(alert_details)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# -----------------------------
# Grok Prescription Generator
# -----------------------------
def get_grok_prescription(alert_details):
    """
    Generates a prescription using xAI's Grok model via HTTP POST.

    Steps:
    1. Checks if API key is available.
    2. Builds prompt and sends POST request to Grok endpoint.
    3. Parses JSON from the model's raw Markdown output.

    Returns:
        dict: JSON-parsed prescription plan or error message.
    """
    if not grok_api_key:
        return {"error": "Grok API key not configured."}

    prompt = get_base_prompt(alert_details)
    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {grok_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "grok-1",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    # Send request to Grok API
    response = requests.post(url, headers=headers, json=data)
    content_str = response.json()['choices'][0]['message']['content']

    # Remove Markdown formatting and parse JSON
    return json.loads(content_str.strip().lstrip("```json\n").rstrip("```"))

# -----------------------------
# Provider Dispatcher
# -----------------------------
def generate_prescription(provider, alert_details):
    """
    Dispatches the prescription request to the selected AI provider.

    Args:
        provider (str): "Gemini", "OpenAI", or "Grok".
        alert_details (dict): Alert metadata to base the prescription on.

    Returns:
        dict: JSON-parsed prescription plan.
    """
    if provider == "Gemini":
        return get_gemini_prescription(alert_details)
    elif provider == "OpenAI":
        return get_openai_prescription(alert_details)
    elif provider == "Grok":
        return get_grok_prescription(alert_details)
    else:
        raise ValueError("Invalid provider selected")

