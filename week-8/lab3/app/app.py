from flask import Flask, redirect, request, jsonify, session
import requests
import json
import os

app = Flask(__name__)
app.secret_key = os.environ.get("APP_SECRET", "supersecret")

# Load config from settings.json
try:
    with open('saml/settings.json') as f:
        config = json.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load settings.json: {e}")

# Replace internal hostname for browser redirects only
def fix_hostname_for_browser(url):
    return url.replace("keycloak", "localhost")

# Keycloak endpoints
CLIENT_ID = config["client_id"]
REDIRECT_URI = config["redirect_uri"]
AUTH_ENDPOINT = fix_hostname_for_browser(config["authorization_endpoint"])  # Browser needs localhost
TOKEN_ENDPOINT = config["token_endpoint"]  # Internal name (keycloak)
USERINFO_ENDPOINT = config["userinfo_endpoint"]

@app.route('/')
def index():
    return '<a href="/login">Login with Keycloak</a>'

@app.route('/login')
def login():
    return redirect(
        f"{AUTH_ENDPOINT}?client_id={CLIENT_ID}"
        f"&response_type=code"
        f"&scope=openid%20email%20profile"
        f"&redirect_uri={REDIRECT_URI}"
    )

@app.route('/callback')
def callback():
    code = request.args.get('code')
    if not code:
        return 'No authorization code received.', 400

    try:
        # Exchange code for tokens
        token_response = requests.post(
            TOKEN_ENDPOINT,
            data={
                'grant_type': 'authorization_code',
                'code': code,
                'redirect_uri': REDIRECT_URI,
                'client_id': CLIENT_ID,
                # 'client_secret': config.get("client_secret")  # Optional
            },
            headers={'Content-Type': 'application/x-www-form-urlencoded'}
        )
        token_response.raise_for_status()
        tokens = token_response.json()

        # Fetch user info
        userinfo_response = requests.get(
            USERINFO_ENDPOINT,
            headers={'Authorization': f"Bearer {tokens['access_token']}"}
        )
        userinfo_response.raise_for_status()
        session['userinfo'] = userinfo_response.json()

        return jsonify(session['userinfo'])

    except requests.exceptions.RequestException as e:
        return f"Token exchange failed: {e}", 500

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
