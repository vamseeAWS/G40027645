from flask import Flask, request, jsonify
from jose import jwt, jwk
import requests

app = Flask(__name__)

KEYCLOAK_URL = "http://keycloak:8080"
REALM = "FintechApp"
CLIENT_ID = "flask-client"

JWKS_URL = f"{KEYCLOAK_URL}/realms/{REALM}/protocol/openid-connect/certs"

def get_public_key(token):
    # Always fetch fresh keys to handle rotating signing keys
    jwks = requests.get(JWKS_URL).json()
    jwks_keys = {key["kid"]: key for key in jwks["keys"]}
    
    headers = jwt.get_unverified_header(token)
    kid = headers.get("kid")
    key = jwks_keys.get(kid)
    if not key:
        raise Exception("Unknown 'kid' in token")
    return jwk.construct(key)

@app.route("/")
def index():
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"error": "Missing token"}), 401

    token = auth_header.split(" ")[1]

    try:
        public_key = get_public_key(token)
        claims = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            audience=CLIENT_ID,
            options={"verify_aud": True}
        )

        return jsonify({
            "email": claims.get("email", "N/A"),
            "name": claims.get("name", "N/A"),
            "sub": claims.get("sub", "N/A")
        })

    except jwt.ExpiredSignatureError:
        return jsonify({"error": "Token expired"}), 401
    except jwt.JWTClaimsError as e:
        return jsonify({"error": f"Invalid claims: {str(e)}"}), 401
    except Exception as e:
        return jsonify({"error": f"Token validation error: {str(e)}"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
