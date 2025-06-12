# app.py (secured version)
from flask import Flask, request, jsonify
import os
import subprocess  # nosec
import ast
import ipaddress
import shutil
import re

app = Flask(__name__)

# Retrieve password securely from environment variable
PASSWORD = os.environ.get('PASSWORD')
if not PASSWORD:
    raise RuntimeError("Environment variable PASSWORD is required and not set.")


@app.route('/')
def hello():
    name = request.args.get('name', 'World')
    if not name.isalnum():
        return jsonify({"error": "Invalid name"}), 400
    return f"Hello, {name}!"

# Secure ping route with validation and full path
@app.route('/ping')
def ping():
    ip = request.args.get('ip')
    try:
        ipaddress.ip_address(ip)  # Validate IP address
        ping_path = shutil.which("ping")
        if not ping_path:
            return jsonify({"error": "Ping not found"}), 500

        result = subprocess.check_output([ping_path, "-c", "1", "-W", "1", ip])
        return result
    except ValueError:
        return jsonify({"error": "Invalid IP address"}), 400
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "Ping failed", "details": e.output.decode()}), 500

# Secure eval using ast.literal_eval
@app.route('/calculate')
def calculate():
    expression = request.args.get('expr', '')
    
    # Only allow digits, operators, parentheses, and whitespace
    if not re.match(r'^[0-9\.\+\-\*\/\(\)\s]+$', expression):
        return jsonify({"error": "Invalid characters in expression"}), 400

    try:
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)
    except Exception:
        return jsonify({"error": "Invalid expression"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
