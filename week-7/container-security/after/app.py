# app.py (secured version)
from flask import Flask, request, jsonify
import os
import subprocess  # nosec
import ast
import ipaddress
import shutil

app = Flask(__name__)

# Retrieve password securely from environment variable
PASSWORD = os.environ.get('PASSWORD', 'default_password')

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
    expression = request.args.get('expr')
    try:
        result = ast.literal_eval(expression)
        return str(result)
    except (SyntaxError, ValueError):
        return jsonify({"error": "Invalid expression"}), 400

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
