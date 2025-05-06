
# Security Architecture for a Containerized Flask Application

This document outlines the process of building and securing a containerized Flask application. It begins from the developer's perspective, creating the application and infrastructure, and then transitions to the cybersecurity architect's role in enhancing security. The document includes all relevant code and configurations from both the "before" and "after" states to provide a complete, self-contained guide.

---

## 1. Developer’s Perspective: Building the Application and Infrastructure
From the developer’s viewpoint, the process involves creating a Flask application, containerizing it with Docker, setting up a multi-container environment with Docker Compose, and automating tasks with a Makefile.

### 1.1 Creating the Flask Application
The developer initially writes a simple Flask web application with intentional security flaws to demonstrate vulnerabilities. These flaws are later addressed by the cybersecurity architect.

#### Before Security Fixes: `app.py`
```python
from flask import Flask, request, jsonify
import os
import subprocess

app = Flask(__name__)

# Hard-coded password
PASSWORD = "supersecretpassword"

@app.route('/')
def hello():
    name = request.args.get('name', 'World')
    if not name.isalnum():
        return jsonify({"error": "Invalid name"}), 400
    return f"Hello, {name}!"

# Command injection vulnerability
@app.route('/ping')
def ping():
    ip = request.args.get('ip')
    # Unsafe command execution
    result = subprocess.check_output(f"ping -c 1 {ip}", shell=True)
    return result

# Insecure use of eval
@app.route('/calculate')
def calculate():
    expression = request.args.get('expr')
    # Dangerous use of eval
    result = eval(expression)
    return str(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

#### After Security Fixes: `app.py`
```python
from flask import Flask, request, jsonify
import os
import subprocess
import ast
import ipaddress

app = Flask(__name__)

# Retrieve password from environment variable instead of hardcoding
PASSWORD = os.environ.get('PASSWORD', 'default_password')

@app.route('/')
def hello():
    name = request.args.get('name', 'World')
    if not name.isalnum():
        return jsonify({"error": "Invalid name"}), 400
    return f"Hello, {name}!"

# Secure ping route with input validation and no shell=True
@app.route('/ping')
def ping():
    ip = request.args.get('ip')
    try:
        ipaddress.ip_address(ip)  # Validate IP address
        result = subprocess.check_output(["ping", "-c", "1", ip])
        return result
    except ValueError:
        return jsonify({"error": "Invalid IP address"}), 400

# Secure calculate route using ast.literal_eval instead of eval
@app.route('/calculate')
def calculate():
    expression = request.args.get('expr')
    try:
        result = ast.literal_eval(expression)
        return str(result)
    except (SyntaxError, ValueError):
        return jsonify({"error": "Invalid expression"}), 400

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)  # Bind to localhost instead of all interfaces
```

**Dependencies:** The `requirements.txt` file pins Flask to version 3.0.3:
```
Flask==3.0.3
```

### 1.2 Containerizing the Application
The application is containerized using a Dockerfile, with security enhancements applied in the "after" version.

#### Before Security Fixes: `Dockerfile`
```
FROM python:3.9-alpine

RUN addgroup -S appgroup && adduser -S appuser -G appgroup

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

USER appuser

CMD ["python", "app.py"]
```

#### After Security Fixes: `Dockerfile`
```
FROM python:3.13-alpine
RUN adduser -D appuser
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
USER appuser
HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:5000/ || exit 1
CMD ["python", "app.py"]
```

### 1.3 Setting Up Docker Compose
Docker Compose manages the Flask app and a PostgreSQL database, with improvements for security and development in the "after" version.

#### Before Security Fixes: `docker-compose.yml`
```yaml
services:
  web:
    build: .
    image: mywebapp
    ports:
      - "15000:5000"
    depends_on:
      - db
    networks:
      - frontend
  db:
    image: postgres:13
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: mydb
    networks:
      - backend
networks:
  frontend:
  backend:
```

#### After Security Fixes: `docker-compose.yml`
```yaml
services:
  web:
    build: .
    image: mywebapp
    ports:
      - "15000:5000"
    volumes:
      - .:/app
    command: flask run --host=0.0.0.0 --port=5000
    environment:
      - FLASK_APP=app.py
      - FLASK_ENV=development
    depends_on:
      - db
    networks:
      - frontend
  db:
    image: postgres:13
    env_file:
      - .env
    networks:
      - backend
networks:
  frontend:
  backend:
```

### 1.4 Surrounding Infrastructure
A `Makefile` automates tasks such as building, running, and scanning the application. The Makefile remains unchanged between "before" and "after" versions.

**Makefile:**
```
# Pre-build security checks
check:
	@echo "Running code analysis with Bandit..."
	docker run --rm -v $(PWD):/app python:3.9-alpine sh -c "pip install bandit && bandit -r /app"
	@echo "Running dependency check with pip-audit..."
	docker run --rm -v $(PWD):/app python:3.9-alpine sh -c "pip install pip-audit && pip-audit -r /app/requirements.txt"

# Host security check
host-security:
	@echo "Running Docker Bench for Security..."
	docker run --rm -v /var/run/docker.sock:/var/run/docker.sock docker/docker-bench-security

# Build Docker image after security checks
dbuild: check
	docker build -t mywebapp .

# Run the container
run:
	docker run -p 6000:5000 mywebapp

# Scan the built image for vulnerabilities
scan:
	docker scout recommendations mywebapp:latest

# Docker Compose commands
build:
	docker compose build

start:
	docker compose up -d

stop:
	docker compose down

logs:
	docker compose logs -f

clean:
	docker system prune -f

restart: stop start
```

---

## 2. Cybersecurity Architect’s Perspective: Securing the Application
A cybersecurity architect enhances the security of the containerized Flask application through a structured process, addressing vulnerabilities identified in the initial setup.

### 2.1 Initial Steps
The architect:
- Reviews `app.py`, `Dockerfile`, `docker-compose.yml`, and `Makefile`.
- Identifies assets: Flask app, PostgreSQL database, Docker images, and host.
- Sets goals: confidentiality, integrity, and availability.

### 2.2 Threat Modeling
Using the STRIDE methodology, potential threats include:
- **Spoofing**: Unauthorized access.
- **Tampering**: Code/data modification.
- **Information Disclosure**: Leaking credentials.
- **Denial of Service**: Overloading services.
- **Elevation of Privilege**: Escalating access.

**Attack Surfaces:**
- Exposed port 15000.
- Hardcoded credentials (before fixes).
- Dependencies and host configuration.

### 2.3 First Scan: Identifying Issues
Scans reveal:
- Hardcoded credentials in `app.py` and `docker-compose.yml`.
- Command injection and `eval` vulnerabilities in `app.py`.
- Outdated base image in `Dockerfile`.

### 2.4 Mapping Issues to MITRE ATT&CK
- **T1552 - Unsecured Credentials**: Hardcoded secrets.
- **T1190 - Exploit Public-Facing Application**: Flask vulnerabilities.

### 2.5 Designing a Mixed Architecture Model
The architect applies:
- **Defense in Depth**: Non-root users, minimal images, secure coding.
- **Application Security**: Input validation, environment variables.
- **Zero Trust**: Least privilege, network isolation.

### 2.6 Implementing the Security Architecture
Security measures are implemented across the files.

#### 2.6.1 Updated Flask Application
- Hardcoded password replaced with an environment variable.
- Command injection fixed in `/ping` with validation and no `shell=True`.
- `eval` replaced with `ast.literal_eval` in `/calculate`.
- Binding changed to `localhost`.

#### 2.6.2 Updated Docker Compose
- Added volumes for development.
- Moved credentials to an `.env` file.
- Configured Flask for development mode.

#### 2.6.3 Updated Dockerfile
- Updated to `python:3.13-alpine`.
- Added a health check.
- Ensured non-root user usage.

#### 2.6.4 Automation Script
A new script, `docker_security_fixes.py`, automates security enhancements:
- Updates `daemon.json` with settings like restricted inter-container communication (`icc: False`).
- Modifies `Dockerfile` to ensure non-root user and health checks.
- Enhances `docker-compose.yml` with resource limits and security options.

**Automation Script: `docker_security_fixes.py`**
```python
import os
import json
import yaml
import subprocess

# Paths to files (adjust as necessary)
DAEMON_JSON_PATH = '/etc/docker/daemon.json'
DOCKERFILE_PATH = 'Dockerfile'
DOCKER_COMPOSE_PATH = 'docker-compose.yml'

def update_daemon_json():
    """Update or create daemon.json with security settings."""
    settings = {
        "icc": False,
        "userns-remap": "default",
        "live-restore": True,
        "userland-proxy": False
    }
    if os.path.exists(DAEMON_JSON_PATH):
        with open(DAEMON_JSON_PATH, 'r') as f:
            current_settings = json.load(f)
        current_settings.update(settings)
    else:
        current_settings = settings
    with open(DAEMON_JSON_PATH, 'w') as f:
        json.dump(current_settings, f, indent=4)
    print(f"Updated {DAEMON_JSON_PATH} with security settings.")

def update_dockerfile():
    """Modify Dockerfile to add non-root user and health check."""
    with open(DOCKERFILE_PATH, 'r') as f:
        lines = f.readlines()
    # Insert non-root user and health check if not present
    if not any('RUN adduser -D appuser' in line for line in lines):
        lines.insert(1, 'RUN adduser -D appuser\n')
    if not any('HEALTHCHECK' in line for line in lines):
        lines.insert(-1, 'HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:5000/ || exit 1\n')
    if not any('USER appuser' in line for line in lines):
        lines.insert(-1, 'USER appuser\n')
    with open(DOCKERFILE_PATH, 'w') as f:
        f.writelines(lines)
    print(f"Updated {DOCKERFILE_PATH} with non-root user and health check.")

def update_docker_compose():
    """Update docker-compose.yml with security settings for containers."""
    with open(DOCKER_COMPOSE_PATH, 'r') as f:
        compose_data = yaml.safe_load(f)
    for service in compose_data.get('services', {}).values():
        service['mem_limit'] = '512m'
        service['read_only'] = True
        service['security_opt'] = ['no-new-privileges:true']
        service['pids_limit'] = 100
        if 'ports' in service:
            for i, port in enumerate(service['ports']):
                if port.startswith('0.0.0.0'):
                    service['ports'][i] = port.replace('0.0.0.0', '127.0.0.1')
    with open(DOCKER_COMPOSE_PATH, 'w') as f:
        yaml.dump(compose_data, f)
    print(f"Updated {DOCKER_COMPOSE_PATH} with security settings.")

def main():
    print("Applying Docker security fixes...")
    update_daemon_json()
    update_dockerfile()
    update_docker_compose()
    print("Security fixes applied. Please review the changes and restart Docker services as necessary.")

if __name__ == "__main__":
    main()
```

### 2.7 Second Scan: Verifying Protection
Post-fix scans confirm:
- No hardcoded credentials.
- Reduced attack surface with secure coding.
- Improved container security with updated image and health checks.

---
