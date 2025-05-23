import os
import json
import yaml

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
    """Overwrite Dockerfile with secure configuration."""
    dockerfile_contents = '''FROM python:3.9-alpine3.21

# Install patched sqlite-libs and other required packages
RUN apk update && apk add --no-cache \\
    gcc \\
    musl-dev \\
    libffi-dev \\
    make \\
    wget \\
    libpq && \\
    pip install --upgrade pip setuptools

# Add non-root user
RUN addgroup -S appgroup && adduser -S appuser -G appgroup

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir -r requirements.txt

COPY . .

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD wget -qO- http://localhost:5000/ || exit 1

CMD ["python", "app.py"]
'''
    with open(DOCKERFILE_PATH, 'w') as f:
        f.write(dockerfile_contents)
    print(f"Overwritten {DOCKERFILE_PATH} with secure base image and best practices.")

def update_docker_compose():
    """Overwrite docker-compose.yml with secure settings and NGINX reverse proxy."""
    compose_data = {
        'version': '3.8',
        'services': {
            'web': {
                'build': '.',
                'image': 'mywebapp:secure',
                'expose': ['5000'],
                'read_only': True,
                'pids_limit': 100,
                'mem_limit': '512m',
                'security_opt': ['no-new-privileges:true'],
                'networks': ['appnet'],
                'environment': ['PASSWORD=${APP_PASSWORD}']
            },
            'nginx': {
                'image': 'nginx:alpine',
                'ports': ['8080:80'],
                'volumes': ['./nginx/default.conf:/etc/nginx/conf.d/default.conf'],
                'depends_on': ['web'],
                'networks': ['appnet']
            }
        },
        'networks': {
            'appnet': {}
        }
    }
    with open(DOCKER_COMPOSE_PATH, 'w') as f:
        yaml.dump(compose_data, f, default_flow_style=False, sort_keys=False)
    print(f"Overwritten {DOCKER_COMPOSE_PATH} with secure configuration and NGINX reverse proxy.")

def main():
    print("Applying Docker security configuration...")
    update_dockerfile()
    update_docker_compose()
    print("Done. Please restart Docker and rebuild your containers if needed.")

if __name__ == '__main__':
    main()
