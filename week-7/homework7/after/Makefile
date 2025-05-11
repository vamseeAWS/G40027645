# Pre-build security checks
check:
	@echo "Running code analysis with Bandit..."
	-docker run --rm -v $(PWD):/app python:3.9-alpine sh -c "pip install bandit && bandit -r /app"
	@echo "Running dependency check with pip-audit..."
	docker run --rm -v $(PWD):/app python:3.9-alpine sh -c "pip install pip-audit && pip-audit -r /app/requirements.txt"

# Host security check
host-security:
	@echo "Running Docker Bench for Security..."
	docker run --rm -v /var/run/docker.sock:/var/run/docker.sock docker/docker-bench-security

# Build Docker image after security checks
dbuild: check
	docker build --no-cache -t mywebapp .

# Run the container
run:
	docker run -p 6000:5000 mywebapp

# Scan the built image for vulnerabilities
scan:
	docker scout cves mywebapp:latest

# Docker Compose commands
build:
	docker compose build --no-cache

start:
	docker compose up -d
	@echo "    Your app is running behind NGINX."
	@echo "🔗 Access it in your browser at: http://localhost:8080/"
	@echo ""
	@echo "   Test your app using these sample URLs:"
	@echo "1. Hello route:"
	@echo "   http://localhost:8080/?name=Alice"
	@echo ""
	@echo "2. Ping route:"
	@echo "   http://localhost:8080/ping?ip=8.8.8.8"
	@echo ""
	@echo "3. Calculate route:"
	@echo "   http://localhost:8080/calculate?expr=5%2B3"


stop:
	docker compose down

logs:
	docker compose logs -f

clean:
	docker system prune -f

restart: stop start
