.PHONY: up down logs test

up:
	@echo "[*] Running setup.sh script..."
	./setup.sh

down:
	@echo "[*] Stopping and removing lab containers..."
	docker-compose down -v

logs:
	@echo "[*] Showing logs (Ctrl+C to stop)..."
	docker-compose logs -f

test:
	@echo "[*] Running Keycloak login test..."
	python3 test_login.py
