.PHONY: up down setup logs test

DOCKER_COMPOSE := docker compose

up:
	$(DOCKER_COMPOSE) up -d --build
	bash setup.sh

down:
	$(DOCKER_COMPOSE) down -v

logs:
	$(DOCKER_COMPOSE) logs -f

test:
	@echo "[*] Fetching access token from Keycloak..."
	@TOKEN=$$(curl -s -X POST http://localhost:8080/realms/FintechApp/protocol/openid-connect/token \
	  -H "Content-Type: application/x-www-form-urlencoded" \
	  -d "grant_type=password" \
	  -d "client_id=flask-client" \
	  -d "client_secret=secret" \
	  -d "username=testuser" \
	  -d "password=password" | jq -r .access_token); \
	if [ "$$TOKEN" = "null" ] || [ -z "$$TOKEN" ]; then \
	  echo "[✖] Failed to obtain access token."; exit 1; \
	fi; \
	echo "[✔] Token acquired. Testing Flask app..."; \
	RESPONSE=$$(curl -s -H "Authorization: Bearer $$TOKEN" http://localhost:15000); \
	echo "[ℹ️ ] Response from Flask:"; echo "$$RESPONSE" | jq .; \
	echo "$$RESPONSE" | jq -e 'has("email") and has("name") and has("sub")' > /dev/null \
	  && echo "[✔] Flask app test passed." \
	  || { echo "[✖] Flask app responded, but required fields missing."; exit 1; }
