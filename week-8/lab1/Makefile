DOCKER_COMPOSE := docker compose

.PHONY: up down reset logs

# Start the full IAM stack
up:
	@echo "[*] Starting Lab 1: Mini-Enterprise IAM Stack..."
	@$(DOCKER_COMPOSE) up -d --build
	@echo "[*] Waiting for LDAP to be ready..."
	@sleep 5
	@echo "[*] Populating LDAP with seed data from seed.ldif..."
	@cat seed.ldif | docker exec -i "$$($(DOCKER_COMPOSE) ps -q ldap)" ldapadd -x -D "cn=admin,dc=example,dc=com" -w adminpw || echo "LDAP add failed (may already be populated)."

# Shut everything down and remove volumes
down:
	@echo "[*] Shutting down Lab 1 stack..."
	@$(DOCKER_COMPOSE) down -v

# Full clean and rebuild
reset: down
	@echo "[*] Pruning Docker system..."
	@docker system prune -f
	@$(MAKE) up

# Follow logs
logs:
	@$(DOCKER_COMPOSE) logs -f

# Run a basic integration test to verify the intranet app is responding
test:
	@echo "[*] Running Lab 1 connectivity test..."
	@sleep 3
	@echo "[*] Sending HTTP request to intranet app at http://localhost:3000 ..."
	@curl -i --max-time 5 http://localhost:3000 || echo "[-] Request to intranet failed"
	@echo ""
	@echo "[*] If you see a login redirect or valid HTML, Keycloak + Node.js integration is working."
