# Makefile for managing the Mini-SOAR application lifecycle and model training.
# It automatically detects whether to use 'docker-compose' or 'docker compose'.

# --- Automatic Command Detection ---
COMPOSE_CMD := $(shell command -v docker-compose 2>/dev/null)
ifeq ($(COMPOSE_CMD),)
  COMPOSE_CMD := docker compose
endif
# --- End of Command Detection ---

.PHONY: all build up down logs clean train retrain

# Default: start the app
all: up

# --- App Lifecycle Targets ---

build:
	@echo "Building Docker image(s) using '$(COMPOSE_CMD)'..."
	@$(COMPOSE_CMD) build

up:
	@echo "Starting Mini-SOAR application using '$(COMPOSE_CMD)'..."
	@$(COMPOSE_CMD) up --build -d
	@echo ""
	@echo "Application is starting up in detached mode."
	@echo "View logs with: make logs"
	@echo "Access the app at: http://localhost:8501"

down:
	@echo "Stopping Mini-SOAR application using '$(COMPOSE_CMD)'..."
	@$(COMPOSE_CMD) down

logs:
	@echo "Following application logs... (Press Ctrl+C to exit)"
	@$(COMPOSE_CMD) logs -f

clean: down
	@echo "Cleaning up generated files and directories..."
	@rm -rf ./models
	@rm -rf ./data
	@echo "Cleanup complete."

# --- Model Training Targets ---

# Train both supervised and unsupervised models if not already present
train:
	@echo "Training phishing_url_detector and threat_actor_profiler models..."
	@python train_model.py
	@echo "Training complete. Models saved in ./models and ./data."

# Force retraining by removing existing models/data first
retrain: clean
	@echo "Forcing full model retrain..."
	@python train_model.py
	@echo "Retraining complete."
