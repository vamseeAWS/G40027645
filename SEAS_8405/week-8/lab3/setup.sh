#!/bin/bash
echo "[*] Starting Keycloak, LDAP, PostgreSQL, and Flask App..."
docker-compose down -v
docker-compose build
docker-compose up -d

echo "[*] Waiting for Keycloak to initialize..."
sleep 20

echo "[*] Containers running:"
docker ps
