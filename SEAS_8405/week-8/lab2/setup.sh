#!/bin/bash
set -e

echo "[*] Starting Keycloak and the Flask app..."
docker compose up -d --build

echo "[*] Waiting for Keycloak to be ready..."
until curl -sf http://localhost:8080/realms/master > /dev/null; do
    echo "Waiting for Keycloak to start..."
    sleep 5
done

echo "[*] Keycloak base is up. Waiting for admin endpoints..."
sleep 5

echo "[*] Fetching admin token from Keycloak..."
for i in {1..10}; do
  ADMIN_TOKEN=$(curl -s -X POST "http://localhost:8080/realms/master/protocol/openid-connect/token" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "username=admin" \
    -d "password=admin" \
    -d "grant_type=password" \
    -d "client_id=admin-cli" | jq -r .access_token)

  if [[ "$ADMIN_TOKEN" != "null" && -n "$ADMIN_TOKEN" ]]; then
    echo "[✔] Admin token acquired."
    break
  else
    echo "[!] Failed to get admin token. Retrying..."
    sleep 3
  fi
done

if [[ -z "$ADMIN_TOKEN" || "$ADMIN_TOKEN" == "null" ]]; then
  echo "[✖] Failed to obtain admin token after multiple attempts."
  exit 1
fi

# Check and create realm
REALM_EXISTS=$(curl -s -H "Authorization: Bearer $ADMIN_TOKEN" http://localhost:8080/admin/realms \
  | jq -r '.[] | select(.realm=="FintechApp") | .realm')

if [ "$REALM_EXISTS" == "FintechApp" ]; then
  echo "[!] Realm 'FintechApp' already exists. Skipping creation."
else
  echo "[*] Creating realm 'FintechApp'..."
  curl -s -X POST "http://localhost:8080/admin/realms" \
    -H "Authorization: Bearer $ADMIN_TOKEN" \
    -H "Content-Type: application/json" \
    -d @realm-config.json
  echo "[✔] Realm 'FintechApp' created."
fi

echo "[*] Testing access token retrieval..."
RESPONSE=$(curl -s -X POST "http://localhost:8080/realms/FintechApp/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=password" \
  -d "client_id=flask-client" \
  -d "client_secret=secret" \
  -d "username=testuser" \
  -d "password=password")

echo "$RESPONSE" | jq

ACCESS_TOKEN=$(echo "$RESPONSE" | jq -r .access_token)
if [[ "$ACCESS_TOKEN" == "null" || -z "$ACCESS_TOKEN" ]]; then
  echo "[✖] Failed to retrieve access token for test user."
  exit 1
fi

echo "[✔] Setup complete. Access the Flask app at: http://localhost:15000"
echo "[ℹ️ ] To test manually:"
echo "curl -H \"Authorization: Bearer $ACCESS_TOKEN\" http://localhost:15000"
