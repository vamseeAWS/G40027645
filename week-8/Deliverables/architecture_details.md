# IAM Architecture Overview – Lab 1, Lab 2 & Lab 3

This repository demonstrates how to design and implement secure Identity and Access Management (IAM) architectures using **Keycloak** as the Identity Provider (IdP) and **Flask** as the protected resource server.

---

## Architecture Components and Purpose

### 🔹 Keycloak (Identity Provider)
- Used in all labs.
- Manages user identity, authentication, and issues tokens (JWT or SAML).
- Realms, Clients, and Users are configured within Keycloak.

> **Significance**: Keycloak is the heart of the IAM setup — it authenticates users and issues identity tokens or SAML assertions consumed by the applications.

---

### 🔹 Flask App (Protected Resource Server)
- Found in:
  - `lab2/app/app.py` – uses OAuth2/OIDC
  - `lab3/app/app.py` – uses SAML
- Protects routes and validates authentication data from Keycloak.

> **Significance**: Flask simulates a microservice that enforces authentication and authorization using identity assertions or tokens.

---

### 🔹 OAuth2 & OpenID Connect (Lab 2)
- Implements modern identity delegation using **OAuth2/OIDC**.
- Users authenticate via Keycloak, receive an access token, and use it to access protected Flask routes.

> **Significance**: Learn how to secure APIs using bearer tokens and validate JWTs securely on the backend.

---

### 🔹 SAML-based Authentication (Lab 3)
- Replaces OIDC with **SAML 2.0** for authentication.
- Keycloak is configured as a **SAML Identity Provider (IdP)**.
- Flask acts as the **Service Provider (SP)** using a SAML client library.

> **Significance**: Understand how SAML differs from OIDC and how it is used to integrate with legacy enterprise systems.

---

### 🔹 Docker Compose
- Used in all labs via `docker-compose.yml`.
- Spins up:
  - Keycloak
  - Flask app or NodeJS intranet service
- Handles container orchestration and networking.

> **Significance**: Enables consistent deployment of IAM architecture for local testing and development.

---

### 🔹 Setup Automation Scripts
- Located in:
  - `lab2/setup.sh`
  - `lab3/setup.sh`
- Pre-configure Keycloak:
  - Create realms, clients, and users.
  - Import test data using `.ldif` or `.json`.

> **Significance**: Automation ensures reproducible, secure, and rapid deployment across environments.

---

## 🧪 Labs Summary

| Lab | Authentication | Protocol | App Technology | Token Format |
|-----|----------------|----------|----------------|--------------|
| Lab 1 | Basic Auth / LDAP | N/A | NodeJS         | N/A          |
| Lab 2 | OIDC            | OAuth2  | Flask (Python) | JWT          |
| Lab 3 | SAML            | SAML 2.0| Flask (Python) | XML Assertion |

---
