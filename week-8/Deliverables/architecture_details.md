# IAM Architecture Overview – Lab 1, Lab 2 & Lab 3

This repository demonstrates how to design and implement secure Identity and Access Management (IAM) architectures using **Keycloak** as the Identity Provider (IdP). It showcases both modern and legacy authentication flows using **OIDC** and **SAML**, integrated with Python and PHP applications.

---

##  Architecture Components and Purpose

### 🔹 Keycloak (Identity Provider)
- Used in **all labs** as the centralized authentication system.
- Manages:
  - Realms (e.g., `CentralIAM`)
  - Users, Groups, Roles
  - OIDC and SAML Clients
  - Signing certificates
- Backed by **PostgreSQL** for persistent configuration and sessions.

> **Key Role**: Trust anchor that issues tokens or assertions after successful authentication.

---

### 🔹 Flask Application (Protected Resource Server)
- A Python app requiring authentication before access.
- **Lab 2**: Uses **OIDC** and validates JWTs directly.
- **Lab 3**: Protected by **SimpleSAMLphp**, which handles SAML assertions.

> **Key Role**: Simulates a microservice validating external identity assertions.

---

### 🔹 SimpleSAMLphp (Service Provider) – Lab 3
- Acts as the **SAML Service Provider (SP)** in the Lab 3 flow.
- Routes authentication to Keycloak via SAML 2.0.
- Validates the signature on SAML assertions using a trusted X.509 certificate.

> **Key Role**: Middleware bridge that supports legacy SAML-based federation.

---

### 🔹 OAuth2 / OIDC – Lab 2
- Used in Lab 2 for secure, modern API protection.
- Flask requests OIDC login from Keycloak, receives a JWT access token, and verifies it.

> **Key Role**: Demonstrates modern token-based authorization flow.

---

### 🔹 SAML 2.0 – Lab 3
- Used in Lab 3 for legacy authentication integration.
- Keycloak serves as a **SAML IdP**, and SimpleSAMLphp acts as the SP.
- Flow:
  1. Browser accesses SP → SP redirects to Keycloak
  2. User authenticates in Keycloak
  3. Signed SAML response returned to SP
  4. SP validates and grants access
- Sessions visible in Keycloak confirm login success. 
-  Note: SP UI post-login may still show blank, but IdP confirms session/Debugging final UI rendering in SimpleSAMLphp

> **Key Role**: Demonstrates enterprise SAML federation and assertion trust.

---

### 🔹 Docker Compose
- Each lab uses Docker containers defined in `docker-compose.yml`.
- Containers include:
  - Keycloak (w/ import realm)
  - PostgreSQL
  - Flask app or SimpleSAMLphp SP
  - LDAP & phpLDAPadmin

> **Key Role**: Provides local, reproducible, and network-isolated IAM environments.

---

### 🔹 Automation Scripts
- Found in each lab as `setup.sh`.
- Automatically:
  - Starts containers
  - Imports Keycloak realm via `realm-export.json`
  - Loads sample users and client configurations

> **Key Role**: Eliminates manual setup and ensures consistency across runs.

---

## 🧪 Labs Summary

| Lab   | Auth Method     | Protocol    | Middleware        | Language     | Token Format   |
|--------|------------------|-------------|--------------------|--------------|----------------|
| Lab 1 | Basic Auth / LDAP | N/A         | N/A                | NodeJS       | N/A            |
| Lab 2 | OIDC              | OAuth2/OIDC | Flask (direct)     | Python       | JWT            |
| Lab 3 | SAML              | SAML 2.0    | SimpleSAMLphp (SP) | PHP + Flask  | XML Assertion  |

---



