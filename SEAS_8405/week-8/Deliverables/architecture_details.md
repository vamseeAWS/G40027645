# IAM Architecture Overview – Lab 1, Lab 2 & Lab 3

This lab demonstrates how to design and implement secure Identity and Access Management (IAM) architectures using **Keycloak** as the Identity Provider (IdP). It showcases authentication flows using **OIDC** and **SAML**, integrated with Python and PHP applications. Each lab builds on the previous one, increasing complexity and depth of IAM integration.

---

## Architecture Components and Purpose

### 🔹 Keycloak (Identity Provider)

* Used in **all labs** as the centralized authentication and authorization platform.
* Keycloak Capabilities Demonstrated:

  * **Realm configuration**: Multiple realms for different applications (e.g., `CentralIAM`, `FintechApp`).
  * **Client configuration**: Public and confidential clients configured for OIDC and SAML.
  * **User federation**: Integrates with LDAP via admin credentials and DN structure.
  * **Protocol support**: Handles both OpenID Connect and SAML 2.0 flows.
  * **Token issuance**: Generates signed JWTs for OIDC and XML assertions for SAML.
* Backed by **PostgreSQL** to persist configurations and session states.

> **Key Role**: Serves as the policy enforcement point (PEP) and token/assertion issuer. Centralizes IAM policies and session management.

---

### 🔹 Flask Application (Protected Resource Server)

* Python microservice requiring token-based authentication.
* Lab 2: Flask validates JWT access tokens against Keycloak's public keys via its discovery endpoint (`/.well-known/openid-configuration`).
* Lab 3: Flask integrated with SAML SP (SimpleSAMLphp) for handling legacy SSO flows.

> **Key Role**: Emulates backend APIs that enforce authentication and validate identity claims before allowing access.

---

### 🔹 Lab 1 – Keycloak + Intranet App (Initial IAM Architecture)

* **Setup Details**:

  * Docker Compose defines Keycloak, LDAP, phpLDAPadmin, and a Node.js intranet frontend.
  * LDAP container initialized with `seed.ldif` containing organizational units (OU), groups, and users.
  * Keycloak realm manually configured or imported using REST API or UI.
* **Authentication**:

  * The Node.js app triggers login flow by redirecting to Keycloak’s login endpoint.
  * Login credentials are checked against the seeded LDAP.
  * Successful authentication establishes session and redirects back to the app.
* **Makefile Instructions**:

  * `make up`: Starts all containers and injects LDAP entries.
  * `make test`: Sends HTTP request to confirm redirect to Keycloak login, indicating setup success.
  * `make logs`: Shows real-time logs for debugging Keycloak or LDAP failures.

> **Key Role**: Introduces containerized IAM infrastructure and enables user authentication using LDAP as an identity store.

---

### 🔹 Lab 2 – Flask + Keycloak (OIDC Integration)

* **Setup Details**:

  * Compose services include Flask (Python 3), Keycloak, PostgreSQL.
  * `setup.sh` provisions `FintechApp` realm, OIDC client `flask-client`, and test users.
  * JWT tokens issued by Keycloak are RS256-signed and verified using the JWKS endpoint.
* **Authentication Flow**:

  1. User sends credentials to Flask.
  2. Flask obtains access token from Keycloak using password grant.
  3. Flask validates token signature, `iss`, `aud`, and `exp` claims.
  4. Protected endpoint returns user profile if authenticated.
* **Makefile Instructions**:

  * `make test`: Automates token acquisition and validation request using `curl` and `jq`.
  * `make reset`: Deletes Keycloak data, re-imports realms, and resets environment.
* **Security Enhancements**:

  * Secrets are loaded via environment variables and not hardcoded.
  * Network traffic between services confined to Docker network.
  * Token validation logic in Flask includes expiration and scope enforcement.

> **Key Role**: Demonstrates stateless API security using industry-standard JWT tokens and dynamic OIDC discovery.

---

### 🔹 Lab 3 – Flask + SimpleSAMLphp + Keycloak (SAML Integration)

* **Setup Details**:

  * Additional services include Apache PHP server running SimpleSAMLphp configured as a SAML SP.
  * Keycloak realm (`CentralIAM`) contains SAML client with configured ACS (Assertion Consumer Service) and metadata.
  * Keycloak metadata exported and loaded into SimpleSAMLphp configuration.
* **Authentication Flow**:

  1. User accesses Flask via SP.
  2. SP redirects to Keycloak IdP.
  3. Keycloak prompts for login (backed by LDAP).
  4. On success, a signed SAML assertion is posted back to SP.
  5. SP validates assertion, establishes session, and returns to Flask.
* **Makefile Instructions**:

  * `make test`: Runs `test_login.py` to simulate SP-initiated login.
  * `make logs`: Useful for viewing Apache SAML logs and verifying assertion receipt.
* **Technical Challenges Solved**:

  * Certificate trust between SP and IdP using self-signed certs.
  * Parsing of SAML assertion contents (NameID, attributes).
  * Correct metadata binding for POST/Redirect SAML bindings.

> **Key Role**: Emulates real-world enterprise federated login setups using SAML assertions and multi-party trust chains.

---

### 🔹 Docker Compose and Containerization

* Shared Docker strategies across labs:

  * Named volumes for persistence (`keycloak_data`, `kcdb_data`, `ldap_data`)
  * Isolated Docker networks for internal communication
  * Service dependencies defined via `depends_on`
* Specific Container Roles:

  * **Keycloak**: Central IAM engine
  * **LDAP**: User and group directory
  * **PostgreSQL**: Configuration DB for Keycloak
  * **Node.js/Flask/PHP**: Resource servers to demonstrate IAM integration

> **Key Role**: Enables consistent, reproducible environments and simplifies IAM platform experimentation.

---

### 🔹 Automation Scripts and Makefile Usage

Each lab's `Makefile` encapsulates DevOps-style operations for consistency and repeatability:

#### Lab 1

* `up`: Builds and runs services, injects LDAP users.
* `test`: Confirms redirect to Keycloak on app access.
* `reset`: Cleans and rebuilds from scratch.

#### Lab 2

* `setup.sh`: Automates realm/client/user configuration using Keycloak Admin CLI (kcadm).
* `test`: Validates full end-to-end OAuth2 password grant flow.
* Includes token integrity check and JSON field presence verification.

#### Lab 3

* `setup.sh`: Sets up SAML metadata, SP configuration, and realm import.
* `test`: Simulates browser login using headless Python script.
* `logs`: Helps with SP error tracking and SAML response debugging.

> **Key Role**: Provides declarative, testable, and automated IAM deployment and validation flow.

---

## Labs Summary Table

| Lab   | Auth Method       | Protocol    | Middleware         | Language    | Token Format  |
| ----- | ----------------- | ----------- | ------------------ | ----------- | ------------- |
| Lab 1 | Basic Auth / LDAP | N/A         | N/A                | NodeJS      | N/A           |
| Lab 2 | OIDC              | OAuth2/OIDC | Flask (direct)     | Python      | JWT           |
| Lab 3 | SAML              | SAML 2.0    | SimpleSAMLphp (SP) | PHP + Flask | XML Assertion |

---

## Conclusion

This lab series progressively builds your understanding of IAM architectures:

* **Lab 1**: Introduces the basics of directory services and identity federation setup.
* **Lab 2**: Demonstrates stateless token-based security using OAuth2/OIDC and validates API protection logic.
* **Lab 3**: Showcases federation with XML-based SAML assertions, providing insight into legacy systems integration.

Together, these labs provide a comprehensive, technical foundation for secure authentication design and implementation using containerized environments and automation best practices.
