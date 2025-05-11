# Summary Report for Flask App + Docker + NGINX
 
**Application Stack:** Flask Web Application, Docker Containers, NGINX Reverse Proxy  
**Frameworks Used:** STRIDE Threat Model, MITRE ATT&CK for Containers, NIST 800-53 Control Mapping

---

## 1. Overview

This report documents the threat modeling and security assessment of a Flask-based web application deployed in Docker containers, fronted by an NGINX reverse proxy. The objective was to identify and mitigate key security vulnerabilities across the application and container infrastructure using STRIDE, MITRE ATT&CK for Containers, and NIST 800-53 controls.

The initial "before" architecture included:
- Flask application running as root inside a Docker container
- Usage of `eval()` and insecure shell commands
- Hardcoded passwords in source code
- An unused PostgreSQL database container exposed internally

The final "after" architecture removed unnecessary components, implemented hardened container configurations, and introduced an NGINX reverse proxy for external access.

---

## 2. Architectural Comparison

### Before State (Insecure)

In the before state, the application and container infrastructure exhibited multiple high-risk vulnerabilities, including hardcoded passwords, command injection, arbitrary code execution via the `eval()` function, and the execution of the application as the root user within a Docker container. Additionally, a PostgreSQL database container was included in the deployment but was not actively used by the Flask application. This introduced unnecessary attack surface and risk, including potential for unauthorized access, exposure of default credentials, and unmonitored persistence.

These flaws collectively exposed the system to risks of privilege escalation, remote code execution, denial of service, information disclosure, and potential lateral movement within the container or host environment.

- Flask app ran as root with full privileges
- `eval()` and `subprocess` calls were unsanitized (`shell=True`)
- Hardcoded credentials in application code
- Docker container had no `HEALTHCHECK`, resource limits, or access controls
- Unused PostgreSQL container introduced avoidable attack surface

### After State (Hardened)

In the after state, the application and container infrastructure were significantly re-architected to reduce security risks, align with container hardening best practices, and eliminate unnecessary components. The Flask application was reconfigured to run as a non-root user within a constrained Docker environment, and the insecure usage of eval() and shell-based subprocess execution was replaced with safer alternatives. Sensitive configuration values such as passwords are now injected securely via environment variables instead of being hardcoded in the source code.

Resource restrictions were introduced in the Docker Compose configuration to defend against denial of service (DoS) conditions by limiting the container's memory and process count, and enforcing a read-only file system to prevent unauthorized file modifications. A HEALTHCHECK directive was also added to monitor container availability and support proactive fault detection.

Additionally, a major architectural change was the removal of the previously unused PostgreSQL container. Though it was not in use, its presence introduced unnecessary complexity and potential risk — including exposure through default settings, possible attack vector chaining, and lack of visibility. Its elimination represents an adherence to the principle of least functionality, reducing the attack surface and simplifying the deployment environment.

The NGINX reverse proxy was introduced as the sole public-facing component. It forwards incoming HTTP requests to the Flask application running internally, allowing for future implementation of access control measures (e.g., personal access tokens or IP allowlists) while keeping the application container isolated from direct external access.

These combined changes significantly improved the security posture of the deployment and reduced exposure to common container-based attack vectors.

- Flask app runs as non-root user (`appuser`)
- Replaced `eval()` with `ast.literal_eval()` or sanitized `eval()` with regex
- Passwords loaded via environment variables (`.env` file)
- Docker Compose restricts memory (`mem_limit`), processes (`pids_limit`), and sets `read_only` filesystem
- NGINX reverse proxy is the only exposed service; Flask app remains internal-only
- PostgreSQL container removed to reduce unnecessary attack surface
- Health monitoring added using Docker `HEALTHCHECK`

---

## 3. STRIDE Threat Model 

Applying the STRIDE framework revealed several threat categories relevant to the original configuration. Spoofing threats were evident due to the hardcoded password in the application, allowing attackers to potentially impersonate legitimate users or access privileged functions. This was addressed by retrieving the password from an environment variable instead of hardcoding it in the application code. Though authentication controls remained minimal, this change reduced the risk of credential exposure [1].

Tampering threats were prominent because the application ran with root privileges, making it easier for an attacker to alter system files or configurations. This issue was mitigated by modifying the Dockerfile to add a non-root user named `appuser` and executing the application under this user. Additionally, Docker Compose settings were updated to enforce read-only file systems, limit the number of process IDs (`pids_limit`), and restrict memory usage (`mem_limit`), thereby reducing the chances of container tampering [2].

Repudiation threats were less explicitly mitigated. Initially, there was no logging of user actions, especially those involving the `eval()` and `ping` endpoints, which could be exploited without traceability. Although error handling was improved in the updated application, full logging and auditing mechanisms were not introduced. To fully address repudiation, future enhancements should include centralized logging and request auditing [3].

Information disclosure risks came from the use of unsanitized input in the `subprocess.check_output()` method and the direct evaluation of user input via the `eval()` function. These issues allowed attackers to gain insights into system behavior and execute arbitrary commands. The revised application addressed these flaws by using the `ipaddress` module to validate IP inputs and `ast.literal_eval()` to safely parse user-provided expressions. These controls significantly reduced the risk of information leakage [4].

Denial of service (DoS) threats were possible due to unrestricted resource usage in the initial deployment. Attackers could send repeated requests to resource-intensive endpoints or exploit subprocess execution to exhaust system memory. After the update, Docker Compose was configured to enforce memory and PID limits on the web container. These constraints reduced the potential for successful DoS attacks [5].

Elevation of privilege was a critical threat in the original state due to the use of `shell=True` in subprocess calls and running the container as the root user. These practices allowed attackers to escalate privileges and potentially compromise the host system. In the updated version, `shell=True` was removed, validated arguments were passed directly to `subprocess`, and the application ran under a limited privilege user. These changes effectively mitigated the risk of privilege escalation [6].

Another key architectural improvement in the after state was the removal of the unused PostgreSQL container. While not directly connected to the Flask application, the database service increased the attack surface, posed potential misconfiguration risks (e.g., default passwords, open ports), and represented an unnecessary asset that could be exploited. Its removal simplifies the infrastructure and aligns with the principle of least functionality.

| STRIDE Category | Before State Issue | After State Mitigation |
|-----------------|---------------------|-------------------------|
| **Spoofing** | Hardcoded passwords | Passwords moved to environment variables; future access tokens possible via NGINX |
| **Tampering** | App ran as root; no FS restrictions | Non-root user, read-only file system, resource limits |
| **Repudiation** | No logs or traceability | Basic error handling added; logging planned via NGINX |
| **Information Disclosure** | Unsafe input to `eval()` and subprocess | Strict input validation with `ipaddress` and sanitized math eval |
| **DoS** | No resource constraints | Memory/PID limits added; health check implemented |
| **Privilege Escalation** | `shell=True` and root access | Removed `shell=True`, validated input, least privilege enforced |

---

## 4. MITRE ATT&CK for Containers Mapping

The MITRE ATT&CK for Containers framework was used to identify tactics and techniques relevant to both the vulnerable and secured states of the application. Prior to remediation, the system was susceptible to multiple ATT&CK techniques, including T1611 (Escape to Host), T1609 (Container Administration Command), T1203 (Exploitation for Privilege Escalation), and T1059 (Command and Scripting Interpreter). These techniques aligned with observed issues such as root container execution, command injection, and unsafe use of shell interpreters. The remediated version addressed these vulnerabilities by implementing user access restrictions, removing shell access, and performing command validation, thereby reducing exposure to these techniques [7].

Additionally, T1525 (Implant Internal Image) was a relevant risk due to the presence of an unused and potentially unmonitored PostgreSQL container. This was resolved by eliminating the unnecessary container, reducing the scope of attackable services.

While some techniques like T1610 (Deploy Container) remained partially addressed due to a lack of image scanning or runtime monitoring tools, further improvements can be made by integrating tools such as Docker Scout, Snyk, or Trivy into the CI/CD pipeline. These tools would automate vulnerability scanning and further improve container security posture [8].


| Technique ID | Name | Before Risk | After Mitigation |
|--------------|------|-------------|------------------|
| T1059 | Command & Scripting Interpreter | Unsafe `eval()`, subprocess | Replaced with sanitized logic |
| T1203 | Exploitation for Privilege Escalation | Root user access | Dropped privileges |
| T1609 | Container Admin Command | Docker misuse, no controls | Secured with Compose limits |
| T1611 | Escape to Host | Poor process isolation | Enforced user namespaces, limits |
| T1525 | Implant Internal Image | Unused PostgreSQL container | Removed entirely |

---

## 5. NIST 800-53 Control Mapping

The report maps specific vulnerabilities to relevant NIST 800-53 controls to align remediation efforts with the Risk Management Framework (RMF). The use of hardcoded passwords violated control IA-5(1) (Authenticator Management), which was addressed by using environment variables. Command injection vulnerabilities violated SI-10 (Input Validation) and SC-39 (Process Isolation), both of which were mitigated by input sanitization and execution control. Running containers as the root user violated AC-6 (Least Privilege) and CM-6 (Configuration Settings), which were addressed by using a dedicated non-root user. Denial of service risks from unrestricted resource usage were linked to SC-5 and SC-6 (Resource Availability and Boundary Protection), mitigated through memory and process limits. The lack of health checks and dependency management were tied to SI-4 (Monitoring) and SI-2 (Flaw Remediation), and were improved by incorporating a `HEALTHCHECK` instruction in the Dockerfile and upgrading base dependencies during the build process. Finally, the removal of the unused PostgreSQL container reduced boundary exposure in accordance with SC-7 (Boundary Protection) [9].


| Control | Description | Before | After |
|---------|-------------|--------|-------|
| IA-5(1) | Authenticator Management | Hardcoded credentials | Env vars |
| SI-10 / SC-39 | Input Validation & Process Isolation | Unsafe eval/subprocess | Sanitized input, no shell |
| AC-6 / CM-6 | Least Privilege & Secure Config | Root user, unconfined | Non-root user, config limits |
| SC-5 / SC-6 | Resource & Boundary Protection | No resource enforcement | mem_limit, pids_limit, read-only |
| SI-4 / SI-2 | Monitoring & Flaw Remediation | No monitoring | Docker `HEALTHCHECK` added |
| SC-7 | Boundary Protection | Exposed, unused DB | PostgreSQL container removed |

---

## 6. Recommendations for Continued Hardening

- Add centralized logging for Flask and NGINX requests
- Enforce token-based access via NGINX or a middleware auth proxy
- Use secure secret stores like **HashiCorp Vault** or **Docker Secrets**
- Add HTTPS termination in NGINX for encrypted browser access

---

## 7. Conclusion

The refactored Docker deployment, with NGINX as a reverse proxy and the removal of PostgreSQL, significantly reduces risk and aligns with container security best practices. This improvement, guided by STRIDE, MITRE ATT&CK, and NIST 800-53, demonstrates a clear uplift in the application’s security posture.

---

## References

- OWASP Docker Top 10: https://owasp.org/www-project-docker-top-10/  
- CIS Docker Benchmark: https://www.cisecurity.org/benchmark/docker  
- NIST SP 800-53 Rev. 5: https://csrc.nist.gov/publications/detail/sp/800-53/rev-5/final  
- MITRE ATT&CK for Containers: https://attack.mitre.org/matrices/enterprise/container/  
- Docker HEALTHCHECK: https://docs.docker.com/engine/reference/builder/#healthcheck  
- Docker Resource Limits: https://docs.docker.com/config/containers/resource_constraints/  
- Trivy Scanner: https://github.com/aquasecurity/trivy  
- Docker Scout: https://docs.docker.com/scout/  
- Snyk Container Scanning: https://snyk.io/product/container-vulnerability-management/
