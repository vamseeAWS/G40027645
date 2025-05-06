## Threat Model Documentation for Flask App and Docker Infrastructure

This report covers the security assessment of a web application built using the Flask framework and containerized with Docker. The assessment employs the STRIDE threat modeling framework, maps relevant attack vectors using the MITRE ATT&CK for Containers matrix, and aligns observed vulnerabilities to the NIST 800-53 security control framework. 

In the before state, the application and container infrastructure exhibited multiple high-risk vulnerabilities, including hardcoded passwords, command injection, arbitrary code execution via the `eval()` function, and the execution of the application as the root user within a Docker container. These flaws collectively exposed the system to risks of privilege escalation, remote code execution, denial of service, and potential lateral movement within the container or host environment.

Applying the STRIDE framework revealed several threat categories relevant to the original configuration. Spoofing threats were evident due to the hardcoded password in the application, allowing attackers to potentially impersonate legitimate users or access privileged functions. This was addressed by retrieving the password from an environment variable instead of hardcoding it in the application code. Though authentication controls remained minimal, this change reduced the risk of credential exposure [1].

Tampering threats were prominent because the application ran with root privileges, making it easier for an attacker to alter system files or configurations. This issue was mitigated by modifying the Dockerfile to add a non-root user named `appuser` and executing the application under this user. Additionally, Docker Compose settings were updated to enforce read-only file systems, limit the number of process IDs (pids_limit), and restrict memory usage (mem_limit), thereby reducing the chances of container tampering [2].

Repudiation threats were less explicitly mitigated. Initially, there was no logging of user actions, especially those involving the `eval()` and `ping` endpoints, which could be exploited without traceability. Although error handling was improved in the updated application, full logging and auditing mechanisms were not introduced. To fully address repudiation, future enhancements should include centralized logging and request auditing [3].

Information disclosure risks come from the use of unsanitized input in the `subprocess.check_output()` method and the direct evaluation of user input via the `eval()` function. These issues allowed attackers to gain insights into system behavior and execute arbitrary commands. The revised application addressed these flaws by using the `ipaddress` module to validate IP inputs and `ast.literal_eval()` to safely parse user-provided expressions. These controls significantly reduced the risk of information leakage [4].

Denial of service (DoS) threats were possible due to unrestricted resource usage in the initial deployment. Attackers could send repeated requests to resource-intensive endpoints or exploit subprocess execution to exhaust system memory. After the update, Docker Compose was configured to enforce memory and PID limits on the web container. These constraints reduced the potential for successful DoS attacks [5].

Elevation of privilege was a critical threat in the original state due to the use of `shell=True` in subprocess calls and running the container as the root user. These practices allowed attackers to escalate privileges and potentially compromise the host system. In the updated version, `shell=True` was removed, validated arguments were passed directly to `subprocess`, and the application ran under a limited privilege user. These changes effectively mitigated the risk of privilege escalation [6].

The MITRE ATT&CK for Containers framework was used to identify tactics and techniques relevant to both the vulnerable and secured states of the application. Prior to remediation, the system was susceptible to multiple ATT&CK techniques, including T1611 (Escape to Host), T1609 (Container Administration Command), T1203 (Exploitation for Privilege Escalation), and T1059 (Command and Scripting Interpreter). These techniques aligned with observed issues such as root container execution, command injection, and unsafe use of shell interpreters. The remediated version addressed these vulnerabilities by implementing user access restrictions, removing shell access, and performing command validation, thereby reducing exposure to these techniques [7].

While some techniques like T1610 (Deploy Container) remained partially addressed due to a lack of image scanning or runtime monitoring tools, further improvements can be made by integrating tools such as Docker Scout, Snyk, or Trivy into the CI/CD pipeline. These tools would automate vulnerability scanning and further improve container security posture [8].

The report maps specific vulnerabilities to relevant NIST 800-53 controls to align remediation efforts with Risk Management Framework(RMF). The use of hardcoded passwords violated control IA-5(1) (Authenticator Management), which was addressed by using environment variables. Command injection vulnerabilities violated SI-10 (Input Validation) and SC-39 (Process Isolation), both of which were mitigated by input sanitization and execution control. Running containers as the root user violated AC-6 (Least Privilege) and CM-6 (Configuration Settings), which were addressed by using a dedicated non-root user. Denial of service risks from unrestricted resource usage were linked to SC-5 and SC-6 (Resource Availability and Boundary Protection), mitigated through memory and process limits. The lack of health checks and dependency management were tied to SI-4 (Monitoring) and SI-2 (Flaw Remediation), and were improved by incorporating a `HEALTHCHECK` instruction in the Dockerfile and upgrading base dependencies during the build process [9].

To summarize, the application and container infrastructure under gone several security enhancements that improved its resistance to known threats. The original state left the system vulnerable to a wide range of attacks, including command injection, code execution, and privilege escalation. These vulnerabilities were addressed through a combination of secure coding practices, Docker hardening techniques, and infrastructure configuration improvements.

Despite these advancements, opportunities for further enhancement remain. It is recommended to implement centralized logging for user actions and security events, integrate image scanning and vulnerability assessment tools in the build process, and adopt secure secrets management solutions such as Docker Secrets or HashiCorp Vault. Additionally, enforcing encrypted communications and secure database connections will strengthen the overall security of the system.

The improvements outlined in thisreport represent a measurable reduction in risk and align the application with best practices in containerized application security. This case study serves as a practical example of how applying STRIDE analysis, MITRE ATT&CK mapping, and NIST control alignment can guide meaningful and actionable security improvements.

---

## References

[1] OWASP Docker Top 10. (n.d.). Retrieved from https://owasp.org/www-project-docker-top-10/

[2] CIS Docker Benchmark. (n.d.). Center for Internet Security. Retrieved from https://www.cisecurity.org/benchmark/docker

[3] NIST SP 800-53 Rev. 5. (2020). Security and Privacy Controls for Information Systems and Organizations. National Institute of Standards and Technology. https://csrc.nist.gov/publications/detail/sp/800-53/rev-5/final

[4] MITRE ATT&CK for Containers. (n.d.). Retrieved from https://attack.mitre.org/matrices/enterprise/container/

[5] Docker Documentation. (n.d.). Resource constraints. Retrieved from https://docs.docker.com/config/containers/resource_constraints/

[6] Snyk Security Scanning. (n.d.). Retrieved from https://snyk.io/product/container-vulnerability-management/

[7] Trivy - Vulnerability Scanner for Containers. (n.d.). Retrieved from https://github.com/aquasecurity/trivy

[8] Docker Scout. (n.d.). Secure your containers. Retrieved from https://docs.docker.com/scout/

[9] Healthcheck instruction in Dockerfile. (n.d.). Retrieved from https://docs.docker.com/engine/reference/builder/#healthcheck
