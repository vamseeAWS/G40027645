from diagrams import Diagram, Cluster, Edge
from diagrams.onprem.network import Internet
from diagrams.programming.language import Java
from diagrams.onprem.client import User
from diagrams.generic.compute import Rack
from diagrams.programming.language import Bash

with Diagram("Log4Shell Architecture: Before vs After", direction="LR", show=False):

    user = User("Attacker")
    internet = Internet("Internet")

    with Cluster("Before Mitigation"):
        with Cluster("App Container"):
            app_vuln = Java("Spring Boot App\n(Log4j 2.14.1)")
            vuln_log = Java("LogController.java\n(No input validation)")
        ldap_vuln = Rack("LDAPRefServer\n(marshalsec)")
        http_server = Rack("HTTP Server\n(Python)")

        user >> internet >> app_vuln
        app_vuln >> vuln_log
        app_vuln >> Edge(label="JNDI\n${jndi:ldap://...}") >> ldap_vuln
        ldap_vuln >> http_server

    with Cluster("After Mitigation"):
        with Cluster("App Container"):
            app_fixed = Java("Spring Boot App\n(Log4j 2.17.0)")
            patched_log = Java("LogController.java\n(JNDI blocked)")
        ldap_fixed = Rack("LDAPRefServer\n(marshalsec)")
        http_fixed = Rack("HTTP Server\n(Python)")
        patch_script = Bash("patch_log4j.sh")

        user >> internet >> app_fixed
        app_fixed >> patched_log
        app_fixed >> Edge(style="dashed", label="Blocked JNDI input") >> ldap_fixed
        patched_log >> Edge(style="dotted", color="gray", label="Updated via script") >> patch_script
        ldap_fixed >> http_fixed
