from diagrams import Cluster, Diagram
from diagrams.programming.language import Python
from diagrams.onprem.client import User
from diagrams.onprem.container import Docker
from diagrams.onprem.network import Nginx
from diagrams.generic.os import LinuxGeneral
from diagrams.onprem.security import Vault
from diagrams.onprem.monitoring import Prometheus
from diagrams.onprem.database import PostgreSQL

with Diagram("Before and After: Hardened Flask App with Docker and NGINX", filename="architecture_diagram", outformat="png", show=False, direction="TB"):
    user = User("User")

    with Cluster("Before State (Insecure)"):
        with Cluster("App Container (Root User)"):
            flask_app_before = Python("Flask App\n- root user\n- eval\n- cmd injection\n- hardcoded pwd")
            docker_engine_before = Docker("Docker Engine\n(No daemon.json)")
            unused_db = PostgreSQL("Unused PostgreSQL\n- No auth config\n- No volume\n- Exposed port")
        user >> flask_app_before >> unused_db

    with Cluster("After State (Hardened)"):
        with Cluster("Docker Hardening"):
            daemon_json = Docker("daemon.json:\n- userns-remap\n- no proxy\n- live-restore")
            secure_image = Docker("Patched Base Image")
            non_root_user = LinuxGeneral("Non-root USER")
            healthcheck = Prometheus("Healthcheck")

        with Cluster("Secure App Stack"):
            nginx = Nginx("NGINX Reverse Proxy")
            flask_app_after = Python("Flask App\n- safe eval\n- input validation\n- ENV password")
            secrets = Vault("ENV Vars (.env)")
            limits = LinuxGeneral("Mem & PID limits")
            read_only_fs = LinuxGeneral("Read-only FS")

        user >> nginx >> flask_app_after
        secrets >> flask_app_after
        limits >> flask_app_after
        read_only_fs >> flask_app_after
        daemon_json >> flask_app_after
        secure_image >> flask_app_after
        non_root_user >> flask_app_after
        healthcheck >> flask_app_after
