from diagrams import Cluster, Diagram
from diagrams.programming.language import Python
from diagrams.onprem.database import PostgreSQL
from diagrams.onprem.client import User
from diagrams.onprem.container import Docker
from diagrams.generic.os import LinuxGeneral
from diagrams.onprem.security import Vault
from diagrams.onprem.monitoring import Prometheus

with Diagram("Before and After: Hardened App and Docker Infrastructure", filename="architecture_diagram", outformat="png", show=False, direction="TB"):
    user = User("User")

    with Cluster("Before State (Insecure)"):
        with Cluster("App Container (Root User)"):
            flask_app_before = Python("Flask App\n- root user\n- eval\n- cmd injection\n- hardcoded pwd")
            docker_engine_before = Docker("Docker Engine\n(No daemon.json)")
        db_before = PostgreSQL("PostgreSQL\n(No volume)")

        user >> flask_app_before >> db_before

    with Cluster("After State (Hardened)"):
        with Cluster("Docker Hardening"):
            daemon_json = Docker("daemon.json:\n- userns-remap\n- no proxy\n- live-restore")
            secure_image = Docker("Patched Base Image")
            non_root_user = LinuxGeneral("Non-root USER")
            healthcheck = Prometheus("Healthcheck")

        with Cluster("Secure App Container"):
            flask_app_after = Python("Flask App\n- ast.literal_eval()\n- input validation\n- ENV password")
            secrets = Vault("ENV Vars")
            limits = LinuxGeneral("Mem & PID limits")
            read_only_fs = LinuxGeneral("Read-only FS")

        db_after = PostgreSQL("PostgreSQL (Volume Mounted)")

        # Logical flows and enforcement
        user >> flask_app_after >> db_after
        secrets >> flask_app_after
        limits >> flask_app_after
        read_only_fs >> flask_app_after
        daemon_json >> flask_app_after
        secure_image >> flask_app_after
        non_root_user >> flask_app_after
        healthcheck >> flask_app_after
