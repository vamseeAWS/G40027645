from diagrams import Diagram, Cluster
from diagrams.onprem.compute import Server
from diagrams.programming.language import Python
from diagrams.onprem.client import User
from diagrams.onprem.network import Internet

with Diagram("Lab 3: SAML-Based IAM Architecture", show=False):
    user = User("Browser/User")
    internet = Internet("Internet Access")

    with Cluster("Docker Compose Stack"):
        idp = Server("Keycloak (SAML IdP)")

        with Cluster("Flask App (Service Provider)"):
            flask_app = Python("Flask App")
            saml_client = Server("SAML Client Library")

        user >> internet >> idp
        idp >> saml_client >> flask_app
        user >> internet >> flask_app
