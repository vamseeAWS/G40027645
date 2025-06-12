from diagrams import Diagram, Cluster
from diagrams.onprem.compute import Server
from diagrams.programming.language import Python
from diagrams.onprem.client import User
from diagrams.onprem.network import Internet
from diagrams.generic.network import Firewall

with Diagram("Lab 2: OAuth2 & OIDC IAM Architecture", show=False):

    user = User("Browser/User")
    internet = Internet("Internet")

    with Cluster("Docker Compose Stack"):
        keycloak = Server("Keycloak (IdP)")
        flask_app = Python("Flask API (Protected)")

        with Cluster("OAuth2/OIDC Flow"):
            auth_token = Server("Access Token (JWT)")
            introspect = Firewall("Token Validation")

        # Flow: user logs in -> gets token -> accesses API
        user >> internet >> keycloak
        keycloak >> auth_token
        auth_token >> flask_app
        user >> internet >> flask_app

        # Flask validates the token via public key / introspection
        flask_app >> introspect
        introspect >> keycloak
