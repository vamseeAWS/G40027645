from diagrams import Diagram, Cluster
from diagrams.onprem.compute import Server
from diagrams.programming.language import Nodejs
from diagrams.onprem.client import User
from diagrams.onprem.network import Internet

with Diagram("Lab 1 IAM Architecture", show=False):
    user = User("Browser/User")
    internet = Internet("Internet")

    with Cluster("Docker Compose Stack"):
        keycloak = Server("Keycloak (IdP)")

        with Cluster("Intranet App"):
            app = Nodejs("NodeJS App")

        user >> internet >> keycloak
        keycloak >> app
        user >> internet >> app

