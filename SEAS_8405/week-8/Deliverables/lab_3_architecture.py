from diagrams import Diagram, Edge
from diagrams.onprem.client import Users
from diagrams.onprem.network import Nginx
from diagrams.programming.language import Php
from diagrams.onprem.database import Postgresql
from diagrams.onprem.compute import Server
from diagrams.custom import Custom
import os

with Diagram("Lab 3: SAML SSO - Keycloak ↔ SimpleSAMLphp", show=False, direction="LR"):

    user = Users("Browser/User")
    sp = Php("SimpleSAMLphp SP")
    nginx = Nginx("SP Frontend (8082)")
    keycloak = Server("Keycloak IdP (CentralIAM)")  # Generic Server instead of Keycloak icon
    postgres = Postgresql("Keycloak DB")

    # Optional cert representation
    cert = Custom("X.509 Cert", "./cert_icon.png") if os.path.exists("./cert_icon.png") else Custom("X.509 Cert", "https://raw.githubusercontent.com/mingrammer/diagrams/master/resources/ssl/certificate.png")

    # Flow
    user >> Edge(label="1. Access SP URL") >> nginx
    nginx >> Edge(label="2. Auth Request") >> sp
    sp >> Edge(label="3. Redirect w/ SAMLRequest") >> keycloak
    keycloak >> Edge(label="4. Login form (HTML)") >> user
    user >> Edge(label="5. Submit creds") >> keycloak
    keycloak >> Edge(label="6. Auth + SAML Response\n+ Assertion Signed") >> sp
    sp >> Edge(label="7. Grant session\n(login success)") >> user

    keycloak >> Edge(style="dotted", label="Stores users") >> postgres
    keycloak >> Edge(style="dotted", label="Provides certificate") >> cert
    sp << Edge(style="dotted", label="Trusts IdP via cert") << cert
