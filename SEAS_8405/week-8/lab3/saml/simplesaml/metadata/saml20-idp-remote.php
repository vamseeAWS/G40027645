<?php

$metadata['http://localhost:8080/realms/CentralIAM'] = [
    'name' => ['en' => 'Keycloak SAML IdP'],
    'description' => 'Keycloak at localhost',
    'SingleSignOnService' => 'http://localhost:8080/realms/CentralIAM/protocol/saml',
    'SingleLogoutService' => 'http://localhost:8080/realms/CentralIAM/protocol/saml',
    'NameIDFormat' => 'urn:oasis:names:tc:SAML:1.1:nameid-format:unspecified',
    'certificate' => 'keycloak-cert.pem',  // Relative to cert dir or absolute path
];


