const express = require('express');
const session = require('express-session');
const Keycloak = require('keycloak-connect');

// Environment config
const KEYCLOAK_URL = process.env.KEYCLOAK_URL || 'http://localhost:8080/auth';  // Fixed
const REALM = process.env.KEYCLOAK_REALM || 'CentralIAM';
const CLIENT_ID = process.env.KEYCLOAK_CLIENT_ID || 'intranet';

const memoryStore = new session.MemoryStore();
const app = express();

// Session middleware
app.use(session({
  secret: 'a very secret key',
  resave: false,
  saveUninitialized: true,
  store: memoryStore
}));

// Keycloak config
const keycloak = new Keycloak({ store: memoryStore }, {
  realm: REALM,
  'auth-server-url': KEYCLOAK_URL,
  resource: CLIENT_ID,
  'public-client': true,
  'confidential-port': 0
});

app.use(keycloak.middleware());

// Protected route
app.get('/', keycloak.protect(), (req, res) => {
  try {
    const token = req.kauth?.grant?.access_token?.content;
    if (!token) {
      return res.status(403).send('Access denied: no token.');
    }

    const username = token.preferred_username || 'unknown';
    const roles = token.realm_access?.roles || [];

    if (!roles.includes('access-app')) {
      return res.status(403).send("Access denied: missing 'access-app' role.");
    }

    res.send(`<h1>Welcome to the intranet, ${username}!</h1>`);
  } catch (error) {
    console.error('Error verifying access:', error);
    res.status(500).send('Internal Server Error');
  }
});

// Start server
const PORT = 3000;
app.listen(PORT, '0.0.0.0', () => {
  console.log(`Intranet app listening on http://0.0.0.0:${PORT}`);
});
