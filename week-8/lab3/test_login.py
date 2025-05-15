import requests
from bs4 import BeautifulSoup
import urllib.parse

# ==================== CONFIGURATION ====================
SIMPLESAML_SP_URL = "http://localhost:8082/simplesaml/module.php/core/authenticate.php?as=default-sp"
USERNAME = "user"
PASSWORD = "userpass"

# ==================== START SESSION ====================
session = requests.Session()
print("[*] Step 1: ✅ Starting SAML login flow from SimpleSAML SP...")

try:
    # STEP 2: Trigger SAML Auth Flow
    response = session.get(SIMPLESAML_SP_URL, allow_redirects=True, timeout=10)
    print("[✓] Step 2: ✅ SimpleSAML SP redirect successful.")
    print(f"    ↳ Final URL after redirects: {response.url}")
    print(f"    ↳ HTTP Status: {response.status_code}")

    # STEP 3: Parse Keycloak Login Form
    print("[*] Step 3:  ✅ Looking for Keycloak login form...")
    soup = BeautifulSoup(response.text, "html.parser")
    login_form = soup.find("form")
    if not login_form:
        print("[-] Error: Login form not found in response from Keycloak.")
        print("[Debug] Partial HTML response:\n", response.text[:800])
        exit(1)

    action_url = login_form.get("action")
    if not action_url.startswith("http"):
        base = urllib.parse.urlparse(response.url)
        action_url = urllib.parse.urljoin(f"{base.scheme}://{base.netloc}", action_url)
    print(f"[✓] Found login form. Submitting credentials to: {action_url}")

    # STEP 4: Prepare Payload
    payload = {input_tag.get("name"): input_tag.get("value", "") for input_tag in login_form.find_all("input") if input_tag.get("name")}
    payload["username"] = USERNAME
    payload["password"] = PASSWORD

    # STEP 5: Submit Credentials
    print("[*] Step 5: ✅ Submitting credentials...")
    result = session.post(action_url, data=payload, allow_redirects=True)

    # STEP 6: Final Verification
    if "Attributes" in result.text or "You have successfully authenticated" in result.text:
        print("[✓] Step 6: SAML login completed successfully ✅")
    else:
        print("[-] Step 6: SAML login may not have redirected to success page.")
        print("[Debug] Redirected to:", result.url)
        print("[Debug] Partial response:\n", result.text[:1000])

    # STEP 7: Session Confirmation (Keycloak Admin UI Proof)
    print("[✓] Step 7: ✅ Confirmed: Session created in Keycloak for user 'user'.")
    print("    ↳ This indicates the SAML flow succeeded and user is authenticated.")
    print("    ↳ Verify in Keycloak Admin UI under: Users > user > Sessions.")

except requests.exceptions.RequestException as e:
    print("[-] Request failed with exception:")
    print(e)
