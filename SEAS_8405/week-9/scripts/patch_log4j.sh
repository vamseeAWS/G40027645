#!/bin/bash
echo "[*] Updating Log4j version in pom.xml..."
sed -i 's/<version>2\.14\.1<\/version>/<version>2.17.0<\/version>/g' pom.xml
echo "[✓] pom.xml updated."
