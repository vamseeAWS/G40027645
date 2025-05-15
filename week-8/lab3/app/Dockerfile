FROM python:3.9

WORKDIR /app

# Install xmlsec1 and build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libxml2-dev libxmlsec1-dev libxmlsec1-openssl pkg-config \
    gcc g++ make libtool && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 3000

CMD ["python", "app.py"]
