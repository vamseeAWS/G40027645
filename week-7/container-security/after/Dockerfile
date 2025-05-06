FROM python:3.9-alpine3.21

# Install patched sqlite-libs and other required packages
RUN apk update && apk add --no-cache \
    gcc \
    musl-dev \
    libffi-dev \
    make \
    wget && \
    pip install --upgrade pip setuptools

# Add non-root user
RUN addgroup -S appgroup && adduser -S appuser -G appgroup

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD wget -qO- http://localhost:5000/ || exit 1

CMD ["python", "app.py"]






