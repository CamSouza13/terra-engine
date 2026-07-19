FROM python:3.11-slim

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -e .

# TERRA_HOME points at the mounted volume so accounts, config, history, and the
# SQLite database survive restarts and deploys. TERRA_AUTH turns on the hosted
# login/paywall. Provide TERRA_SMTP_* and STRIPE_* as Fly secrets, not here.
ENV TERRA_HOME=/data \
    TERRA_AUTH=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8080
CMD ["terra", "serve", "--port", "8080"]
