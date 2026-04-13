#!/usr/bin/env bash
set -Eeuo pipefail

log() { echo "[$(date +'%F %T')] $*"; }
fail() { log "ERROR: $*"; exit 1; }

NEW_NAME="uplift-streamlit-next"
OLD_NAME="uplift-streamlit"
LIVE_PORT=8501
TEST_PORT=8502

# Check registry credentials
: "${IMAGE_REF:?IMAGE_REF is required}"
: "${REGISTRY_URL:?REGISTRY_URL is required}"
: "${REGISTRY_USER:?REGISTRY_USER is required}"
: "${REGISTRY_PASSWORD:?REGISTRY_PASSWORD is required}"


command -v docker >/dev/null 2>&1 || fail "docker not found"
if ! command -v curl >/dev/null 2>&1; then
  command -v wget >/dev/null 2>&1 || fail "curl or wget is required for health check"
fi

log "[DEPLOY] Logging into registry: $REGISTRY_URL as $REGISTRY_USER"
echo "$REGISTRY_PASSWORD" | docker login -u "$REGISTRY_USER" --password-stdin "$REGISTRY_URL"

log "[DEPLOY] Pulling image: $IMAGE_REF"
docker pull "$IMAGE_REF"

# Clean previous test container
if docker ps -a --format '{{.Names}}' | grep -q "^${NEW_NAME}\$"; then
  docker rm -f "$NEW_NAME" >/dev/null 2>&1 || true
fi

# Run new (test) container
log "[DEPLOY] Starting new container (test): $NEW_NAME on :$TEST_PORT"
docker run --env-file "$HOME/.env" \
  -p "$TEST_PORT:8501" \
  -v "$HOME/artifacts:/app/artifacts" \
  --label "deploy.role=test" \
  --log-driver json-file --log-opt max-size=10m --log-opt max-file=3 \
  --restart unless-stopped \
  -d --name "$NEW_NAME" "$IMAGE_REF"

# Health check
log "[DEPLOY] Waiting for health check on :$TEST_PORT ..."
attempts=20
sleep_between=3
healthy=0
for ((i=1; i<=attempts; i++)); do
  if curl -fsS "http://localhost:${TEST_PORT}/" >/dev/null 2>&1 || wget -q -O /dev/null "http://localhost:${TEST_PORT}/"; then
    log "[DEPLOY] Health check passed on attempt $i"
    healthy=1
    break
  fi
  log "[DEPLOY] Health check $i/$attempts failed; retrying in ${sleep_between}s..."
  sleep "$sleep_between"
done

if [[ "$healthy" -ne 1 ]]; then
  log "[DEPLOY] New container failed health check. Rolling back."
  docker logs "$NEW_NAME" || true
  docker rm -f "$NEW_NAME" || true
  exit 1
fi

log "[DEPLOY] Switching traffic."

# Stop/remove old live container, if any
if docker ps -a --format '{{.Names}}' | grep -q "^${OLD_NAME}\$"; then
  log "[DEPLOY] Stopping old container: $OLD_NAME"
  docker stop -t 10 "$OLD_NAME" || true
  docker rm "$OLD_NAME" || true
fi

# Promote tested container to live
docker stop "$NEW_NAME" >/dev/null 2>&1 || true
docker rm "$NEW_NAME"   >/dev/null 2>&1 || true

log "[DEPLOY] Starting new container on live port :$LIVE_PORT as $OLD_NAME"
docker run --env-file "$HOME/.env" \
  -p "$LIVE_PORT:8501" \
  -v "$HOME/artifacts:/app/artifacts" \
  --label "deploy.role=live" \
  --log-driver json-file --log-opt max-size=10m --log-opt max-file=3 \
  --restart unless-stopped \
  -d --name "$OLD_NAME" "$IMAGE_REF"

# Clean dangling-images (if any)
docker image prune -f >/dev/null 2>&1 || true

log "[DEPLOY] Deployment complete! Live at port $LIVE_PORT"