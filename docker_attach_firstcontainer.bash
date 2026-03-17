#!/usr/bin/env bash

set -e

# Find running devcontainer (containers labeled by VS Code devcontainers)
CONTAINER_ID=$(docker ps \
  --format "{{.Names}}" \
  | head -n 1)

if [ -z "$CONTAINER_ID" ]; then
  echo "No running devcontainer found."
  exit 1
fi

echo "Attaching to devcontainer: $CONTAINER_ID"

docker exec -it "$CONTAINER_ID" /bin/bash