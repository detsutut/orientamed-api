#!/bin/bash
cd "$(dirname "$0")"
docker compose -f ./docker-compose.yml -p orientamed --env-file ./.env up
