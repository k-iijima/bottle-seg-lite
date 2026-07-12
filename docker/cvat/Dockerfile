# syntax=docker/dockerfile:1.7

# Local wrapper for the official CVAT server image.
# Keep CVAT_VERSION aligned with .env and compose.yml.
ARG CVAT_VERSION=v2.68.0
FROM cvat/server:${CVAT_VERSION}

# This file exists mainly so you can add local packages, CA certificates,
# or corporate proxy settings later while still starting from the official image.
# Example:
# USER root
# RUN apt-get update \
#     && apt-get install -y --no-install-recommends ca-certificates \
#     && rm -rf /var/lib/apt/lists/*
# USER django
