# Flutter web dev server. Source is bind-mounted via docker-compose so edits
# on the host are picked up (restart the container to rebuild).
FROM ghcr.io/cirruslabs/flutter:stable

WORKDIR /app

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
