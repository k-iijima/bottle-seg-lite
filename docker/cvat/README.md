# CVAT local Docker Compose

CVATをローカルDockerで起動するための構成です。
CVAT本体と主要依存イメージは、2026-06-13時点で確認した公式 `v2.68.0` 系の構成に寄せています。

## 方針

- CVAT: `v2.68.0`
- PostgreSQL: `15-alpine`
  - CVAT公式ドキュメント上、外部DBを使う場合も公式Composeと同じPostgreSQLメジャー版がサポート対象です。
  - そのため、PostgreSQLだけ17/18などへ上げる構成にはしていません。
- Redis: `7.2.11-alpine`
- Kvrocks: `2.15.0`
- Traefik: `v3.6`
- OPA: `1.12.2`

この構成はローカル検証・学習データ作成用です。HTTPS、外部公開、メール認証、バックアップ、監視、Nuclio/serverless自動アノテーションは含めていません。

## 起動

```bash
docker compose --env-file .env -f compose.yml up -d --build
```

## 管理者ユーザー作成

```bash
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

Windows PowerShellで対話入力が崩れる場合は、Git BashやWSLのターミナルから実行してください。

## アクセス

```text
http://localhost:4810
```

LAN内の別PCからアクセスしたい場合は `.env` の `CVAT_HOST` をホストPCのIPアドレスに変更します。

例:

```env
CVAT_HOST=192.168.1.50
CVAT_PORT=8080
```

その後、再起動します。

```bash
docker compose --env-file .env -f compose.yml down
docker compose --env-file .env -f compose.yml up -d --build
```

## 停止

```bash
docker compose --env-file .env -f compose.yml down
```

## データも含めて完全削除

注意: タスク、アノテーション、DB、アップロードデータも消えます。

```bash
docker compose --env-file .env -f compose.yml down -v
```

## 共有フォルダ

`./share` は `/home/django/share` に読み取り専用でマウントしています。
CVATのタスク作成時に共有ストレージからデータを参照したい場合に使えます。

## 更新方針

`CVAT_VERSION` を上げるときは、CVATの公式 `docker-compose.yml` で依存イメージのタグも確認してください。
特にPostgreSQLのメジャーアップグレードはDB移行作業が必要になる可能性があります。

## トラブルシュート

起動状況:

```bash
docker compose --env-file .env -f compose.yml ps
```

サーバーログ:

```bash
docker logs -f cvat_server
```

UIが `Cannot connect to the server` になる場合:

```bash
docker logs cvat_server

docker logs cvat_opa

docker logs cvat_db
```

ディスク使用率が高いとCVATのヘルスチェックに失敗する場合があります。
`.env` の `CVAT_HEALTH_DISK_USAGE_MAX` を調整できますが、基本は空き容量を増やす方が安全です。
