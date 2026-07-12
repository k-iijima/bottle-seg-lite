.PHONY: rtmdet-onnx apk up logs down clean rebuild seg-build seg-preview seg \
        cvat-up cvat-down cvat-logs cvat-superuser cvat-export

# 学習済み RTMDet-Ins-s を ONNX 化（app/assets/models/rtmdet_ins.onnx、ローカル GPU 必須）
rtmdet-onnx:
	docker compose --profile tools build rtmdet-onnx
	docker compose --profile tools run --rm rtmdet-onnx

# Android APK をビルド（出力: app/build/app/outputs/flutter-apk/app-release.apk）
apk:
	docker compose --profile tools run --rm apk

# SAM 3 でセグメンテーションを再生成（GPU 必須・root .env の HF_TOKEN を使用）
seg-build:
	docker compose --profile tools build seg
seg-preview:        # 動作確認: 30画像だけ処理して qa_sam3/preview_*.jpg を生成
	docker compose --profile tools run --rm seg
seg:                # 全件処理
	docker compose --profile tools run --rm seg python make_sam3_segmentation.py

# CVAT で確認・補正（VSLAM の docker-compose-cvat.yml を流用）。http://localhost:4810
cvat-up:            # 起動（初回は server-local イメージをビルド）
	docker compose -f docker-compose-cvat.yml up -d --build
cvat-down:
	docker compose -f docker-compose-cvat.yml down
cvat-logs:
	docker compose -f docker-compose-cvat.yml logs -f cvat_server
cvat-superuser:     # 初回のみ: 管理ユーザー作成
	docker exec -it cvat_server python3 ~/manage.py createsuperuser
cvat-export:        # COCO -> CVAT インポート zip（全 split, _sam3merge）を生成
	docker compose --profile tools run --rm seg python export_coco_for_cvat.py

# Start the Flutter web dev server at http://localhost:8081 (host 8081 -> container 8080)
# (要 app/assets/models/rtmdet_ins.onnx — README「学習済みモデル」参照)
up:
	docker compose up --build web

logs:
	docker compose logs -f web

down:
	docker compose down

# Rebuild the web image from scratch (e.g. after changing the Dockerfile)
rebuild:
	docker compose build --no-cache web

# Remove containers and volumes
clean:
	docker compose down -v
