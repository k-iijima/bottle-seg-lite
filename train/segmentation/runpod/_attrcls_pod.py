#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""属性分類器学習用の単一 pod 管理(_refine_fleet.py の縮小版、マルチGPU対応)。
転送・実行は README_RUNPOD.md の教訓どおりネイティブ ssh/scp で行うので、
ここでは pod の作成/情報/削除だけを担当する。

  python _attrcls_pod.py create [GPU数=2]  : H100 優先で pod 作成、ssh 情報を表示
  python _attrcls_pod.py info              : 保存済み pod の ssh 情報
  python _attrcls_pod.py term              : pod 削除(課金停止。必ず実行)
"""
import json, os, sys, time, urllib.error, urllib.request
from pathlib import Path

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

HERE = Path(__file__).resolve().parent
RP = HERE / ".rp"
KEY = str(RP / "id_rsa")
STATE = RP / "attrcls_pod.json"
API = "https://rest.runpod.io/v1"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
GPUS = ["NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe",
        "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"]
DISK_GB = 40


def envval(name):
    v = os.environ.get(name)
    if v:
        return v
    for p in [HERE, *HERE.parents]:
        f = p / ".env"
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith(name + "="):
                    return line.split("=", 1)[1].strip().strip("\"'")
    raise SystemExit(f"{name} が .env にも環境変数にもない")


def api(method, path):
    try:
        r = urllib.request.urlopen(urllib.request.Request(
            API + path, headers={"Authorization": "Bearer " + envval("RUNPOD_KEY")},
            method=method), timeout=60)
        t = r.read().decode()
        return r.status, (json.loads(t) if t.strip() else {})
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]


def gql(query, variables=None):
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(
        "https://api.runpod.io/graphql?api_key=" + envval("RUNPOD_KEY"),
        data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        r = urllib.request.urlopen(req, timeout=60)
        return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]


def cmd_create(args):
    n_gpu = int(args[0]) if args else 2
    pub = open(KEY + ".pub").read().strip()
    MUT = """
mutation($in: PodFindAndDeployOnDemandInput) {
  podFindAndDeployOnDemand(input: $in) { id costPerHr machine { gpuDisplayName } }
}"""
    pid = None
    for attempt in range(8):
        gpu = GPUS[min(attempt // 2, len(GPUS) - 1)]
        cloud = "SECURE" if attempt % 2 == 0 else "ALL"
        st, r = gql(MUT, {"in": {
            "cloudType": cloud, "gpuCount": n_gpu, "gpuTypeId": gpu,
            "name": "attrcls-train", "imageName": IMAGE,
            "containerDiskInGb": DISK_GB, "volumeInGb": 0, "ports": "22/tcp",
            "env": [{"key": "PUBLIC_KEY", "value": pub}]}})
        pod = r.get("data", {}).get("podFindAndDeployOnDemand") if isinstance(r, dict) else None
        if st == 200 and pod:
            pid = pod["id"]
            print(f"created {pid} ${pod.get('costPerHr')}/h ({gpu} x{n_gpu}, {cloud})")
            break
        print(f"create retry {attempt+1} ({gpu} x{n_gpu}, {cloud})", st, str(r)[:150])
        time.sleep(5)
    if not pid:
        raise SystemExit("pod create failed")
    json.dump({"id": pid}, open(STATE, "w"))
    for _ in range(60):
        s, p = api("GET", "/pods/" + pid)
        if isinstance(p, dict):
            ip = p.get("publicIp") or ""
            pm = p.get("portMappings") or {}
            port = pm.get("22") if isinstance(pm, dict) else None
            if ip and port:
                json.dump({"id": pid, "ip": ip, "port": port}, open(STATE, "w"))
                print(f"ssh -i runpod/.rp/id_rsa -p {port} root@{ip}")
                return
        time.sleep(10)
    print("SSH 情報が取れなかった。python _attrcls_pod.py info で再確認を")


def cmd_info(args):
    st = json.load(open(STATE))
    s, p = api("GET", "/pods/" + st["id"])
    if isinstance(p, dict):
        ip = p.get("publicIp") or ""
        pm = p.get("portMappings") or {}
        port = pm.get("22") if isinstance(pm, dict) else None
        print(json.dumps({"id": st["id"], "ip": ip, "port": port,
                          "status": p.get("desiredStatus"),
                          "cost": p.get("costPerHr")}))
        if ip and port:
            json.dump({"id": st["id"], "ip": ip, "port": port}, open(STATE, "w"))
    else:
        print("api error", s, p)


def cmd_term(args):
    st = json.load(open(STATE))
    print("term", st["id"], *api("DELETE", "/pods/" + st["id"]))


if __name__ == "__main__":
    {"create": cmd_create, "info": cmd_info, "term": cmd_term}[sys.argv[1]](sys.argv[2:])
