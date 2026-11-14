#!/usr/bin/env python3
"""Watermelon Aimbot 本地参数控制台（仅标准库）

功能:
  - 读取 config.ini（保留注释、顺序）
  - 写回时只替换对应行，不破坏注释和排版
  - 启动 / 停止 run.py（新控制台窗口，便于输入卡密）

用法:
  python webui/server.py --open
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.ini"
BACKUP_PATH = ROOT / "config.ini.webui.bak"
WEB_DIR = Path(__file__).resolve().parent

SECTION_RE = re.compile(r"^\[([^\]]+)\]\s*$")
OPTION_RE = re.compile(r"^(\s*)([^=\s;#][^=]*?)\s*=\s*(.*?)\s*$")

TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}

# 值看起来是整数、但语义上应按浮点处理的键
FLOAT_KEYS = {
    "body_x_offset", "body_y_offset",
    "mouse_sensitivity", "mouse_min_speed_multiplier", "mouse_max_speed_multiplier",
    "prediction_interval", "prediction_jitter_threshold", "prediction_max_offset",
    "bscope_multiplier", "ai_conf",
    "recoil_scale", "recoil_fire_rate", "recoil_duration",
}

# 这些键允许混合输入数字和 auto / 键名等文本，始终按字符串处理
TEXT_KEYS = {
    "obs_camera_id",
    "arduino_port", "ch9329_port",
    "ch9329_screen_width", "ch9329_screen_height",
    "ai_model_name", "ai_device",
    "hotkey_targeting", "hotkey_exit", "hotkey_pause", "hotkey_reload_config",
    "debug_window_screenshot_key", "recoil_pattern",
}

CAPTURE_KEYS = ("bettercam_capture", "obs_capture", "mss_capture")

_proc: subprocess.Popen | None = None
_proc_lock = threading.Lock()
_ext_cache = {"at": 0.0, "pid": None}


class ApiError(Exception):
    def __init__(self, status: int, message: str, extra: dict | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.extra = extra or {}


# ─────────────────────────── config.ini 解析 ───────────────────────────

def infer_type(key: str, value: str) -> str:
    v = value.strip()
    lv = v.lower()
    if key in TEXT_KEYS:
        return "text"
    if lv in ("true", "false"):
        return "bool"
    if key in FLOAT_KEYS:
        return "float"
    if re.fullmatch(r"-?\d+", v):
        return "int"
    if re.fullmatch(r"-?\d+\.\d+", v):
        return "float"
    return "text"


def parse_config(text: str) -> list[dict]:
    """按行解析，保留注释作为字段说明。"""
    sections: list[dict] = []
    current = None
    pending: list[str] = []

    for line_no, raw in enumerate(text.splitlines()):
        line = raw.strip()
        if not line:
            pending = []
            continue
        if line.startswith(";") or line.startswith("#"):
            pending.append(line[1:].strip())
            continue

        m = SECTION_RE.match(line)
        if m:
            current = {
                "name": m.group(1),
                "description": "\n".join(pending),
                "items": [],
                "line": line_no,
            }
            sections.append(current)
            pending = []
            continue

        m = OPTION_RE.match(raw)
        if m and current is not None:
            key = m.group(2).strip()
            value = m.group(3).strip()
            current["items"].append({
                "key": key,
                "value": value,
                "type": infer_type(key, value),
                "widget": "textarea" if key == "recoil_pattern" else "default",
                "description": "\n".join(pending),
            })
            pending = []
        else:
            pending = []

    return sections


def _read_config_text() -> str:
    with CONFIG_PATH.open("r", encoding="utf-8", newline="") as f:
        return f.read()


def read_config_json() -> dict:
    if not CONFIG_PATH.exists():
        raise ApiError(500, f"未找到配置文件: {CONFIG_PATH}")
    return {
        "sections": parse_config(_read_config_text()),
        "mtime": CONFIG_PATH.stat().st_mtime,
        "path": str(CONFIG_PATH),
    }


# ─────────────────────────── 校验与写回 ───────────────────────────

def _flat(sections: list[dict]) -> dict[str, dict[str, str]]:
    return {s["name"]: {i["key"]: i["value"] for i in s["items"]} for s in sections}


def _merge_and_validate(current: dict[str, dict[str, str]], payload: dict) -> tuple[dict[str, dict[str, str]], list[str]]:
    incoming = payload.get("sections")
    if not isinstance(incoming, dict) or not incoming:
        raise ApiError(400, "没有需要保存的字段")

    updates: dict[str, dict[str, str]] = {}

    for section, items in incoming.items():
        if section not in current:
            raise ApiError(400, f"未知配置分区: {section}")
        if not isinstance(items, dict):
            raise ApiError(400, f"配置分区 {section} 格式错误")
        for key, raw in items.items():
            if key not in current[section]:
                raise ApiError(400, f"未知配置项: [{section}] {key}")
            if not isinstance(raw, (str, int, float, bool)):
                raise ApiError(400, f"[{section}] {key} 值类型无效")

            value = str(raw).strip()
            if "\n" in value or "\r" in value:
                raise ApiError(400, f"[{section}] {key} 不能包含换行")
            if len(value) > 4096:
                raise ApiError(400, f"[{section}] {key} 内容过长")

            vtype = infer_type(key, current[section][key])
            if vtype == "bool":
                low = value.lower()
                if low in TRUE_VALUES:
                    value = "true"
                elif low in FALSE_VALUES:
                    value = "false"
                else:
                    raise ApiError(400, f"[{section}] {key} 应为 true / false")
            elif vtype == "int":
                try:
                    int(value)
                except ValueError:
                    raise ApiError(400, f"[{section}] {key} 应为整数") from None
            elif vtype == "float":
                try:
                    float(value)
                except ValueError:
                    raise ApiError(400, f"[{section}] {key} 应为数字") from None

            updates.setdefault(section, {})[key] = value

    merged = {name: dict(values) for name, values in current.items()}
    for section, items in updates.items():
        merged[section].update(items)

    def is_on(section: str, key: str) -> bool:
        return merged.get(section, {}).get(key, "false").strip().lower() in TRUE_VALUES

    capture_on = [k for k in CAPTURE_KEYS if is_on("Capture Methods", k)]
    if len(capture_on) != 1:
        raise ApiError(
            400,
            "截取后端必须且只能启用一个（BetterCam / OBS / MSS），当前启用了 %d 个" % len(capture_on),
        )

    if is_on("Arduino", "arduino_move") and is_on("CH9329", "ch9329_move"):
        raise ApiError(400, "Arduino 与 CH9329 的鼠标移动不能同时启用，请只保留一种")

    warnings: list[str] = []
    if is_on("Arduino", "arduino_shoot") and is_on("CH9329", "ch9329_shoot"):
        warnings.append("Arduino 与 CH9329 的射击同时启用时，程序只会使用 CH9329。")
    if is_on("Shooting", "auto_shoot") and not (is_on("CH9329", "ch9329_shoot") or is_on("Arduino", "arduino_shoot")):
        warnings.append("已开启自动射击但未启用硬件点击，程序会使用 win32api 软件点击，封号风险更高。")

    try:
        if int(merged["Detection window"]["detection_window_width"]) >= 600:
            warnings.append("检测窗口宽度 ≥ 600，可能明显影响性能。")
        if int(merged["Detection window"]["detection_window_height"]) >= 600:
            warnings.append("检测窗口高度 ≥ 600，可能明显影响性能。")
        if int(merged["Capture Methods"]["capture_fps"]) >= 120:
            warnings.append("截取帧率 ≥ 120，自动瞄准稳定性可能下降。")
    except (KeyError, ValueError):
        pass

    use_enc = is_on("AI", "ai_model_use_enc")
    enc_file = ROOT / "models" / "sunxds_0.8.0.enc"
    model_name = merged.get("AI", {}).get("ai_model_name", "")
    if use_enc and not enc_file.exists():
        warnings.append("已开启加密模型，但 models/sunxds_0.8.0.enc 不存在，程序启动会失败。")
    if not use_enc:
        if model_name and not (ROOT / "models" / model_name).exists():
            warnings.append(f"明文模型 models/{model_name} 不存在，程序启动会失败。")

    if not merged.get("Hotkeys", {}).get("hotkey_targeting", "").strip():
        warnings.append("没有配置瞄准激活键，将只能开启“自动瞄准”才会移动鼠标。")

    return updates, warnings


def render_config(text: str, updates: dict[str, dict[str, str]]) -> str:
    """只改写受影响的行，保留全部注释、空行和顺序。"""
    eol_match = re.search(r"\r\n|\r|\n", text)
    eol = eol_match.group(0) if eol_match else "\n"

    out: list[str] = []
    current_name = None

    for line in text.splitlines():
        m = SECTION_RE.match(line.strip())
        if m:
            current_name = m.group(1)
            out.append(line)
            continue

        if current_name in updates:
            om = OPTION_RE.match(line)
            if om:
                key = om.group(2).strip()
                if key in updates[current_name]:
                    out.append(f"{om.group(1)}{key} = {updates[current_name][key]}")
                    continue

        out.append(line)

    return eol.join(out).rstrip("\r\n") + eol


def save_config(payload: dict) -> dict:
    text = _read_config_text()
    current = _flat(parse_config(text))
    updates, warnings = _merge_and_validate(current, payload)

    new_text = render_config(text, updates)
    changed = new_text != text

    if changed:
        try:
            BACKUP_PATH.write_bytes(CONFIG_PATH.read_bytes())
        except OSError:
            pass
        tmp = CONFIG_PATH.with_name(CONFIG_PATH.name + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="") as f:
            f.write(new_text)
        os.replace(tmp, CONFIG_PATH)

    fresh = read_config_json()
    return {
        "ok": True,
        "changed": changed,
        "warnings": warnings,
        "sections": fresh["sections"],
        "mtime": fresh["mtime"],
    }


def restore_backup() -> dict:
    if not BACKUP_PATH.exists():
        raise ApiError(404, "还没有备份，先保存一次配置后才会生成。")
    CONFIG_PATH.write_bytes(BACKUP_PATH.read_bytes())
    fresh = read_config_json()
    return {"ok": True, "sections": fresh["sections"], "mtime": fresh["mtime"],
            "warnings": ["已还原到上一次保存前的配置。"]}


# ─────────────────────────── 进程控制 ───────────────────────────

def python_exe() -> str:
    venv_py = ROOT / "venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def _find_run_py_pid() -> int | None:
    now = time.time()
    if now - _ext_cache["at"] < 2.0:
        return _ext_cache["pid"]

    pid = None
    try:
        import psutil

        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmdline = " ".join(proc.info.get("cmdline") or [])
                if "run.py" in cmdline and "server.py" not in cmdline and proc.pid != os.getpid():
                    pid = proc.pid
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        pid = None

    _ext_cache.update(at=now, pid=pid)
    return pid


def get_status() -> dict:
    global _proc

    running = bool(_proc is not None and _proc.poll() is None)
    pid = _proc.pid if running else None

    if not running:
        external = _find_run_py_pid()
        if external:
            running = True
            pid = external

    return {
        "running": running,
        "pid": pid,
        "venv": (ROOT / "venv" / "Scripts" / "python.exe").exists(),
        "enc_model": (ROOT / "models" / "sunxds_0.8.0.enc").exists(),
        "config_mtime": CONFIG_PATH.stat().st_mtime if CONFIG_PATH.exists() else 0,
        "python": python_exe(),
    }


def start_program() -> dict:
    global _proc
    with _proc_lock:
        if _proc is not None and _proc.poll() is None:
            raise ApiError(409, f"程序已在运行（PID {_proc.pid}）")
        external = _find_run_py_pid()
        if external:
            raise ApiError(409, f"检测到 run.py 已在运行（PID {external}）")

        py = python_exe()
        flags = 0
        if os.name == "nt":
            flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0x00000010)

        try:
            _proc = subprocess.Popen(
                [py, "-u", "run.py"],
                cwd=str(ROOT),
                creationflags=flags,
            )
        except OSError as exc:
            raise ApiError(500, f"启动失败: {exc}") from exc

        _ext_cache.update(at=0.0, pid=None)
        return {"ok": True, "pid": _proc.pid,
                "message": "已在新窗口启动，请在该窗口中输入卡密"}


def stop_program() -> dict:
    global _proc
    with _proc_lock:
        pid = None
        if _proc is not None and _proc.poll() is None:
            pid = _proc.pid
        else:
            pid = _find_run_py_pid()

        if not pid:
            raise ApiError(409, "程序当前没有运行")

        try:
            if os.name == "nt":
                # 不依赖 PATH（部分机器的系统 PATH 缺少 System32）
                taskkill = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "taskkill.exe"
                subprocess.run(
                    [str(taskkill), "/PID", str(pid), "/T", "/F"],
                    capture_output=True,
                    timeout=10,
                )
            else:
                os.kill(pid, 15)
        except Exception as exc:
            raise ApiError(500, f"停止失败: {exc}") from exc

        if _proc is not None and _proc.poll() is not None:
            _proc = None
        _ext_cache.update(at=0.0, pid=None)
        return {"ok": True, "message": f"已结束进程 {pid}"}


# ─────────────────────────── HTTP 服务 ───────────────────────────

class Handler(BaseHTTPRequestHandler):
    server_version = "WatermelonUI/1.0"

    def log_message(self, fmt, *args):  # 安静运行
        pass

    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_index(self):
        index = WEB_DIR / "index.html"
        if not index.exists():
            self._send_json({"ok": False, "error": "index.html 不存在"}, 500)
            return
        body = index.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                self._send_index()
            elif path == "/api/config":
                self._send_json({"ok": True, **read_config_json()})
            elif path == "/api/status":
                self._send_json({"ok": True, **get_status()})
            else:
                self._send_json({"ok": False, "error": "not found"}, 404)
        except ApiError as exc:
            self._send_json({"ok": False, "error": exc.message, **exc.extra}, exc.status)
        except Exception as exc:
            self._send_json({"ok": False, "error": f"服务端异常: {exc}"}, 500)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8") or "{}")

            if path == "/api/config":
                self._send_json(save_config(payload))
            elif path == "/api/start":
                self._send_json(start_program())
            elif path == "/api/stop":
                self._send_json(stop_program())
            elif path == "/api/restore":
                self._send_json(restore_backup())
            else:
                self._send_json({"ok": False, "error": "not found"}, 404)
        except ApiError as exc:
            body = {"ok": False, "error": exc.message, "errors": [exc.message]}
            body.update(exc.extra)
            self._send_json(body, exc.status)
        except json.JSONDecodeError:
            self._send_json({"ok": False, "error": "请求体不是合法 JSON"}, 400)
        except Exception as exc:
            self._send_json({"ok": False, "error": f"服务端异常: {exc}"}, 500)


def main():
    parser = argparse.ArgumentParser(description="Watermelon Aimbot 参数控制台")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"

    try:
        httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError:
        print(f"[提示] 端口 {args.port} 已被占用，配置界面可能已经在运行: {url}")
        if args.open:
            webbrowser.open(url)
        return

    print("=" * 56)
    print("  Watermelon Aimbot 参数控制台")
    print(f"  地址: {url}")
    print(f"  配置: {CONFIG_PATH}")
    print("  说明: 保存后程序会在 0.25 秒内自动热重载")
    print("  关闭本窗口即可停止控制台")
    print("=" * 56)

    if args.open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
