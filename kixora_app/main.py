"""
main.py — Kixora FastAPI backend
WebSocket endpoints:
  /ws/imu/{player_id}   — phone streams IMU samples here
  /ws/ui                — browser UI receives real-time events here
  /ws/ble/{player_id}   — browser relays Web Bluetooth packets here (client-side BLE)

REST endpoints:
  POST /session/start  POST /session/stop  GET /session/status
  POST /ble/scan  POST /ble/connect  POST /ble/disconnect
  POST /kick/manual
  GET  /sessions  GET /api/info

NOTE: BLE is now handled client-side via the Web Bluetooth API in index.html.
      The browser scans/connects directly to the ESP32 sock and forwards packets
      to this server over /ws/ble/{player_id}.  Server-side bleak endpoints are
      kept for backwards-compat but are no longer called by the default UI.
"""
import asyncio
import json
import logging
import os
import socket
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ble_manager import BLEManager
from data_recorder import SessionRecorder, KickEvent
from classifier_core import RealtimeClassifier

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
log = logging.getLogger("kixora")

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
MODEL_DIR  = os.path.abspath(os.path.join(BASE_DIR, "..", "classifier", "output"))
MODEL_PATH = os.path.join(MODEL_DIR, "kick_classifier_v9.pkl")
META_PATH  = os.path.join(MODEL_DIR, "model_metadata_v9.json")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# ── Global state ───────────────────────────────────────────────
ble_mgr: BLEManager = None
session: Optional[SessionRecorder] = None
classifiers: dict = {}           # pid -> RealtimeClassifier
ui_clients: list[WebSocket] = [] # connected browser UIs
imu_ws: dict = {}                # pid -> WebSocket (phone IMU stream)
model_available = False
imu_frame_count: dict = {1: 0, 2: 0}  # for throttling UI broadcast

def get_lan_ip() -> str:
    """
    Get real LAN IP — skips VirtualBox/VMware/Hyper-V virtual adapters.
    Uses PowerShell to find the best physical WiFi/Ethernet interface.
    """
    import subprocess
    # Best method: PowerShell — filters out virtual adapters by description
    try:
        cmd = (
            "Get-NetIPAddress -AddressFamily IPv4 | "
            "Where-Object { "
            "  $_.IPAddress -notlike '127.*' -and "
            "  $_.IPAddress -notlike '169.*' -and "
            "  $_.IPAddress -notlike '192.168.56.*' -and "
            "  $_.IPAddress -notlike '192.168.99.*' -and "
            "  $_.IPAddress -notlike '172.16.*' -and "
            "  $_.IPAddress -notlike '172.17.*' -and "
            "  $_.IPAddress -notlike '10.0.75.*' "
            "} | Sort-Object InterfaceIndex | "
            "Select-Object -First 1 -ExpandProperty IPAddress"
        )
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", cmd],
            timeout=4, stderr=subprocess.DEVNULL,
            creationflags=0x08000000  # CREATE_NO_WINDOW
        ).decode().strip()
        if out and out.count('.') == 3:
            log.info(f"LAN IP (PowerShell): {out}")
            return out
    except Exception as e:
        log.warning(f"PowerShell IP detection failed: {e}")

    # Fallback: hostname lookup
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if not ip.startswith("127.") and not ip.startswith("192.168.56."):
            return ip
    except Exception:
        pass

    return "127.0.0.1"

LAN_IP = get_lan_ip()
log.info(f"Detected LAN IP: {LAN_IP}")

# ── App ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ble_mgr, model_available
    ble_mgr = BLEManager()
    ble_mgr.on_packet(_on_ble_packet)
    ble_mgr.on_status(_on_ble_status)
    model_available = os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)
    if model_available:
        log.info(f"Model found: {MODEL_PATH}")
    else:
        log.warning(f"Model NOT found at {MODEL_PATH}. Classification disabled.")
    log.info(f"""\n
    ╔══════════════════════════════════════════╗
    ║  KIXORA — Server Ready                  ║
    ║  Dashboard : http://{LAN_IP}:8000        ║
    ║  Player 1  : http://{LAN_IP}:8000/phone/1║
    ║  Player 2  : http://{LAN_IP}:8000/phone/2║
    ╚══════════════════════════════════════════╝
    """)
    yield
    if session:
        session.close()

app = FastAPI(title="Kixora", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ═══════════════════════════════════════════════════════════════
# BLE callbacks (called from BLE background tasks)
# ═══════════════════════════════════════════════════════════════
async def _on_ble_packet(player_id: int, pkt: dict):
    """Called when ESP32 sends a kick packet via BLE notify."""
    if session and session.is_recording:
        session.record_ble(player_id, pkt)
    await _broadcast_ui({
        "type": "ble_packet",
        "player": player_id,
        "data": pkt,
        "ts": round(time.time() * 1000),
    })


async def _on_ble_status(player_id: int, status: str, extra: dict):
    await _broadcast_ui({
        "type": "ble_status",
        "player": player_id,
        "status": status,
        "extra": extra,
        "ts": round(time.time() * 1000),
    })


# ═══════════════════════════════════════════════════════════════
# UI broadcast helper
# ═══════════════════════════════════════════════════════════════
async def _broadcast_ui(msg: dict):
    dead = []
    text = json.dumps(msg)
    for ws in ui_clients:
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    for ws in dead:
        ui_clients.remove(ws)


# ═══════════════════════════════════════════════════════════════
# WebSocket: IMU stream from phone
# ═══════════════════════════════════════════════════════════════
@app.websocket("/ws/imu/{player_id}")
async def ws_imu(websocket: WebSocket, player_id: int):
    if player_id not in (1, 2):
        await websocket.close(code=4000)
        return
    await websocket.accept()
    imu_ws[player_id] = websocket
    log.info(f"[P{player_id}] Phone IMU WebSocket connected")
    await _broadcast_ui({"type": "phone_connected", "player": player_id})

    # Init classifier for this player
    if model_available and player_id not in classifiers:
        try:
            classifiers[player_id] = RealtimeClassifier(MODEL_PATH, META_PATH, src_fs=500)
            log.info(f"[P{player_id}] Classifier initialized")
        except Exception as e:
            log.error(f"[P{player_id}] Classifier init failed: {e}")

    try:
        async for raw in websocket.iter_text():
            try:
                pkt = json.loads(raw)
                x = float(pkt.get("x", 0))
                y = float(pkt.get("y", 0))
                z = float(pkt.get("z", 0))
                ts = float(pkt.get("ts", time.time() * 1000))

                # Record raw IMU (always, even without BLE)
                if session and session.is_recording:
                    session.record_imu(player_id, x, y, z, src_fs=500)

                # Feed classifier (if model available)
                clf = classifiers.get(player_id)
                if clf:
                    clf.feed_imu(x, y, z, ts)
                    kicks = clf.get_kicks()
                    for k in kicks:
                        last_ble = ble_mgr.devices[player_id].last_packet if (ble_mgr and player_id in ble_mgr.devices) else None
                        kick_event = KickEvent(
                            timestamp_ms=ts, player_id=player_id,
                            kick_type=k["label"], confidence=k["confidence"],
                            source="auto",
                            fsr_peak=last_ble.get("f") if last_ble else None,
                            ball_peak=last_ble.get("b") if last_ble else None,
                            instep_peak=last_ble.get("i") if last_ble else None,
                            duration_ms=last_ble.get("d") if last_ble else None,
                            probs=k.get("probs"),
                        )
                        if session and session.is_recording:
                            session.record_kick(kick_event)
                        await _broadcast_ui({
                            "type": "kick_detected", "player": player_id,
                            "kick": {
                                "type": k["label"], "confidence": k["confidence"],
                                "probs": k.get("probs", {}), "win_sec": k.get("win_sec"),
                                "source": "auto",
                                "fsr": last_ble.get("f") if last_ble else None,
                            }, "ts": round(ts),
                        })

                # Throttle IMU broadcast to UI: every 10th sample (~50 Hz)
                imu_frame_count[player_id] += 1
                if imu_frame_count[player_id] % 10 == 0:
                    mag = round((x**2 + y**2 + z**2)**0.5, 2)
                    await _broadcast_ui({
                        "type": "imu_sample", "player": player_id,
                        "x": round(x, 2), "y": round(y, 2),
                        "z": round(z, 2), "mag": mag, "ts": round(ts),
                    })

            except json.JSONDecodeError:
                pass
            except Exception as e:
                log.error(f"[P{player_id}] IMU error: {e}")

    except WebSocketDisconnect:
        log.info(f"[P{player_id}] Phone disconnected")
        imu_ws.pop(player_id, None)
        await _broadcast_ui({"type": "phone_disconnected", "player": player_id})


# ═══════════════════════════════════════════════════════════════
# WebSocket: UI real-time events
# ═══════════════════════════════════════════════════════════════
@app.websocket("/ws/ui")
async def ws_ui(websocket: WebSocket):
    await websocket.accept()
    ui_clients.append(websocket)
    # Send current state on connect
    await websocket.send_text(json.dumps({
        "type": "init",
        "ble_status": ble_mgr.status_dict(),
        "session": session.get_summary() if session else None,
        "model_available": model_available,
        "phones_connected": list(imu_ws.keys()),
    }))
    try:
        async for _ in websocket.iter_text():
            pass  # UI only listens; commands go via REST
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in ui_clients:
            ui_clients.remove(websocket)


# ═══════════════════════════════════════════════════════════════
# WebSocket: BLE packet relay from browser (Web Bluetooth)
# ═══════════════════════════════════════════════════════════════
@app.websocket("/ws/ble/{player_id}")
async def ws_ble(websocket: WebSocket, player_id: int):
    """
    The browser connects to the ESP32 via Web Bluetooth API, receives JSON
    notify packets, then forwards them here over this WebSocket.
    Packet format (from ESP32): {"f":1420,"d":84,"b":45,"i":890,"h":12}
      f = FSR peak force (N)
      d = event duration (ms)
      b = ball piezo ADC (0-4095)
      i = instep piezo ADC (0-4095)
      h = heel piezo ADC (0-4095)
    Special control messages:
      {"ctrl":"connected",  "name":"Kixora-L"}
      {"ctrl":"disconnected"}
    """
    if player_id not in (1, 2):
        await websocket.close(code=4000)
        return
    await websocket.accept()
    log.info(f"[P{player_id}] Browser BLE relay WebSocket connected")

    # Mirror connected status into ble_mgr so /session/status stays accurate
    if ble_mgr and player_id in ble_mgr.devices:
        ble_mgr.devices[player_id].connected = False  # reset until we get ctrl:connected

    try:
        async for raw in websocket.iter_text():
            try:
                pkt = json.loads(raw)

                # ── Control messages ──────────────────────────────────
                ctrl = pkt.get("ctrl")
                if ctrl == "connected":
                    device_name = pkt.get("name", f"Kixora-{player_id}")
                    log.info(f"[P{player_id}] BLE relay: device connected — {device_name}")
                    if ble_mgr and player_id in ble_mgr.devices:
                        ble_mgr.devices[player_id].connected = True
                        ble_mgr.devices[player_id].address   = device_name
                        ble_mgr.devices[player_id].error     = None
                    await _broadcast_ui({
                        "type":   "ble_status",
                        "player": player_id,
                        "status": "connected",
                        "extra":  {"name": device_name},
                        "ts":     round(time.time() * 1000),
                    })
                    continue

                if ctrl == "disconnected":
                    log.info(f"[P{player_id}] BLE relay: device disconnected")
                    if ble_mgr and player_id in ble_mgr.devices:
                        ble_mgr.devices[player_id].connected = False
                    await _broadcast_ui({
                        "type":   "ble_status",
                        "player": player_id,
                        "status": "disconnected",
                        "extra":  {},
                        "ts":     round(time.time() * 1000),
                    })
                    continue

                # ── Sensor packet ─────────────────────────────────────
                if ble_mgr and player_id in ble_mgr.devices:
                    dev = ble_mgr.devices[player_id]
                    dev.last_packet   = pkt
                    dev.packet_count += 1

                await _on_ble_packet(player_id, pkt)

            except json.JSONDecodeError:
                log.warning(f"[P{player_id}] BLE relay: bad JSON — {raw[:80]}")
            except Exception as e:
                log.error(f"[P{player_id}] BLE relay error: {e}")

    except WebSocketDisconnect:
        log.info(f"[P{player_id}] Browser BLE relay WebSocket closed")
        if ble_mgr and player_id in ble_mgr.devices:
            ble_mgr.devices[player_id].connected = False
        await _broadcast_ui({
            "type":   "ble_status",
            "player": player_id,
            "status": "disconnected",
            "extra":  {"relay_closed": True},
            "ts":     round(time.time() * 1000),
        })


# ═══════════════════════════════════════════════════════════════
# REST: Session
# ═══════════════════════════════════════════════════════════════
class SessionStartRequest(BaseModel):
    session_name: str = "Training"
    player1_name: str = "Player 1"
    player2_name: str = "Player 2"


@app.post("/session/start")
async def session_start(req: SessionStartRequest):
    global session
    if session and session.is_recording:
        raise HTTPException(400, "Session already recording. Stop it first.")
    if session:
        session.close()
    session = SessionRecorder(
        session_name=req.session_name,
        player_names={1: req.player1_name, 2: req.player2_name},
    )
    session.start()
    # Reset classifiers so gravity is re-estimated
    for clf in classifiers.values():
        clf.reset_gravity()
    await _broadcast_ui({"type": "session_started", "summary": session.get_summary()})
    return session.get_summary()


@app.post("/session/stop")
async def session_stop():
    global session
    if not session:
        raise HTTPException(400, "No active session.")
    session.stop()
    summary = session.get_summary()
    await _broadcast_ui({"type": "session_stopped", "summary": summary})
    return summary


@app.get("/session/status")
async def session_status():
    return {
        "session": session.get_summary() if session else None,
        "ble": ble_mgr.status_dict(),
        "phones": list(imu_ws.keys()),
        "model_available": model_available,
    }


# ═══════════════════════════════════════════════════════════════
# REST: BLE
# ═══════════════════════════════════════════════════════════════
@app.post("/ble/scan")
async def ble_scan():
    devices = await ble_mgr.scan(duration=6.0)
    return {"devices": devices}


class BLEConnectRequest(BaseModel):
    player_id: int
    address: str


@app.post("/ble/connect")
async def ble_connect(req: BLEConnectRequest):
    if req.player_id not in (1, 2):
        raise HTTPException(400, "player_id must be 1 or 2")
    asyncio.create_task(ble_mgr.connect(req.player_id, req.address))
    return {"status": "connecting", "player": req.player_id, "address": req.address}


class BLEDisconnectRequest(BaseModel):
    player_id: int


@app.post("/ble/disconnect")
async def ble_disconnect(req: BLEDisconnectRequest):
    await ble_mgr.disconnect(req.player_id)
    return {"status": "disconnected", "player": req.player_id}


# ═══════════════════════════════════════════════════════════════
# REST: Manual kick annotation
# ═══════════════════════════════════════════════════════════════
class ManualKickRequest(BaseModel):
    player_id: int
    kick_type: str   # "Ap Chagi" | "Dolyo Chagi" | "Other"


@app.post("/kick/manual")
async def kick_manual(req: ManualKickRequest):
    if req.player_id not in (1, 2):
        raise HTTPException(400, "player_id must be 1 or 2")
    valid = ["Ap Chagi", "Dolyo Chagi", "Other"]
    if req.kick_type not in valid:
        raise HTTPException(400, f"kick_type must be one of {valid}")
    # BLE is optional — work fine without it
    last_ble = None
    if ble_mgr and req.player_id in ble_mgr.devices:
        last_ble = ble_mgr.devices[req.player_id].last_packet
    kick_event = KickEvent(
        timestamp_ms=time.time() * 1000,
        player_id=req.player_id,
        kick_type=req.kick_type,
        confidence=1.0,
        source="manual",
        fsr_peak=last_ble.get("f") if last_ble else None,
        ball_peak=last_ble.get("b") if last_ble else None,
        instep_peak=last_ble.get("i") if last_ble else None,
        duration_ms=last_ble.get("d") if last_ble else None,
    )
    if session and session.is_recording:
        session.record_kick(kick_event)
    await _broadcast_ui({
        "type": "kick_detected",
        "player": req.player_id,
        "kick": {
            "type": req.kick_type, "confidence": 1.0,
            "source": "manual",
            "fsr": last_ble.get("f") if last_ble else None,
        },
        "ts": round(time.time() * 1000),
    })
    return {"status": "recorded", "kick": req.kick_type, "player": req.player_id}


# ═══════════════════════════════════════════════════════════════
# REST: Sessions list
# ═══════════════════════════════════════════════════════════════
@app.get("/sessions")
async def list_sessions():
    sessions_root = os.path.join(BASE_DIR, "sessions")
    if not os.path.exists(sessions_root):
        return {"sessions": []}
    sessions_list = []
    for d in sorted(os.listdir(sessions_root), reverse=True):
        meta_path = os.path.join(sessions_root, d, "session_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            sessions_list.append(meta)
    return {"sessions": sessions_list}


@app.get("/api/info")
async def api_info():
    """Returns server LAN IP so dashboard can build correct phone links."""
    return {
        "lan_ip": LAN_IP,
        "port": 8000,
        "phone1_url": f"http://{LAN_IP}:8000/phone/1",
        "phone2_url": f"http://{LAN_IP}:8000/phone/2",
        "model_available": model_available,
        "phones_connected": list(imu_ws.keys()),
    }


# ═══════════════════════════════════════════════════════════════
# Serve UI
# ═══════════════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
async def root():
    index = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index):
        with open(index, encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>Kixora — place index.html in static/</h1>")


@app.get("/sessions_ui", response_class=HTMLResponse)
async def sessions_page():
    page = os.path.join(STATIC_DIR, "sessions.html")
    if os.path.exists(page):
        with open(page, encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>Sessions UI missing</h1>")


@app.get("/phone/{player_id}", response_class=HTMLResponse)
async def phone_page(player_id: int):
    """Served to the phone browser — streams IMU via WebSocket."""
    phone = os.path.join(STATIC_DIR, "phone.html")
    if os.path.exists(phone):
        with open(phone, encoding="utf-8") as f:
            content = f.read().replace("{{PLAYER_ID}}", str(player_id))
            return content
    return HTMLResponse(f"<h1>Phone page for Player {player_id} — place phone.html in static/</h1>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
