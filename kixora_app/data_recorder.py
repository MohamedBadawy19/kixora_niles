"""
data_recorder.py — Thread-safe session recorder.
Records FSR/piezo (from BLE) + IMU (from phone) + kick classifications.
Saves everything to CSV files per player per session.
"""
import os
import csv
import json
import time
import threading
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

RECORDS_DIR = os.path.join(os.path.dirname(__file__), "sessions")


@dataclass
class KickEvent:
    timestamp_ms: float
    player_id: int
    kick_type: str          # Ap Chagi / Dolyo Chagi / Manual
    confidence: float
    source: str             # "auto" | "manual"
    fsr_peak: Optional[int]   = None
    ball_peak: Optional[int]  = None
    instep_peak: Optional[int]= None
    duration_ms: Optional[int]= None
    probs: Optional[dict]     = None


class SessionRecorder:
    """
    One recorder per session (one match / training session).
    Maintains separate writers for:
      - imu_{player}.csv        (raw IMU samples from phone)
      - ble_{player}.csv        (BLE packets from ESP32)
      - kicks_{player}.csv      (detected + manual kick events)
      - session_meta.json       (session metadata)
    """

    def __init__(self, session_name: str, player_names: Dict[int, str]):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in session_name)
        self.session_dir = os.path.join(RECORDS_DIR, f"{ts}_{safe_name}")
        os.makedirs(self.session_dir, exist_ok=True)

        self.player_names = player_names
        self.start_time   = time.time()
        self.start_ms     = time.time() * 1000
        self.is_recording = False
        self._lock        = threading.Lock()

        # Writers per player
        self._imu_files:  Dict[int, Any] = {}
        self._imu_writers:Dict[int, csv.writer] = {}
        self._ble_files:  Dict[int, Any] = {}
        self._ble_writers:Dict[int, csv.writer] = {}
        self._kick_files: Dict[int, Any] = {}
        self._kick_writers:Dict[int, csv.writer] = {}

        self._kick_counts: Dict[int, Dict[str, int]] = {1: {}, 2: {}}
        self._total_kicks: Dict[int, int] = {1: 0, 2: 0}

        for pid in player_names:
            self._init_player_files(pid)

        self._save_meta("created")

    def _init_player_files(self, pid: int):
        name = self.player_names.get(pid, f"Player{pid}")

        # IMU file
        p = os.path.join(self.session_dir, f"imu_player{pid}_{name}.csv")
        f = open(p, "w", newline="", encoding="utf-8")
        w = csv.writer(f)
        w.writerow(["timestamp_ms", "elapsed_ms", "X", "Y", "Z",
                    "X_raw", "Y_raw", "Z_raw", "source_fs"])
        self._imu_files[pid]   = f
        self._imu_writers[pid] = w

        # BLE file
        p = os.path.join(self.session_dir, f"ble_player{pid}_{name}.csv")
        f = open(p, "w", newline="", encoding="utf-8")
        w = csv.writer(f)
        w.writerow(["timestamp_ms", "elapsed_ms", "fsr", "ball_piezo",
                    "instep_piezo", "duration_ms"])
        self._ble_files[pid]   = f
        self._ble_writers[pid] = w

        # Kicks file
        p = os.path.join(self.session_dir, f"kicks_player{pid}_{name}.csv")
        f = open(p, "w", newline="", encoding="utf-8")
        w = csv.writer(f)
        w.writerow(["timestamp_ms", "elapsed_ms", "kick_type", "confidence",
                    "source", "fsr_peak", "ball_peak", "instep_peak",
                    "duration_ms", "probs_json"])
        self._kick_files[pid]   = f
        self._kick_writers[pid] = w

    # ── Start / Stop ─────────────────────────────────────────
    def start(self):
        self.is_recording = True
        self.start_ms = time.time() * 1000
        self._save_meta("recording")

    def stop(self):
        self.is_recording = False
        self._flush_all()
        self._save_meta("stopped")

    def close(self):
        self.is_recording = False
        self._flush_all()
        for f in self._imu_files.values():   f.close()
        for f in self._ble_files.values():   f.close()
        for f in self._kick_files.values():  f.close()
        self._save_meta("closed")

    def _flush_all(self):
        for f in self._imu_files.values():   f.flush()
        for f in self._ble_files.values():   f.flush()
        for f in self._kick_files.values():  f.flush()

    # ── IMU ──────────────────────────────────────────────────
    def record_imu(self, pid: int, x: float, y: float, z: float,
                   x_raw: float = None, y_raw: float = None, z_raw: float = None,
                   src_fs: int = 500):
        if not self.is_recording: return
        now = time.time() * 1000
        elapsed = now - self.start_ms
        with self._lock:
            if pid in self._imu_writers:
                self._imu_writers[pid].writerow([
                    round(now, 1), round(elapsed, 1),
                    round(x, 4), round(y, 4), round(z, 4),
                    round(x_raw or x, 4), round(y_raw or y, 4),
                    round(z_raw or z, 4), src_fs
                ])

    # ── BLE ──────────────────────────────────────────────────
    def record_ble(self, pid: int, pkt: dict):
        if not self.is_recording: return
        now = time.time() * 1000
        elapsed = now - self.start_ms
        with self._lock:
            if pid in self._ble_writers:
                self._ble_writers[pid].writerow([
                    round(now, 1), round(elapsed, 1),
                    pkt.get("f", 0), pkt.get("b", 0),
                    pkt.get("i", 0), pkt.get("d", 0),
                ])

    # ── Kicks ─────────────────────────────────────────────────
    def record_kick(self, kick: KickEvent):
        now = time.time() * 1000
        elapsed = now - self.start_ms
        pid = kick.player_id
        with self._lock:
            if pid in self._kick_writers:
                self._kick_writers[pid].writerow([
                    round(now, 1), round(elapsed, 1),
                    kick.kick_type, round(kick.confidence, 3),
                    kick.source,
                    kick.fsr_peak, kick.ball_peak,
                    kick.instep_peak, kick.duration_ms,
                    json.dumps(kick.probs or {}),
                ])
            self._total_kicks[pid] = self._total_kicks.get(pid, 0) + 1
            kc = self._kick_counts.get(pid, {})
            kc[kick.kick_type] = kc.get(kick.kick_type, 0) + 1
            self._kick_counts[pid] = kc

    # ── Meta ─────────────────────────────────────────────────
    def _save_meta(self, status: str):
        meta = {
            "session_dir": self.session_dir,
            "status": status,
            "players": self.player_names,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "kick_counts": {str(k): v for k, v in self._kick_counts.items()},
            "total_kicks": {str(k): v for k, v in self._total_kicks.items()},
            "elapsed_s": round(time.time() - self.start_time, 1),
        }
        with open(os.path.join(self.session_dir, "session_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def get_summary(self) -> dict:
        self._save_meta("recording" if self.is_recording else "stopped")
        return {
            "session_dir": self.session_dir,
            "is_recording": self.is_recording,
            "elapsed_s": round(time.time() - self.start_time, 1),
            "kick_counts": self._kick_counts,
            "total_kicks": self._total_kicks,
            "player_names": self.player_names,
        }
