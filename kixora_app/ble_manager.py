"""
ble_manager.py — Async BLE manager for ESP32-S3 Kixora sock devices.
Handles scan, connect, disconnect, auto-reconnect for 2 players.
"""
import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict
from bleak import BleakClient, BleakScanner
from bleak.exc import BleakError

log = logging.getLogger("ble_manager")

SERVICE_UUID        = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
DEVICE_NAME_PREFIX  = "Kixora"


@dataclass
class DeviceState:
    player_id: int           # 1 or 2
    address: Optional[str] = None
    client: Optional[BleakClient] = None
    connected: bool = False
    rssi: Optional[int] = None
    last_packet: Optional[dict] = None
    packet_count: int = 0
    error: Optional[str] = None


class BLEManager:
    def __init__(self):
        self.devices: Dict[int, DeviceState] = {
            1: DeviceState(player_id=1),
            2: DeviceState(player_id=2),
        }
        self._packet_callbacks: list[Callable] = []
        self._status_callbacks: list[Callable] = []
        self._scan_results: list[dict] = []
        self._scanning = False
        self._stop_event = asyncio.Event()

    # ── Callbacks ────────────────────────────────────────────
    def on_packet(self, cb: Callable):
        self._packet_callbacks.append(cb)

    def on_status(self, cb: Callable):
        self._status_callbacks.append(cb)

    async def _emit_packet(self, player_id: int, data: dict):
        for cb in self._packet_callbacks:
            try:
                await cb(player_id, data)
            except Exception as e:
                log.error(f"Packet callback error: {e}")

    async def _emit_status(self, player_id: int, status: str, extra: dict = None):
        for cb in self._status_callbacks:
            try:
                await cb(player_id, status, extra or {})
            except Exception as e:
                log.error(f"Status callback error: {e}")

    # ── Scan ─────────────────────────────────────────────────
    async def scan(self, duration: float = 5.0) -> list[dict]:
        """Scan for Kixora BLE devices. Returns list of {name, address, rssi}."""
        self._scanning = True
        self._scan_results = []
        log.info(f"Scanning for {duration}s...")
        try:
            devices = await BleakScanner.discover(timeout=duration)
            for d in devices:
                name = d.name or ""
                if DEVICE_NAME_PREFIX in name or SERVICE_UUID.lower() in str(d.metadata).lower():
                    entry = {
                        "name": name or "Kixora",
                        "address": d.address,
                        "rssi": d.rssi,
                    }
                    self._scan_results.append(entry)
                    log.info(f"  Found: {entry}")
        except Exception as e:
            log.error(f"Scan error: {e}")
        self._scanning = False
        return self._scan_results

    # ── Connect ──────────────────────────────────────────────
    async def connect(self, player_id: int, address: str):
        dev = self.devices[player_id]
        if dev.connected:
            await self.disconnect(player_id)
        dev.address = address
        dev.error   = None
        await self._emit_status(player_id, "connecting", {"address": address})
        try:
            client = BleakClient(address, disconnected_callback=self._make_disconnect_cb(player_id))
            await client.connect(timeout=10.0)
            dev.client    = client
            dev.connected = True
            dev.error     = None
            log.info(f"[P{player_id}] Connected to {address}")
            await self._emit_status(player_id, "connected", {"address": address})
            # Subscribe to notifications
            await client.start_notify(CHARACTERISTIC_UUID, self._make_notify_cb(player_id))
        except BleakError as e:
            dev.error     = str(e)
            dev.connected = False
            log.error(f"[P{player_id}] Connect failed: {e}")
            await self._emit_status(player_id, "error", {"error": str(e)})
        except asyncio.TimeoutError:
            dev.error     = "Connection timed out"
            dev.connected = False
            await self._emit_status(player_id, "error", {"error": "Connection timed out"})

    async def disconnect(self, player_id: int):
        dev = self.devices[player_id]
        if dev.client:
            try:
                await dev.client.disconnect()
            except Exception:
                pass
        dev.client    = None
        dev.connected = False
        await self._emit_status(player_id, "disconnected", {})

    # ── Internal callbacks ────────────────────────────────────
    def _make_notify_cb(self, player_id: int):
        def _cb(sender, data: bytearray):
            asyncio.create_task(self._handle_notify(player_id, data))
        return _cb

    async def _handle_notify(self, player_id: int, data: bytearray):
        try:
            text = data.decode("utf-8").strip()
            pkt  = json.loads(text)
            dev  = self.devices[player_id]
            dev.last_packet  = pkt
            dev.packet_count += 1
            await self._emit_packet(player_id, pkt)
        except Exception as e:
            log.warning(f"[P{player_id}] Bad packet: {e} | raw={data}")

    def _make_disconnect_cb(self, player_id: int):
        def _cb(client: BleakClient):
            asyncio.create_task(self._on_disconnect(player_id))
        return _cb

    async def _on_disconnect(self, player_id: int):
        dev = self.devices[player_id]
        dev.connected = False
        dev.client    = None
        log.warning(f"[P{player_id}] Disconnected unexpectedly")
        await self._emit_status(player_id, "disconnected", {"unexpected": True})

    # ── Status ───────────────────────────────────────────────
    def status_dict(self) -> dict:
        result = {}
        for pid, dev in self.devices.items():
            result[str(pid)] = {
                "player_id": pid,
                "address":   dev.address,
                "connected": dev.connected,
                "packet_count": dev.packet_count,
                "error":     dev.error,
                "last_packet": dev.last_packet,
            }
        return result
