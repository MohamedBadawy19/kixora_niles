# Kixora Smart System: Quick Start & Field Guide

This document is your reference manual for setting up the Kixora system in any new environment (like a gym or tournament), preparing the athletes, and saving clean data.

---

## 1. Moving to a New Location / New WiFi

When you take your laptop and phones to a new location, the IP addresses will change. The Kixora system is designed to handle this automatically, provided you follow these steps:

### Network Setup
1. **Connect Everything to the Same Network:** Ensure your laptop, Player 1's phone, and Player 2's phone are all connected to the **exact same WiFi network**.
2. **Start the Server:** Open a terminal on your laptop, navigate to the `kixora_app` folder, and run:
   ```bash
   python main.py
   ```
3. **Open the Dashboard:** Open your laptop's web browser and go to:
   ```
   http://localhost:8000
   ```
4. **Get the Phone Links:** Scroll to the bottom of the dashboard to the **"📱 Phone Setup"** panel. Kixora will automatically detect your new WiFi IP address (e.g., `192.168.x.x`). Use the URLs provided there for the phones.

### Phone Setup (Browser Rules)
* **iPhone:** You **MUST** use Safari. Go to `Settings -> Safari -> Advanced -> Motion & Orientation Access` and ensure it is turned **ON**.
* **Android:** Do **NOT** use Chrome (Chrome blocks motion sensors on local IPs). Use **Firefox for Android** or **Samsung Internet**.

Navigate to the provided URL on the phone, tap **"Start Streaming"**, and verify the `|a|` (magnitude) and `Motion events` counters are moving.

---

## 2. Athlete Preparation (Phone Placement)

For the machine learning model to work correctly (and for future data collection to be consistent), the phone must be securely attached to the athlete's leg.

### Placement Rules
1. **Location:** Strap the phone securely to the **lower leg (calf/shin area)**, just below the knee.
2. **Orientation:** The screen should face outward (away from the leg). The exact rotation (upside down, tilted) does not matter significantly because the `RealtimeClassifier` automatically estimates and aligns the data to gravity at the start of every session.
3. **Tightness:** Use a tight sports band. If the phone wiggles or shakes independently of the leg, the data will be noisy and kicks will be harder to classify.

### The "Gravity Calibration" Phase
Whenever you click **"▶ Start Recording"** on the dashboard, the system resets its gravity estimate.
* **Important:** Have the athlete stand **still in a neutral stance for the first 2-3 seconds** after you click Start. This allows the system to establish a perfect baseline of which way is "down."

---

## 3. Data Collection (Session Recording)

If your goal is to collect raw data to train a better model later, follow this exact workflow:

1. **Name the Session:** Use descriptive names in the dashboard.
   * *Example Session Name:* `Dataset_ApChagi_Mohamed`
2. **Start the Recording:** Click **"▶ Start Recording"**. (Remember to have the athlete stand still for 2 seconds).
3. **Perform the Action:** Have the athlete perform 20-30 of the *same* kick (e.g., only Ap Chagi).
4. **Use Manual Labels (Optional but highly recommended):** While they kick, tap the **"Ap Chagi"** button under the "Manual Kick" section on the dashboard every time they land a strike. This guarantees a perfect label is saved in the CSV, even if the automatic model misses it.
5. **Stop the Recording:** Click **"■ Stop"**. The data is instantly saved.
6. **Repeat:** Change the session name (e.g., `Dataset_Dolyo_Mohamed`) and record the next kick type.

### Where is the data?
Inside `kixora_app/sessions/`, you will find a new folder for every session.
* `imu_playerX.csv` contains the raw 50Hz motion data for signal processing.
* `kicks_playerX.csv` contains the timestamps of every kick detected (both auto and manual).

---

## 4. Hardware Guide: The ESP32 Smart Sock

When you are ready to build the physical sock with sensors:

### The Hardware
* **Microcontroller:** ESP32 or ESP32-S3 (requires a small LiPo battery).
* **Sensors:** 
  * FSR (Force Sensitive Resistor) under the heel/sole.
  * 2x Piezo discs (one on the ball of the foot, one on the instep).

### The Firmware Requirements
The ESP32 must act as a BLE Server.
* **Service UUID:** `4fafc201-1fb5-459e-8fcc-c5c9c331914b`
* **Characteristic UUID:** `beb5483e-36e1-4688-b7f5-ea07361b26a8`

When a strike occurs, the ESP32 must send a JSON string via a BLE Notification to that characteristic.
* **Format:** `{"f":4095, "b":800, "i":120, "d":85}`
  * `f` = FSR peak value
  * `b` = Ball piezo peak value
  * `i` = Instep piezo peak value
  * `d` = Duration of the impact event in milliseconds

### Connecting the Sock (PC Setup)
Before scanning on the dashboard, ensure your Windows PC is ready to receive BLE connections:
1. **Turn on Bluetooth:** Go to Windows Settings -> Bluetooth & devices and ensure Bluetooth is toggled **ON**.
2. **Turn on the ESP32:** Ensure it is powered on and flashing/advertising.
3. **Scan on Dashboard:** On the Kixora dashboard, under the Player's card, click **Scan**.
4. **Connect:** Select your ESP32 from the dropdown and click **Connect**.
5. The dashboard will now merge the ESP32 impact data with the Phone's IMU data automatically when recording.
