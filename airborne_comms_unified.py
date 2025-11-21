#!/usr/bin/env python3
"""
airborne_comms_with_enriched_csv_v5_random5.py

Final consolidated script:

- 70s video (1750 frames @25fps)
- RSA sign/verify, forced signature-fail simulation preserved
- Communications logging preserved
- Faults scheduled only while flight is inside radar (exit frame computed)
- 9 possible fault types:
    1) Fuel Leak
    2) Engine Failure
    3) Control Malfunction
    4) Electrical Failure
    5) Pressurization Failure
    6) Sensor Failure
    7) Autopilot Issue
    8) Fire Warning
    9) Structural Issue
- On EACH RUN: exactly 5 fault types are chosen at random (out of the 9),
  assigned to 5 different flights, at well-separated times.
- Any faulty flight blinks red until it exits the radar:
    - ON phase: bright red, slightly larger
    - OFF phase: its underlying ML color (or red if ML already red), slightly smaller
- All fault starts are logged once into CSV.
- Updated:
    * Each faulty flight is scheduled so that, as far as possible, it remains inside
      the radar for at least 8 seconds (~200 frames) AFTER the fault occurs.
    * The time difference between one faulty flight and another is increased
      (larger MIN_FAULT_GAP) to look more realistic.
    * NEW: Every 5 seconds (125 frames), each alive flight sends an ADS-B style
      "HEALTH_BROADCAST" to ATC, logging its state and active faults into the CSV.
"""

import random
import math
import time
import csv
import datetime
from dataclasses import dataclass

# headless-friendly backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes

# ---------------- CONFIG ----------------
NUM_FLIGHTS = 30
FRAME_COUNT = 1750               # 70 seconds @25fps
RANGE = 100
SPEED_SCALE = 0.18
COMM_RANGE = 40
COMM_PROB = 0.04
COMM_MAX_ACTIVE = 4
COMM_DURATION_FRAMES = 50
ML_RETRAIN_INTERVAL = 150
OUTPUT_FILENAME = "airborne_comms_unified_with_faults_random5.mp4"
CSV_LOG = "comms_enriched_log_v5_random5.csv"

# ADS-B style health broadcast every 5 seconds
HEALTH_BROADCAST_INTERVAL_FRAMES = 125  # 5s * 25fps

# NOTE: we do NOT seed random/np.random on purpose for variability between runs.
RNG_SEED = 42  # only used for IsolationForest, not for global random()

CALLSIGNS = [
    "Emirates-302", "Lufthansa-455", "Indigo-211", "AirIndia-441",
    "BritishAirways-207", "AirFrance-133", "Delta-102", "QatarAirways-229",
    "SingaporeAir-317", "American-904", "Vistara-824", "Etihad-421",
    "Turkish-701", "SpiceJet-605", "GoFirst-372", "United-811",
    "Cathay-503", "Swiss-725", "KLM-431", "Aeroflot-221",
    "Finnair-117", "Malaysia-712", "Thai-323", "KoreanAir-501",
    "JapanAir-214", "ChinaAir-410", "AirCanada-903", "Ethiopian-610",
    "EgyptAir-655", "Qantas-806"
]

# ---------------- RSA helpers ----------------
def generate_rsa_keypair():
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pub = priv.public_key()
    return priv, pub

def sign_message(priv_key, message_bytes: bytes):
    return priv_key.sign(
        message_bytes,
        padding.PKCS1v15(),
        hashes.SHA256()
    )

def verify_message(pub_key, message_bytes: bytes, signature: bytes):
    try:
        pub_key.verify(signature, message_bytes, padding.PKCS1v15(), hashes.SHA256())
        return True
    except Exception:
        return False

# ---------------- Flight class ----------------
@dataclass
class Flight:
    callsign: str
    x: float
    y: float
    vx: float
    vy: float
    private_key: object
    public_key: object
    alive: bool = True
    color: str = "lime"
    # fuel leak
    fuel_kg: float = 20000.0
    base_fuel_flow: float = 1.8
    leak_rate: float = 0.0
    leak_active: bool = False
    leak_start_frame: int = None
    # engine
    engine_failed: bool = False
    engine_fail_frame: int = None
    # control
    control_failed: bool = False
    control_malf_name: str = ""
    control_fail_frame: int = None
    # electrical
    electrical_failed: bool = False
    electrical_malf_name: str = ""
    electrical_fail_frame: int = None
    # pressurization
    pressurization_failed: bool = False
    pressurization_malf_name: str = ""
    pressurization_fail_frame: int = None
    # sensor
    sensor_failed: bool = False
    sensor_malf_name: str = ""
    sensor_fail_frame: int = None
    # autopilot
    autopilot_failed: bool = False
    autopilot_malf_name: str = ""
    autopilot_fail_frame: int = None
    # fire
    fire_warning: bool = False
    fire_warn_frame: int = None
    # structural
    structural_failed: bool = False
    structural_malf_name: str = ""
    structural_fail_frame: int = None
    # fuel history
    fuel_history: list = None

    def move(self):
        if not self.alive:
            return
        self.x += self.vx
        self.y += self.vy
        if math.hypot(self.x, self.y) > RANGE:
            self.alive = False

    def sign(self, text: str) -> bytes:
        return sign_message(self.private_key, text.encode("utf-8"))

    def consume_fuel(self, dt):
        if self.fuel_history is None:
            self.fuel_history = []
        total_burn = (self.base_fuel_flow + self.leak_rate) * dt
        self.fuel_kg = max(0.0, self.fuel_kg - total_burn)
        now = time.time()
        self.fuel_history.append((now, self.fuel_kg))
        cutoff = now - 10.0
        while len(self.fuel_history) > 0 and self.fuel_history[0][0] < cutoff:
            self.fuel_history.pop(0)

# ---------------- Initialization ----------------
def spawn_flight_from_boundary():
    angle = random.uniform(0, 2 * math.pi)
    x = RANGE * math.cos(angle)
    y = RANGE * math.sin(angle)
    inward_angle = angle + math.pi + random.uniform(-0.3, 0.3)
    speed = random.uniform(0.6, 1.4) * SPEED_SCALE
    vx = speed * math.cos(inward_angle)
    vy = speed * math.sin(inward_angle)
    return x, y, vx, vy

flights = []
for i in range(NUM_FLIGHTS):
    x, y, vx, vy = spawn_flight_from_boundary()
    priv, pub = generate_rsa_keypair()
    flights.append(
        Flight(
            callsign=CALLSIGNS[i],
            x=x, y=y, vx=vx, vy=vy,
            private_key=priv, public_key=pub
        )
    )

# ---------------- Helper: exit frame ----------------
FRAME_DURATION_SEC = 1.0 / 25.0
def compute_exit_frame_for_flight(f: Flight, max_search=FRAME_COUNT):
    x, y, vx, vy = f.x, f.y, f.vx, f.vy
    for fr in range(0, max_search + 1):
        if math.hypot(x, y) > RANGE:
            return fr
        x += vx
        y += vy
    return max_search + 1

exit_frames = [compute_exit_frame_for_flight(f) for f in flights]

# ---------------- ML model ----------------
scaler = StandardScaler()
def train_ml_model(flights_list):
    data = []
    for f in flights_list:
        if f.alive:
            data.append([f.x, f.y, f.vx, f.vy])
    if len(data) < 10:
        data = np.random.normal(0, RANGE / 6, size=(200, 4))
    else:
        data = np.array(data)
        aug = np.random.normal(0, RANGE / 6, size=(200, 4))
        data = np.vstack([data, aug])
    scaled = scaler.fit_transform(data)
    model = IsolationForest(contamination=0.06, random_state=RNG_SEED)
    model.fit(scaled)
    return model

ml_model = train_ml_model(flights)
last_ml_train_frame = 0

# ---------------- CSV LOGGING ----------------
csv_file = open(CSV_LOG, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Frame", "Timestamp", "Sender", "Receiver", "Distance", "Status", "Reason"])

def write_log(frame, sender, receiver, dist, status, reason):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_writer.writerow([frame, ts, sender, receiver, f"{dist:.1f}", status, reason])
    csv_file.flush()

# ---------------- Matplotlib ----------------
fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
ax.set_facecolor("black")
ax.set_xlim(-RANGE - 10, RANGE + 10)
ax.set_ylim(-RANGE - 10, RANGE + 10)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("ATC Radar — Secure Airborne Communication", color="lime", fontsize=14)
for r in range(20, RANGE + 1, 20):
    ax.add_patch(plt.Circle((0, 0), r, color="green", fill=False, linestyle="dotted", alpha=0.45))
ax.axhline(0, color="green", lw=0.5, alpha=0.4)
ax.axvline(0, color="green", lw=0.5, alpha=0.4)

sweep_line, = ax.plot([], [], color="lime", lw=1.5, alpha=0.9)
dots = [ax.plot([], [], "o", color="lime", markersize=6)[0] for _ in flights]
labels = [ax.text(0, 0, "", color="white", fontsize=7) for _ in flights]
active_comms = []

# ATC alert panel
alert_text = ax.text(
    -RANGE + 5, RANGE - 8, "", color="red", fontsize=9, va="top", ha="left",
    bbox=dict(facecolor='black', alpha=0.6, edgecolor='red')
)

# ---------------- Fault scheduling: choose 5 random fault types ----------------
ALL_FAULT_TYPES = [
    "fuel_leak",
    "engine_failure",
    "control_failure",
    "electrical_failure",
    "pressurization_failure",
    "sensor_failure",
    "autopilot_failure",
    "fire_warning",
    "structural_failure",
]

# Choose 5 random fault types each run
chosen_fault_types = random.sample(ALL_FAULT_TYPES, 5)

# Assign them to 5 distinct flights
available_indices = list(range(NUM_FLIGHTS))
random.shuffle(available_indices)
fault_flights = {}
for ft in chosen_fault_types:
    fault_flights[ft] = available_indices.pop()

# Default preferred frames for each fault type
DEFAULT_FRAMES = {
    "fuel_leak": 350,
    "engine_failure": 950,
    "control_failure": 1150,
    "electrical_failure": 1350,
    "pressurization_failure": 600,   # preferred special frame
    "sensor_failure": 1050,
    "autopilot_failure": 1250,
    "fire_warning": 750,             # preferred special frame
    "structural_failure": 830,       # preferred special frame
}

# Time spacing + in-radar constraints
MIN_FRAME = 80
SAFETY_MARGIN = 15        # distance from exit frame so fault is visible
MIN_VISIBLE_FRAMES = 200  # at least 8 seconds (200 frames) after fault inside radar
MIN_FAULT_GAP = 260       # min gap (frames) between faults (~10.4 s)

# Prepare schedule dict: fault_type -> {"idx": flight_idx, "frame": frame}
fault_schedule = {}

# Sort chosen faults by their default frame so they appear in a logical order
sorted_faults = sorted(
    [(ft, fault_flights[ft], DEFAULT_FRAMES[ft]) for ft in chosen_fault_types],
    key=lambda x: x[2]
)

last_fault_frame = MIN_FRAME - MIN_FAULT_GAP

def choose_frame_for_flight(desired_min, desired_max, exit_f):
    """
    Choose a frame such that:
      - frame is >= desired_min
      - frame is <= desired_max
      - frame is <= exit_f - SAFETY_MARGIN - 1 - MIN_VISIBLE_FRAMES
        (so at least MIN_VISIBLE_FRAMES remain inside radar after fault)
    If no such frame exists, schedule as late as possible while still trying
    to honor the MIN_VISIBLE_FRAMES requirement as best as we can.
    """
    upper_limit = exit_f - SAFETY_MARGIN - 1 - MIN_VISIBLE_FRAMES
    upper = min(desired_max, upper_limit)
    lower = desired_min

    if upper < lower:
        candidate = min(desired_max, max(MIN_FRAME, upper_limit))
        return max(MIN_FRAME, candidate)
    return random.randint(lower, upper)

for ft, idx, base_frame in sorted_faults:
    exit_f = exit_frames[idx]
    candidate_min = max(base_frame, last_fault_frame + MIN_FAULT_GAP, MIN_FRAME)
    candidate_max = FRAME_COUNT - 1  # we'll clamp inside choose_frame_for_flight

    frame = choose_frame_for_flight(candidate_min, candidate_max, exit_f)
    fault_schedule[ft] = {"idx": idx, "frame": frame}
    last_fault_frame = frame

print("Scheduled faults (5 random types, flights inside radar long enough after fault):")
for ft in chosen_fault_types:
    data = fault_schedule[ft]
    idx = data["idx"]
    frame = data["frame"]
    print(f"  {ft} -> {flights[idx].callsign} at frame {frame} (exit={exit_frames[idx]})")

# Parameters for some faults
LEAK_RATE_MIN = 0.3
LEAK_RATE_MAX = 1.5
BLINK_PERIOD_FRAMES = 6  # blink toggle granularity

CONTROL_MALF_TYPES = ["Aileron Jam", "Elevator Runaway", "Rudder Hardover", "Flaps Malfunction", "Autopilot Disconnect"]
ELECTRICAL_MALF_TYPES = ["Generator Failure", "Main Bus Trip", "Battery Fault", "Transformer Failure", "TRU Failure"]
PRESSURIZATION_MALF_TYPES = ["Outflow Valve Failure", "Cabin Altitude Controller Failure", "Cabin Pressure Leak"]
SENSOR_MALF_TYPES = ["Pitot Failure", "AoA Sensor Fault", "ADC Disagreement", "Static Port Blocked"]
AUTOPILOT_MALF_TYPES = ["Auto Trim Fault", "AP Disconnect", "AP Mode Erratic"]
FIRE_MALF_TYPES = ["Engine Fire", "Cargo Fire", "APU Fire"]
STRUCTURAL_MALF_TYPES = ["Flight Control Surface Crack", "Fuselage Deformation", "Door Seal Failure"]

# ---------------- Color logic ----------------
def get_ml_color(f):
    feat = np.array([[f.x, f.y, f.vx, f.vy]])
    scaled = scaler.transform(feat)
    score = ml_model.decision_function(scaled)[0]
    if score > 0.05:
        return "lime"
    elif score > -0.02:
        return "yellow"
    else:
        return "red"

# ---------------- Update ----------------
def update(frame):
    global ml_model, last_ml_train_frame, active_comms

    # move + fuel consumption
    for f in flights:
        f.move()
        if f.alive:
            f.consume_fuel(FRAME_DURATION_SEC)

    # ---------------- ADS-B style health broadcasts every 5 seconds ----------------
    if frame % HEALTH_BROADCAST_INTERVAL_FRAMES == 0:
        for f in flights:
            if not f.alive:
                continue
            faults = []
            if f.leak_active:
                faults.append("fuel_leak")
            if f.engine_failed:
                faults.append("engine_failure")
            if f.control_failed:
                faults.append("control_failure")
            if f.electrical_failed:
                faults.append("electrical_failure")
            if f.pressurization_failed:
                faults.append("pressurization_failure")
            if f.sensor_failed:
                faults.append("sensor_failure")
            if f.autopilot_failed:
                faults.append("autopilot_failure")
            if f.fire_warning:
                faults.append("fire_warning")
            if f.structural_failed:
                faults.append("structural_failure")

            faults_str = ",".join(faults) if faults else "none"
            reason = (
                f"STATE x={f.x:.1f} y={f.y:.1f} "
                f"vx={f.vx:.2f} vy={f.vy:.2f} "
                f"faults={faults_str}"
            )
            write_log(frame, f.callsign, "ATC", 0.0, "HEALTH_BROADCAST", reason)

    # ---------------- Fault triggers (each exactly once) ----------------

    # Fuel leak
    if "fuel_leak" in fault_schedule and frame == fault_schedule["fuel_leak"]["frame"]:
        idx = fault_schedule["fuel_leak"]["idx"]
        f = flights[idx]
        if f.alive and not f.leak_active:
            f.leak_rate = random.uniform(LEAK_RATE_MIN, LEAK_RATE_MAX)
            f.leak_active = True
            f.leak_start_frame = frame
            reason = f"LEAK_START leak_rate={f.leak_rate:.2f}kg/s fuel_left={f.fuel_kg:.1f}kg"
            write_log(frame, f.callsign, "ATC", 0.0, "LEAK_START", reason)

    # Engine failure
    if "engine_failure" in fault_schedule and frame == fault_schedule["engine_failure"]["frame"]:
        idx = fault_schedule["engine_failure"]["idx"]
        f = flights[idx]
        if f.alive and not f.engine_failed:
            f.engine_failed = True
            f.engine_fail_frame = frame
            reason = "ENGINE_FAILURE detected"
            write_log(frame, f.callsign, "ATC", 0.0, "ENGINE_FAILURE", reason)

    # Control malfunction
    if "control_failure" in fault_schedule and frame == fault_schedule["control_failure"]["frame"]:
        idx = fault_schedule["control_failure"]["idx"]
        f = flights[idx]
        if f.alive and not f.control_failed:
            f.control_failed = True
            f.control_fail_frame = frame
            f.control_malf_name = random.choice(CONTROL_MALF_TYPES)
            reason = f"CONTROL_FAILURE {f.control_malf_name}"
            write_log(frame, f.callsign, "ATC", 0.0, "CONTROL_FAILURE", reason)

    # Electrical failure
    if "electrical_failure" in fault_schedule and frame == fault_schedule["electrical_failure"]["frame"]:
        idx = fault_schedule["electrical_failure"]["idx"]
        f = flights[idx]
        if f.alive and not f.electrical_failed:
            f.electrical_failed = True
            f.electrical_fail_frame = frame
            f.electrical_malf_name = random.choice(ELECTRICAL_MALF_TYPES)
            reason = f"ELECTRICAL_FAILURE {f.electrical_malf_name}"
            write_log(frame, f.callsign, "ATC", 0.0, "ELECTRICAL_FAILURE", reason)

    # Pressurization failure
    if "pressurization_failure" in fault_schedule and frame == fault_schedule["pressurization_failure"]["frame"]:
        idx = fault_schedule["pressurization_failure"]["idx"]
        f = flights[idx]
        if f.alive and not f.pressurization_failed:
            f.pressurization_failed = True
            f.pressurization_fail_frame = frame
            f.pressurization_malf_name = random.choice(PRESSURIZATION_MALF_TYPES)
            reason = f"PRESSURIZATION_FAILURE {f.pressurization_malf_name}"
            write_log(frame, f.callsign, "ATC", 0.0, "PRESSURIZATION_FAILURE", reason)

    # Sensor failure
    if "sensor_failure" in fault_schedule and frame == fault_schedule["sensor_failure"]["frame"]:
        idx = fault_schedule["sensor_failure"]["idx"]
        f = flights[idx]
        if f.alive and not f.sensor_failed:
            f.sensor_failed = True
            f.sensor_fail_frame = frame
            f.sensor_malf_name = random.choice(SENSOR_MALF_TYPES)
            reason = f"SENSOR_FAILURE {f.sensor_malf_name}"
            write_log(frame, f.callsign, "ATC", 0.0, "SENSOR_FAILURE", reason)

    # Autopilot issue
    if "autopilot_failure" in fault_schedule and frame == fault_schedule["autopilot_failure"]["frame"]:
        idx = fault_schedule["autopilot_failure"]["idx"]
        f = flights[idx]
        if f.alive and not f.autopilot_failed:
            f.autopilot_failed = True
            f.autopilot_fail_frame = frame
            f.autopilot_malf_name = random.choice(AUTOPILOT_MALF_TYPES)
            reason = f"AUTOPILOT_FAILURE {f.autopilot_malf_name}"
            write_log(frame, f.callsign, "ATC", 0.0, "AUTOPILOT_FAILURE", reason)

    # Fire warning
    if "fire_warning" in fault_schedule and frame == fault_schedule["fire_warning"]["frame"]:
        idx = fault_schedule["fire_warning"]["idx"]
        f = flights[idx]
        if f.alive and not f.fire_warning:
            f.fire_warning = True
            f.fire_warn_frame = frame
            reason = f"FIRE_WARNING {random.choice(FIRE_MALF_TYPES)}"
            write_log(frame, f.callsign, "ATC", 0.0, "FIRE_WARNING", reason)

    # Structural issue
    if "structural_failure" in fault_schedule and frame == fault_schedule["structural_failure"]["frame"]:
        idx = fault_schedule["structural_failure"]["idx"]
        f = flights[idx]
        if f.alive and not f.structural_failed:
            f.structural_failed = True
            f.structural_fail_frame = frame
            f.structural_malf_name = random.choice(STRUCTURAL_MALF_TYPES)
            reason = f"STRUCTURAL_FAILURE {f.structural_malf_name}"
            write_log(frame, f.callsign, "ATC", 0.0, "STRUCTURAL_FAILURE", reason)

    # retrain ml occasionally
    if frame - last_ml_train_frame >= ML_RETRAIN_INTERVAL:
        ml_model = train_ml_model(flights)
        last_ml_train_frame = frame

    # update ML color
    for f in flights:
        if f.alive:
            f.color = get_ml_color(f)

    # clean expired comm lines
    for comm in active_comms[:]:
        comm["frames"] -= 1
        if comm["frames"] <= 0:
            try:
                comm["line"].remove()
            except Exception:
                pass
            active_comms.remove(comm)

    # COMM attempt (unchanged)
    if random.random() < COMM_PROB:
        alive_idx = [i for i, f in enumerate(flights) if f.alive]
        if len(alive_idx) >= 2:
            i, j = random.sample(alive_idx, 2)
            f1, f2 = flights[i], flights[j]
            dist = math.hypot(f1.x - f2.x, f1.y - f2.y)
            if dist <= COMM_RANGE:
                msg = f"POS:{f1.callsign}:{f1.x:.1f},{f1.y:.1f}:{int(time.time())}"
                sig = f1.sign(msg)
                verified = verify_message(f1.public_key, msg.encode("utf-8"), sig)
                if verified and random.random() < 0.30:
                    verified = False
                if verified:
                    line_color = "lime"
                    write_log(frame, f1.callsign, f2.callsign, dist, "ACCEPTED", "")
                else:
                    line_color = "red"
                    write_log(frame, f1.callsign, f2.callsign, dist, "FAILED", "Signature Failed")
                if len(active_comms) < COMM_MAX_ACTIVE:
                    line_obj, = ax.plot(
                        [f1.x, f2.x], [f1.y, f2.y],
                        color=line_color, lw=1.3, alpha=0.8
                    )
                    active_comms.append({"frames": COMM_DURATION_FRAMES, "line": line_obj})
            else:
                write_log(frame, f1.callsign, f2.callsign, dist, "FAILED", "No Link Established")

    # build active alerts
    active_alerts = []
    for f in flights:
        if not f.alive:
            continue
        if f.leak_active:
            active_alerts.append(f"{f.callsign}: LEAK {f.leak_rate:.2f}kg/s")
        if f.engine_failed:
            active_alerts.append(f"{f.callsign}: ENGINE FAILURE")
        if f.control_failed:
            active_alerts.append(f"{f.callsign}: CONTROL FAIL ({f.control_malf_name})")
        if f.electrical_failed:
            active_alerts.append(f"{f.callsign}: ELECTRICAL FAIL ({f.electrical_malf_name})")
        if f.pressurization_failed:
            active_alerts.append(f"{f.callsign}: PRESSURE FAIL ({f.pressurization_malf_name})")
        if f.sensor_failed:
            active_alerts.append(f"{f.callsign}: SENSOR FAIL ({f.sensor_malf_name})")
        if f.autopilot_failed:
            active_alerts.append(f"{f.callsign}: AUTOPILOT FAIL ({f.autopilot_malf_name})")
        if f.fire_warning:
            active_alerts.append(f"{f.callsign}: FIRE WARN")
        if f.structural_failed:
            active_alerts.append(f"{f.callsign}: STRUCT FAIL ({f.structural_malf_name})")

    # radar sweep
    ang = math.radians((frame * 2) % 360)
    sweep_line.set_data(
        [0, (RANGE + 8) * math.cos(ang)],
        [0, (RANGE + 8) * math.sin(ang)]
    )

    # update dots and labels; faulty flights blink until exit
    for idx, f in enumerate(flights):
        if f.alive:
            has_fault = (
                f.leak_active or f.engine_failed or f.control_failed or
                f.electrical_failed or f.pressurization_failed or
                f.sensor_failed or f.autopilot_failed or
                f.fire_warning or f.structural_failed
            )
            base_color = f.color
            if has_fault:
                # blink in red vs underlying ML color,
                # and also slightly change size so it's clearly visible
                div = max(1, (BLINK_PERIOD_FRAMES // 2))
                on_off = ((frame // div) % 2) == 0
                if on_off:
                    display_color = "red"
                    size = 9
                else:
                    display_color = base_color
                    size = 4 if base_color == "red" else 6
            else:
                display_color = base_color
                size = 6

            dots[idx].set_data([f.x], [f.y])
            dots[idx].set_color(display_color)
            dots[idx].set_markersize(size)
            labels[idx].set_position((f.x + 2, f.y + 2))
            labels[idx].set_text(f.callsign)
        else:
            dots[idx].set_data([], [])
            labels[idx].set_text("")

    # ATC panel (show up to 12)
    if len(active_alerts) == 0:
        alert_text.set_text("")
    else:
        alert_text.set_text("\n".join(active_alerts[:12]))

    artists = [sweep_line] + dots + labels + [alert_text]
    for comm in active_comms:
        artists.append(comm["line"])
    return artists

# ---------------- Run & Save ----------------
print("🚀 Starting radar + enriched CSV logging (random5 faults, health beacons, blink until exit, 8s visibility)...")
anim = FuncAnimation(fig, update, frames=FRAME_COUNT, interval=40, blit=True)
video_writer = FFMpegWriter(fps=25)
anim.save(OUTPUT_FILENAME, writer=video_writer, dpi=160)
csv_file.close()
print(f"🎬 Video saved: {OUTPUT_FILENAME}")
print(f"📄 Enriched log saved: {CSV_LOG}")
