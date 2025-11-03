#!/usr/bin/env python3
"""
airborne_comms_with_enriched_csv_v3.py

Updates:
- Introduces a forced signature-failure probability for in-range comms
  (FORCED_SIG_FAIL_PROB = 0.30 => 30% of otherwise-verified messages are forced to fail)
- Keeps animation & visuals unchanged (green = accepted, red = failed)
- Logs EVERY attempt; distance always recorded
- CSV format: Frame, Timestamp, Sender, Receiver, Distance, Status, Reason
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
FRAME_COUNT = 1200
RANGE = 100
SPEED_SCALE = 0.18
COMM_RANGE = 40
COMM_PROB = 0.04
COMM_MAX_ACTIVE = 4
COMM_DURATION_FRAMES = 50
ML_RETRAIN_INTERVAL = 150
OUTPUT_FILENAME = "airborne_comms_unified.mp4"
CSV_LOG = "comms_enriched_log_v3.csv"
RNG_SEED = 42

# Forced signature-failure probability for in-range messages (30%)
FORCED_SIG_FAIL_PROB = 0.30

random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

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

    def move(self):
        if not self.alive:
            return
        self.x += self.vx
        self.y += self.vy
        if math.hypot(self.x, self.y) > RANGE:
            self.alive = False

    def sign(self, text: str) -> bytes:
        return sign_message(self.private_key, text.encode("utf-8"))

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
    flights.append(Flight(
        callsign=CALLSIGNS[i],
        x=x, y=y, vx=vx, vy=vy,
        private_key=priv, public_key=pub
    ))

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

# ---------------- Matplotlib setup ----------------
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

    for f in flights:
        f.move()

    if frame - last_ml_train_frame >= ML_RETRAIN_INTERVAL:
        ml_model = train_ml_model(flights)
        last_ml_train_frame = frame

    for f in flights:
        if f.alive:
            f.color = get_ml_color(f)

    # Clean expired comm lines
    for comm in active_comms[:]:
        comm["frames"] -= 1
        if comm["frames"] <= 0:
            try:
                comm["line"].remove()
            except:
                pass
            active_comms.remove(comm)

    # COMM ATTEMPT (probabilistic)
    if random.random() < COMM_PROB:
        alive_idx = [i for i, f in enumerate(flights) if f.alive]
        if len(alive_idx) >= 2:
            i, j = random.sample(alive_idx, 2)
            f1, f2 = flights[i], flights[j]
            dist = math.hypot(f1.x - f2.x, f1.y - f2.y)

            if dist <= COMM_RANGE:
                # In range → create + sign + verify
                msg = f"POS:{f1.callsign}:{f1.x:.1f},{f1.y:.1f}:{int(time.time())}"
                sig = f1.sign(msg)
                verified = verify_message(f1.public_key, msg.encode("utf-8"), sig)

                # Force some verified messages to fail to simulate tampering
                if verified and random.random() < FORCED_SIG_FAIL_PROB:
                    verified = False  # forced failure

                if verified:
                    # ACCEPTED
                    line_color = "lime"
                    write_log(frame, f1.callsign, f2.callsign, dist, "ACCEPTED", "")
                else:
                    # RSA FAIL (forced or actual)
                    line_color = "red"
                    write_log(frame, f1.callsign, f2.callsign, dist, "FAILED", "Signature Failed")

                # Draw comm line visually (unchanged behavior)
                if len(active_comms) < COMM_MAX_ACTIVE:
                    line_obj, = ax.plot([f1.x, f2.x], [f1.y, f2.y], color=line_color, lw=1.3, alpha=0.8)
                    active_comms.append({"frames": COMM_DURATION_FRAMES, "line": line_obj})

            else:
                # OUT OF RANGE → still log actual distance
                write_log(frame, f1.callsign, f2.callsign, dist, "FAILED", "No Link Established")

    # Radar sweep update
    ang = math.radians((frame * 2) % 360)
    sweep_line.set_data([0, (RANGE + 8) * math.cos(ang)], [0, (RANGE + 8) * math.sin(ang)])

    for idx, f in enumerate(flights):
        if f.alive:
            dots[idx].set_data([f.x], [f.y])
            dots[idx].set_color(f.color)
            labels[idx].set_position((f.x + 2, f.y + 2))
            labels[idx].set_text(f.callsign)
        else:
            dots[idx].set_data([], [])
            labels[idx].set_text("")

    artists = [sweep_line] + dots + labels
    for comm in active_comms:
        artists.append(comm["line"])
    return artists

# ---------------- Run & Save ----------------
print("🚀 Starting radar + enriched CSV logging (v3 with forced signature failures)...")
anim = FuncAnimation(fig, update, frames=FRAME_COUNT, interval=70, blit=True)
video_writer = FFMpegWriter(fps=25)
anim.save(OUTPUT_FILENAME, writer=video_writer, dpi=160)
csv_file.close()
print(f"🎬 Video saved: {OUTPUT_FILENAME}")
print(f"📄 Enriched log saved: {CSV_LOG}")
