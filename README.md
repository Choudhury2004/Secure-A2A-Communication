# Secure-A2A-Communication
A **Linux-based radar simulation system** that visualizes secure **air-to-air (A2A)** communication between aircraft with **cryptographic authentication** and **AI-driven threat detection**.  
The project simulates both **verified and unverified** communications between aircraft, color-coded on a radar interface, and logs all interactions to a CSV file.

## 🚀 Features

- **Real-Time Radar Visualization**  
  - Simulates 30 aircraft flying across a radar screen with smooth motion.  
  - Flights appear dynamically over time (not all at once).  

- **Secure Communication (Crypto Layer)**  
  - Messages between flights are authenticated using **RSA digital signatures**.  
  - Verified (green) and unverified (red) communication lines are color-coded.  

- **AI-Based Threat Detection**  
  - Uses **IsolationForest (scikit-learn)** to detect suspicious or spoofed flights.  
  - Radar colors dynamically reflect aircraft security status.  

- **Comprehensive CSV Logging**  
  - Logs both verified and unverified communications with timestamps, flight IDs, distance, and status.  

- **Video Export**  
  - Generates an **MP4 radar simulation** using **FFmpeg** for presentation or research demonstrations.

---

## 🧠 Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| Language | Python 3.12 |
| OS / Platform | Linux |
| Visualization | Matplotlib |
| Data Processing | NumPy, Pandas |
| AI / Anomaly Detection | Scikit-learn (IsolationForest) |

## 📁 Project Structure
airborne-comms-sim/
│
├── airborne_comms_unified.py # Main simulation script
├── comms_log.csv # Generated CSV log (output)
├── radar_simulation.mp4 # Video output (generated)
├── venv/ # Python virtual environment
└── README.md # Project documentation
| Security | Cryptography (RSA) |
| Video Encoding | FFmpeg |
| Environment | venv (Virtual Environment) |
