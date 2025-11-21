# Secure-A2A-Communication
This project simulates a secure Air Traffic Control (ATC) radar that tracks commercial flights, performs RSA-based secure communication, detects anomalies using Machine Learning (Isolation Forest), and dynamically visualizes critical aircraft system failures in real-time.

## 🚀 Key Features

🔐 Secure A2A Communication (ADS-B-Style Messaging)
  1. Flights within communication range exchange encrypted position messages.
  2. Each message is digitally signed using RSA-2048.
  3. Signature verification simulated using PKCS1 v1.5 + SHA-256.
  4. Communication failures are visually highlighted in red on radar.

🧠 ML-Powered Flight Anomaly Detection
  1. Uses Isolation Forest to analyze flight movement in real time.
  2. Flights change color based on detected behavior:
    Normal	        🟢 Lime
    Mild anomaly    🟡 Yellow
    Severe anomaly	🔴 Red
---

## 🧠 Tech Stack

| Domain           | Tools Used                                                 |
| ---------------- | ---------------------------------------------------------- |
| Language         | Python (Matplotlib, NumPy, Dataclasses)                    |
| Cryptography     | RSA-2048, SHA-256 (cryptography library)                   |
| Machine Learning | Isolation Forest (Scikit-Learn)                            |
| Visualization    | Matplotlib                                                 |
| Logging          | CSV event recorder                                         |

## 📁 Project Structure
📁 Secure-Airborne-ATC-Simulation/
│
├── 📂 myenv/                      # Local virtual environment (should be ignored on GitHub)
├── 📂 venv/                       # Another virtual environment (should be ignored)
│
├── 🎥 airborne_comms_unified_with_faults_fixed.mp4
├── 🎥 airborne_comms_unified_with_faults_random5.mp4
│        # Radar animation videos with secure comms & faults
│
├── 🐍 airborne_comms_with_enriched_csv_v3.py
│        # Python source (generates video + CSV logs)
│
├── 📄 comms_enriched_log_fixed.csv
├── 📄 comms_enriched_log_v5_random5.csv
