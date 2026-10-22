<!-- ============================  HEADER  ============================ -->
<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,50:203a43,100:2c5364&height=220&section=header&text=Intelligent%20IoT%20Anomaly%20Detection&fontSize=40&fontColor=ffffff&animation=fadeIn&fontAlignY=36&desc=Autoencoder%20%E2%80%A2%20Federated%20Learning%20%E2%80%A2%20AES-256%20Security&descAlignY=56&descSize=18" width="100%" alt="header"/>

<a href="#">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=36BCF7&center=true&vCenter=true&multiline=false&width=720&lines=Detecting+machine+failures+before+they+happen+%E2%9A%99%EF%B8%8F;Deep+Autoencoder+trained+on+5+IIoT+sensor+streams+%F0%9F%A7%A0;Privacy-preserving+Federated+Learning+across+factories+%F0%9F%8F%AD;AES-256+encrypted+sensor+pipeline+%F0%9F%94%90;Real-time+Streamlit+monitoring+dashboard+%F0%9F%93%A1" alt="Typing SVG"/>
</a>

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Flower](https://img.shields.io/badge/Flower-Federated_Learning-F2B705?style=for-the-badge&logo=flower&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
<br/>
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![AES-256](https://img.shields.io/badge/Security-AES--256--CBC-2EA44F?style=for-the-badge&logo=letsencrypt&logoColor=white)
![SHAP](https://img.shields.io/badge/XAI-SHAP-8A2BE2?style=for-the-badge)

</div>

---

## 📖 Overview

**Intelligent IoT Anomaly Detection** is an end-to-end Industrial IoT (IIoT) research platform that detects abnormal machine behaviour from live sensor readings. A **deep autoencoder** learns what "normal" looks like; anything it cannot reconstruct well is flagged as an **anomaly**.

The system goes beyond a single model:

- 🏭 **Federated Learning** — multiple factories collaboratively train one global model **without sharing raw data** (Flower + FedAvg).
- 🔐 **AES-256 encryption** — every sensor payload is encrypted/decrypted in the inference pipeline to simulate secure IIoT transport.
- 🧠 **Explainability** — SHAP highlights which sensor drove a given prediction.
- 📡 **Interactive dashboard** — a Streamlit control room with live predictions, trend charts, history, analytics and model diagnostics.

Built on the **[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)** (10,000 samples).

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## ✨ Key Features

<table>
<tr>
<td width="50%" valign="top">

### 🧠 Deep Autoencoder
Symmetric `5 → 32 → 16 → 8 → 16 → 32 → 5` network trained to reconstruct healthy sensor states. Reconstruction MSE above the **99th-percentile threshold** ⇒ anomaly.

</td>
<td width="50%" valign="top">

### 🏭 Federated Learning
Dataset split across **Factory A / B / C**. Each client trains locally; a Flower server aggregates weights with **FedAvg** over 5 rounds.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 🔐 AES-256 Data Security
Sensor readings are serialised, encrypted with **AES-256-CBC** (random IV per message), then decrypted before inference.

</td>
<td width="50%" valign="top">

### 📊 Real-time Dashboard
Gauges, live trend charts, machine-health score, prediction history with CSV export, analytics and model-performance graphs.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 🔎 Explainable AI
`src/explain.py` uses **SHAP KernelExplainer** to rank each sensor's contribution to the anomaly score.

</td>
<td width="50%" valign="top">

### 📝 IEEE-style Evaluation
Accuracy, Precision, Recall, F1, confusion matrix and full classification report on a stratified hold-out set.

</td>
</tr>
</table>

## 🏗️ System Architecture

```mermaid
flowchart LR
    subgraph Edge["🏭 IIoT Edge"]
        S1[🌡️ Air Temp]
        S2[🔥 Process Temp]
        S3[⚙️ RPM]
        S4[🔩 Torque]
        S5[🛠️ Tool Wear]
    end

    Edge --> ENC[🔐 AES-256-CBC<br/>Encrypt]
    ENC --> DEC[🔓 Decrypt]
    DEC --> SC[📏 StandardScaler]
    SC --> AE[🧠 Autoencoder]
    AE --> ERR[📉 Reconstruction<br/>Error MSE]
    ERR --> TH{Error ><br/>Threshold?}
    TH -- Yes --> AN[🚨 Anomaly]
    TH -- No --> NO[✅ Normal]
    AN --> DB[📡 Streamlit Dashboard]
    NO --> DB
    DB --> LOG[(📜 Prediction<br/>History CSV)]
```

### 🌐 Federated Learning Topology

```mermaid
flowchart TB
    SRV["🖥️ Flower Server<br/>FedAvg · 5 rounds<br/>127.0.0.1:8080"]
    A["🏭 Factory A<br/>3,333 samples"]
    B["🏭 Factory B<br/>3,333 samples"]
    C["🏭 Factory C<br/>3,334 samples"]
    A <-- "local weights ⇄ global weights" --> SRV
    B <-- "local weights ⇄ global weights" --> SRV
    C <-- "local weights ⇄ global weights" --> SRV
    SRV --> G[("💾 federated_autoencoder.keras")]
```

> 🔒 Raw sensor data **never leaves** a factory — only model weights are exchanged.

### 🧬 Autoencoder Design

```mermaid
flowchart LR
    I["Input<br/>5"] --> E1["Dense 32<br/>ReLU"] --> E2["Dense 16<br/>ReLU"] --> Z(("Latent<br/>8")) --> D1["Dense 16<br/>ReLU"] --> D2["Dense 32<br/>ReLU"] --> O["Output<br/>5"]
    style Z fill:#2c5364,color:#fff,stroke:#36BCF7,stroke-width:3px
```

| Setting | Value |
|---|---|
| Loss | Mean Squared Error |
| Optimizer | Adam |
| Batch size | 256 (centralised) · 32 (federated) |
| Anomaly threshold | 99th percentile of reconstruction error |
| Split | 80 / 20 stratified on `Machine failure` |

## 📈 Results

<div align="center">

| Training Loss | Reconstruction Error Histogram |
|:---:|:---:|
| <img src="results/training_loss.png" width="420"/> | <img src="results/reconstruction_error_histogram.png" width="420"/> |

| Reconstruction Error over Samples |
|:---:|
| <img src="results/reconstruction_error_line.png" width="860"/> |

</div>

<details>
<summary><b>📂 Feature distributions (click to expand)</b></summary>
<br/>

| Air Temperature | Process Temperature | Rotational Speed |
|:---:|:---:|:---:|
| <img src="results/Air_Temp.png" width="280"/> | <img src="results/Process_Temp.png" width="280"/> | <img src="results/RPM.png" width="280"/> |

| Torque | Tool Wear |
|:---:|:---:|
| <img src="results/Torque.png" width="280"/> | <img src="results/Tool_Wear.png" width="280"/> |

</details>

## 📁 Project Structure

```text
intelligent-iot-anomaly-detection-v2/
├── 📂 dashboard/
│   └── app.py                    # Streamlit monitoring dashboard
├── 📂 data/
│   ├── ai4i2020.csv              # AI4I 2020 predictive maintenance dataset
│   └── factory_{A,B,C}.csv       # Federated client partitions
├── 📂 models/
│   ├── autoencoder.keras         # Trained autoencoder
│   ├── scaler.pkl                # Fitted StandardScaler
│   ├── threshold.pkl             # Anomaly threshold
│   └── X_test.pkl / y_test.pkl   # Hold-out evaluation set
├── 📂 results/                   # Training & evaluation plots
├── 📂 src/
│   ├── preprocess.py             # Load, split & scale data
│   ├── train.py / train_ieee.py  # Autoencoder training
│   ├── evaluate.py               # Reconstruction-error evaluation
│   ├── evaluate_ieee.py          # Accuracy / Precision / Recall / F1
│   ├── predict.py                # Encrypted inference pipeline
│   ├── encryption.py             # AES-256-CBC helpers
│   ├── explain.py                # SHAP explainability
│   ├── graphs.py                 # Result plot generation
│   ├── history.py                # Prediction logging
│   ├── split_dataset.py          # Factory A/B/C split
│   └── 📂 federated/
│       ├── server.py             # Flower server (FedAvg)
│       ├── client.py             # Flower factory client
│       ├── strategy.py           # Aggregation strategy
│       ├── model.py              # Shared model definition
│       └── save_global_model.py  # Persist global weights
├── test_model.py                 # Inference smoke test
└── requirements.txt
```

## 🚀 Getting Started

### 1️⃣ Clone & install

```bash
git clone https://github.com/rahulmj2004/intelligent-iot-anomaly-detection-v2.git
cd intelligent-iot-anomaly-detection-v2

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install flwr matplotlib shap  # federated learning, plots & XAI extras
```

### 2️⃣ Launch the dashboard

Pre-trained artefacts are included in `models/`, so you can start right away:

```bash
streamlit run dashboard/app.py
```

### 3️⃣ Train & evaluate (optional)

```bash
python src/train_ieee.py        # train autoencoder + compute threshold
python src/evaluate_ieee.py     # accuracy, precision, recall, F1
python src/graphs.py            # regenerate result plots
python test_model.py            # quick inference smoke test
```

### 4️⃣ Run Federated Learning

```bash
python src/split_dataset.py                       # create factory_A/B/C.csv

# Terminal 1 — server
python src/federated/server.py

# Terminals 2, 3, 4 — one client per factory
python src/federated/client.py --factory factory_A
python src/federated/client.py --factory factory_B
python src/federated/client.py --factory factory_C
```

## 🖥️ Dashboard Tour

| Tab | What it shows |
|---|---|
| 🔍 **Live Prediction** | Enter 5 sensor values → anomaly verdict, reconstruction error, risk gauge, encrypted / IV / decrypted payload |
| 📡 **Live Dashboard** | Rolling sensor trends and machine-health gauge |
| 📜 **History** | Every prediction, timestamped, with CSV download |
| 📊 **Analytics** | Normal vs. anomaly breakdown and error statistics |
| 🧠 **Model Performance** | Training loss & reconstruction-error graphs |
| 📂 **Feature Distribution** | Per-sensor distribution plots |

## 📊 Dataset

| Feature | Unit | Description |
|---|---|---|
| Air temperature | K | Ambient air temperature |
| Process temperature | K | Machine process temperature |
| Rotational speed | rpm | Spindle speed |
| Torque | Nm | Applied torque |
| Tool wear | min | Cumulative tool usage |
| **Machine failure** | 0 / 1 | Ground-truth label (used only for evaluation) |

## 🛡️ Security Note

`src/encryption.py` ships with a **hard-coded demo key** for reproducibility. For any real deployment, load the AES key from an environment variable or a secrets manager and never commit it to source control.

## 🗺️ Roadmap

- [x] Deep autoencoder anomaly detector
- [x] AES-256 encrypted inference pipeline
- [x] Federated learning across 3 simulated factories
- [x] Streamlit monitoring dashboard
- [x] SHAP explainability module
- [ ] MQTT ingestion from real IoT devices
- [ ] Secure aggregation & differential privacy for FL
- [ ] Docker deployment

## 🤝 Contributing

Contributions, issues and feature requests are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m "Add amazing feature"`
4. Push the branch and open a Pull Request

## 👤 Author

<div align="center">

**Rahul M J**

[![GitHub](https://img.shields.io/badge/GitHub-rahulmj2004-181717?style=for-the-badge&logo=github)](https://github.com/rahulmj2004)

⭐ **If you found this project useful, consider giving it a star!** ⭐

</div>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2c5364,50:203a43,100:0f2027&height=120&section=footer" width="100%" alt="footer"/>
