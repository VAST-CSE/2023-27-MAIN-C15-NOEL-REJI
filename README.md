<!-- ================================  HERO  ================================ -->
<div align="center">

<img src="assets/hero.svg" width="100%" alt="Intelligent IoT Anomaly Detection — animated sensor waveform with anomaly spike"/>

<a href="#-getting-started">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&duration=3000&pause=800&color=36BCF7&center=true&vCenter=true&width=760&lines=Detecting+machine+failures+before+they+happen+%E2%9A%99%EF%B8%8F;Deep+Autoencoder+on+5+IIoT+sensor+streams+%F0%9F%A7%A0;Privacy-preserving+Federated+Learning+%F0%9F%8F%AD%F0%9F%8F%AD%F0%9F%8F%AD;AES-256+encrypted+sensor+pipeline+%F0%9F%94%90;Explainable+AI+with+SHAP+%F0%9F%94%8E;Real-time+Streamlit+control+room+%F0%9F%93%A1" alt="Typing SVG"/>
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
<br/>
![Stars](https://img.shields.io/github/stars/rahulmj2004/intelligent-iot-anomaly-detection-v2?style=social)
![Forks](https://img.shields.io/github/forks/rahulmj2004/intelligent-iot-anomaly-detection-v2?style=social)
![Last commit](https://img.shields.io/github/last-commit/rahulmj2004/intelligent-iot-anomaly-detection-v2?style=flat-square&color=36BCF7)
![Repo size](https://img.shields.io/github/repo-size/rahulmj2004/intelligent-iot-anomaly-detection-v2?style=flat-square&color=2EE6A6)

<p>
  <a href="#-overview">Overview</a> •
  <a href="#-key-features">Features</a> •
  <a href="#%EF%B8%8F-system-architecture">Architecture</a> •
  <a href="#-results">Results</a> •
  <a href="#-getting-started">Getting Started</a> •
  <a href="#-authors">Authors</a>
</p>

</div>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Smilies/Robot.png" width="34" align="center"/> Overview

**Intelligent IoT Anomaly Detection** is an end-to-end Industrial IoT (IIoT) research platform that spots abnormal machine behaviour from live sensor readings. A **deep autoencoder** learns what *healthy* looks like. When it can't reconstruct a reading well, that reading is flagged as an **anomaly**.

<table>
<tr>
<td>

- 🏭 **Federated Learning**: factories train one global model together **without sharing raw data** (Flower + FedAvg).
- 🔐 **AES-256 encryption**: every sensor payload goes through an encrypt → decrypt step to simulate secure IIoT transport.
- 🔎 **Explainability**: SHAP shows which sensor drove a prediction.
- 📡 **Interactive dashboard**: a Streamlit control room with live predictions, trends, history and diagnostics.

</td>
</tr>
</table>

Built on the **[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)** (10,000 samples · 5 sensors).

## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/High%20Voltage.png" width="34" align="center"/> Key Features

<table>
<tr>
<td width="50%" valign="top" align="center">
<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Smilies/Robot.png" width="56"/>

### Deep Autoencoder
Symmetric `5 → 32 → 16 → 8 → 16 → 32 → 5` network trained to reconstruct healthy sensor states. A reconstruction MSE above the **99th-percentile threshold** means an anomaly.

</td>
<td width="50%" valign="top" align="center">
<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Factory.png" width="56"/>

### Federated Learning
The dataset is split across **Factory A / B / C**. Each client trains locally, and a Flower server aggregates the weights with **FedAvg** over 5 rounds.

</td>
</tr>
<tr>
<td width="50%" valign="top" align="center">
<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Locked.png" width="56"/>

### AES-256 Data Security
Sensor readings are serialised and encrypted with **AES-256-CBC** (a fresh random IV per message), then decrypted before inference.

</td>
<td width="50%" valign="top" align="center">
<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Bar%20Chart.png" width="56"/>

### Real-time Dashboard
Gauges, live trend charts, a machine-health score, prediction history with CSV export, analytics and model-performance graphs.

</td>
</tr>
<tr>
<td width="50%" valign="top" align="center">
<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Magnifying%20Glass%20Tilted%20Left.png" width="56"/>

### Explainable AI
`src/explain.py` uses **SHAP KernelExplainer** to rank how much each sensor contributes to the anomaly score.

</td>
<td width="50%" valign="top" align="center">
<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Chart%20Increasing.png" width="56"/>

### IEEE-style Evaluation
Accuracy, Precision, Recall, F1, a confusion matrix and a full classification report on a stratified hold-out set.

</td>
</tr>
</table>

## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Gear.png" width="34" align="center"/> System Architecture

### 🔄 Inference Pipeline

<div align="center">
<img src="assets/pipeline.svg" width="100%" alt="Animated inference pipeline: sensors → AES encryption → decrypt & scale → autoencoder → reconstruction error → dashboard"/>
</div>

### 🌐 Federated Learning Topology

<div align="center">
<img src="assets/federated.svg" width="85%" alt="Animated federated learning: three factories exchange weights with a global FedAvg server"/>
</div>

> 🔒 Raw sensor data **never leaves** a factory. Only model weights are exchanged.

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

<details>
<summary><b>📐 Detailed data-flow diagram (click to expand)</b></summary>
<br/>

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

</details>

## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Laptop.png" width="34" align="center"/> Tech Stack

<div align="center">

<img src="https://skillicons.dev/icons?i=py,tensorflow,sklearn,github,vscode&theme=dark" alt="tech stack"/>

| Layer | Tools |
|---|---|
| 🧠 Modelling | TensorFlow / Keras · scikit-learn · NumPy · pandas |
| 🏭 Federated | Flower (`flwr`) · FedAvg |
| 🔐 Security | PyCryptodome (AES-256-CBC) |
| 🔎 Explainability | SHAP |
| 📡 Visualisation | Streamlit · Plotly · Matplotlib |

</div>

## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Chart%20Increasing.png" width="34" align="center"/> Results

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

## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Card%20Index%20Dividers.png" width="34" align="center"/> Project Structure

```text
intelligent-iot-anomaly-detection-v2/
├── 📂 assets/                    # Animated README graphics
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

## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Rocket.png" width="34" align="center"/> Getting Started

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

The pre-trained model files are already in `models/`, so you can start right away:

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

# Terminal 1: server
python src/federated/server.py

# Terminals 2, 3, 4: one client per factory
python src/federated/client.py --factory factory_A
python src/federated/client.py --factory factory_B
python src/federated/client.py --factory factory_C
```

## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Satellite%20Antenna.png" width="34" align="center"/> Dashboard Tour

| Tab | What it shows |
|---|---|
| 🔍 **Live Prediction** | Enter 5 sensor values to get an anomaly verdict, reconstruction error, risk gauge, and the encrypted / IV / decrypted payload |
| 📡 **Live Dashboard** | Rolling sensor trends and a machine-health gauge |
| 📜 **History** | Every prediction, timestamped, with CSV download |
| 📊 **Analytics** | Normal vs. anomaly breakdown and error statistics |
| 🧠 **Model Performance** | Training loss & reconstruction-error graphs |
| 📂 **Feature Distribution** | Per-sensor distribution plots |

## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Books.png" width="34" align="center"/> Dataset

| Feature | Unit | Description |
|---|---|---|
| 🌡️ Air temperature | K | Ambient air temperature |
| 🔥 Process temperature | K | Machine process temperature |
| ⚙️ Rotational speed | rpm | Spindle speed |
| 🔩 Torque | Nm | Applied torque |
| 🛠️ Tool wear | min | Cumulative tool usage |
| 🎯 **Machine failure** | 0 / 1 | Ground-truth label (used only for evaluation) |

## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Shield.png" width="34" align="center"/> Security Note

> [!WARNING]
> `src/encryption.py` ships with a **hard-coded demo key** so results are reproducible. For any real deployment, load the AES key from an environment variable or a secrets manager, and never commit it to source control.

## 🗺️ Roadmap

- [x] Deep autoencoder anomaly detector
- [x] AES-256 encrypted inference pipeline
- [x] Federated learning across 3 simulated factories
- [x] Streamlit monitoring dashboard
- [x] SHAP explainability module
- [ ] MQTT ingestion from real IoT devices
- [ ] Secure aggregation & differential privacy for FL
- [ ] Docker deployment

## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Hand%20gestures/Handshake.png" width="34" align="center"/> Contributing

Contributions, issues and feature requests are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m "Add amazing feature"`
4. Push the branch and open a Pull Request

## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/People/Busts%20in%20Silhouette.png" width="34" align="center"/> Authors

<div align="center">

<table>
<tr>
<td align="center" width="200">
<a href="https://github.com/rahulmj2004"><img src="https://github.com/rahulmj2004.png?size=120" width="100" style="border-radius:50%" alt="Rahul M J"/></a><br/>
<b>Rahul M J</b><br/><br/>
<a href="https://github.com/rahulmj2004"><img src="https://img.shields.io/badge/GitHub-rahulmj2004-181717?style=flat-square&logo=github"/></a>
</td>
<td align="center" width="200">
<a href="https://github.com/Noel-Reji"><img src="https://github.com/Noel-Reji.png?size=120" width="100" style="border-radius:50%" alt="Noel Reji"/></a><br/>
<b>Noel Reji</b><br/><br/>
<a href="https://github.com/Noel-Reji"><img src="https://img.shields.io/badge/GitHub-Noel--Reji-181717?style=flat-square&logo=github"/></a>
</td>
<td align="center" width="200">
<a href="https://github.com/Rasinjohn4444"><img src="https://github.com/Rasinjohn4444.png?size=120" width="100" style="border-radius:50%" alt="Rasin John"/></a><br/>
<b>Rasin John</b><br/><br/>
<a href="https://github.com/Rasinjohn4444"><img src="https://img.shields.io/badge/GitHub-Rasinjohn4444-181717?style=flat-square&logo=github"/></a>
</td>
</tr>
</table>

Repository transfer to [VAST-CSE](https://github.com/VAST-CSE) was carried out by [Noel Reji](https://github.com/Noel-Reji).

<br/>

<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Glowing%20Star.png" width="28" align="center"/> **If you found this project useful, consider giving it a star!** <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Glowing%20Star.png" width="28" align="center"/>

</div>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2c5364,50:203a43,100:0f2027&height=120&section=footer" width="100%" alt="footer"/>
