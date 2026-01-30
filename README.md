# Delta-Based Trajectory Gap Filling for Aircraft Flights using Seq2Seq LSTM

This project focuses on **trajectory gap completion for aircraft flights** using a
**delta-based Seq2Seq LSTM model** trained on real-world **ADS-B flight data**.

The goal is to reconstruct missing segments (gaps) in flight trajectories by learning
**physical motion patterns** instead of directly predicting absolute positions.

This problem naturally appears in real aviation data due to:
- sensor dropouts,
- communication losses,
- incomplete surveillance coverage.

---

## Problem Definition

Given a flight trajectory:

Past trajectory  →  [ GAP ]  →  Future trajectory

The task is to predict the missing trajectory segment using:
- past motion context,
- future motion context.

Instead of predicting absolute latitude/longitude values, the model predicts
**delta movements**:

Δlatitude, Δlongitude, Δaltitude

This makes the learning problem more stable and physically meaningful.

---

## Approach

The system is built around a **Seq2Seq (Encoder–Decoder) LSTM architecture**:

- One encoder processes the **past trajectory**
- Another encoder processes the **future trajectory**
- A decoder generates the **missing gap sequence**

Key ideas:
- Time-series modeling with LSTM
- Delta-based prediction
- Sliding windows over trajectories
- Artificial gap injection for supervised training
- Feature normalization with `StandardScaler`

---

## Model Architecture

Input features per timestep:
- latitude  
- longitude  
- geometric altitude  
- velocity  
- vertical rate  
- heading (sin, cos)  
- on-ground flag  

Output:
- Δlatitude  
- Δlongitude  
- Δaltitude  

The model learns how aircraft move in real physical space and reconstructs
missing segments accordingly.

---

## Dataset & Reproducibility

This project uses real-world ADS-B flight trajectory data obtained from the  
**OpenSky Network**: https://opensky-network.org/

Due to data size and licensing restrictions, raw datasets are **not included**
in this repository. Users must generate the dataset locally.

---

### Step 1 – OpenSky Account

To access historical ADS-B data, you need to create a free account:

https://opensky-network.org/apidoc/index.html

(Some historical queries may require approval.)

---

### Step 2 – Download Flight Data

Use the SQL query provided in:

sql/query.sql

Run this query on the OpenSky Trino interface or historical database.

The query extracts flight trajectories between:
- **LTFM (Istanbul Airport)**
- **LFPG (Paris Charles de Gaulle)**

over a 1-year period.

Export the result as a CSV file.

---

### Step 3 – Place Raw Data

Create a local `data/` directory and place the CSV file:

data/flights_raw.csv

(This directory is ignored by Git and not part of the repository.)

---

### Step 4 – Preprocessing & Dataset Construction

Run:

python src/build_gap_dataset.py

This script will:
- clean invalid samples  
- filter on-ground states  
- sort by flight and timestamp  
- apply sliding windows  
- inject artificial gaps  
- normalize features  
- generate training tensors  

---

### Step 5 – Train the Model

python src/train_gapfill_model.py

---

### Step 6 – Evaluate

python src/evaluate_gapfill_model.py

---

## Project Structure

.
├── src/                 # Core model and pipeline  
├── sql/                 # OpenSky query  
├── data/                # Local dataset (not tracked)  
├── requirements.txt  
└── README.md  

---

## Technologies Used

- Python  
- PyTorch  
- NumPy / Pandas  
- Scikit-learn  
- LSTM / Seq2Seq models  

---

## Why This Project Matters

This project demonstrates:

- real-world time-series modeling  
- sequence-to-sequence learning  
- data engineering with large datasets  
- reproducible machine learning pipelines  
- physically meaningful prediction strategies  

It is not a toy dataset or a Kaggle notebook, but a **full end-to-end ML system**
built on real aviation data.

---

## Notes

- Raw dataset size: ~1–3 GB  
- Processed training tensors: ~200–500 MB  
- Trained models are generated locally and not versioned.  

This design follows standard machine learning best practices:  
**code is shared, data is reproducible, models are regenerable.**

---

## Disclaimer

This project uses publicly available ADS-B data for **research and educational purposes only**.  
No personally identifiable information is stored or processed.
