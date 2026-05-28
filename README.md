
# 🤖 Groq & Roll Data Laundromat: LLM-Powered Spreadsheet Washer

A high-performance Streamlit web application that utilizes advanced Large Language Models (LLMs) to automatically ingest, clean, analyze, and query large datasets.

By leveraging the speed of **Llama 3.1 8B** via the Groq API and coupling it with optimized background metadata preprocessing, this application can seamlessly handle datasets scaling up to thousands of rows without encountering token exhaustion or latency bottlenecks.

---

## 🚀 Core Features

* **Intelligent Data Scrubber (Step 1):** Automatically identifies and cleans structural anomalies, fills missing data values using contextual statistical safeties, drops clone duplicates, and fixes tracking data types via a reinforced autonomous agent loop.
* **The Spin Cycle & Insight Press (Step 2):** Generates a high-density executive business report complete with dynamic local data visualizations (Matplotlib/Seaborn) embedded directly into the narrative stream.
* **Interactive Spot Cleaner (Step 3):** An advanced contextual chatbot container that allows users to query their data records natively using a high-density pivot matrix background engine.

---

## 🛠️ Architecture & High-Scale Optimization

Unlike traditional data-to-text architectures that feed massive raw CSV rows straight into an LLM context window—which crashes from token inflation—this app utilizes a **Metadata & Summary-Driven Aggregation** pattern.

Heavy arithmetic, sorting, multi-dimensional grouping, and pivot profile compilations are calculated instantly using native Python background threads. The LLM receives pre-digested statistical truths, keeping token consumption perfectly flat and lightning-fast whether processing 20 rows or 50,000 rows.

---

## 📋 Repository Directory Structure

```text
groq-data-laundromat/
│
├── .gitignore                 # Prevents committing local secrets (.env)
├── README.md                  # Project documentation
├── app.py                     # Main production-ready Streamlit application
└── requirements.txt           # Environment package dependencies

```

---

## ⚙️ Installation and Local Setup

If you want to clone this repository and run the engine locally on your machine, you must **bring your own Groq API Key**. Your private keys must never be committed to source control.

### 1. Clone the Repository

```bash
git clone https://github.com/shubham0o7/groq-data-laundromat.git
cd groq-data-laundromat

```

### 2. Install Required Dependencies

```bash
pip install -r requirements.txt

```

### 3. Configure Your Private API Environment

Create a file named exactly `.env` in the root directory of the project and add your unique Groq API credential string:

```text
GROQ_API_KEY=your_actual_groq_api_key_here

```

*(Note: The included `.gitignore` file is pre-configured to ensure this `.env` file is never pushed to public servers).*

### 4. Launch the Application

```bash
streamlit run app.py

```

---

## 🌐 Production Cloud Deployment Notes

When deploying this application to **Streamlit Community Cloud**:

1. Do **not** upload your `.env` file to GitHub.
2. Navigate to your App Dashboard, click **Advanced Settings -> Secrets**, and insert your secure environment payload manually:
```toml
GROQ_API_KEY = "gsk_your_private_key_here"

```



```
3. To protect infrastructure limits and quota footprints after a live presentation or evaluation window concludes, the deployment administrator can seamlessly revoke the key active string directly inside the **Groq Console** or delete the container allocation from the Streamlit deployment manager dashboard.

---

## 🧰 Tech Stack
* **Frontend UI:** Streamlit Cloud Layer
* **Inference Engine:** Groq Cloud API (`llama-3.1-8b-instant`)
* **Orchestration:** LangChain Core Components
* **Data Processing:** Pandas, NumPy & Tabulate
* **Visualizations:** Matplotlib & Seaborn

```
