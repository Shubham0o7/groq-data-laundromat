# 🤖 Groq & Roll Data Laundromat

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

## ⚙️ Installation and Local Setup

1. Clone the Repository
git clone https://github.com/shubham0o7/groq-data-laundromat.git
cd groq-data-laundromat

2. Set Up a Virtual Environment (Recommended)
python -m venv env
# On Windows:
env\Scripts\activate
# On macOS/Linux:
source env/bin/activate

3. Install Required Dependencies
pip install -r requirements.txt

4. Configure Your Private API Environment
Create a file named exactly .env in the root directory of the project and add your unique Groq API credential string:
GROQ_API_KEY=your_actual_groq_api_key_here

5. Launch the Application
streamlit run app.py
