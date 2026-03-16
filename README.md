# AI Smart Queue Management System

This project is an **AI-powered hospital queue management system** that predicts:

* **Patient No-Show Probability**
* **Expected Waiting Time**
* **Queue Position and Token Generation**

The system combines **Machine Learning models, backend APIs, and a frontend dashboard** to help hospitals manage patient queues efficiently.

---

# Project Requirements

* **Python Version:** 3.11.9
* **Virtual Environment:** `myenv`

---

# Project Setup

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <project-folder>
```

---

## 2. Create Virtual Environment

Create a virtual environment named **myenv**.

```bash
python -m venv myenv
```

Activate the environment:

### Windows

```bash
myenv\Scripts\activate
```

### Mac/Linux

```bash
source myenv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Database Setup

If the database has **not been created**, run the following script:

```bash
python create_db.py
```

This will create the required database and tables.

---

# Machine Learning Models

The system contains two ML models:

1. **No-Show Prediction Model**
2. **Waiting Time Prediction Model**

If the models are not already generated, run the preprocessing and training scripts.

### No-Show Model

```bash
python no_show_preprocessing.py
python no_show_train.py
```

### Wait Time Model

```bash
python wait_time_preprocessing.py
python wait_time_train.py
```

These scripts will:

* preprocess the dataset
* train the models
* save the trained models for backend use

---

# Running the Backend

Start the backend server using:

```bash
python main.py
```

The backend will:

* load trained ML models
* expose prediction APIs
* handle queue logic
* communicate with the frontend dashboard

---

# Frontend Output

The frontend dashboard displays:

* Token Number
* Members Before You
* Queue Position
* Estimated Waiting Time
* No-Show Probability
* SHAP based explanation of predictions

---

# System Workflow

1. Patient enters queue
2. Backend generates **token**
3. ML models predict:

   * no-show probability
   * waiting time
4. Backend sends predictions to frontend
5. Dashboard displays queue analytics and patient details

---

# Project Structure (Example)

```
project/
│
├── main.py
├── create_db.py
├── requirements.txt
│
├── models/
│   ├── no_show_model.pkl
│   └── wait_time_model.pkl
│
├── preprocessing/
│   ├── no_show_preprocessing.py
│   └── wait_time_preprocessing.py
│
├── frontend/
│   └── index.html
│
└── README.md
```

