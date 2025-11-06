# 🧱 LEGO Classifier

A Streamlit web application for classifying LEGO pieces using deep learning.

## Setup

1. Clone the repository
2. Create virtual environment:

```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
   pip install -r requirements.txt
```

4. Add your model file to `models/best_broad_classifier.pt`
5. Add your class names to `data/class_names.json`

## Run

```bash
streamlit run main.py
```

## Project Structure

```
lego-classifier-app/
├── main.py           # Entry point
├── src/              # Source code
│   ├── app.py        # Streamlit app logic
│   ├── model.py      # Model architecture
│   ├── utils.py      # Helper functions
│   └── config.py     # Configuration
├── models/           # Trained models
├── data/             # Data files
└── requirements.txt  # Dependencies
```

## Development

Run the app in development mode:

```bash
streamlit run main.py --server.runOnSave true
```
