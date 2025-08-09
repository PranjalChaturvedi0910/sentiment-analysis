# Sentiment Analysis Project

## Overview

This project performs sentiment analysis on airline-related tweets. It classifies text into positive, negative, or neutral sentiments using a pre-trained machine learning model.

## Project Structure

```
sentiment-analysis/
├── data/                  # Dataset files
├── models/                # Saved ML models and vectorizers
├── notebooks/             # Jupyter notebooks for EDA and experimentation
├── scripts/               # Python scripts for training, evaluation, and prediction
├── README.md               # Project documentation
└── requirements.txt        # Dependencies
```

## Features

* Preprocessing of text data
* Training sentiment analysis model
* Prediction script with neutral word handling
* Jupyter notebook for EDA (Exploratory Data Analysis)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/PranjalChaturvedi0910/sentiment-analysis.git
cd sentiment-analysis
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Run EDA Notebook

Open the `eda.ipynb` in `notebooks/` using Jupyter Notebook or Jupyter Lab to explore the dataset.

### 2. Train Model

```bash
python scripts/train.py
```

### 3. Evaluate Model

```bash
python scripts/evaluate.py
```

### 4. Predict Sentiment

```bash
python scripts/predict.py
```

## Dataset

The dataset used is `Tweets.csv`, containing airline-related tweets with sentiment labels.

## Example Prediction

```bash
Enter text (or 'quit' to exit): The flight was amazing!
Predicted sentiment: positive
```

## Contributing

1. Fork the repo
2. Create a new branch (`git checkout -b feature-name`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Create a Pull Request

## License

This project is licensed under the MIT License.
