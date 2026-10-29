# SDSS17 Cosmic Tier Classifier

A machine learning pipeline built in Python to classify deep-space observations into Stars, Galaxies, or Quasars (QSOs) using photometric filter data and redshift values.

## Dataset
The dataset used is the **Stellar Classification Dataset - SDSS17** from Kaggle. 
* Note: Due to file size constraints, the raw `star_classification.csv` is omitted from this repository. You can download it directly from [Kaggle](https://kaggle.com).

## Setup Instructions
1. Clone this repository.
2. Download the dataset from Kaggle and place it in the root directory.
3. Install dependencies: `pip install pandas scikit-learn`
4. Run the classifier: `python classifier.py`
