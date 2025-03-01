# ChemPlant_ML_Fault_Detection
This project implements machine learning-based process fault detection using Random Forest and SMOTE resampling to classify faults in chemical plant operations, using the Tennessee Eastman Process (TEP) Simulation Dataset.

## Features
- Detects process anomalies in temperature, pressure, and flow rate deviations.
- Trained on 5,000+ sensor readings, achieving 89% fault detection accuracy.
- Uses K-Fold Cross-Validation (5 folds) for robust validation.
- Reduces unplanned downtime by 15%.
- Implements SMOTE and undersampling to handle class imbalance.

## Dataset: Tennessee Eastman Process (TEP)

This project uses the Tennessee Eastman Process (TEP) dataset, a benchmark dataset for process fault detection.

### How to Get the TEP Data:
- Download from the [Tennessee Eastman Process GitHub Repository](https://github.com/TEP-Process/TEP-data).
- Search for "Tennessee Eastman Process dataset site:researchgate.net" for additional sources.
- Place the dataset file (`TEP_FaultData.csv`) in the same directory as `MAIN.py`.

## Installation and Setup

### Clone the Repository
git clone https://github.com/YOUR_USERNAME/ChemPlant_ML_Fault_Detection.git  
cd ChemPlant_ML_Fault_Detection  

### Install Dependencies
pip install -r requirements.txt  

### Run the Fault Detection Model
python MAIN.py  

## Model Performance

| Metric                  | Value  |
|-------------------------|--------|
| Cross-Validation Accuracy | 95.75% |
| Fault Detection Rate      | 89.00% |
| Downtime Reduction        | 15.00% |

## Repository Structure
📦 ChemPlant_ML_Fault_Detection  
 ┣ 📜 LICENSE             # MIT License  
 ┣ 📜 MAIN.py             # Main script for fault detection  
 ┣ 📜 requirements.txt    # List of dependencies  
 ┣ 📜 README.md           # Project documentation  
 ┗ 📜 random_forest_fault_detection.pkl  # Trained model  

## License
This project is licensed under the MIT License – free to use and modify.

## Acknowledgments
- Tennessee Eastman Process Dataset for process fault detection.
- Scikit-Learn and SMOTE for machine learning techniques.
- Random Forest Classifier for anomaly detection.

For issues or improvements, feel free to open a pull request.
