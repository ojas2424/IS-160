
# **Fresno County Housing Development Prediction with Deep Learning**

## **Overview**
This project implements Deep Learning (DL) models to predict and analyze housing development trends, land-use patterns, and transportation demands in Fresno County. The dataset includes features like zoning, housing density, and proximity to infrastructure, which are used to train and evaluate three DL models:
1. **Baseline Model**
2. **Weighted Model**
3. **SMOTE Balanced Model**

---

### **Features and Objectives**
The project addresses key urban development questions:
- Predict land-use patterns and successful housing development areas.
- Forecast streets and highways experiencing high demand due to developments.
- Recommend new transportation options, such as bus stops, to maximize accessibility.

The models focus on improving minority class performance while maintaining overall accuracy.

---

### **Datasets**
The project uses the following datasets for Fresno County:

1. **Bus Routes Data (`bus_routes_data.csv`)**  
   - Contains details about bus routes in the region.

2. **Bus Stops Data (`bus_stop_data.csv`)**  
   - Contains coordinates and details of existing bus stops.

3. **City Limits Data (`city_limits_data.csv`)**  
   - Defines the geographic boundaries of Fresno County.

4. **Streets Data (`street_data.csv`)**  
   - Includes information on major streets and highways.

5. **Zoning Data (`zoning_data.csv`)**  
   - Contains zoning classifications for different parcels of land.

6. **Fresno Addresses Data (`fresno_addresses_data.zip`)**  
   - Contains address-level data, including parcel locations and additional features.  
   - **Note**: This file is provided as a zip folder and must be extracted before use.

---

### **Deep Learning Models**
#### **1. Baseline Model**
- A simple feedforward neural network trained on the raw dataset.
- **Objective**: Provide a baseline performance for classification without handling class imbalance.
- **Challenges**: Performance on minority classes is limited due to class imbalance in the dataset.

#### **2. Weighted Model**
- A feedforward neural network with **class weights** to handle class imbalance.
- **Implementation**:
  - Adjusted weights during training to assign more importance to minority classes.
- **Outcome**: Improved recall and F1-score for minority classes compared to the baseline model.

#### **3. SMOTE Balanced Model**
- A feedforward neural network trained on a **balanced dataset** created using **Synthetic Minority Oversampling Technique (SMOTE)**.
- **Implementation**:
  - Oversampled the minority classes in the dataset before training.
- **Outcome**: Achieved the highest performance on minority classes while maintaining good overall accuracy.

---

### **Usage**
The project is implemented in a Jupyter Notebook using Python. Each model is run sequentially by **commenting out the previous model** and uncommenting the desired model. Follow the steps below to reproduce the results:

#### **1. Run the Baseline Model**
Uncomment the Baseline Model code block in the notebook and comment out Weighted and SMOTE models:
```python
# Uncomment this block to run the baseline model
# model = Sequential([
#     Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
#     Dropout(0.2),
#     Dense(64, activation='relu'),
#     Dropout(0.2),
#     Dense(len(label_encoder.classes_), activation='softmax')
# ])
```

#### **2. Run the Weighted Model**
Comment out the Baseline Model and uncomment the Weighted Model block. Add the class weights to the `model.fit()` function:
```python
# Uncomment this block to run the weighted model
# class_weights = {0: 0.5, 1: 5.0, 2: 3.0, ...}  # Example class weights
# model.fit(X_train, y_train, epochs=20, batch_size=32, class_weight=class_weights)
```

#### **3. Run the SMOTE Balanced Model**
Preprocess the dataset using SMOTE, then train the model. Uncomment the SMOTE preprocessing and model block:
```python
# Uncomment this block to run the SMOTE balanced model
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
# model.fit(X_resampled, y_resampled, epochs=20, batch_size=32)
```

---

### **Results**

| **Model**          | **Accuracy** | **Precision (Macro Avg)** | **Recall (Macro Avg)** | **F1-Score (Macro Avg)** |
|---------------------|--------------|---------------------------|-------------------------|---------------------------|
| Baseline Model      | 86%          | 48%                       | 47%                     | 46%                       |
| Weighted Model      | 56%          | 35%                       | 58%                     | 36%                       |
| SMOTE Balanced Model| 97%          | 97%                       | 97%                     | 96%                       |

---

### **Requirements**
- Python 3.10+
- Required libraries:
  - `tensorflow`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `imbalanced-learn`

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

### **Future Work**
1. Extend the project with **Reinforcement Learning** to simulate adaptive transportation placement.
2. Incorporate real population density and traffic data for improved accuracy.

---

### **How to Run**
1. Clone the repository:
```bash
git clone https://github.com/your-repo-url.git
```
2. Open the Jupyter Notebook (`IS_160_Project_DL_model.ipynb`) in Google Colab or locally.
3. Follow the usage instructions to run the desired model.

