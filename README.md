# Customer Churn Prediction using ANN

## Overview
This project leverages **Artificial Neural Networks (ANN)** to predict customer churn in a bank. Using deep learning techniques, the model determines the likelihood of a customer leaving the bank based on various demographic and financial factors.

## Dataset
The dataset used is **Churn_Modelling.csv**,(https://www.kaggle.com/datasets/amirrezaei97/churn-dataset) which consists of **10,000 customer records** with 14 columns:
- **Demographic Features:** `Geography`, `Gender`, `Age`
- **Financial Features:** `CreditScore`, `Balance`, `NumOfProducts`, `EstimatedSalary`
- **Behavioral Features:** `HasCrCard`, `IsActiveMember`, `Tenure`
- **Target Variable:** `Exited` (1 = churned, 0 = retained)

## Technologies Used
- **Python**: Primary programming language
- **TensorFlow & Keras**: For building and training the ANN model
- **NumPy & Pandas**: Data manipulation and preprocessing
- **Jupyter Notebook**: For interactive analysis and model training

## Model Architecture
The ANN model consists of:
1. **Input Layer**: Accepts feature inputs
2. **Hidden Layers**: Fully connected layers with activation functions (ReLU)
3. **Output Layer**: Single neuron with sigmoid activation for binary classification

## Training Process
1. **Data Preprocessing**
   - Removed irrelevant columns (`RowNumber`, `CustomerId`, `Surname`)
   - Split dataset into features (`x`) and target (`y`)
   - Normalized numerical features for optimal ANN performance
2. **Model Compilation**
   - Loss Function: `binary_crossentropy`
   - Optimizer: `Adam`
   - Evaluation Metric: `accuracy`
3. **Model Training**
   - Trained the ANN using a training dataset
   - Validated using test data

## Results & Analysis
- Evaluated model performance using accuracy and loss metrics
- Predictions analyzed using confusion matrix
- The model provides insights into key factors influencing customer churn

## How to Run the Project
1. Install dependencies:
   ```bash
   pip install tensorflow pandas numpy
   ```
2. Run the Jupyter Notebook `Bank_customer_churn_ANN.ipynb`.
3. Observe model predictions and evaluation metrics.

## Future Improvements
- Hyperparameter tuning for better accuracy
- Experimenting with different architectures (LSTM, CNN for tabular data)
- Incorporating additional customer behavior data

## License
This project is open-source under the MIT License.

