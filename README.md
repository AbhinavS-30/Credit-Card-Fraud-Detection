# Credit Card Fraud Detection

## Project Overview
This project analyzes credit card transactions to detect fraudulent activities. The dataset used is `creditcard.csv`, which contains transaction records labeled as fraudulent or non-fraudulent.

## Dataset
The dataset consists of anonymized transaction features along with a `Class` column:
- `Class = 0`: Non-fraudulent transactions
- `Class = 1`: Fraudulent transactions

## Methodology
1. **Data Loading**: The dataset is loaded using pandas.
2. **Exploratory Data Analysis (EDA)**:
   - Class distribution is examined to understand the imbalance.
   - Histograms are plotted to visualize feature distributions.
3. **Preprocessing**:
   - Handling class imbalance using techniques such as oversampling or undersampling.
   - Feature scaling and transformation.
4. **Model Training**:
   - A machine learning model is trained to classify transactions as fraudulent or non-fraudulent.
   - Common algorithms: Logistic Regression, Decision Trees, Random Forest, or Neural Networks.
5. **Evaluation**:
   - Metrics such as accuracy, precision, recall, and F1-score are used to assess model performance.

## Results
### Imbalanced Dataset
| Model                           | Precision | Recall | F1-score |
|---------------------------------|-----------|--------|----------|
| Logistic Regression on val      | 0.73      | 0.53   | 0.61     |
| Shallow Neural Net on val       | 0.70      | 0.72   | 0.71     |
| Random Forest Classifier        | 0.80      | 0.44   | 0.57     |
| Gradient Boosting Classifier    | 0.67      | 0.67   | 0.67     |
| Linear SVC on val               | 0.71      | 0.75   | 0.73     |

### Balanced Dataset
| Model                           | Precision | Recall | F1-score |
|---------------------------------|-----------|--------|----------|
| Logistic Regression on val      | 0.93      | 0.96   | 0.94     |
| Shallow Neural Net on val       | 0.93      | 0.91   | 0.92     |
| Shallow Neural Net (1 ReLU)     | 0.94      | 0.93   | 0.94     |
| Random Forest Classifier        | 1.00      | 0.54   | 0.70     |
| Gradient Boosting Classifier    | 1.00      | 0.76   | 0.86     |
| Linear SVC on val               | 1.00      | 0.84   | 0.91     |

## Observations
1. **Effect of Class Imbalance**:
   - Models trained on the imbalanced dataset have significantly lower recall, indicating that many fraudulent transactions are missed.
   - Random Forest and Gradient Boosting struggle with recall in the imbalanced dataset, showing that class imbalance affects their ability to detect fraud effectively.
   
2. **Improvement with Balanced Dataset**:
   - Balancing the dataset significantly improves recall across all models, showing that the models can detect fraudulent transactions more effectively.
   - Logistic Regression and Neural Networks demonstrate strong performance, with F1-scores around 0.94.
   - The Random Forest classifier, while achieving perfect precision, still has a relatively lower recall (0.54), suggesting it is highly conservative in labeling transactions as fraudulent.
   
3. **Best Performing Models**:
   - The best overall performance is observed with **Logistic Regression, Neural Networks, and Linear SVC** on the balanced dataset, as they maintain high precision and recall.
   - Gradient Boosting also shows a strong balance between precision and recall (F1-score = 0.86), making it a viable choice.

4. **Trade-offs Between Precision and Recall**:
   - High precision models (e.g., Random Forest) tend to avoid false positives but at the cost of missing fraudulent transactions (low recall).
   - Models with higher recall (e.g., Logistic Regression, Neural Networks) are better suited for fraud detection, as missing fraud cases can be more costly than having a few false positives.

## How to Run
1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```
2. Run the script:
   ```bash
   python fraud_detection.py
   ```

## Future Improvements
- Implement deep learning models (e.g., LSTMs or Autoencoders)
- Optimize class imbalance handling methods
- Deploy the model using Flask or FastAPI

## Acknowledgments
- The dataset was sourced from [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud).

