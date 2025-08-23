# Text Classification with Multiple Algorithms

This project demonstrates text classification using three different machine learning algorithms: **Naive Bayes**, **Logistic Regression**, and **Support Vector Machine (SVM)**. We classify economic news articles as relevant or not relevant to the US Economy.

## Dataset

- **Source**: Economic news articles from Figure-Eight
- **Size**: ~8,000 articles
- **Classes**: 
  - Relevant to US Economy (18%)
  - Not Relevant to US Economy (82%)
  - Not Sure (0.1% - removed during preprocessing)
- **Features**: Text content of news articles

## Technologies Used

- **Python 3.11**
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Visualization
- **numpy**: Numerical operations

## Project Structure

```
my-lab/
├── TextClassification.ipynb    # Main Jupyter notebook
├── README.md                   # This file
└── Data/                       # Dataset directory
    └── Full-Economic-News-DFE-839861.csv
```

## Process Overview

### 1. Data Loading and Exploration
- Load CSV file with economic news articles
- Explore dataset structure and class distribution
- Identify class imbalance (82% not relevant vs 18% relevant)

### 2. Data Preprocessing
- Remove "not sure" entries (9 articles)
- Convert labels to binary: 'yes' → 1, 'no' → 0
- Keep only essential columns: 'text' and 'relevance'

### 3. Text Preprocessing
- Remove HTML tags (`<br/>`)
- Remove punctuation and numbers
- Remove stop words using CountVectorizer
- Create cleaning function for text normalization

### 4. Feature Extraction
- Use CountVectorizer to convert text to numerical features
- Apply text cleaning function as preprocessor
- Create document-term matrix (DTM)

### 5. Model Training and Evaluation

#### Algorithm 1: Multinomial Naive Bayes
- **Features**: 5,000 most frequent words
- **Class weights**: Default (no balancing)
- **Strengths**: Fast training, good baseline performance
- **Expected accuracy**: ~82% (due to class imbalance)

#### Algorithm 2: Logistic Regression
- **Features**: 5,000 most frequent words
- **Class weights**: Balanced (handles class imbalance)
- **Strengths**: Probabilistic outputs, handles class imbalance
- **Expected accuracy**: ~73-75%

#### Algorithm 3: Support Vector Machine (SVM)
- **Features**: 1,000 most frequent words (reduced)
- **Class weights**: Balanced
- **Strengths**: Good for high-dimensional sparse data
- **Expected accuracy**: Varies based on feature reduction

### 6. Evaluation Metrics
- **Accuracy**: Overall correct predictions
- **AUC (Area Under Curve)**: Model discrimination ability
- **Confusion Matrix**: Detailed class-wise performance
- **Class-wise Performance**: Focus on minority class performance

## Key Code Sections

### Data Loading
```python
our_data = pd.read_csv("path/to/Full-Economic-News-DFE-839861.csv", 
                       encoding="ISO-8859-1")
```

### Data Cleaning
```python
# Remove 'not sure' entries and convert to binary
our_data = our_data[our_data.relevance != "not sure"]
our_data['relevance'] = our_data.relevance.map({'yes':1, 'no':0})
our_data = our_data[["text","relevance"]]
```

### Text Preprocessing
```python
def clean(doc):
    doc = doc.replace("</br>", " ")
    doc = "".join([char for char in doc if char not in string.punctuation and not char.isdigit()])
    doc = " ".join([token for token in doc.split() if token not in stopwords])
    return doc
```

### Model Training
```python
# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)

# Logistic Regression
logreg = LogisticRegression(class_weight="balanced")
logreg.fit(X_train_dtm, y_train)

# SVM
svm = LinearSVC(class_weight='balanced')
svm.fit(X_train_dtm_1000, y_train)
```

## Results and Insights

### Class Imbalance Challenge
- **Problem**: 82% not relevant vs 18% relevant
- **Impact**: Models tend to predict majority class
- **Solution**: Use class weights and focus on minority class performance

### Algorithm Comparison
1. **Naive Bayes**: Good baseline, fast, but struggles with minority class
2. **Logistic Regression**: Better balanced performance with class weights
3. **SVM**: Efficient with reduced features, good for sparse data

### Key Learnings
- **Feature reduction** can improve model efficiency
- **Class weights** are crucial for imbalanced datasets
- **No single algorithm** performs best on all datasets
- **Confusion matrix** reveals more than accuracy alone

## Best Practices Demonstrated

1. **Data Exploration**: Always understand your data first
2. **Text Preprocessing**: Clean and normalize text data
3. **Feature Engineering**: Choose appropriate feature extraction
4. **Model Selection**: Try multiple algorithms
5. **Evaluation**: Use multiple metrics, not just accuracy
6. **Class Imbalance**: Handle with appropriate techniques

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```

2. **Download Dataset**:
   - Place `Full-Economic-News-DFE-839861.csv` in the Data directory

3. **Run Notebook**:
   ```bash
   jupyter notebook TextClassification.ipynb
   ```

## Next Steps

- **Hyperparameter Tuning**: Optimize model parameters
- **Cross-Validation**: More robust evaluation
- **Feature Selection**: Try different feature extraction methods
- **Ensemble Methods**: Combine multiple models
- **Deep Learning**: Try neural network approaches

## Contributing

Feel free to experiment with:
- Different preprocessing techniques
- Additional algorithms
- Feature engineering approaches
- Evaluation metrics

---

**Note**: This project demonstrates fundamental text classification concepts and serves as a foundation for more advanced NLP tasks.

## Lab Results Summary

### Model Performance Comparison

| Algorithm | Features | Accuracy | AUC | Notes |
|-----------|----------|----------|-----|-------|
| **Naive Bayes** | 5,000 | 68.8% | 72.8% | Good baseline performance |
| **Logistic Regression** | 5,000 | 73.6% | 65.9% | Best overall accuracy |
| **SVM** | 1,000 | 68.4% | N/A | Efficient with fewer features |

### Key Findings

1. **Logistic Regression performed best** with 73.6% accuracy
2. **Naive Bayes had the highest AUC** (72.8%), indicating better class separation
3. **SVM maintained good performance** despite using 80% fewer features
4. **All models struggled** with the class imbalance (82% vs 18%)

### Practical Insights

- **Feature reduction** (5,000 → 1,000) had minimal impact on SVM performance
- **Class balancing** in Logistic Regression helped achieve the best accuracy
- **No single metric** tells the complete story - both accuracy and AUC matter
- **Model choice depends** on specific requirements (speed vs accuracy vs interpretability)

### Recommendations

- **For production**: Use Logistic Regression for best overall performance
- **For interpretability**: Choose Naive Bayes for probabilistic outputs
- **For efficiency**: Use SVM when computational resources are limited
- **For improvement**: Focus on handling class imbalance and feature engineering
