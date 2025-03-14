# IMDb-Sentiment-Analysis-Machine-Learning-Project


This project focuses on sentiment analysis of IMDb movie reviews. The objective is to classify movie reviews as positive or negative using various machine learning techniques.

## Project Overview

- The dataset consists of **50,000** IMDb movie reviews labeled as positive or negative.
- The analysis involves **data cleaning, exploratory data analysis (EDA), and machine learning modeling**.
- Three machine learning models were implemented and compared:
  - **Linear Regression**
  - **Random Forest**
  - **Logistic Regression** (achieved **89% accuracy**)

## Dataset & Preprocessing

- **Dataset:** IMDb movie reviews dataset with two columns: `review` (text) and `sentiment` (label: positive/negative).
- **Preprocessing Steps:**
  - **Text Cleaning:** Removal of HTML tags using BeautifulSoup.
  - **Tokenization:** Splitting text into individual words.
  - **Stopword Removal:** Eliminating commonly used words (`a, an, the`, etc.).
  - **Lemmatization:** Converting words to their base form.
  - **Vectorization:** Using **Count Vectorizer** and **TF-IDF** for text representation.

## Machine Learning Models Used

1. **Linear Regression**
   - A simple yet effective model for predicting numerical values.
   - Applied for binary classification with probability estimation.
2. **Random Forest**
   - Ensemble model consisting of multiple decision trees.
   - Provides robust predictions and minimizes overfitting.
3. **Logistic Regression**
   - A highly efficient binary classification model.
   - Achieved the highest accuracy (**89%**) in sentiment classification.

## Model Performance

- **Logistic Regression** outperformed the other models with the highest accuracy.
- **Random Forest** provided strong results but had longer training times.
- **Linear Regression** had lower accuracy but was still effective for basic classification.

## Installation & Usage

### Requirements
- **Python 3.7+**
- **Jupyter Notebook**
- **Required Libraries:** `pandas`, `numpy`, `sklearn`, `nltk`, `matplotlib`

### Running the Project
1. Clone the repository or download the project files.
2. Install required dependencies:
   ```sh
   pip install pandas numpy scikit-learn nltk matplotlib
   ```
3. Open the Jupyter Notebook and execute the cells.

## Conclusion

- **Logistic Regression** proved to be the best model for IMDb sentiment analysis.
- The project highlights the effectiveness of traditional machine learning methods in NLP tasks.
- Further improvements can be made using deep learning models like LSTMs and Transformers.

## License

This project is open-source. Feel free to modify and improve it as needed.

