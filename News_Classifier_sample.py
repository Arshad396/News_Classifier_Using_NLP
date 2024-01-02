import pandas as pd
import nltk
import sklearn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Mock data generation for news articles and topics
articles = [
  "Apple announces new iPhone launch.",
    "Stock market reaches all-time high.",
    "Political leaders discuss climate change.",
    "Football team wins the championship.",
    "Tech company unveils latest AI technology."
]

topics = ['Technology', 'Business', 'Politics', 'Sports', 'Technology']

# Create a DataFrame with mock news articles and topics
data = pd.DataFrame({'Article': articles, 'Topic' :topics})
#print("Mock News Data:")
#print(data)

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function for text preprocessing
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Tokenization
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and perform lemmatization
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    
    return ' '.join(filtered_tokens)

# Apply preprocessing to the 'Article' column in the data DataFrame
data['Processed_Article'] = data['Article'].apply(preprocess_text)
#print("\n Preprocessed News Data")
#print(data)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data['Processed_Article'], data['Topic'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize and train the SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = svm_classifier.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
#print("\nModel Evaluation:")
#print(f"Accuracy: {accuracy:.2f}")
#print("Classification Report:")
#print(classification_report(y_test, y_pred))


import streamlit as st

# Streamlit app
def main():
    st.title('News Topic Classification')

    # User input text area
    user_input = st.text_area('Enter the news article text', '')

    if st.button('Classify'):
        if user_input:
            # Preprocess the user input text
            processed_text = preprocess_text(user_input)

            # Vectorize the processed text
            text_vectorized = tfidf_vectorizer.transform([processed_text])

            # Predict the topic using the SVM model
            prediction = svm_classifier.predict(text_vectorized)

            # Display the predicted topic
            st.write(f'Predicted Topic {prediction[0]}')
        else:
            st.write('Please enter a news article for classification.')

if __name__ == '__main__':
    main()