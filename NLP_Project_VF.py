
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
import warnings
warnings.filterwarnings('ignore')

############################################################
# 1. Load the data into a pandas data frame
# Since the data contains non-ASCII characters, we need to specify the encoding type. (emoji)
# We have tried to use null(should use when the file contain encoding details), 
# unicode_escape (should use when there are Unicode characters using backslash escapes), 
# ISO-8859-1 (only for latin characters)
############################################################
df = pd.read_csv('data/Youtube01-Psy.csv', encoding='utf-8')


############################################################
# 2.1 Basic Data Exploration & drop unnecessary columns:
# Print the number of rows and columns using df.shape (row by column)
# To check which columns are useful in our machine learning using df.info()
# Also, to check if there is any missing data (we could use df.isnull().sum() or df.info())
############################################################
print(f'DataFrame shape: {df.shape}')
print(f'Get information about the dataframe: {df.info()}')
print(f'Get description about the dataframe: {df.describe()}')
print(f'First 5 rows of the DataFrame:\n{df.head()}')

# Further more, we print out all the urls using regex
url_pattern = re.compile(r'https?://(?:www\.)?\S+')
all_urls = []
for content in df['CONTENT']:
    urls = url_pattern.findall(content)
    all_urls.extend(urls)
print('\n\nAll URLs found in CONTENT column:')
unique_urls = list(set(all_urls))
print('\n'.join(unique_urls))

# Further more, we print out all the emoticons using regex 
#(only unicode emojis, doesn't include ASCII emoticons or non-unicode emojis like ♡, :D, ❤️ etc.)
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U0001F910-\U0001F917"  # emojis
        u"\U0001F926-\U0001F930"  # face
        "]+", flags=re.UNICODE)
all_emojis = []
for content in df['CONTENT']:
    emojis = emoji_pattern.findall(content)
    all_emojis.extend(emojis)
print('\n\nAll emojis found in CONTENT column:')
unique_emojis = list(set(all_emojis))
print('\n'.join(unique_emojis))

non_eng_pattern = re.compile(r'[^a-zA-Z0-9]+')
all_non_eng = []
for content in df['CONTENT']:
    non_eng = non_eng_pattern.findall(content)
    all_non_eng.extend(non_eng)
print('\n\nAll non-English, numeric characters found in CONTENT column:')
unique_non_eng = list(set(all_non_eng))
print('\n'.join(unique_non_eng))

############################################################
# 2.2 Drop unnecessary columns because they are not related to text classification
# We determine that the columns COMMENT_ID, AUTHOR, and DATE are not useful for our machine learning
# We use df.drop() to drop the columns
# The axis=1 parameter specifies that the columns should be dropped, default value is 0 (rows)
# The inplace=True parameter specifies that the changes should be made to the df DataFrame
# df['CLASS'] is our target variable, 1 = spam, 0 = not spam
# df['CONTENT'] is our feature variable
############################################################
df.drop(['COMMENT_ID', 'AUTHOR', 'DATE'], axis=1, inplace=True)



############################################################
# 3. Using nltk for preparing the data for model building
# Based on the initial exporation, we found that CONTENT contains a variety of characters, including punctuation, special characters, and emojis
# Some of the CONTENT contains text in all caps while others contain mixed case text.
# We may need to pre-process by removing  stop words and punctuation, converting all text to lowercase or uppercase 
# stemming - base form by removing affixes or suffixes 
# lemmatization - base form by part of speech
# nltk.download('punkt') to download the Punkt Tokenizer to split the text into sentences
# nltk.download('stopwords') to download the Stopwords Tokenizer to remove non-informative words (meaningless words, i.e. the a, an, the, etc.)
############################################################
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
# Define a function to preprocess the text
def preprocess_text(text):
    # Replace URLs with a special tag
    text = re.sub(r'https?://\S+|www\.\S+', 'hyperlink', text)
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation marks
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize the text into words
    words = word_tokenize(text)
    # Remove stop words
    #stop_words = set(stopwords.words('english'))
    #words = [word for word in words if word not in stop_words]
    # Stem the words
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    # Join the words back into a string
    text = ' '.join(words)
    return text
# Use pandas apply() to apply the preprocess_text function to the CONTENT column
df['CONTENT'] = df['CONTENT'].apply(preprocess_text)
# Creates a CountVectorizer object that will be used to preprocess the text data
# No need to specify any parameters, since we have already preprocessed the text
vectorizer = CountVectorizer()
# Involves tokenizing the text data, counting the frequency of each token
X = vectorizer.fit_transform(df['CONTENT'])
# Assigns the target variable to y
y = df['CLASS']



############################################################
# 4. Highlights of the Output
############################################################
print("Sparse matrix X:", X.toarray())
print("Shape of feature matrix X:", X.shape)
print("Size of vocabulary:", len(vectorizer.vocabulary_))
print("Tokens :", vectorizer.vocabulary_.keys())
vocab_dict = {'Token': list(vectorizer.vocabulary_.keys()), 'Count': list(vectorizer.vocabulary_.values())}
vocab_df = pd.DataFrame.from_dict(vocab_dict)
sorted_vocab_df = vocab_df.sort_values(by=['Count'], ascending=False)
print(f"\n\nTop 20 tokens: \n{sorted_vocab_df.head(20)}")
print(f"\n\nBottom 20 tokens: \n{sorted_vocab_df.tail(20)}")



############################################################
# 5. Downscaling the transformed data using tf-idf
# Weights for each word in each document
############################################################
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X)
print(f'X_tfidf shape: {X_tfidf.shape}')



############################################################
# 6. Use pandas.sample to shuffle the dataset, set frac =1 
# frac=1 specifies that we want to shuffle all the rows of the DataFrame.
# random_state=42 sets a seed value for the random number generator used for shuffling the dataset.
# Use a common index to ensure the correspondence between the feature matrix and the target vector is maintained
############################################################
df_shuffled = df.sample(frac=1, random_state=1)
X_shuffled = X_tfidf[df_shuffled.index]
y_shuffled = y[df_shuffled.index]



############################################################
# 7. Splitting the Dataset into 75% Training and 25% Testing
# By default, train_test_split shuffles the data before splitting it into training and testing sets
# 25% of the data is used for testing the model
# 75% of the data is used for training the model
# random_state parameter is set to 42 for reproducibility
############################################################
#X_train, X_test, y_train, y_test = train_test_split(X_shuffled, y_shuffled, test_size=0.25, random_state=42)
train_size = int(0.75*len(df_shuffled))
X_train = X_shuffled[:train_size,:]
X_test = X_shuffled[train_size:,:]
y_train =  y_shuffled[:train_size]
y_test = y_shuffled[train_size:]
#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape)

############################################################
# 8. Fitting the Model into Naive Bayes Classifier:
# fit() method is used to train the model on the input data
############################################################
clf = MultinomialNB()
clf.fit(X_train, y_train)



############################################################
# 9. Cross Validating the Model on the Training Data:
# cv=5 specifies that we want to use 5-fold cross validation
# The cross_val_score() function returns an array of scores for each fold
# The mean of the scores is the accuracy of the model
############################################################
scores = cross_val_score(clf, X_train, y_train, cv=5)
print(f'\n\nResult of Model Accuracy: {scores}')
print(f'\n\nMean Result of Model Accuracy: {scores.mean()}')



############################################################
# 10. Testing the Model on the Test Data:
# 40 true positive predictions (TP)
# 2 false positive prediction (FP)
# 3 false negative prediction (FN)
# 43 true negative predictions (TN)
# class 0 is the negative class
# class 1 is the positive class
# precision = TP / (TP + FP) (how well the classifier correctly identifies the positive instances out of all predicted positive instances)
# recall = TP / (TP + FN) (how well the classifier correctly identifies the positive instances out of all actual positive instances)
# f1-score = 2 * (precision * recall) / (precision + recall) (harmonic mean of precision and recall, higher score indicates better performance)
# support = number of occurrences of each class in y_test
# accuracy = (TP + TN) / (TP + TN + FP + FN)
# macro avg = average by number of classes
# weighted avg = average by weighting the support for each class
# The mean result of model accuracy gives a more general idea of the model's performance on unseen data.
############################################################
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(f'\n\nConfusion Matrix:\n{cm}')
print(f'\n\nModel Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))
cm_df = pd.DataFrame(cm, index=['0','1'], columns=['0','1'])
sns.heatmap(cm_df, annot=True, fmt='d')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()



############################################################
# 11 Testing the Model on New Comments:
# 4 comments should be non spam and 2 comment spam
############################################################
new_comments = [
    "He dances well",
    "i hate this song 😡",
    'OMGGGGG ❤️ this mv so much', 
    '😊😊😊😊😊',
    'Click here to get a free iPhone', 
    'Go to https://tinyurl.com/3h4h3h4h to get free movie ticket'
]

new_comments_processed = [preprocess_text(comment) for comment in new_comments]
new_comments_tfidf = tfidf_transformer.transform(vectorizer.transform(new_comments_processed))
new_comments_pred = clf.predict(new_comments_tfidf)
for i in range(len(new_comments)):
    print(f'\nComment: {new_comments[i]}')
    print(f'Predicted class: {new_comments_pred[i]}')