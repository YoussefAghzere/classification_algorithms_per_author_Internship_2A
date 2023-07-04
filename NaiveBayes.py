# Import libraries
import os
import glob
import pandas as pd
import numpy as nm
import matplotlib.pyplot as mtp
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle



def readFile(filepath):
    with open(filepath, "r") as df:
        content = df.read()
    return content


def dir_to_csv(folder, ai_generated): # ai_generated = 0 or 1
    txt_files = glob.glob(folder + '/*.txt')
    texts_list = []
    for file_path in txt_files:
        texts_list.append(readFile(file_path))

    texts_dataframe = pd.DataFrame({'Text' : texts_list, 'AI-Generated': [ai_generated for i in range(len(texts_list))]})
    texts_dataframe.to_csv(f'{folder}.csv', index=False)

def combine_csv_files(parent_folder, filename):
    csv_files = glob.glob(parent_folder + '/*.csv')
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)
    merged_df = pd.concat(dfs)
    shuffled_df = merged_df.sample(frac=1)
    shuffled_df.to_csv(f'{parent_folder}/{filename}_ai_human_merged.csv', index=False)

# csv files creation and combination
"""root_dir = '/home/youssef/Desktop/classification_algorithms_per_author/Datasets_per_author'
authors_dirs = glob.glob(root_dir + "/*")
for author_dir in authors_dirs:
    author_basename = author_dir.split("/")[-1]
    ai_dir = author_dir + f"/{author_basename}_AI"
    human_dir = author_dir + f"/{author_basename}_Human"
    csv_ai_file = dir_to_csv(ai_dir, 1)
    csv_human_file = dir_to_csv(human_dir, 0)
    combine_csv_files(author_dir, author_basename)
    os.remove(f"{author_dir}/{author_basename}_AI.csv")
    os.remove(f"{author_dir}/{author_basename}_Human.csv")"""


def process_text(text):
  # Remove punctuation
  nopunc = [char for char in text if char not in string.punctuation]
  nopunc = ''.join(nopunc)

  # Remove stopwords
  clean_words = [word for word in nopunc.split() if word not in stopwords.words('english')]
  return clean_words


def naive_bayes(csv_file):
    dataset = pd.read_csv(csv_file)

    # Remove rows containing NaN values
    dataset.dropna(inplace=True)

    # Convert a collection of text to a matrix of tokens
    vectorizer = CountVectorizer(analyzer=process_text)
    text_bow = vectorizer.fit_transform(dataset['Text'])
    x = dataset.iloc[:, 0]
    y = dataset.iloc[:, 1]

    # splitting the dataset into the training set and test set
    x_train, x_test, y_train, y_test = train_test_split(text_bow, y, test_size = 0.2, random_state = 0)

    # Fitting Naive Bayes to the Training set
    classifier = MultinomialNB().fit(x_train, y_train)

    # Evaluate the model of the testing dataset
    print("--- Evaluation of the classifier on the test dataset ---")
    pred_test = classifier.predict(x_test)
    print(classification_report(y_test, pred_test))
    print()
    print('Confusion Matrix :\n', confusion_matrix(y_test, pred_test))
    print()
    print('Accuracy : ', accuracy_score(y_test, pred_test))

    return vectorizer, classifier



def save_classifier_and_vectorizer(csv_file, destination_folder, author_basename):
    vectorizer, classifier = naive_bayes(csv_file)
    with open(f'{destination_folder}/naive_bayes_classifier_{author_basename}.pickle', 'wb') as f:
        pickle.dump(classifier, f)
    with open(f'{destination_folder}/naive_bayes_vectorizer_{author_basename}.pickle', 'wb') as f:
        pickle.dump(vectorizer, f)


def read_classifier_and_vectorizer():
    with open('naive_bayes_classifier.pickle', 'rb') as f:
        classifier = pickle.load(f)
    with open('naive_bayes_vectorizer.pickle', 'rb') as f:
        vectorizer = pickle.load(f)

    return vectorizer, classifier


root_dir = '/home/youssef/Desktop/classification_algorithms_per_author/Datasets_per_author'
classifiers_dir = '/home/youssef/Desktop/classification_algorithms_per_author/classifiers'
authors_dirs = glob.glob(root_dir + "/*")
for author_dir in authors_dirs:
    author_basename = author_dir.split("/")[-1]
    print(f"***************** {author_basename}*****************")
    csv_file = f"{author_dir}/{author_basename}_ai_human_merged.csv"
    save_classifier_and_vectorizer(csv_file, f"{classifiers_dir}/{author_basename}", author_basename)
    print()
