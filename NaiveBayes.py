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


def dir_to_csv(folder, ai_generated, writing_style): # ai_generated = 0 or 1
    txt_files = glob.glob(folder + '/*.txt')
    texts_list = []
    for file_path in txt_files:
        texts_list.append(readFile(file_path))

    texts_dataframe = pd.DataFrame({'Text' : texts_list, 'Writing Style': [writing_style for i in range(len(texts_list))], 'AI-Generated': [ai_generated for i in range(len(texts_list))]})
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


    ai_dir = author_dir + f"/{author_basename}_Reedsy_Prompts_short_stories_AI_version"
    aip_dir = author_dir + f"/{author_basename}_Reedsy_Prompts_short_stories_AI_Person_style"
    aih_dir = author_dir + f"/{author_basename}_Reedsy_Prompts_short_stories_AI_Human_Style"
    sai_dir = author_dir + f"/{author_basename}_Stories_AI"
    saih_dir = author_dir + f"/{author_basename}_Stories_AI_human_style"
    saip_dir = author_dir + f"/{author_basename}_Stories_AI_person_style"
    human_dir = author_dir + f"/{author_basename}_Reedsy_Prompts_short_stories"


    dir_to_csv(ai_dir, 1, 'ReedSy-AI')
    dir_to_csv(aip_dir, 1, 'ReedSy-AI-Person-Style')
    dir_to_csv(aih_dir, 1, 'ReedSy-AI-Human-Style')
    dir_to_csv(sai_dir, 1, 'Stories-AI')
    dir_to_csv(saih_dir, 1, 'Stories-AI-Human-Style')
    dir_to_csv(saip_dir, 1, 'Stories-AI-Person-Style')
    dir_to_csv(human_dir, 0, 'ReedSy-Human')

    combine_csv_files(author_dir, author_basename)
    os.remove(f"{author_dir}/{author_basename}_Stories_AI_person_style.csv")
    os.remove(f"{author_dir}/{author_basename}_Stories_AI_human_style.csv")
    os.remove(f"{author_dir}/{author_basename}_Stories_AI.csv")
    os.remove(f"{author_dir}/{author_basename}_Reedsy_Prompts_short_stories_AI_version.csv")
    os.remove(f"{author_dir}/{author_basename}_Reedsy_Prompts_short_stories_AI_Person_style.csv")
    os.remove(f"{author_dir}/{author_basename}_Reedsy_Prompts_short_stories_AI_Human_Style.csv")
    os.remove(f"{author_dir}/{author_basename}_Reedsy_Prompts_short_stories.csv")
"""
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

    unique_styles = dataset['Writing Style'].unique()

    test_data = pd.DataFrame(columns=dataset.columns)
    train_data = pd.DataFrame(columns=dataset.columns)

    for style in unique_styles:
        style_data = dataset[dataset['Writing Style'] == style]
        style_train, style_test = train_test_split(style_data, test_size=0.2, random_state=0)
        test_data = pd.concat([test_data, style_test])
        train_data = pd.concat([train_data, style_train])

    x_train = vectorizer.fit_transform(train_data['Text'])  # Transform the training text data
    y_train = train_data['AI-Generated'].astype(int)

    x_test = vectorizer.transform(test_data['Text'])  # Transform the testing text data
    y_test = test_data['AI-Generated'].astype(int)

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
    print(f"***************** {author_basename} *****************")
    csv_file = f"{author_dir}/{author_basename}_ai_human_merged.csv"
    save_classifier_and_vectorizer(csv_file, f"{classifiers_dir}/{author_basename}", author_basename)
    print()



"""
dataset = pd.read_csv("/home/youssef/Desktop/classification_algorithms_per_author/Datasets_per_author/Aries_Walker/Aries_Walker_ai_human_merged.csv")

# Remove rows containing NaN values
dataset.dropna(inplace=True)

# Convert a collection of text to a matrix of tokens
vectorizer = CountVectorizer(analyzer=process_text)
# text_bow = vectorizer.fit_transform(dataset['Text'])

unique_styles = dataset['Writing Style'].unique()

test_data = pd.DataFrame(columns=dataset.columns)
train_data = pd.DataFrame(columns=dataset.columns)

for style in unique_styles:
    style_data = dataset[dataset['Writing Style'] == style]
    style_train, style_test = train_test_split(style_data, test_size=0.2, random_state=0)
    test_data = pd.concat([test_data, style_test])
    train_data = pd.concat([train_data, style_train])

x_train = vectorizer.fit_transform(train_data['Text'])  # Transform the training text data
y_train = train_data['AI-Generated']

x_test = vectorizer.transform(test_data['Text'])  # Transform the testing text data
y_test = test_data['AI-Generated']

print(y_test)

"""
