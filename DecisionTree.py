import glob
import string
import nltk
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle



def process_text(text):
  # Remove punctuation
  nopunc = [char for char in text if char not in string.punctuation]
  nopunc = ''.join(nopunc)

  # Remove stopwords
  clean_words = [word for word in nopunc.split() if word not in stopwords.words('english')]
  return clean_words



def decision_tree_classifier(csv_file):
    # Load the dataset from CSV
    dataset = pd.read_csv(csv_file)

    # Create TF-IDF vectorizer to convert text into numerical features
    vectorizer = CountVectorizer(analyzer=process_text)


    # Split the data into training and testing sets
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

    #Fitting K-NN classifier to the training set
    classifier = DecisionTreeClassifier()
    classifier.fit(x_train, y_train)

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
    vectorizer, classifier = decision_tree_classifier(csv_file)
    with open(f'{destination_folder}/decision_tree_classifier_{author_basename}.pickle', 'wb') as f:
        pickle.dump(classifier, f)
    with open(f'{destination_folder}/decision_tree_vectorizer_{author_basename}.pickle', 'wb') as f:
        pickle.dump(vectorizer, f)

def read_classifier_and_vectorizer():
    with open('classifier.pickle', 'rb') as f:
        classifier = pickle.load(f)
    with open('vectorizer.pickle', 'rb') as f:
        vectorizer = pickle.load(f)

    return vectorizer, classifier

def test_classifier(text_test, vectorizer, classifier):
  text_test_words = process_text(text_test)
  text_test = ' '.join(text_test_words)
  text_bow = vectorizer.transform([text_test])
  text_test_pred = classifier.predict(text_bow)
  return text_test_pred





root_dir = '/home/youssef/Desktop/classification_algorithms_per_author/Datasets_per_author'
classifiers_dir = '/home/youssef/Desktop/classification_algorithms_per_author/classifiers'
authors_dirs = glob.glob(root_dir + "/*")
for author_dir in authors_dirs:
    author_basename = author_dir.split("/")[-1]
    print(f"***************** {author_basename} *****************")
    csv_file = f"{author_dir}/{author_basename}_ai_human_merged.csv"
    # decision_tree_classifier(csv_file)
    save_classifier_and_vectorizer(csv_file, f"{classifiers_dir}/{author_basename}", author_basename)
    print()



