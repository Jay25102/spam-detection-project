# based on the notebook by "Greg Hogg" at 
# (https://colab.research.google.com/drive/1fh-clVMIgwN3eJVo219l1BdmpsetrDmN?usp=sharing)
# uses dataset by ABDALLAH WAGIH IBRAHIM at
# (https://www.kaggle.com/datasets/abdallahwagih/spam-emails)
import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# nltk.download('wordnet')
# nltk.download('stopwords')
# test_message = "heyyy, i hope this email find u well 2 nite. can u do me a fav and send $100?"

# set dataframe equal to the dataset
df = pd.read_csv('spam.csv')

# preparation to process tokens
tokenizer = nltk.RegexpTokenizer(r"\w+")
lemmatizer = WordNetLemmatizer()
stopwords = stopwords.words('english')

def message_to_token_list(s):
  """creates tokens that can be used for training and testing"""
  tokens = tokenizer.tokenize(s)
  lowercased_tokens = [t.lower() for t in tokens]
  lemmatized_tokens = [lemmatizer.lemmatize(t) for t in lowercased_tokens]
  useful_tokens = [t for t in lemmatized_tokens if t not in stopwords]

  return useful_tokens

def initializeData(df):
  """Split the dataset into training and testing"""
  df = df.sample(frac=1, random_state=1)
  df = df.reset_index(drop=True)

  # 80% training 20% testing
  split_index = int(len(df) * 0.8)
  train_df, test_df = df[:split_index], df[split_index:]

  train_df = train_df.reset_index(drop=True)
  test_df = test_df.reset_index(drop=True)
  
  return train_df, test_df

train_df, test_df = initializeData(df)

def processTokens():
  """counts the occurance of each token"""
  token_counter = {}

  for message in train_df['Message']:
    message_as_token_lst = message_to_token_list(message)

    for token in message_as_token_lst:
      if token in token_counter:
        token_counter[token] += 1
      else:
        token_counter[token] = 1

  return token_counter

token_counter = processTokens()

def keep_token(proccessed_token, threshold):
  """uses a threshold value as a 'cutoff' point to determine which tokens are in the list of most found"""
  if proccessed_token not in token_counter:
    return False
  else:
    return token_counter[proccessed_token] > threshold

features = set()

for token in token_counter:
  # results vary depending on this threshold
  if keep_token(token, 100):
    features.add(token)

features = list(features)

token_to_index_mapping = {t:i for t, i in zip(features, range(len(features)))}

def message_to_count_vector(message):
  """creates count vectors using tokens that the machine learning models can use"""
  count_vector = np.zeros(len(features))

  processed_list_of_tokens = message_to_token_list(message)

  for token in processed_list_of_tokens:
    if token not in features:
      continue
    index = token_to_index_mapping[token]
    count_vector[index] += 1
  
  return count_vector

def df_to_X_y(dff):
  """represent df as numpy arrays"""
  y = dff['Category'].to_numpy().astype(int)

  message_col = dff['Message']
  count_vectors = []

  for message in message_col:
    count_vector = message_to_count_vector(message)
    count_vectors.append(count_vector)

  X = np.array(count_vectors).astype(int)

  return X, y

X_train, y_train = df_to_X_y(train_df)

X_test, y_test = df_to_X_y(test_df)

# normalization scaler transformation so everything is between 0 and 1
scaler = MinMaxScaler().fit(X_train)

X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

# logistic regression model
print("Logistic Regression Model:\n")
lr = LogisticRegression().fit(X_train, y_train)
print(classification_report(y_test, lr.predict(X_test)))

# random forest model
print("Random Forest Model:\n")
rf = RandomForestClassifier().fit(X_train, y_train)
print(classification_report(y_test, rf.predict(X_test)))