from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow
import sys
import pandas as pd

num_words = 26
skip_top = 25
oov_char = -1

def main():
  # Getting the information necessary from the command line
  if len(sys.argv) != 3:
    sys.exit("Usage: python NaiveBayes.py {string I. Y to use IMDB dataset/N otherwise} {string D. Y for diagnostic test/N for default execution}")
    
  if (len(sys.argv) == 3):
    try:
      imdb = str(sys.argv[1])
    except ValueError:
      print("Not a string")

  if (len(sys.argv) == 3):
    try:
      diagnostics = str(sys.argv[2])
    except ValueError:
      print("Not a string")

  if (imdb == 'Y'):
    train, test, inverted_word_index, word_index = IMDB_dataset()
    properties_train = create_properties(train)
    properties_test = create_properties(test)
  else: 
    path = input("Please enter the path:\n")


  if (diagnostics == 'N'):
    #NaiveBayes()
    None
  elif (diagnostics == 'Y' and imdb == 'Y'):
    diagnostic_tests(properties_train, properties_test)
  elif (diagnostics == 'Y' and imdb != 'Y'):
    print("Diagnostic tests are available only for the IMDB dataset")


def IMDB_dataset():
  train, test = tensorflow.keras.datasets.imdb.load_data(num_words=num_words, skip_top=skip_top, oov_char=oov_char)
  word_index = keras.datasets.imdb.get_word_index()
  inverted_word_index = dict((i, word) for (word, i) in word_index.items())
  return train, test, inverted_word_index, word_index
  

def create_properties(examples):
  properties = []
  # Properties form
  # [
  #  [1, 0, 1, 1, 0, ..., 1]
  #  [0, 0, 0, 1, 0, ..., 1]
  #  ...
  # ]
  
  # For every review...
  for r in range(len(examples[0])):
    # ...create a new list
    properties.append([])
    for i in range(skip_top, num_words):
      # For every word of the vocabulary
      # add a cell in the list of the review with value 0
        properties[r].append(0)
        # For every word of the review...
        for w in examples[0][r]:
          # ...if it isn't in the vocabulary continue to the next one
          if w == -1: continue
          # Otherwise, if the word of the vocabulary exists in the review
          # change the value of the cell in the properties list to 1
          if i == w:
            properties[r][i-skip_top] = 1
            break
    # Add the label of the review (positive:1 , negative: 0) at the end of each list
    properties[r].append(examples[1][r])

  return properties


def calculate_possibility_properties(properties):
  negative_count = 0 # Count of negative reviews
  properties_count_neg = [] # Count of times a given property is found in a negative review
  properties_count_pos = [] # Count of times a given property is found in a positive review

  for i in range(len(properties[0])):
    properties_count_neg.append(0)
    properties_count_pos.append(0)

  for p in properties:
    if p[len(p) - 1] == 0:
        negative_count += 1
    for i in range(len(p)):
      if p[len(p) - 1] == 0:
        if p[i] == 1:
          properties_count_neg[i] += 1
      else:
        if p[i] == 1:
          properties_count_pos[i] += 1
  PC0 = negative_count/len(properties)
  PC1 = 1 - PC0

  possibility_neg = [] # Possibility that a word exists in a negative review, P(X=1|C=0)
  possibility_not_neg = [] # Possibility that a word doesn't exist in a negative review, P(X=0|C=0)
  possibility_pos = [] # Possibility that a word exists in a possitive review, P(X=1|C=1)
  possibility_not_pos = [] # Possibility that a word doesn't exist in a positive review, P(X=0|C=1)
  for i in range(len(properties[0])):
    # Laplace smoothinγ
    possibility_neg.append((properties_count_neg[i] + 1) / (negative_count + 2))
    possibility_not_neg.append((negative_count - properties_count_neg[i] + 1) / (negative_count + 2))
    possibility_pos.append((properties_count_pos[i] + 1) / (len(properties) - negative_count + 2))
    possibility_not_pos.append((len(properties) - negative_count - properties_count_pos[i] + 1) / (len(properties) - negative_count + 2))

  return possibility_neg, possibility_pos, possibility_not_neg, possibility_not_pos, PC0, PC1


def NaiveBayes(properties, possibility_neg, possibility_pos, possibility_not_neg, possibility_not_pos, PC0, PC1):
  categories = []
  for p in properties:
    product1 = PC1
    product0 = PC0
    for i in range(len(p)):
      if p[i] == 0:
        product1 *= possibility_not_pos[i]
        product0 *= possibility_not_neg[i]
      else:
        product1 *= possibility_pos[i]
        product0 *= possibility_neg[i]

    if product0 > product1:
      categories.append(0)
    else:
      categories.append(1)

  return categories


def calculate_classifiers(categories, properties):
  correct_predictions = 0
  wrong_predictions = 0
  true_positives = 0
  false_positives = 0
  false_negatives = 0
  for i in range(len(properties)):
    if categories[i] == properties[i][-1]:
      correct_predictions += 1
    else:
      wrong_predictions += 1

    if categories[i] == 1 and properties[i][-1] == 1:
      true_positives += 1
    if categories[i] == 1 and properties[i][-1] == 0:
      false_positives += 1
    if categories[i] == 0 and properties[i][-1] == 1:
      false_negatives += 1

  accuracy = correct_predictions / (correct_predictions + wrong_predictions)
  precision = true_positives / (true_positives + false_positives)
  recall = true_positives / (true_positives + false_negatives)
  F1 = (2 * precision * recall) / (precision + recall)

  return accuracy, precision, recall, F1


def diagnostic_tests(properties_train, properties_test):
  accuracy_train = []
  accuracy_test = []
  precision_train = []
  recall_train = []
  F1_train = []
  num_of_examples = []

  partitions = 5
  for i in range(partitions, 0, -1):
    num_of_examples.append((len(properties_train)//partitions * (partitions - i + 1))-1)
    print("Executing the algorithm with " + str(num_of_examples[partitions-i]) + " reviews...")
    possibility_neg, possibility_pos, possibility_not_neg, possibility_not_pos, PC0, PC1 = calculate_possibility_properties(properties_train[:num_of_examples[partitions-i]])
    categories = NaiveBayes(properties_train[:num_of_examples[partitions-i]], possibility_neg, possibility_pos, possibility_not_neg, possibility_not_pos, PC0, PC1)
    print(categories[i], properties_train[i][-1])
    classifiers_train = calculate_classifiers(categories, properties_train[:num_of_examples[partitions-i]])
    accuracy_train.append(classifiers_train[0])
    precision_train.append(classifiers_train[1])
    recall_train.append(classifiers_train[2])
    F1_train.append(classifiers_train[3])
    categories = NaiveBayes(properties_test[:num_of_examples[partitions-i]], possibility_neg, possibility_pos, possibility_not_neg, possibility_not_pos, PC0, PC1)
    classifiers_test = calculate_classifiers(categories, properties_test[:num_of_examples[partitions-i]])
    accuracy_test.append(classifiers_test[0])
  print(pd.DataFrame({'Reviews': num_of_examples, 'Accuracy (Train)': accuracy_train, 'Accuracy (Test)': accuracy_test}))
  print(pd.DataFrame({'Reviews': num_of_examples, 'Precision (Train)': precision_train, 'Recall (Train)': recall_train, 'F1 (Train)': F1_train}))
  plt.plot(num_of_examples, accuracy_train, label='Total train accuracy')
  plt.plot(num_of_examples, accuracy_test, label='Total test accuracy')
  #plt.plot(num_of_examples, precision_train, label='Precision for train data')
  #plt.plot(num_of_examples, recall_train, label='Recall for train data')
  #plt.plot(num_of_examples, F1_train, label='F-measure for train data where β=1')
  plt.xlabel('Number of examples')
  plt.title('Classifiers')
  plt.axis([0, 30000, 0, 1.1])
  plt.legend()
  plt.show()


main()