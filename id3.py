from math import log2
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow
import sys
import pandas as pd

num_words = 100
skip_top = 25
oov_char = -1

def main():
  sys.setrecursionlimit(5000)
  # Getting the information necessary from the command line
  if len(sys.argv) != 2:
    sys.exit("Usage: python id3.py {string D. Y for diagnostic test/N for default execution}")
    
  try:
    diagnostics = str(sys.argv[1])
  except ValueError:
    print("Not a string")
  
  if (diagnostics != 'N' and diagnostics != 'Y'):
    sys.exit("Incorrect input. Please choose between 'Y' and 'N'.")

  train, test, inverted_word_index, word_index = IMDB_dataset()
  properties_train = create_properties(train)
  properties_test = create_properties(test)

  if (diagnostics == 'N'):
    print("Executing the algorithm with train data (all examples)...")
    tree = id3(train, properties_train, inverted_word_index)
    print("Evaluating the algorithm with the test data...")
    classifiers = calculate_classifiers(tree, properties_test, word_index)
    print("The accuracy for the given dataset is: ", classifiers[0])
    print("The precision for the given dataset is: ", classifiers[1])
    print("The recall for the given dataset is: ", classifiers[2])
    print("The F-measure with β=1 for the given dataset is: ", classifiers[3])
  elif (diagnostics == 'Y'):
    diagnostic_tests(train, properties_train, properties_test, inverted_word_index, word_index)


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


def calculate_entropy(prob):
  if (prob == 0 or prob == 1):
    return 0
  else:
    return - (prob * log2(prob)) - ((1.0 - prob) * log2(1.0 - prob))


def calculate_IG(properties):
    # Number of examples
    numOfExamples = len(properties) 
    # Number of features (-1 for last column which is C)
    numOfFeatures = len(properties[0]) - 1

    # Information gain for every feature
    IG = []

    # How many examples are positive (C=1)
    positives = 0
    for i in range(numOfExamples):
      if properties[i][numOfFeatures] == 1: positives +=1
    
    PC1 = positives / numOfExamples  # Probability of C=1, P(C=1)
    HC = calculate_entropy(PC1) # H(C)
    PX1 = [] # Probability of X=1, P(X=1). One for each feature
    PC1X1 = [] # Probability of C=1 given X=1, P(C=1|X=1). One for each feature
    PC1X0 = [] # Probability of C=1 given X=0, P(C=1|X=0). One for each feature
    HCX1 = [] # H(C=1|X=1)
    HCX0 = [] # H(C=1|X=0)

    for j in range(numOfFeatures):
      count_X1 = 0 # For every feature count the examples in which X=1
      count_C1X1 = 0 # For every feature count the examples in which C=1 given X=1
      count_C1X0 = 0 # For every feature count the examples in which C=1 given X=0

      for i in range(numOfExamples):
        if properties[i][j] == 1: count_X1 += 1
        if (properties[i][j] == 1 and properties[i][numOfFeatures] == 1): count_C1X1 += 1
        if (properties[i][j] == 0 and properties[i][numOfFeatures] == 1): count_C1X0 += 1

      PX1.append(count_X1 / numOfExamples)
      
      if (count_X1 == 0): PC1X1.append(0) # If all examples have X=0 then P(C=1|X=1) is 0
      else: PC1X1.append(count_C1X1 / count_X1)

      if (count_X1 == numOfExamples): PC1X0.append(0) # If all examples have X=1 then P(C=1|X=0) is 0
      else: PC1X0.append(count_C1X0 / (numOfExamples - count_X1))

      HCX1.append(calculate_entropy(PC1X1[j])) # Entropy for C when X is 1
      HCX0.append(calculate_entropy(PC1X0[j])) # Entropy for C when X is 0

      IG.append(HC - ((PX1[j] * HCX1[j]) + ((1.0 - PX1[j]) * HCX0[j])))

    return IG


def select_best_property(properties):
  IG = calculate_IG(properties)
  # Return the index of the property with the maximum IG from in the IG list
  return IG.index(max(IG))


def sub_tree(best_property, properties):
  # Create a dictionary for the given property where the keys are the values of the property (0 or 1) and 
  # the values (of the dictionary) represent the amount of times the property has that value
  property_dict = {}
  property_count = 0
  # Count the amount of times where property is 0
  for p in properties:
    if p[best_property] == 0:
      property_count += 1
  
  # Save the count variables in the dictionary
  property_dict[0] = property_count
  property_dict[1] = len(properties) - property_count
  
  tree = {}

  for property_value, count in property_dict.items():
    # Create a new list for every possible value of the property
    filtered_properties = []
    for p in properties:
      # Every list contains the reviews where the given property is equal to 'property_value'
      if p[best_property] == property_value:
        filtered_properties.append(p)
      
    # When for the given property value we are sure about the label (positive or negative) of the review the
    # 'pure_label' variable is equal to True. For example, if the word "bad" exists in a review (property_value = 1)
    # and every review that contains that word is negative, the the 'pure_label' variable will be equal to True
    pure_label = False
    for label in range(2):
      label_count = 0
      for p in filtered_properties:
        # For every review in the list that contains the reviews where the given property is
        # equal to 'property_value' keep track of the label count (for every label)
        if p[len(p) - 1] == label:
          label_count += 1
      
      if (count != 0):
        common_percentage = (label_count/count)  * 100
      else:
        common_percentage = 100

      # Check whether every review in the list mentioned above has the same label
      if common_percentage > 90:
        # If true add an element to the 'tree' dictionary with key the 'property_value' and with value the label.
        # In the example mentioned in row 168 that entry would be "1 : "0""
        tree[property_value] = str(label)
        # Remove all the reviews where the property is equal to the 'property_value' from the properties list
        for p in properties:
          if p[best_property] != property_value:
            properties.remove(p)
            pure_label = True
    
    # If the 'pure_label' variable is False add an entry to the dictionary with the key 'property_value' and with value 'Extend'
    if not pure_label:
        tree[property_value] = "Extend"
    
  return tree, properties


def create_tree(root, previous_property_value, properties, inverted_word_index):
  # Check whether the properties list is empty after updating it
  if len(properties) != 0:
    best_property = select_best_property(properties)
    tree, properties = sub_tree(best_property, properties)
    next_root = None

    ''' 
    Tree/ Root form
    {
      Bad: 
      {
        0: 
        {
          Good:
          {
            ...
          }
        }
        1: "0"
      }
    }
    '''

    # Create an entry with key the word that gives us 
    # the best IG and value the sub_tree of that word
    if previous_property_value == None: 
      root[inverted_word_index[best_property + skip_top]] = tree
      next_root = root[inverted_word_index[best_property + skip_top]]
    else:
      root[previous_property_value] = dict()
      root[previous_property_value][inverted_word_index[best_property + skip_top]] = tree
      next_root = root[previous_property_value][inverted_word_index[best_property + skip_top]]
    
    # Node = a value of a word in the tree (0 or 1)
    # Branch = it's equal either to "Extend" or to ("0" or "1")
    for node, branch in next_root.items():
      # If the tree needs to be further extended
      if branch == "Extend":
        # Use only the reviews where the 'best_property' is equal to node. In other words use only the 
        # reviews that either contain the word we are looking at or don't (based on the value of the 'node' variable)
        filtered_properties = []
        for p in properties:
          if p[best_property] == node:
            filtered_properties.append(p)
        create_tree(next_root, node, filtered_properties, inverted_word_index)


def id3(examples, properties, inverted_word_index, default_C=0):

  # If there are no examples return the default category
  if len(examples) == 0:
    return default_C
  
  # Check whether all the examples belong in the same category
  # and keep count of the examples in each category
  flag = True
  count_0 = 0
  for i in range(len(examples[1])):
    if flag and examples[1][0] != examples[1][i]:
      flag = False
    if examples[1][i] == 0:
      count_0 += 1
  
  count_1 = len(examples[1]) - count_0

  # If all the examples belong in the same category
  # return it
  if flag:
    return examples[1][0]

  # If there are no properties return the category with the most examples
  if len(properties) == 0:
    return 0 if count_0 >= count_1 else 1

  tree = {}
  create_tree(tree, None, properties, inverted_word_index)
  return tree


def predict(tree, example, word_index):
  if isinstance(tree, str):
    return tree
  else:
    root = next(iter(tree))
    property_value = example[word_index[root] - skip_top]
    if property_value in tree[root]:
      return predict(tree[root][property_value], example, word_index)
    else:
      return None
  

def calculate_classifiers(tree, properties_test, word_index, default_C=0):
  correct_predictions = 0
  wrong_predictions = 0
  true_positives = 0
  false_positives = 0
  false_negatives = 0
  for p in properties_test:
    result = predict(tree, p, word_index)
    if result == "1":
      result = 1
    elif result == "0":
      result = 0
    else:
      result = default_C

    if result == p[-1]:
      correct_predictions += 1
    else:
      wrong_predictions += 1

    if result == 1 and p[-1] == 1:
      true_positives += 1
    if result == 1 and p[-1] == 0:
      false_positives += 1
    if result == 0 and p[-1] == 1:
      false_negatives += 1

  #Calculating the classifiers
  accuracy = correct_predictions / (correct_predictions + wrong_predictions)
  precision = true_positives / (true_positives + false_positives)
  recall = true_positives / (true_positives + false_negatives)
  F1 = (2 * precision * recall) / (precision + recall)

  return accuracy, precision, recall, F1


def diagnostic_tests(train, properties_train, properties_test, inverted_word_index, word_index):
  accuracy_train = []
  accuracy_test = []
  precision_train = []
  recall_train = []
  F1_train = []
  num_of_examples = []
  
  partitions = 5
  for i in range(partitions, 0, -1):
    num_of_examples.append((len(properties_train)//partitions * (partitions - i + 1)))
    print("Executing the algorithm with train data (" + str(num_of_examples[partitions-i]) + " reviews)...")
    tree = id3(train, properties_train[:num_of_examples[partitions-i]], inverted_word_index)
    classifiers_train = calculate_classifiers(tree, properties_train[:num_of_examples[partitions-i]], word_index)
    accuracy_train.append(classifiers_train[0])
    precision_train.append(classifiers_train[1])
    recall_train.append(classifiers_train[2])
    F1_train.append(classifiers_train[3])
    classifiers_test = calculate_classifiers(tree, properties_test, word_index)
    accuracy_test.append(classifiers_test[0])

  print(pd.DataFrame({'Reviews': num_of_examples, 'Accuracy (Train)': accuracy_train, 'Accuracy (Test)': accuracy_test}))
  print(pd.DataFrame({'Reviews': num_of_examples, 'Precision (Train)': precision_train, 'Recall (Train)': recall_train, 'F1 (Train)': F1_train}))
  plt.plot(num_of_examples, accuracy_train, label='Total train accuracy')
  plt.plot(num_of_examples, accuracy_test, label='Total test accuracy')
  plt.plot(num_of_examples, precision_train, label='Precision for train data')
  plt.plot(num_of_examples, recall_train, label='Recall for train data')
  plt.plot(num_of_examples, F1_train, label='F-measure for train data where β=1')
  plt.xlabel('Number of examples')
  plt.title('Classifiers')
  plt.axis([0, 30000, 0, 1])
  plt.legend()
  plt.show()


main()
