import json
from random import shuffle

def split_json(filename, test_size=0.2):
  """
  Splits a JSON file into training and test sets.

  Args:
      filename: Path to the JSON file.
      test_size: Proportion of data for the test set (default: 0.2).

  Returns:
      A dictionary containing training and test data as lists.
  """

  with open(filename, 'r') as f:
    data = json.load(f)

  shuffle(data)  # Shuffle data for random distribution

  split_point = int(len(data) * (1 - test_size))
  trainn_data = data[:split_point]
  valid_data = data[split_point:]

  return {'trainn': trainn_data, 'valid': valid_data}

# Split the data
data = split_json('C:/Users/Beste/Documents/AdvancedAnalytics_Steam_images_DL/train_data.json')

with open('C:/Users/Beste/Documents/AdvancedAnalytics_Steam_images_DL/trainn_data.json', 'w') as f:
  json.dump(data['trainn'], f)

with open('C:/Users/Beste/Documents/AdvancedAnalytics_Steam_images_DL/valid_data.json', 'w') as f:
  json.dump(data['valid'], f)

print("Data split completed! Training and valid sets created.")