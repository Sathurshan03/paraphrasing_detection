import json
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd

reference_dir = os.path.join('/app/input/', 'ref')
prediction_dir = os.path.join('/app/input/', 'res')
score_dir = '/app/output/'

print('Reading prediction')

prediction = []

val_label_list = []

with open(os.path.join(prediction_dir, 'predictions.txt'), 'r') as prediction_file:
    
    prediction = [float(x) for x in prediction_file.read().splitlines()]
    ## 

    val_labels = pd.read_json(os.path.join(reference_dir, 'val_label.json'))

    ## convert to to float list
    val_label_list = val_labels['label'].tolist()




print('Checking Accuracy')
accuracy = (1.0 - mean_squared_error(val_label_list, prediction)) * 100.0
print('Scores:')

scores = {
    'accuracy': accuracy,
    'duration': 0.1
}
print(scores)

with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file:
    score_file.write(json.dumps(scores))
