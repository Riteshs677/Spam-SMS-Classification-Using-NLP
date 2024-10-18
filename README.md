#  Spam SMS Classification Using NLP [98% Accuracy]

## About Dataset

### Data Collection
Kaggle
This corpus has been collected from free or free for research sources at the Internet:
The SMS Spam Collection is a public set of SMS labeled messages that have been collected for mobile phone spam research.


## Imports

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, LSTM, Dense, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
```

## Model Implementation

The LSTM model is built using TensorFlow or PyTorch. The model architecture can be adjusted based on hyperparameter tuning to improve accuracy.
Key Libraries Required

TensorFlow or PyTorch for building the LSTM model.
Pandas for data manipulation.
Seaborn and Matplotlib for data visualization

## Conclusion

This project demonstrates how to effectively use LSTM networks for spam detection in SMS messages. The achieved accuracy can be further improved through techniques such as hyperparameter tuning, additional feature engineering, and using advanced architectures.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
