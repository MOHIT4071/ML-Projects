#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

file_path = 'train.txt'

with open(file_path, 'r') as file:
    data = file.read()

lines = data.strip().split('\n')

ids = []
times = []
outputs = []
features = [] 
for i in range(0, len(lines)):
    
    if(i%11!=10):
    
        feature_values = [float(value) for value in lines[i].split(',')[2:27]]
        id_, time = map(int, lines[i].split(',')[:2])
        ids.append(id_)
        times.append(time)
        features.append(feature_values)
        
    else:
        output = int(lines[i])
        print(output)
        
        outputs.append(output)


features_columns = [f'Feature_{i}' for i in range(1, 26)]
columns = ['ID', 'Time'] + features_columns 

df_new = pd.DataFrame(columns=columns)

for i in range(0, len(df), 10):
    df_slice = df.iloc[i:i+10]
    df_slice = df_slice.set_index('ID')
    df_slice.columns = [f'{col}_{j}' for j in range(1, 11) for col in df_slice.columns]
    df_new = pd.concat([df_new, df_slice], axis=0)

df_new = df_new.reset_index().drop(columns=['index'])
print(df_new.shape)
print(df_new.head())


# In[ ]:


df_new = pd.DataFrame(features)
df_dict = {'ID': ids, 'Time': times}
df_new2 = pd.DataFrame(df_dict)
frames = [df_new2,df_new]
df = pd.concat(frames,axis = 1)
df.head()


# In[ ]:


df3_dict = {'Output':outputs}
df3 = pd.DataFrame(df3_dict)
df3


# In[ ]:


import pandas as pd
df2 = pd.read_csv('test.txt')
columns = ['ID', 'Time Stamp'] + [f'{i}' for i in range(25)]
df2.columns = columns
df2


# In[ ]:


import pandas as pd
timestamp_value = 10
filtered_df = df[df['Time'] == timestamp_value]


# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Activation
input_shape = (25, 1)
X = filtered_df.iloc[:, 2:].values
y = df3['Output'].values 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train_cnn = X_train.reshape(-1, 25, 1)


def swish(x):
    return x * tf.keras.activations.sigmoid(x)
model = Sequential([
    Conv1D(32, 3, input_shape=(25, 1)),
    BatchNormalization(),
    Activation(swish),
    Conv1D(64, 3),
    BatchNormalization(),
    Activation(swish),
    Conv1D(128, 3),
    BatchNormalization(),
    Activation(swish),
    LSTM(64, return_sequences=False),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_cnn, y_train, epochs=10, batch_size=20, validation_split=0.15)
X_test_cnn = X_test.reshape(-1, 25, 1)
_, accuracy = model.evaluate(X_test_cnn, y_test)
print(f"Accuracy: {accuracy}")


# In[ ]:


X_2 = filtered_df2.iloc[:, 2:].values
X_2 = X_2.reshape(-1, 25, 1)
predicted_classes = model.predict(X_2)
predicted_classes = np.argmax(predicted_classes, axis=1)
print(predicted_classes)
import pandas as pd
y_pred_rf_thresholded_df = pd.DataFrame({'ID': range(50000, 50000 + len(predicted_classes)),
                                         'Column ID': predicted_classes})
y_pred_rf_thresholded_df.to_csv('y_pred_rf_thresholded.csv', index=False)

