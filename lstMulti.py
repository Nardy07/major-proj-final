#importing required libraries
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential,save_model
from keras.layers import LSTM, Dense, Dropout,Conv1D,MaxPooling1D
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.utils import class_weight

#reading the csv files(preprocessed data and labels), selecting only the features extracted via RFC and storing the required in X and y
data = pd.read_csv('X_bb1.csv')
labels = pd.read_csv('Y_bb1.csv')
#features = ['Src_Port', 'Dst_Port', 'Protocol', 'Flow_Duration', 'Tot_Bwd_Pkts', 'TotLen_Fwd_Pkts', 'Fwd_Pkt_Len_Mean', 'Bwd_Pkt_Len_Max', 'Bwd_Pkt_Len_Mean', 'Bwd_IAT_Tot', 'Bwd_IAT_Mean', 'Bwd_IAT_Max', 'Bwd_IAT_Min', 'Bwd_Header_Len', 'Bwd_Pkts/s', 'Pkt_Len_Max', 'Pkt_Len_Mean', 'Pkt_Len_Std', 'Pkt_Len_Var', 'SYN_Flag_Cnt', 'Pkt_Size_Avg', 'Fwd_Seg_Size_Avg', 'Subflow_Fwd_Byts', 'Subflow_Bwd_Byts', 'Init_Bwd_Win_Byts']
#X = data.values
y = labels['Label'].values
le = LabelEncoder()
y = le.fit_transform(y)
#scaling
scaler = StandardScaler()
X = scaler.fit_transform(data)
#converting labels to one hot encoded format
num_classes = 3
y = to_categorical(y,num_classes=num_classes)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)


X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

##y_train = to_categorical(y_train,num_classes=num_classes)
#y_test = to_categorical(y_test,num_classes=num_classes)

model = Sequential()
model.add(LSTM(units=80, input_shape=(1,X_train.shape[2]), return_sequences=True,kernel_regularizer='l2'))
model.add(Dropout(0.2))
model.add(LSTM(units=40,kernel_regularizer='l2'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define the early stopping monitor and patience
early_stopping_monitor = 'loss'
patience = 10

# Create an instance of the EarlyStopping callback
early_stopping = EarlyStopping(monitor=early_stopping_monitor, patience=patience, verbose=1, restore_best_weights=True)

# Add the early stopping callback to the model's callbacks
callbacks = [early_stopping]

# Train the model with early stopping
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks)


y_pred = model.predict(X_test)
y_pred = np.round(y_pred)

print(classification_report(y_test, y_pred,zero_division='warn'))

#saving the model 
#save_model(model,filepath='/home/iskadoodle/Downloads/InSDN_DatasetCSV/micMultiBB1',save_format='tf')
model.save('micmultiBB1.keras')
