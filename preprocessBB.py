from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np 

# Load the NIDS dataset
df1 = pd.read_csv('datasets/Normal_data.csv')
df2 = pd.read_csv('datasets/metasploitable-2.csv')
df = df1.merge(df2,how='outer')

#putting underscored in place of spaces in the column names
cols = df.columns
cols = cols.map(lambda x: x.replace(' ', '_'))
df.columns = cols
print(df.head())

#dropping some label values
df = df.drop(df[df['Label'].isin(['U2R','BFA','DoS'])].index)

#dropping some columns
df = df.drop(columns=['Timestamp', 'Flow_ID', 'Src_IP', 'Dst_IP'])
print('Timestamp, ', 'Flow_ID, ', 'Src_IP, ', 'Dst_IP ','columns are dropped')

#normalising the numerical features
numeric_col = df.select_dtypes(include='number').columns
std_scaler = StandardScaler()
def normalization(df,col):
  for i in col:
    arr = df[i]
    arr = np.array(arr)
    df[i] = std_scaler.fit_transform(arr.reshape(len(arr),1))
  return df

df = normalization(df.copy(),numeric_col)

# Prepare the dataset for feature selection
X = df.drop('Label', axis=1)
y = df['Label']

# Initialize the Random Forest Classifier
clf = RandomForestClassifier(random_state=42)

# Use RFE to select the top 25 features
#rfe = RFE(clf, n_features_to_select=25)
#rfe.fit(X, y)

# Get the names of the top 25 features
#top_features = X.columns[rfe.support_].tolist()
#print("Top 25 features:", top_features)

top_features = ['Src_Port', 'Dst_Port', 'Protocol', 'Fwd_Pkt_Len_Mean', 'Bwd_Pkt_Len_Max', 'Bwd_Pkt_Len_Mean', 'Flow_Pkts/s', 'Bwd_IAT_Tot', 'Bwd_IAT_Mean', 'Bwd_IAT_Max', 'Bwd_IAT_Min', 'Bwd_Header_Len', 'Fwd_Pkts/s', 'Bwd_Pkts/s', 'Pkt_Len_Max', 'Pkt_Len_Mean', 'Pkt_Len_Std', 'Pkt_Len_Var', 'SYN_Flag_Cnt', 'Pkt_Size_Avg', 'Fwd_Seg_Size_Avg', 'Subflow_Fwd_Byts', 'Subflow_Bwd_Pkts', 'Subflow_Bwd_Byts', 'Init_Bwd_Win_Byts']
X[top_features].to_csv('X_bb1.csv',index=False)
y.to_csv('Y_bb1.csv',index=False)
