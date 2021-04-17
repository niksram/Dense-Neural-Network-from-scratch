# PESU-MI_1903_1957_1972

import numpy as np
import pandas as pd

def one_hot(df,l,h,col): # performs one hot encoding on categorical attributes
    df[col].fillna(df[col].mode(),inplace=True) #replace NaN with mode of categorical columns
    d={col+str(i):[] for i in range(l,h+1)}
    for i in df.index: #traverses dataset according to index
        for r in range(l,h+1): # sets the appropriate one hot value for the corresponding categories of the column
            if(df[col][i]==r):
                d[col+str(r)].append(1)
            else:
                d[col+str(r)].append(0)
    for i in d: # creating the new columns
        df[i]=d[i]
    del df[col] # deleting original categorical column

def return_normalized(df,col): # performs normalisation of data on numerical values
    return (df[col]-df[col].mean())/df[col].std()

def clean_data(X): # the main wrapper function that performs data pre processing 
    df=X
    one_hot(df,1,4,"Community") # one hot encoding
    one_hot(df,1,2,"Delivery phase")
    one_hot(df,1,2,"Residence")
    df["Education"].fillna(df["Education"].mode()[0],inplace=True)
    df["Education"]=df["Education"]/10  #min max normalization for scaled variable Education
    mean_list=["Age","Weight","HB","BP"]
    for col in mean_list: # the NaN values are replaced with mean and are normalised
        df[col].fillna(df[col].mean(),inplace=True)
        df[col]=return_normalized(df,col)
    return df

X = pd.read_csv('Original_Dataset.csv')
y=X.pop('Result')
X=clean_data(X)
X['Result']=y
X.to_csv('../data/pre_processed.csv')