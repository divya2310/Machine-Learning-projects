import streamlit as st

import pickle

## importing the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb') ,'UTF-8')

st.title("Laptop Predictor")

## collecting info from the user

## brand
company = st.selectbox('Brand',df['Company'].unique())

## type of laptop
type = st.selectbox('Type',df['TypeName'].unique())

## ram
Ram = st.selectbox('Ram(in GB)',[2,4,6,8,12,16,24,32,64])


## tocuhscreen
Touchscreen = st.selectbox('Touchscreen',['No','Yes'])

## IPS
IPS = st.selectbox('IPS',['No','Yes'])

## screen_size
screen_size = st.number_input('Screen Size')

## resolution
resolution = st.selectbox('Screen Resolution',['1920*1080','1366*768',
                                               '1600*900','3840*1800','2880*1800',
                                               '2560*1600'])

## cpu brand

CPU = st.selectbox('Brand',df['Cpu Brand'].unique())

HDD =  st.selectbox('HDD(in GB)',['0','128','256','512','1024','2048'])

SSD  = st.selectbox('SSD(in GB)',['0','128','256','512','1024'])

GPU = st.selectbox('Brand',df['Brand Name'].unique())
OS = st.selectbox('OS',df['OS'].unique())

if st.button('Predict Price'):
    pass




