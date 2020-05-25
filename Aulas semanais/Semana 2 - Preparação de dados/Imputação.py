#import streamlit as st
import pandas as pd

def main():
    df = pd.read_csv('iris.csv',sep=',')
    print(df.shape[0], df.shape[1])

if __name__ == "__main__":
    main()