import streamlit as st
import pandas as pd

def main():
    st.title('Hello world')
    file = st.file_uploader('Escolha o arquivo csv', type="csv")
    if file is not None:
        slider = st.slider('Valores Head dataframe', min_value=1,max_value=50)
        df = pd.read_csv(file)
        st.dataframe(df.head(slider))
        st.markdown('============')
        st.table(df.head(slider))
        #st.write(df.columns)
        st.table(df.groupby('species')['petal_width'].mean())
if __name__ == "__main__":
    main()