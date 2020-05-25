#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler


# In[2]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[3]:


countries = pd.read_csv("countries.csv")


# In[4]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[5]:


# Sua análise começa aqui.
countries.isna().sum()


# In[6]:


countries.info()


# * Etapas necessárias <br/>
# 1 - Retirar os espaços em branco da coluna Region<br/>
# 2 - Converter os valores númericos que possuem vírgula para float e as que não possuem em int64<br/>
# 3 - Retirar nulos da coluna Climate<br/>

# ### Passo 1 - Retirar os espaços da coluna Region

# In[7]:


# função que vai retirar os espaços em branco do texto passado. Vou usar num lambda
def retirarEspacos(texto):
    return str.strip(texto)


# In[8]:


# faço um lambda para retirar os espaços da coluna Region
countries['Region'] = countries.apply(lambda x: retirarEspacos(x['Region']),axis=1 )


# ### Passo 2 - Converter as variáveis adequadamente conforme configuração

# In[9]:


# Função que converterá todos as colunas com número de object como float
def converterFloat(valor):
    if type(valor) == str:
        return float(valor.replace(',','.'))
    return valor


# In[10]:


colunas = countries.columns[4:]
for i,l in countries.iterrows():
    for coluna in colunas:
        retorno = converterFloat(l[coluna])
        #print('%s - %s - %s' % (coluna,retorno,type(retorno)))
        countries.at[i,coluna] = retorno
        


# In[11]:


for coluna in colunas:
    countries[coluna]=countries[coluna].astype('float')


# In[12]:


countries.info()


# ### Passo 3 - Retirar nulos da coluna Climate

# In[13]:


media = countries['Climate'].mean()
countries['Climate'].fillna(media,inplace=True)


# In[14]:


countries['Climate'].unique()


# In[15]:


countries.head()


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[16]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return list(countries.Region.sort_values(ascending=True).unique())
q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[17]:


def q2():
    # Retorne aqui o resultado da questão 2.
    discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
    discretizer.fit(countries[["Pop_density"]])
    score_bins = discretizer.transform(countries[["Pop_density"]])
    return int((score_bins >= 9.).sum())
q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[18]:


def q3():
    # Retorne aqui o resultado da questão 3.
    # realizo o onehotencoder e aqui já tirei os nulos do climate
    one_hot_encoder_sparse = OneHotEncoder(sparse=True) # sparse=True é o default.

    course_encoded_sparse = one_hot_encoder_sparse.fit_transform(countries[["Region",'Climate']])

    return course_encoded_sparse.shape[1]
q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[19]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[20]:


teste = pd.DataFrame([test_country], columns=countries.columns)
teste


# In[21]:


def q4():
    # Retorne aqui o resultado da questão 4.
    colunas = countries.select_dtypes(['int64','float64']).columns
    pipeline = Pipeline(steps=[("imputer",SimpleImputer(strategy="median")),('scaler',StandardScaler())])    
    transformer = ColumnTransformer(transformers = [('number', pipeline,colunas)], n_jobs=-1)    
    transformer.fit(countries)
    resultado = transformer.transform(teste)
    return round(float(resultado[0][colunas.get_loc('Arable')]),3)
q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[ ]:





# In[22]:


def q5():
    # Retorne aqui o resultado da questão 4.
    quantil1,quantil3 = countries['Net_migration'].quantile([.25,.75])
    iqr = quantil3 - quantil1
    
    outlier_inferior = quantil1 - 1.5 * iqr
    outlier_superior = quantil3 + 1.5 * iqr
    
    outlier_abaixo = int(countries[countries['Net_migration'] < outlier_inferior].shape[0])
    outlier_acima = int(countries[countries['Net_migration'] > outlier_superior].shape[0])
    devo_remover = bool(((outlier_abaixo+outlier_acima)/int(countries['Net_migration'].shape[0])) < 0.1)
    return (outlier_abaixo, outlier_acima,devo_remover)
q5()    


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[23]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[24]:


def q6():
    count_vectorizer = CountVectorizer()
    result = count_vectorizer.fit_transform(newsgroup.data)
    return int(result[:,count_vectorizer.vocabulary_.get("phone")].sum())
    
q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[25]:


def q7():
    tfid=TfidfVectorizer()
    result = tfid.fit_transform(newsgroup.data)
    return float(result[:, tfid.vocabulary_.get('phone')].sum().round(3))
q7()


# In[ ]:




