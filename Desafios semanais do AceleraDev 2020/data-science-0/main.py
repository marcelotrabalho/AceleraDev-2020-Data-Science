#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[3]:


black_friday.info()


# In[4]:


black_friday.head()


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[5]:


def q1():
    return black_friday.shape    


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[11]:


def q2():
    return black_friday.query('Age == "26-35" & Gender=="F"')['User_ID'].count()


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[7]:


def q3():
    return black_friday['User_ID'].nunique()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[223]:


def q4():
    return black_friday.dtypes.nunique()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[224]:


def q5():
    a = black_friday['Product_Category_3'].isna().sum() / black_friday.shape[0]
    return float(a)


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[225]:


def q6():
    return int(black_friday['Product_Category_3'].isna().sum())


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[226]:


def q7():
    return black_friday.groupby('Product_Category_3').count().sort_values(by='User_ID',ascending=False).index[0]


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[233]:


def q8():
    # Retorne aqui o resultado da questão 8.
    valor_maximo = black_friday['Purchase'].max()
    valor_minimo = black_friday['Purchase'].min()
    result = (black_friday['Purchase'] - valor_minimo)/(valor_maximo - valor_minimo)
    return float(result.mean())


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[262]:


def q9():
    # Retorne aqui o resultado da questão 9.
    compra = black_friday['Purchase']
    padronizacao = (compra - compra.mean()) / (compra.std())
    return int(((padronizacao >= -1) & (padronizacao <= 1)).sum())


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[282]:


def q10():
    # Retorne aqui o resultado da questão 10.
    nulo_2_e_3 = black_friday.query('Product_Category_2.isnull() & Product_Category_3.isnull()', engine='python')['User_ID'].count()
    nulo_2= black_friday['Product_Category_2'].isna().sum()
    if nulo_2_e_3==nulo_2:
        return True
    return False


# In[ ]:




