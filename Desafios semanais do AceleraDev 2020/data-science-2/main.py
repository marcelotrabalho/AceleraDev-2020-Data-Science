#!/usr/bin/env python
# coding: utf-8

# # Desafio 4
# 
# Neste desafio, vamos praticar um pouco sobre testes de hipóteses. Utilizaremos o _data set_ [2016 Olympics in Rio de Janeiro](https://www.kaggle.com/rio2016/olympic-games/), que contém dados sobre os atletas das Olimpíadas de 2016 no Rio de Janeiro.
# 
# Esse _data set_ conta com informações gerais sobre 11538 atletas como nome, nacionalidade, altura, peso e esporte praticado. Estaremos especialmente interessados nas variáveis numéricas altura (`height`) e peso (`weight`). As análises feitas aqui são parte de uma Análise Exploratória de Dados (EDA).
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[127]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm


# In[128]:


#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[167]:


athletes = pd.read_csv("athletes.csv")


# In[130]:


def get_sample(df, col_name, n=100, seed=42):
    """Get a sample from a column of a dataframe.
    
    It drops any numpy.nan entries before sampling. The sampling
    is performed without replacement.
    
    Example of numpydoc for those who haven't seen yet.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    col_name : str
        Name of the column to be sampled.
    n : int
        Sample size. Default is 100.
    seed : int
        Random seed. Default is 42.
    
    Returns
    -------
    pandas.Series
        Sample of size n from dataframe's column.
    """
    np.random.seed(seed)
    
    random_idx = np.random.choice(df[col_name].dropna().index, size=n, replace=False)
    
    return df.loc[random_idx, col_name]


# ## Inicia sua análise a partir daqui

# In[131]:


# Sua análise começa aqui.
athletes.shape


# In[132]:


athletes.info()


# In[133]:


athletes.head()


# In[134]:


athletes.nationality.value_counts()


# In[135]:


athletes.describe()


# ## Questão 1
# 
# Considerando uma amostra de tamanho 3000 da coluna `height` obtida com a função `get_sample()`, execute o teste de normalidade de Shapiro-Wilk com a função `scipy.stats.shapiro()`. Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[136]:


df_heigh = get_sample(athletes,'height',3000)


# In[137]:


def q1():
    # Retorne aqui o resultado da questão 1.
    shapiro_result = sct.shapiro(df_heigh)
    return bool(shapiro_result[1] > 0.05)


# In[138]:


q1()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Plote o qq-plot para essa variável e a analise.
# * Existe algum nível de significância razoável que nos dê outro resultado no teste? (Não faça isso na prática. Isso é chamado _p-value hacking_, e não é legal).

# In[139]:


sns.distplot(df_heigh,bins=25)
plt.show()


# Não são condizentes, porque me parece muito com uma distribuição normal.
# Fazendo com uma distribuição menor, vejo que o shapiro aponta para uma distribuição normal.
# Não permitindo invalidar a Hipótese nula

# In[140]:


shapiro_result = sct.shapiro(get_sample(athletes,'height',150))
shapiro_result[1] > 0.05


# In[141]:


sm.qqplot(df_heigh,fit=True,line='45')
plt.show()


# O Gráfico qq-plot sugere que segue uma distribuição normal

# ## Questão 2
# 
# Repita o mesmo procedimento acima, mas agora utilizando o teste de normalidade de Jarque-Bera através da função `scipy.stats.jarque_bera()`. Agora podemos afirmar que as alturas são normalmente distribuídas (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[142]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return bool(sct.jarque_bera(df_heigh)[1] > 0.05)


# In[143]:


q2()


# __Para refletir__:
# 
# * Esse resultado faz sentido?

# In[144]:


sct.jarque_bera(df_heigh)


# Não faz sentido porquê por este teste não é uma distribuição normal

# ## Questão 3
# 
# Considerando agora uma amostra de tamanho 3000 da coluna `weight` obtida com a função `get_sample()`. Faça o teste de normalidade de D'Agostino-Pearson utilizando a função `scipy.stats.normaltest()`. Podemos afirmar que os pesos vêm de uma distribuição normal ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[145]:


weight_sample = get_sample(athletes,'weight',3000)


# In[146]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return bool(sct.normaltest(weight_sample)[1] > 0.05)


# In[147]:


q3()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Um _box plot_ também poderia ajudar a entender a resposta.

# In[148]:


f, (eixo1,eixo2) = plt.subplots(1,2)
sns.distplot(weight_sample, bins=25,ax=eixo1)
sns.boxplot(weight_sample,ax = eixo2)
plt.show()


# <strong>Olhando a plotagem não parece uma distribuição normal, confirmando o teste de Person.</strong>

# ## Questão 4
# 
# Realize uma transformação logarítmica em na amostra de `weight` da questão 3 e repita o mesmo procedimento. Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[149]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return bool(sct.normaltest(np.log(weight_sample))[1] > 0.05)


# In[150]:


q4()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Você esperava um resultado diferente agora?

# In[151]:


f, (eixo1,eixo2) = plt.subplots(1,2)
sns.distplot(np.log(weight_sample), bins=25,ax=eixo1)
sns.boxplot(np.log(weight_sample),ax = eixo2)
plt.show()


# Com a logaritima, a distribuição ficou mais normalizada, mas ainda sim tenho muitos outliers
# como visto no boxplot

# > __Para as questão 5 6 e 7 a seguir considere todos testes efetuados ao nível de significância de 5%__.

# ## Questão 5
# 
# Obtenha todos atletas brasileiros, norte-americanos e canadenses em `DataFrame`s chamados `bra`, `usa` e `can`,respectivamente. Realize um teste de hipóteses para comparação das médias das alturas (`height`) para amostras independentes e variâncias diferentes com a função `scipy.stats.ttest_ind()` entre `bra` e `usa`. Podemos afirmar que as médias são estatisticamente iguais? Responda com um boolean (`True` ou `False`).

# In[171]:


bra = athletes.query('nationality == "BRA"').height.dropna()
usa = athletes.query('nationality == "USA"').height.dropna()
can = athletes.query('nationality == "CAN"').height.dropna()


# In[173]:


bra.mean(), can.mean(), usa.mean()


# In[174]:


bra.size, usa.size, can.size


# In[175]:


def q5():
    # Retorne aqui o resultado da questão 5.
    return bool(sct.stats.ttest_ind(bra,usa,equal_var=False)[1] > 0.05)


# In[176]:


q5()


# ## Questão 6
# 
# Repita o procedimento da questão 5, mas agora entre as alturas de `bra` e `can`. Podemos afimar agora que as médias são estatisticamente iguais? Reponda com um boolean (`True` ou `False`).

# In[177]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return bool(sct.stats.ttest_ind(bra,can,equal_var=False)[1] > 0.05)


# In[178]:


q6()


# ## Questão 7
# 
# Repita o procedimento da questão 6, mas agora entre as alturas de `usa` e `can`. Qual o valor do p-valor retornado? Responda como um único escalar arredondado para oito casas decimais.

# In[179]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return float(round(sct.ttest_ind(usa,can,equal_var=False)[1],8))


# In[180]:


q7()


# __Para refletir__:
# 
# * O resultado faz sentido?
# * Você consegue interpretar esse p-valor?
# * Você consegue chegar a esse valor de p-valor a partir da variável de estatística?

# Esse resultado me diz que as médias não são próximas nem idênticas entre as alturas de usa e can.
# <br/>Isso porque o p-value é menor que área de aceitação > 0.05
# <br/>Já entre Brasil e canadá tem e vendo as médias independentes, pode ver que elas são bem próximas, evidenciando o objetivo da sct.stats.ttest_ind que é exatamente ver se as médias são próximas

# In[ ]:




