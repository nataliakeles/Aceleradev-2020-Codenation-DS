
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[4]:


df = black_friday


# In[5]:


df.head()


# In[6]:


df_exploracao = pd.DataFrame({'nomes': df.columns, 'tipos':df.dtypes, 'Valores NaN': df.isna().sum()})


# In[7]:


df_exploracao['valores NaN (%)'] = df_exploracao['Valores NaN']/df.shape[0]


# In[8]:


df_exploracao


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[9]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return df.shape
    pass 


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[10]:


def q2():
    df_F = df[df['Gender']=='F']
    df_Age = df_F[df_F['Age']=='26-35']
    return df_Age['User_ID'].nunique()
    pass


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[11]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return df['User_ID'].nunique()
    pass


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[215]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return df_exploracao['tipos'].nunique()
    pass


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[26]:


def q5():
    # Retorne aqui o resultado da questão 5.
    count_NaN = df.shape[0] - len(df.dropna())
    porcent_NaN = count_NaN / df.shape[0]
    return porcent_NaN
    pass


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[27]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return df_exploracao[df_exploracao['Valores NaN']!=0]['Valores NaN'].max()
    pass


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[28]:


def q7():
    # Retorne aqui o resultado da questão 7.
    freq = df['Product_Category_3'].value_counts()
    
    return freq.index[0]
    
    pass


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[29]:


def q8():
    # Retorne aqui o resultado da questão 8.
    valor_max = df['Purchase'].max()
    valor_min = df['Purchase'].min()
    norm = (df['Purchase'] - valor_min) / (valor_max - valor_min)
    media = norm.mean()
    
    return media
    
    pass


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[30]:


def q9():
    # Retorne aqui o resultado da questão 9.   
    valor_max = df['Purchase'].max()
    valor_min = df['Purchase'].min()
    df['Purschase_padro'] = (df['Purchase'] - df['Purchase'].mean()) / df['Purchase'].std()
    df_padro = df[df['Purschase_padro']>-1]
    df_padro = df_padro[df_padro['Purschase_padro']<1]
    
    return df_padro.shape[0]
    
    pass


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[31]:


def q10():
    # Retorne aqui o resultado da questão 10.
    # Denfine quais indices referentes a Product_Category_2 são nulos
    nulos_1 = df.loc[ df.isnull()['Product_Category_2'] == True] 
    
    # Denfine quais indices referentes a Product_Category_3 são nulos
    nulos_2 = df.loc[ df.isnull()['Product_Category_2'] == True]
    
    #Compara se os indices são iguais
    comum = nulos_1.index == nulos_2.index
    
    
    #Se não existir elemento Falso no vetor comum
    # implica que em ambas colunas os elementos são nulos
    
    if (False not in comum) == True:
        resp_bool = True 
    else:
        resp_bool = False

    return resp_bool
    pass

