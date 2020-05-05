
# coding: utf-8

# # Desafio 4
# 
# Neste desafio, vamos praticar um pouco sobre testes de hipóteses. Utilizaremos o _data set_ [2016 Olympics in Rio de Janeiro](https://www.kaggle.com/rio2016/olympic-games/), que contém dados sobre os atletas das Olimpíadas de 2016 no Rio de Janeiro.
# 
# Esse _data set_ conta com informações gerais sobre 11538 atletas como nome, nacionalidade, altura, peso e esporte praticado. Estaremos especialmente interessados nas variáveis numéricas altura (`height`) e peso (`weight`). As análises feitas aqui são parte de uma Análise Exploratória de Dados (EDA).
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[137]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns


# In[138]:


#%matplotlib inline

#from IPython.core.pylabtools import figsize


#figsize(12, 8)

#sns.set()


# In[139]:


athletes = pd.read_csv("athletes.csv")


# In[140]:


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

# In[141]:


# Sua análise começa aqui.
athletes.head()


# In[142]:


athletes.shape


# In[143]:


athletes['height'].mode()


# In[144]:


athletes['height'].mean()


# In[145]:


athletes['height'].median()


# In[146]:


sample = get_sample(athletes, 'height', n=3000)


# ## Questão 1
# 
# Considerando uma amostra de tamanho 3000 da coluna `height` obtida com a função `get_sample()`, execute o teste de normalidade de Shapiro-Wilk com a função `scipy.stats.shapiro()`. Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# **1. Formulação da Hipótese:**
# 
#    $H_0$ = A amostra provém de uma população normal
#    
#    $H_1$ = A amostra provém de uma população normal
#    
# **2. Estabelecer o Nível de significância do teste ($\alpha$)**
#     
#    $\alpha$ = 0.05
#     
# **3. Calcular o coeficiente de Shapiro-Wilk e o p-valor**
# 
#     sct.shapiro(sample)
#     
# **4. Fazer a análise do resultado**
# 
#     Tomar a decisão: Rejeitar $H_0$ ao nível de significância  $\alpha$ se p > $\alpha$ 

# In[147]:


stats, p = sct.shapiro(sample)


# In[148]:


p


# In[149]:


alpha = 0.05


# In[150]:


def q1():
    alpha = 0.05
    sample = get_sample(athletes, 'height', n=3000)
    stats, p = sct.shapiro(sample)
    print(p)
    if p > alpha:
        return True  #Aceitar a hipotese nula (É uma distribuição normal)
    else:
        return False #Rejeitar a hipotese nula (Não é uma distribuição normal)
    


# In[151]:


q1()


# Nesse caso, observamos que o p-valor é menor que o nível de significância, ou seja devemos rejeitar a hipótese nula. Ou seja, não temos evidências suficientes para afirmar que a distribuição seja normal

# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?

# In[152]:


sample.hist(bins=25);


# * Plote o qq-plot para essa variável e a analise.

# In[153]:


import numpy as np 
import statsmodels.api as sm 
import pylab as py 


# In[154]:


sm.qqplot(sample, fit=True, line ='45') ;


# A distribuição dos dados sobre a reta no qqplot pode nos dar informações interessantes sobre o comportamento dos dados.
# Nesse caso verificamos que na parte inferior do gráfico temos pontos que não estão sobre a reta, inidica que existe uma cauda suave no lado esquerdo da curva (possivel verificar no histograma)

# ## Questão 2
# 
# Repita o mesmo procedimento acima, mas agora utilizando o teste de normalidade de Jarque-Bera através da função `scipy.stats.jarque_bera()`. Agora podemos afirmar que as alturas são normalmente distribuídas (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[155]:


stats, p = sct.jarque_bera(sample)
p


# In[156]:


def q2():
    alpha = 0.05
    sample = get_sample(athletes, 'height', n=3000)
    stats, p = sct.jarque_bera(sample)
    if p > alpha: 
        return True # Aceitar a hipotese nula (É uma distribuição normal) 
    else:
        return False #Rejeitar a hipotese nula (Não é uma distribuição normal)


# In[157]:


q2()


# __Para refletir__:
# 
# * Esse resultado faz sentido?
# 
# Novamente o p-valor foi menor do que o nível de significância, o que indica que devemos rejeitar a hipótese nula, ou seja não temos argumentos suficientes para indicar que a distribuição é normal

# ## Questão 3
# 
# Considerando agora uma amostra de tamanho 3000 da coluna `weight` obtida com a função `get_sample()`. Faça o teste de normalidade de D'Agostino-Pearson utilizando a função `scipy.stats.normaltest()`. Podemos afirmar que os pesos vêm de uma distribuição normal ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[158]:


sample_weight = get_sample(athletes, 'weight', n=3000)
stats, p = sct.normaltest(sample_weight)
p


# In[159]:


def q3():
    alpha = 0.05
    sample_weight = get_sample(athletes, 'weight', n=3000)
    stats, p = sct.normaltest(sample_weight)
    if p > alpha:
        return True #Aceitar a hipotese nula (É uma distribuição normal)
    else:
        return False #Rejeitar a hipotese nula (Não é uma distribuição normal)


# In[160]:


q3()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?

# In[161]:


sample_weight.hist(bins=25);


# O gráfico mostra que existe um prolongamento da cauda direita, ou seja não é uma distribuição normal conforme refutado pelo teste.
# 

# * Um _box plot_ também poderia ajudar a entender a resposta.

# In[162]:


sns.boxplot(sample_weight);


# O boxplot confirma que a distribuição da variavel weight não é uma distribuição normal, mostrando que há um prolongamento da cauda direita indicando uma assimetria positiva.

# In[163]:


sm.qqplot(sample_weight, fit=True, line="45");


# ## Questão 4
# 
# Realize uma transformação logarítmica em na amostra de `weight` da questão 3 e repita o mesmo procedimento. Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[164]:


weight_log = np.log(sample_weight)
stats, p = sct.normaltest(weight_log)
p


# In[165]:


def q4():
    alpha = 0.05
    sample_weight = get_sample(athletes, 'weight', n=3000)
    weight_log = np.log(sample_weight)
    stats, p = sct.normaltest(weight_log)
    if p > alpha:
        return True #Aceitar a hipotese nula (É uma distribuição normal)
    else:
        return False #Rejeitar a hipotese nula (Não é uma distribuição normal)


# In[166]:


q4()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Você esperava um resultado diferente agora?

# In[167]:


weight_log.hist(bins=25);


# In[168]:


sm.qqplot(weight_log, fit=True, line="45");


# In[169]:


sns.boxplot(weight_log);


# Visualmente os gráficos aplicando a transformação logarítimica se aproximaram mais da curva normal do que sem a transformação.
# O que é esperado pelo efeito do log, porém visualmente ainda conseguimos verificar a assimetria nos 3 gráficos mostrados. O valor p no teste normal confirma isso. Ainda conseguimos verificar o aumento do p-valor quando utilizamos os dados transformados.

# > __Para as questão 5 6 e 7 a seguir considere todos testes efetuados ao nível de significância de 5%__.

# ## Questão 5
# 
# Obtenha todos atletas brasileiros, norte-americanos e canadenses em `DataFrame`s chamados `bra`, `usa` e `can`,respectivamente. Realize um teste de hipóteses para comparação das médias das alturas (`height`) para amostras independentes e variâncias diferentes com a função `scipy.stats.ttest_ind()` entre `bra` e `usa`. Podemos afirmar que as médias são estatisticamente iguais? Responda com um boolean (`True` ou `False`).

# In[170]:


athletes.head()


# In[171]:


bra = athletes[athletes['nationality']=='BRA']
can = athletes[athletes['nationality']=='CAN']
usa = athletes[athletes['nationality']=='USA']


# In[172]:


stats_bc, p_braxcan = sct.ttest_ind(bra['height'], can['height'], equal_var=False, nan_policy='omit')


# In[173]:


stats_bu, p_braxusa = sct.ttest_ind(bra['height'], usa['height'], equal_var=False, nan_policy='omit')


# In[174]:


stats_cu, p_canxusa = sct.ttest_ind(can['height'], usa['height'], equal_var=False, nan_policy='omit')


# In[175]:


p_braxcan


# In[176]:


p_braxusa


# In[177]:


p_canxusa


# In[178]:


def q5():
    # Retorne aqui o resultado da questão 5.
    bra = athletes[athletes['nationality']=='BRA']
    usa = athletes[athletes['nationality']=='USA']
    stats_bu, p_braxusa = sct.ttest_ind(bra['height'], usa['height'], equal_var=False, nan_policy='omit')
    alpha = 0.05
    if p_braxusa > alpha:
        return True #Aceita a hipotese nula (As médias são iguais)
    else:
        return False #Rejeita a hipotese nula (As médias não são iguais)


# In[179]:


q5()


# In[180]:


bra['height'].mean()


# In[181]:


usa['height'].mean()


# In[182]:


bra['height'].std()


# In[183]:


usa['height'].std()


# In[184]:


sns.distplot(bra['height'].dropna(), label='Brasil', color='green')
sns.distplot(usa['height'].dropna(), label='USA', color ='blue')
plt.legend();


# O teste encontrou um p-valor menor que o valor de significância $\alpha$ o que indica deve-se rejeitar a hipótese nula. Ou seja as médias das distribuições entre o Brasil e os Estados Unidos não são iguais.

# ## Questão 6
# 
# Repita o procedimento da questão 5, mas agora entre as alturas de `bra` e `can`. Podemos afimar agora que as médias são estatisticamente iguais? Reponda com um boolean (`True` ou `False`).

# In[185]:


def q6():
    # Retorne aqui o resultado da questão 6.
    bra = athletes[athletes['nationality']=='BRA']
    can = athletes[athletes['nationality']=='CAN']
    stats_bu, p_braxcan = sct.ttest_ind(bra['height'], can['height'], equal_var=False, nan_policy='omit')
    alpha = 0.05
    if p_braxcan > alpha:
        return True #Aceita a hipotese nula (As médias são iguais)
    else:
        return False #Rejeita a hipotese nula (As médias não são iguais)


# In[186]:


q6()


# In[187]:


bra['height'].mean()


# In[188]:


can['height'].mean()


# In[189]:


bra['height'].std()


# In[190]:


can['height'].std()


# In[191]:


sns.distplot(bra['height'].dropna(), label='Brasil', color='green')
sns.distplot(can['height'].dropna(), label='Canada', color ='red')
plt.legend();


# O teste obteve um p-valor maior do que o valor de significância $\alpha$ isso indica que pode-se aceitar a hipotese nula, ou seja considerar as médias de altura entre os atletas do brasil e dos estados unidos igual.

# ## Questão 7
# 
# Repita o procedimento da questão 6, mas agora entre as alturas de `usa` e `can`. Qual o valor do p-valor retornado? Responda como um único escalar arredondado para oito casas decimais.

# In[192]:


def q7():
    usa = athletes[athletes['nationality']=='USA']
    can = athletes[athletes['nationality']=='CAN']
    stats_bu, p_usaxcan = sct.ttest_ind(usa['height'], can['height'], equal_var=False, nan_policy='omit')
    alpha = 0.05
    return round(p_usaxcan,8)


# In[193]:


q7()


# In[194]:


usa['height'].mean()


# In[195]:


can['height'].mean()


# In[196]:


usa['height'].std()


# In[197]:


can['height'].std()


# In[198]:


sns.distplot(usa['height'].dropna(), label='USA', color='blue')
sns.distplot(can['height'].dropna(), label='Canada', color ='red')
plt.legend();


# O p-valor obtido no teste é menor que o valor de significância escolhido $\alpha$=0,05 o que refuta a hipótese nula, ou seja,
# não se pode afirmar  que as médias são iguais

# __Para refletir__:
# 
# O resultado mostra que as médias de altura entre os atletas do Brasil e do Canadá são estatísticamente iguais e a média de altura dos atletas dos Estados Unidos é maior do que as do atletas do Brasil e do Canadá.
