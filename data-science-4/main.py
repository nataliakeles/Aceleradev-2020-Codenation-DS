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
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfTransformer, TfidfVectorizer
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder, Binarizer, KBinsDiscretizer,
    MinMaxScaler, StandardScaler, PolynomialFeatures
)


# In[2]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

#from IPython.core.pylabtools import figsize


#figsize(12, 8)

#sns.set()


# In[3]:


countries = pd.read_csv("countries.csv")


# In[4]:


countries['Region'] = countries['Region'] = countries['Region'].apply(lambda y: y.strip())


# In[5]:


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

# In[6]:


countries.tail()


# In[7]:


countries.shape


# In[8]:


Geral  = pd.DataFrame({'Colunas': countries.columns, 
              'Tipo': countries.dtypes, 
              'Qtd NaN': countries.isna().sum(), 
              '% Qtd NaN': countries.isna().sum()/countries.shape[0],
              'Qtd Valores Únicos': countries.nunique()})
Geral 


# In[9]:


countries.describe()


# A tabela geral nos mostra diversas variáveis foram cadastradas com o tipo errada (são variáveis numéricas cadastradas como variáveis categóricas), comprovado pelo método describe que retorna valores apenas para variáveis numéricas. Provavelmente esse erro ocorreu devido ao uso do separador ',' ao invés do ponto.

# In[10]:


for col in countries.select_dtypes(include='object').columns:
    countries[col] = countries[col].str.replace(',','.')


# In[11]:


countries.dtypes


# In[12]:


change_category = countries.select_dtypes(include='object').columns
change_category


# In[13]:


change_category = ['Pop_density', 'Coastline_ratio', 'Net_migration',
       'Infant_mortality', 'Literacy', 'Phones_per_1000', 'Arable', 'Crops',
       'Other', 'Climate', 'Birthrate', 'Deathrate', 'Agriculture', 'Industry',
       'Service']


# In[14]:


countries[change_category] = countries[change_category].astype('float64')


# In[15]:


countries.dtypes


# In[16]:


countries.head()


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[17]:


def q1():
    return list(sorted(countries['Region'].unique()))


# In[18]:


q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[19]:


def q2():
    #Instanciando a discretização em 10 intervalos utilizando encode ordinal e a esrtratégia quantile
    kbins_disc = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
    #ajuste das variáveis da feature Pop_density
    kbins_disc.fit(countries[['Pop_density']])
    score = kbins_disc.transform(countries[["Pop_density"]])
    # Quantidade maiores que 90
    return score[score>=9].shape[0]


# A binarização é uma operação comum em dados de contagem de texto em que o analista pode decidir considerar apenas a presença ou ausência de um recurso em vez de um número quantificado de ocorrências, por exemplo.
# 
# Também pode ser usado como uma etapa de pré-processamento para estimadores que consideram variáveis, aleatórias booleanas (por exemplo, modeladas usando a distribuição de Bernoulli em uma configuração bayesiana).

# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[20]:


def q3():
    return  countries[['Region']].nunique(dropna=False)[0] + countries[['Climate']].nunique(dropna=False)[0]


# In[21]:


q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[22]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]

#Dataframe de teste 
df_test_country = pd.DataFrame([test_country], columns=countries.columns)

df_test_country


# In[23]:


# definindo o pipeline 
pipe = Pipeline(steps=[('imputer', SimpleImputer(missing_values=np.nan, strategy='median')), 
                       ('scaler', StandardScaler())])

#fit & transform
pipe.fit(countries.drop(columns=['Country','Region'], axis=1))
test_country_numeric = pipe.transform(df_test_country.drop(columns=['Country','Region'], axis=1))


# In[24]:


df_test = pd.DataFrame(test_country_numeric, columns=df_test_country.drop(columns=['Country','Region'], axis=1).columns)
df_test


# In[25]:


def q4():
    return round(df_test['Arable'][0],3)


# In[26]:


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

# In[27]:


#Visualização utilizando boxplot
sns.boxplot(countries['Net_migration'], orient='vertical')


# In[28]:


sns.distplot(countries['Net_migration']);


# In[63]:


#primeiro quartil 0.25
qt1 = countries['Net_migration'].quantile(0.25)

#terceiro quartil 0.75
qt3 = countries['Net_migration'].quantile(0.75)

#Amplitude Interquartil
IQR = qt3 - qt1


# In[64]:


#Ourliers Superiores
higher_outliers_point = qt3 + 1.5*IQR

#Ourliers Inferiores
lower_outliers_point = qt1 - 1.5*IQR


# In[31]:


higher_outliers = countries[countries['Net_migration'] > higher_outliers_point]['Net_migration']
higher_outliers


# In[32]:


#Registros abaixo do ponto lower_outlieres_point

lower_outliers = countries[countries['Net_migration'] < lower_outliers_point]['Net_migration']
lower_outliers 


# In[33]:


#Registros abaixo do ponto lower_outlieres_point

higher_outliers = countries[countries['Net_migration'] > higher_outliers_point]['Net_migration']
higher_outliers


# In[34]:


countries.shape


# In[35]:


total_outliers = len(higher_outliers) + len(lower_outliers)
total_outliers 


# Cerca de 22% dos dados são outliers, portanto iria haver muita perda de informação se eles fossem excluídos

# In[36]:


def q5():
    return(len(lower_outliers), len(higher_outliers),False)


# In[37]:


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

# In[38]:


from sklearn.datasets import fetch_20newsgroups


# In[39]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroups = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# Para começar a usar o TfidfTransformer, primeiro você terá que criar um CountVectorizer para contar o número de palavras (termo frequência), limitar o tamanho do seu vocabulário, aplicar palavras de parada e etc

# In[40]:


countvectorizer = CountVectorizer()


# In[41]:


#Contagem das palavras no corpus
word_count_vector = countvectorizer.fit_transform(newsgroups.data)


# In[48]:


#Verificando o tamanho (palavras presetes nos bancos)
word_count_vector.shape


# Significa que esse corpus possui 1773 documentos (número de linhas) e  27335 palavras únicas (número de colunas)

# Agora vamos calcular os valores da IDF

# In[61]:


tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector) #tfidf ajustado de acordo com a contagem de palavras


# In[62]:


df_idf = pd.DataFrame(tfidf_transformer.idf_, index=countvectorizer.get_feature_names(),columns=["idf_weights"])
 
df_idf.sort_values(by=['idf_weights'])


# O dataframe acima indica o valor de idf para cada palavra em ordem crescente. Quanto menor o valor IDF de uma palavra, menos exclusivo é para qualquer documento em particular, por exemplo 'from' e subject aparece em todos os documentos

# Depois de obter os valores do IDF, agora você pode calcular as pontuações tf-idf para qualquer documento ou conjunto de documentos.

# In[51]:


tfidf_vectorizer = TfidfVectorizer(use_idf=True)
tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(newsgroups.data)


# In[52]:


tfidf_transformer.fit(word_count_vector)
newsgroups_tfidf = tfidf_transformer.transform(word_count_vector)


# In[53]:


#calculando tdif da palavra phone
tdif_phone = newsgroups_tfidf.getcol(countvectorizer.vocabulary_.get('phone')) 
tdif_phone.sum()


# Encontrando a quantidade de palavras existentes em cada documento

# In[54]:


fitted_vectorizer=tfidf_vectorizer.fit(newsgroups.data)
tfddidf_vectorizer_vectors=fitted_vectorizer.transform(newsgroups.data)


# In[55]:


words_idx = sorted([countvectorizer.vocabulary_.get('phone')])
words = pd.DataFrame(word_count_vector.toarray(), columns=countvectorizer.get_feature_names())


# In[56]:


words['phone'].sum()


# In[57]:


def q6():
    return words['phone'].sum()


# In[58]:


q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[ ]:





# In[59]:


def q7():
    return round(tdif_phone.sum(),3)


# In[60]:


q7()


# In[ ]:





# In[ ]:


tdif_phone.sum()


# In[ ]:




