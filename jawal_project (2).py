#!/usr/bin/env python
# coding: utf-8

# Introduction :
# Le nettoyage des données constitue une étape essentielle 
# dans tout projet d’analyse ou de data science. 
# Avant de pouvoir explorer, modéliser ou 
# les informations d’un jeu de données, 
# il est indispensable de s’assurer de sa qualité.
# Cette étape intervient généralement au début du processus, 
# juste après la collecte ou l’importation des données.
# Elle permet d’identifier et de corriger les valeurs manquantes, les doublons,
# les erreurs de saisie, les incohérences et les valeurs aberrantes.
# Un dataset propre et structuré garantit la fiabilité des analyses futures, 
# améliore la performance des modèles prédictifs et limite les biais.
# le nettoyage des données représente le fondement sur lequel repose tout travail analytique rigoureux interpréter

# on va commencer par l'importation des bib pandas afin de rendre facile le traitement des données tabulaires (comme les fichiers Excel ou CSV) et le calcul statistique.

# In[2]:


import pandas as pd


# voici le chemin de note base de données

# In[3]:


data=pd.read_csv(r'C:\Users\PC\Documents\maroc_telecom_data.csv.xls')


# Maintenant, on va jeter un coup d’œil à nos données.

# In[4]:


data


# le nombre des ligne et des colonnes 

# In[5]:


data.shape


# voici les types des données de chaque colonne

# In[6]:


data.dtypes


# In[7]:


#les colonnes qui contiennent des valeurs manquants
data.isna().sum()


# In[8]:


# Afficher seulement les colonnes avec des valeurs manquante
les_valeur_manquantes=data.isna().sum()
les_valeur_manquantes=les_valeur_manquantes[les_valeur_manquantes>1]
les_valeur_manquantes


# In[9]:


#les étiquettes des lignes
data.index


# In[10]:


#le nombre des doublons
data.duplicated().sum()


# In[11]:


#la diversité des données dans chaque colonne
for col in data.columns:
    print(col, "→", data[col].nunique(), "valeurs uniques")


# In[12]:


#résumé rapide des données
data.info()


# In[14]:


#les doubpons
data[data.duplicated()]


# In[15]:


#tous les lignes repetes
data[data.duplicated(keep=False)]


# In[18]:


#supprimer les doublons
data = data.drop_duplicates()


# In[19]:


#Pourcentage de valeurs manquantes par colonne
pourcentage = (data.isna().mean() * 100).round(2)
print(pourcentage[pourcentage > 0])


# In[20]:


#Afficher les lignes qui contiennent AU MOINS une valeur manquante
data[data.isna().any(axis=1)]


# In[26]:


#Afficher les lignes sans aucune valeur manquante (juste pour vérifier)
data[data.notna().all(axis=1)]


# In[22]:


# Remplir facture_moyenne_dhs par la moyenne
data['facture_moyenne_dhs'] = data['facture_moyenne_dhs'].fillna(
    data['facture_moyenne_dhs'].mean()
)


# In[23]:


data["facture_moyenne_dhs"]


# In[24]:


# 2. commentaire_client → "inconnu"
data['commentaire_client'] = data['commentaire_client'].fillna("inconnu")


# In[25]:


data["commentaire_client"]


# In[27]:


# 3. age → médiane (valeurs manquantes + négatives)
median_age = data.loc[data['age'] >= 0, 'age'].median()

data.loc[data['age'] < 0, 'age'] = median_age
data['age'] = data['age'].fillna(median_age)


# In[28]:


data["age"].head(13)


# In[29]:


#Vérifier les types de chaque colonne
print(data.dtypes)


# In[30]:


data["historique_conso_data"]


# In[31]:


print(type(data["historique_conso_data"].iloc[0]))
print(data["historique_conso_data"].iloc[0])


# In[32]:


import numpy as np


# In[33]:


import re

def extraire_parentheses(ligne):
    # Si c'est déjà une liste → on ne touche pas
    if isinstance(ligne, list):
        return ligne

    # Si c'est un nombre → on ne touche pas
    if isinstance(ligne, (int, float)):
        return ligne

    # Sinon, on traite la ligne comme texte
    ligne_str = str(ligne)
    chiffres = re.findall(r"np\.float64\(([-+]?\d*\.\d+|\d+)\)", ligne_str)
    return [float(x) for x in chiffres] if chiffres else ligne

# Application uniquement sur cette colonne
data['historique_conso_data'] = data['historique_conso_data'].apply(extraire_parentheses)



# In[35]:


data["historique_conso_data"]


# In[36]:


# Voir les lignes qui ne sont pas des listes
lignes_problemes = data[~data['historique_conso_data'].apply(lambda x: isinstance(x, list))]
print(lignes_problemes)


# In[37]:


def contient_pas_num(x):
    if isinstance(x, list):
        return any(not isinstance(v, (int, float)) for v in x)
    return False

print(data[data['historique_conso_data'].apply(contient_pas_num)])


# In[38]:


def moyenne_sure(x):
    if isinstance(x, list) and len(x) > 0:
        nums = [v for v in x if isinstance(v, (int, float))]
        return sum(nums)/len(nums) if len(nums) > 0 else None
    return None

data['historique_conso_data'] = data['historique_conso_data'].apply(moyenne_sure)


# In[39]:


data.head(72)


# In[40]:


data['historique_conso_data'] = data['historique_conso_data'].fillna(0)


# In[41]:


data.head(72)


# In[42]:


# Afficher seulement les colonnes avec des valeurs manquantes
missing_values = data.isna().sum()
missing_values = missing_values[missing_values > 0]
print(missing_values)


# In[43]:


data[data.isna().any(axis=1)]


# In[44]:


data['commentaire_client'] = data['commentaire_client'].str.lower().str.strip()


# In[45]:


median_age = data.loc[data['age'] < 120, 'age'].median()
data['age'] = data['age'].apply(lambda x: median_age if x >= 120 else x)


# In[47]:


data.head(72)


# In[48]:


data['ville'] = data['ville'].str.strip().str.title()
data['sexe'] = data['sexe'].str.lower().str.strip()
data['type_offre'] = data['type_offre'].str.strip()


# In[49]:


#un résumé statistique
data.describe()


# Une fois le nettoyage des données terminé, nous pouvons poursuivre avec l’étape suivante du projet, 
# disponible sur notre compte GitHub.

# In[ ]:




