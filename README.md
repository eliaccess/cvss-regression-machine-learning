# Dataset and problematic

The dataset given is nammed CVE. It is available right [here](https://www.kaggle.com/datasets/andrewkronser/cve-common-vulnerabilities-and-exposures "source dataset cve"), but a clean version can be downloaded at the root of the repository.

This dataset contains several columns, each of them listing data about security breach and cyber attacks. They are nammed using their CWE name (Common Weakness Enumeration, which is a list of software vulnerabilities).

The criticity of these 90 000+ breaches is evaluated with the CVSS, which goes from 0 (no risk) to 10 (extremely critical). The Common Vulnerability Scoring System is an evaluation standard of the vulnerabilities criticity based on objective and measurable factors.

Two simple approaches are available to see what Machine Learning can do:
- regression, to try to evaluate the exact value of the CVSS
- classification, if the CVSS value is split into classes (LOW, MEDIUM, HIGH, CRITICAL for example)

In this Notebook, we will try the regressive solution, because it allows to avoid the information loss between two elements of the same class.

However, the original dataset is impossible to use, because lots of comments have been written in it, breaking the CSV format. Therefore a clean version is available at the root of the project, deleting about 2000 broken rows.

# Imports and constants
## Librairies


```python
# NUMPY
import numpy as np

# STATS
import scipy.stats as stats
from scipy.stats import norm, skew, kstest

# MATPLOTLIB
import matplotlib as mlp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
%matplotlib inline 
# plt.style.use('fivethirtyeight') 

# PLOTLY 
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import plot

# WARNINGS
import warnings
warnings.filterwarnings('ignore')

# PANDAS
import pandas as pd 
pd.set_option("display.max_rows", None, "display.max_columns", None) 

# SEABORN
import seaborn as sns

# SCIKIT-LEARN: SELECTION DE VARIABLES
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression

# SCIKIT-LEARN: PRE-PROCESSING
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder # Encodage des variables catégorielles ordinales
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder # Encodage des variables catégorielles nominales
from sklearn.preprocessing import StandardScaler # Normalisation des variables numériques
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer # Imputation
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# SCIKIT-LEARN: MODELES
from sklearn.dummy import DummyClassifier
from sklearn import linear_model # Classe Modèle linéaire 
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet # Régression linéaire
from sklearn.linear_model import LogisticRegression, SGDRegressor, LinearRegression # Régression logistique
from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR # Machines à vecteurs de support

# SCIKIT-LEARN: VALIDATION CROISEE + OPTIMISATION
from sklearn.model_selection import train_test_split # Découpage en données train et test
from sklearn.model_selection import cross_val_score # Validation croisée pour comparaison entre modèles
from sklearn.model_selection import validation_curve # Courbe de validation: visulaisr les scores lors du choix d'un hyperparamétre
from sklearn.model_selection import GridSearchCV # Tester plusieurs hyper_paramètres
from sklearn.model_selection import learning_curve # Courbe d'apprentissage: visualisation les scores du train et du validation sets en fonction des quanitiés des données
 
## EVALUATION
from sklearn import metrics
from sklearn.metrics import accuracy_score # Exactitude (accuracy)
from sklearn.metrics import f1_score # F1-score
from sklearn.metrics import confusion_matrix # Matrice de confusion
from sklearn.metrics import plot_confusion_matrix # Graphique de la matrice de confusion
from sklearn.metrics import classification_report # Rapport pour le modèle de classification
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

## EVALUATION: COURBE ROC
from sklearn.metrics import auc # Aire sous la courbe 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve 

# SCHIKIT-LEARN: PIPELINE et TRANSFORMATEUR
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

## ARBRES, FORETS, APRRENTISSAGE D'ENSEMBLE
from sklearn.tree import DecisionTreeClassifier # Arbres de décision (classification)
from sklearn.ensemble import RandomForestClassifier # Forêts aléatoires (classification)
from sklearn.ensemble import BaggingClassifier # Classifier Bagging (classification)
from sklearn.ensemble import AdaBoostClassifier # Classifier Adaboost (classification)
from sklearn.ensemble import GradientBoostingClassifier  # Gradient de boosting (classification)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

# VALIDATION CROISEE + OPTIMISATION
from sklearn.model_selection import train_test_split # Séparation des données en train et test set
from sklearn.model_selection import cross_val_score # Validation croisée pour comparaison entre modèles
from sklearn.model_selection import validation_curve # Courbe de validation: visulaisr les scores lors du choix d'un hyperparamétre
from sklearn.model_selection import GridSearchCV # tester plusieurs hyperparamètres
from sklearn.model_selection import RandomizedSearchCV # tester arbitrairement plusieurs hyperparamètres
from sklearn.model_selection import learning_curve # courbe d'apprentissage: visualisation les scores du train et du validation sets en fonction des quanitiés des données
 
# WARNINGS
import warnings
warnings.filterwarnings('ignore')
```

## Constants definition


```python
# Setting of the random state to reproduce identically tests
_RANDOM_STATE_ = 7
# Target name
targetName = 'cvss'
```

## Dataset import


```python
dataset = pd.read_csv("cve.csv", sep = ';')
```

# Exploratory data analysis


```python
 # Copy of the dataset
df_data = dataset.copy()
```


```python
df_data.head()
```





  <div id="df-0515d2ed-4fae-4f7f-9e4c-04570cbe81cf">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cve</th>
      <th>mod_date</th>
      <th>pub_date</th>
      <th>cvss</th>
      <th>cwe_code</th>
      <th>cwe_name</th>
      <th>summary</th>
      <th>access_authentication</th>
      <th>access_complexity</th>
      <th>access_vector</th>
      <th>impact_availability</th>
      <th>impact_confidentiality</th>
      <th>impact_integrity</th>
      <th>Unnamed: 13</th>
      <th>Unnamed: 14</th>
      <th>Unnamed: 15</th>
      <th>Unnamed: 16</th>
      <th>Unnamed: 17</th>
      <th>Unnamed: 18</th>
      <th>Unnamed: 19</th>
      <th>Unnamed: 20</th>
      <th>Unnamed: 21</th>
      <th>Unnamed: 22</th>
      <th>Unnamed: 23</th>
      <th>Unnamed: 24</th>
      <th>Unnamed: 25</th>
      <th>Unnamed: 26</th>
      <th>Unnamed: 27</th>
      <th>Unnamed: 28</th>
      <th>Unnamed: 29</th>
      <th>Unnamed: 30</th>
      <th>Unnamed: 31</th>
      <th>Unnamed: 32</th>
      <th>Unnamed: 33</th>
      <th>Unnamed: 34</th>
      <th>Unnamed: 35</th>
      <th>Unnamed: 36</th>
      <th>Unnamed: 37</th>
      <th>Unnamed: 38</th>
      <th>Unnamed: 39</th>
      <th>Unnamed: 40</th>
      <th>Unnamed: 41</th>
      <th>Unnamed: 42</th>
      <th>Unnamed: 43</th>
      <th>Unnamed: 44</th>
      <th>Unnamed: 45</th>
      <th>Unnamed: 46</th>
      <th>Unnamed: 47</th>
      <th>Unnamed: 48</th>
      <th>Unnamed: 49</th>
      <th>Unnamed: 50</th>
      <th>Unnamed: 51</th>
      <th>Unnamed: 52</th>
      <th>Unnamed: 53</th>
      <th>Unnamed: 54</th>
      <th>Unnamed: 55</th>
      <th>Unnamed: 56</th>
      <th>Unnamed: 57</th>
      <th>Unnamed: 58</th>
      <th>Unnamed: 59</th>
      <th>Unnamed: 60</th>
      <th>Unnamed: 61</th>
      <th>Unnamed: 62</th>
      <th>Unnamed: 63</th>
      <th>Unnamed: 64</th>
      <th>Unnamed: 65</th>
      <th>Unnamed: 66</th>
      <th>Unnamed: 67</th>
      <th>Unnamed: 68</th>
      <th>Unnamed: 69</th>
      <th>Unnamed: 70</th>
      <th>Unnamed: 71</th>
      <th>Unnamed: 72</th>
      <th>Unnamed: 73</th>
      <th>Unnamed: 74</th>
      <th>Unnamed: 75</th>
      <th>Unnamed: 76</th>
      <th>Unnamed: 77</th>
      <th>Unnamed: 78</th>
      <th>Unnamed: 79</th>
      <th>Unnamed: 80</th>
      <th>Unnamed: 81</th>
      <th>Unnamed: 82</th>
      <th>Unnamed: 83</th>
      <th>Unnamed: 84</th>
      <th>Unnamed: 85</th>
      <th>Unnamed: 86</th>
      <th>Unnamed: 87</th>
      <th>Unnamed: 88</th>
      <th>Unnamed: 89</th>
      <th>Unnamed: 90</th>
      <th>Unnamed: 91</th>
      <th>Unnamed: 92</th>
      <th>Unnamed: 93</th>
      <th>Unnamed: 94</th>
      <th>Unnamed: 95</th>
      <th>Unnamed: 96</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CVE-2019-16548</td>
      <td>21/11/2019 15:15</td>
      <td>21/11/2019 15:15</td>
      <td>6.8</td>
      <td>352</td>
      <td>Cross-Site Request Forgery (CSRF)</td>
      <td>A cross-site request forgery vulnerability in ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CVE-2019-16547</td>
      <td>21/11/2019 15:15</td>
      <td>21/11/2019 15:15</td>
      <td>4.0</td>
      <td>732</td>
      <td>Incorrect Permission Assignment for Critical ...</td>
      <td>Missing permission checks in various API endpo...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CVE-2019-16546</td>
      <td>21/11/2019 15:15</td>
      <td>21/11/2019 15:15</td>
      <td>4.3</td>
      <td>639</td>
      <td>Authorization Bypass Through User-Controlled Key</td>
      <td>Jenkins Google Compute Engine Plugin 4.1.1 and...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CVE-2013-2092</td>
      <td>20/11/2019 21:22</td>
      <td>20/11/2019 21:15</td>
      <td>4.3</td>
      <td>79</td>
      <td>Improper Neutralization of Input During Web P...</td>
      <td>Cross-site Scripting (XSS) in Dolibarr ERP/CRM...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CVE-2013-2091</td>
      <td>20/11/2019 20:15</td>
      <td>20/11/2019 20:15</td>
      <td>7.5</td>
      <td>89</td>
      <td>Improper Neutralization of Special Elements u...</td>
      <td>SQL injection vulnerability in Dolibarr ERP/CR...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-0515d2ed-4fae-4f7f-9e4c-04570cbe81cf')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-0515d2ed-4fae-4f7f-9e4c-04570cbe81cf button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-0515d2ed-4fae-4f7f-9e4c-04570cbe81cf');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




The display of the first 5 lines of the dataset gives us an idea of its composition. Here, we notice the target, which is a floating continuous value, 'cwe_code' which is integer, and the others which seem either textual or missing.


```python
# Size of the dataset
n_samples, n_features = df_data.shape
print("Rows:", n_samples)
print("Columns:", n_features)
```

    Rows: 89531
    Columns: 97
    


```python
df_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 89531 entries, 0 to 89530
    Data columns (total 97 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   cve                     89531 non-null  object 
     1   mod_date                89531 non-null  object 
     2   pub_date                89531 non-null  object 
     3   cvss                    89531 non-null  float64
     4   cwe_code                89531 non-null  int64  
     5   cwe_name                89531 non-null  object 
     6   summary                 89531 non-null  object 
     7   access_authentication   85401 non-null  object 
     8   access_complexity       85401 non-null  object 
     9   access_vector           85401 non-null  object 
     10  impact_availability     85401 non-null  object 
     11  impact_confidentiality  85401 non-null  object 
     12  impact_integrity        85401 non-null  object 
     13  Unnamed: 13             0 non-null      float64
     14  Unnamed: 14             0 non-null      float64
     15  Unnamed: 15             0 non-null      float64
     16  Unnamed: 16             0 non-null      float64
     17  Unnamed: 17             0 non-null      float64
     18  Unnamed: 18             0 non-null      float64
     19  Unnamed: 19             0 non-null      float64
     20  Unnamed: 20             0 non-null      float64
     21  Unnamed: 21             0 non-null      float64
     22  Unnamed: 22             0 non-null      float64
     23  Unnamed: 23             0 non-null      float64
     24  Unnamed: 24             16 non-null     object 
     25  Unnamed: 25             16 non-null     object 
     26  Unnamed: 26             15 non-null     object 
     27  Unnamed: 27             15 non-null     object 
     28  Unnamed: 28             15 non-null     object 
     29  Unnamed: 29             15 non-null     object 
     30  Unnamed: 30             15 non-null     object 
     31  Unnamed: 31             14 non-null     object 
     32  Unnamed: 32             13 non-null     object 
     33  Unnamed: 33             12 non-null     object 
     34  Unnamed: 34             11 non-null     object 
     35  Unnamed: 35             11 non-null     object 
     36  Unnamed: 36             10 non-null     object 
     37  Unnamed: 37             10 non-null     object 
     38  Unnamed: 38             9 non-null      object 
     39  Unnamed: 39             9 non-null      object 
     40  Unnamed: 40             9 non-null      object 
     41  Unnamed: 41             9 non-null      object 
     42  Unnamed: 42             9 non-null      object 
     43  Unnamed: 43             9 non-null      object 
     44  Unnamed: 44             9 non-null      object 
     45  Unnamed: 45             9 non-null      object 
     46  Unnamed: 46             9 non-null      object 
     47  Unnamed: 47             9 non-null      object 
     48  Unnamed: 48             9 non-null      object 
     49  Unnamed: 49             9 non-null      object 
     50  Unnamed: 50             9 non-null      object 
     51  Unnamed: 51             9 non-null      object 
     52  Unnamed: 52             9 non-null      object 
     53  Unnamed: 53             9 non-null      object 
     54  Unnamed: 54             9 non-null      object 
     55  Unnamed: 55             9 non-null      object 
     56  Unnamed: 56             9 non-null      object 
     57  Unnamed: 57             9 non-null      object 
     58  Unnamed: 58             9 non-null      object 
     59  Unnamed: 59             9 non-null      object 
     60  Unnamed: 60             9 non-null      object 
     61  Unnamed: 61             9 non-null      object 
     62  Unnamed: 62             9 non-null      object 
     63  Unnamed: 63             9 non-null      object 
     64  Unnamed: 64             9 non-null      object 
     65  Unnamed: 65             8 non-null      object 
     66  Unnamed: 66             4 non-null      object 
     67  Unnamed: 67             4 non-null      object 
     68  Unnamed: 68             4 non-null      object 
     69  Unnamed: 69             4 non-null      object 
     70  Unnamed: 70             4 non-null      object 
     71  Unnamed: 71             4 non-null      object 
     72  Unnamed: 72             4 non-null      object 
     73  Unnamed: 73             4 non-null      object 
     74  Unnamed: 74             4 non-null      object 
     75  Unnamed: 75             4 non-null      object 
     76  Unnamed: 76             4 non-null      object 
     77  Unnamed: 77             4 non-null      object 
     78  Unnamed: 78             4 non-null      object 
     79  Unnamed: 79             4 non-null      object 
     80  Unnamed: 80             4 non-null      object 
     81  Unnamed: 81             4 non-null      object 
     82  Unnamed: 82             4 non-null      object 
     83  Unnamed: 83             3 non-null      object 
     84  Unnamed: 84             3 non-null      object 
     85  Unnamed: 85             3 non-null      object 
     86  Unnamed: 86             3 non-null      object 
     87  Unnamed: 87             3 non-null      object 
     88  Unnamed: 88             3 non-null      object 
     89  Unnamed: 89             3 non-null      object 
     90  Unnamed: 90             3 non-null      object 
     91  Unnamed: 91             3 non-null      object 
     92  Unnamed: 92             3 non-null      object 
     93  Unnamed: 93             3 non-null      object 
     94  Unnamed: 94             3 non-null      object 
     95  Unnamed: 95             3 non-null      object 
     96  Unnamed: 96             3 non-null      object 
    dtypes: float64(12), int64(1), object(84)
    memory usage: 66.3+ MB
    

Most of the columns have only a few non-zero values. They are therefore not usable as such, and we must question their usefulness. However, we can delete columns 13 to 23 which contain only null values.


```python
df_data.drop(df_data.iloc[:, 13:24], inplace = True, axis = 1)
```

We can now observe the behavior of the variables when the features 96 is not null:


```python
df_data.loc[df_data['Unnamed: 96'].notnull()].head()
```





  <div id="df-9f792491-3732-4f1a-b9f5-fedd24742a35">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cve</th>
      <th>mod_date</th>
      <th>pub_date</th>
      <th>cvss</th>
      <th>cwe_code</th>
      <th>cwe_name</th>
      <th>summary</th>
      <th>access_authentication</th>
      <th>access_complexity</th>
      <th>access_vector</th>
      <th>impact_availability</th>
      <th>impact_confidentiality</th>
      <th>impact_integrity</th>
      <th>Unnamed: 24</th>
      <th>Unnamed: 25</th>
      <th>Unnamed: 26</th>
      <th>Unnamed: 27</th>
      <th>Unnamed: 28</th>
      <th>Unnamed: 29</th>
      <th>Unnamed: 30</th>
      <th>Unnamed: 31</th>
      <th>Unnamed: 32</th>
      <th>Unnamed: 33</th>
      <th>Unnamed: 34</th>
      <th>Unnamed: 35</th>
      <th>Unnamed: 36</th>
      <th>Unnamed: 37</th>
      <th>Unnamed: 38</th>
      <th>Unnamed: 39</th>
      <th>Unnamed: 40</th>
      <th>Unnamed: 41</th>
      <th>Unnamed: 42</th>
      <th>Unnamed: 43</th>
      <th>Unnamed: 44</th>
      <th>Unnamed: 45</th>
      <th>Unnamed: 46</th>
      <th>Unnamed: 47</th>
      <th>Unnamed: 48</th>
      <th>Unnamed: 49</th>
      <th>Unnamed: 50</th>
      <th>Unnamed: 51</th>
      <th>Unnamed: 52</th>
      <th>Unnamed: 53</th>
      <th>Unnamed: 54</th>
      <th>Unnamed: 55</th>
      <th>Unnamed: 56</th>
      <th>Unnamed: 57</th>
      <th>Unnamed: 58</th>
      <th>Unnamed: 59</th>
      <th>Unnamed: 60</th>
      <th>Unnamed: 61</th>
      <th>Unnamed: 62</th>
      <th>Unnamed: 63</th>
      <th>Unnamed: 64</th>
      <th>Unnamed: 65</th>
      <th>Unnamed: 66</th>
      <th>Unnamed: 67</th>
      <th>Unnamed: 68</th>
      <th>Unnamed: 69</th>
      <th>Unnamed: 70</th>
      <th>Unnamed: 71</th>
      <th>Unnamed: 72</th>
      <th>Unnamed: 73</th>
      <th>Unnamed: 74</th>
      <th>Unnamed: 75</th>
      <th>Unnamed: 76</th>
      <th>Unnamed: 77</th>
      <th>Unnamed: 78</th>
      <th>Unnamed: 79</th>
      <th>Unnamed: 80</th>
      <th>Unnamed: 81</th>
      <th>Unnamed: 82</th>
      <th>Unnamed: 83</th>
      <th>Unnamed: 84</th>
      <th>Unnamed: 85</th>
      <th>Unnamed: 86</th>
      <th>Unnamed: 87</th>
      <th>Unnamed: 88</th>
      <th>Unnamed: 89</th>
      <th>Unnamed: 90</th>
      <th>Unnamed: 91</th>
      <th>Unnamed: 92</th>
      <th>Unnamed: 93</th>
      <th>Unnamed: 94</th>
      <th>Unnamed: 95</th>
      <th>Unnamed: 96</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>44010</th>
      <td>CVE-2017-17144</td>
      <td>29/03/2018 14:21</td>
      <td>05/03/2018 19:29</td>
      <td>5.0</td>
      <td>119</td>
      <td>Improper Restriction of Operations within the...</td>
      <td>Backup feature of SIP module in Huawei DP300 V...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>V100R001C10SPC300</td>
      <td>V100R001C10SPC500</td>
      <td>V100R001C10SPC600</td>
      <td>V100R001C10SPC700B010</td>
      <td>V100R001C10SPC800</td>
      <td>V500R002C00SPC200</td>
      <td>V500R002C00SPC500</td>
      <td>V500R002C00SPC600</td>
      <td>V500R002C00SPC700</td>
      <td>V500R002C00SPC900</td>
      <td>V500R002C00SPCb00</td>
      <td>V600R006C00</td>
      <td>TE40 V500R002C00SPC600</td>
      <td>V500R002C00SPC700</td>
      <td>V500R002C00SPC900</td>
      <td>V500R002C00SPCb00</td>
      <td>V600R006C00</td>
      <td>V600R006C00SPC200</td>
      <td>TE50 V500R002C00SPC600</td>
      <td>V500R002C00SPC700</td>
      <td>V500R002C00SPCb00</td>
      <td>V600R006C00</td>
      <td>V600R006C00SPC200</td>
      <td>TE60 V100R001C01SPC100</td>
      <td>V100R001C01SPC107TB010</td>
      <td>V100R001C10</td>
      <td>V100R001C10SPC300</td>
      <td>V100R001C10SPC400</td>
      <td>V100R001C10SPC500</td>
      <td>V100R001C10SPC600</td>
      <td>V100R001C10SPC700</td>
      <td>V100R001C10SPC800</td>
      <td>V100R001C10SPC900</td>
      <td>V500R002C00</td>
      <td>V500R002C00SPC100</td>
      <td>V500R002C00SPC200</td>
      <td>V500R002C00SPC300</td>
      <td>V500R002C00SPC600</td>
      <td>V500R002C00SPC700</td>
      <td>V500R002C00SPC800</td>
      <td>V500R002C00SPC900</td>
      <td>V500R002C00SPCa00</td>
      <td>V500R002C00SPCb00</td>
      <td>V500R002C00SPCd00</td>
      <td>V600R006C00</td>
      <td>V600R006C00SPC100</td>
      <td>V600R006C00SPC200</td>
      <td>V600R006C00SPC300</td>
      <td>TP3106 V100R002C00</td>
      <td>V100R002C00SPC200</td>
      <td>V100R002C00SPC400</td>
      <td>V100R002C00SPC600</td>
      <td>V100R002C00SPC700</td>
      <td>V100R002C00SPC800</td>
      <td>TP3206 V100R002C00</td>
      <td>V100R002C00SPC200</td>
      <td>V100R002C00SPC400</td>
      <td>V100R002C00SPC600</td>
      <td>V100R002C00SPC700</td>
      <td>V100R002C10</td>
      <td>ViewPoint 9030 V100R011C02SPC100</td>
      <td>V100R011C03B012SP15</td>
      <td>V100R011C03B012SP16</td>
      <td>V100R011C03B015SP03</td>
      <td>V100R011C03LGWL01SPC100</td>
      <td>V100R011C03SPC100</td>
      <td>V100R011C03SPC200</td>
      <td>V100R011C03SPC300</td>
      <td>V100R011C03SPC400</td>
      <td>V100R011C03SPC500</td>
      <td>eSpace U1960 V200R003C30SPC200</td>
      <td>eSpace U1981 V100R001C20SPC700</td>
      <td>V200R003C20SPCa00 has an overflow vulnerabili...</td>
    </tr>
    <tr>
      <th>44089</th>
      <td>CVE-2017-17143</td>
      <td>27/03/2018 20:41</td>
      <td>05/03/2018 19:29</td>
      <td>5.0</td>
      <td>119</td>
      <td>Improper Restriction of Operations within the...</td>
      <td>SIP module in Huawei DP300 V500R002C00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>V100R001C10SPC300</td>
      <td>V100R001C10SPC500</td>
      <td>V100R001C10SPC600</td>
      <td>V100R001C10SPC700B010</td>
      <td>V100R001C10SPC800</td>
      <td>V500R002C00SPC200</td>
      <td>V500R002C00SPC500</td>
      <td>V500R002C00SPC600</td>
      <td>V500R002C00SPC700</td>
      <td>V500R002C00SPC900</td>
      <td>V500R002C00SPCb00</td>
      <td>V600R006C00</td>
      <td>TE40 V500R002C00SPC600</td>
      <td>V500R002C00SPC700</td>
      <td>V500R002C00SPC900</td>
      <td>V500R002C00SPCb00</td>
      <td>V600R006C00</td>
      <td>V600R006C00SPC200</td>
      <td>TE50 V500R002C00SPC600</td>
      <td>V500R002C00SPC700</td>
      <td>V500R002C00SPCb00</td>
      <td>V600R006C00</td>
      <td>V600R006C00SPC200</td>
      <td>TE60 V100R001C01SPC100</td>
      <td>V100R001C01SPC107TB010</td>
      <td>V100R001C10</td>
      <td>V100R001C10SPC300</td>
      <td>V100R001C10SPC400</td>
      <td>V100R001C10SPC500</td>
      <td>V100R001C10SPC600</td>
      <td>V100R001C10SPC700</td>
      <td>V100R001C10SPC800</td>
      <td>V100R001C10SPC900</td>
      <td>V500R002C00</td>
      <td>V500R002C00SPC100</td>
      <td>V500R002C00SPC200</td>
      <td>V500R002C00SPC300</td>
      <td>V500R002C00SPC600</td>
      <td>V500R002C00SPC700</td>
      <td>V500R002C00SPC800</td>
      <td>V500R002C00SPC900</td>
      <td>V500R002C00SPCa00</td>
      <td>V500R002C00SPCb00</td>
      <td>V500R002C00SPCd00</td>
      <td>V600R006C00</td>
      <td>V600R006C00SPC100</td>
      <td>V600R006C00SPC200</td>
      <td>V600R006C00SPC300</td>
      <td>TP3106 V100R002C00</td>
      <td>V100R002C00SPC200</td>
      <td>V100R002C00SPC400</td>
      <td>V100R002C00SPC600</td>
      <td>V100R002C00SPC700</td>
      <td>V100R002C00SPC800</td>
      <td>TP3206 V100R002C00</td>
      <td>V100R002C00SPC200</td>
      <td>V100R002C00SPC400</td>
      <td>V100R002C00SPC600</td>
      <td>V100R002C00SPC700</td>
      <td>V100R002C10</td>
      <td>ViewPoint 9030 V100R011C02SPC100</td>
      <td>V100R011C03B012SP15</td>
      <td>V100R011C03B012SP16</td>
      <td>V100R011C03B015SP03</td>
      <td>V100R011C03LGWL01SPC100</td>
      <td>V100R011C03SPC100</td>
      <td>V100R011C03SPC200</td>
      <td>V100R011C03SPC300</td>
      <td>V100R011C03SPC400</td>
      <td>V100R011C03SPC500</td>
      <td>eSpace U1960 V200R003C30SPC200</td>
      <td>eSpace U1981 V100R001C20SPC700</td>
      <td>V200R003C20SPCa00 has an overflow vulnerabili...</td>
    </tr>
    <tr>
      <th>44090</th>
      <td>CVE-2017-17142</td>
      <td>27/03/2018 20:40</td>
      <td>05/03/2018 19:29</td>
      <td>5.0</td>
      <td>119</td>
      <td>Improper Restriction of Operations within the...</td>
      <td>SIP module in Huawei DP300 V500R002C00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>V100R001C10SPC300</td>
      <td>V100R001C10SPC500</td>
      <td>V100R001C10SPC600</td>
      <td>V100R001C10SPC700B010</td>
      <td>V100R001C10SPC800</td>
      <td>V500R002C00SPC200</td>
      <td>V500R002C00SPC500</td>
      <td>V500R002C00SPC600</td>
      <td>V500R002C00SPC700</td>
      <td>V500R002C00SPC900</td>
      <td>V500R002C00SPCb00</td>
      <td>V600R006C00</td>
      <td>TE40 V500R002C00SPC600</td>
      <td>V500R002C00SPC700</td>
      <td>V500R002C00SPC900</td>
      <td>V500R002C00SPCb00</td>
      <td>V600R006C00</td>
      <td>V600R006C00SPC200</td>
      <td>TE50 V500R002C00SPC600</td>
      <td>V500R002C00SPC700</td>
      <td>V500R002C00SPCb00</td>
      <td>V600R006C00</td>
      <td>V600R006C00SPC200</td>
      <td>TE60 V100R001C01SPC100</td>
      <td>V100R001C01SPC107TB010</td>
      <td>V100R001C10</td>
      <td>V100R001C10SPC300</td>
      <td>V100R001C10SPC400</td>
      <td>V100R001C10SPC500</td>
      <td>V100R001C10SPC600</td>
      <td>V100R001C10SPC700</td>
      <td>V100R001C10SPC800</td>
      <td>V100R001C10SPC900</td>
      <td>V500R002C00</td>
      <td>V500R002C00SPC100</td>
      <td>V500R002C00SPC200</td>
      <td>V500R002C00SPC300</td>
      <td>V500R002C00SPC600</td>
      <td>V500R002C00SPC700</td>
      <td>V500R002C00SPC800</td>
      <td>V500R002C00SPC900</td>
      <td>V500R002C00SPCa00</td>
      <td>V500R002C00SPCb00</td>
      <td>V500R002C00SPCd00</td>
      <td>V600R006C00</td>
      <td>V600R006C00SPC100</td>
      <td>V600R006C00SPC200</td>
      <td>V600R006C00SPC300</td>
      <td>TP3106 V100R002C00</td>
      <td>V100R002C00SPC200</td>
      <td>V100R002C00SPC400</td>
      <td>V100R002C00SPC600</td>
      <td>V100R002C00SPC700</td>
      <td>V100R002C00SPC800</td>
      <td>TP3206 V100R002C00</td>
      <td>V100R002C00SPC200</td>
      <td>V100R002C00SPC400</td>
      <td>V100R002C00SPC600</td>
      <td>V100R002C00SPC700</td>
      <td>V100R002C10</td>
      <td>ViewPoint 9030 V100R011C02SPC100</td>
      <td>V100R011C03B012SP15</td>
      <td>V100R011C03B012SP16</td>
      <td>V100R011C03B015SP03</td>
      <td>V100R011C03LGWL01SPC100</td>
      <td>V100R011C03SPC100</td>
      <td>V100R011C03SPC200</td>
      <td>V100R011C03SPC300</td>
      <td>V100R011C03SPC400</td>
      <td>V100R011C03SPC500</td>
      <td>eSpace U1960 V200R003C30SPC200</td>
      <td>eSpace U1981 V100R001C20SPC700</td>
      <td>V200R003C20SPCa00 has an overflow vulnerabili...</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-9f792491-3732-4f1a-b9f5-fedd24742a35')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-9f792491-3732-4f1a-b9f5-fedd24742a35 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-9f792491-3732-4f1a-b9f5-fedd24742a35');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




We notice that on the 3 non-zero rows of the feature 96, all the columns starting from 24 are filled.

## Continuous features (Target)


```python
con_features = df_data.select_dtypes(include=['float64']).columns
print(con_features)
```

    Index(['cvss'], dtype='object')
    


```python
print(f'{np.sort(df_data[targetName].unique())}')
```

    [ 0.   1.2  1.3  1.5  1.7  1.8  1.9  2.1  2.3  2.4  2.6  2.7  2.8  2.9
      3.   3.2  3.3  3.5  3.6  3.7  3.8  4.   4.1  4.3  4.4  4.6  4.7  4.8
      4.9  5.   5.1  5.2  5.3  5.4  5.5  5.6  5.7  5.8  5.9  6.   6.1  6.2
      6.3  6.4  6.5  6.6  6.7  6.8  6.9  7.   7.1  7.2  7.3  7.4  7.5  7.6
      7.7  7.8  7.9  8.   8.2  8.3  8.5  8.7  8.8  9.   9.3  9.4  9.7 10. ]
    

Only the target is a continuous column, going from 0 to 10. No transformation seems necessary to use it. However, for most Machine Learning models, it is necessary to check that the density curve of the target is approximately normal (Gaussian shape) in order to use it.


```python
sns.distplot(df_data[targetName], fit=norm)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f184da15350>




    
![png](output_23_1.png)
    


As we can see here, the target is not at all Gaussian (black curve: expected Gaussian, blue curve: target distribution). We will therefore have to try to reduce and center it with a log-transformation.


```python
con_features.drop(targetName)
```




    Index([], dtype='object')



We can remove the target from the continuous features, because we don't want to apply any pre-processing to them (scaling etc.).

## Categorical features


```python
cat_features = df_data.select_dtypes(include=['int64','object']).columns
print(cat_features)
```

    Index(['cve', 'mod_date', 'pub_date', 'cwe_code', 'cwe_name', 'summary',
           'access_authentication', 'access_complexity', 'access_vector',
           'impact_availability', 'impact_confidentiality', 'impact_integrity',
           'Unnamed: 24', 'Unnamed: 25', 'Unnamed: 26', 'Unnamed: 27',
           'Unnamed: 28', 'Unnamed: 29', 'Unnamed: 30', 'Unnamed: 31',
           'Unnamed: 32', 'Unnamed: 33', 'Unnamed: 34', 'Unnamed: 35',
           'Unnamed: 36', 'Unnamed: 37', 'Unnamed: 38', 'Unnamed: 39',
           'Unnamed: 40', 'Unnamed: 41', 'Unnamed: 42', 'Unnamed: 43',
           'Unnamed: 44', 'Unnamed: 45', 'Unnamed: 46', 'Unnamed: 47',
           'Unnamed: 48', 'Unnamed: 49', 'Unnamed: 50', 'Unnamed: 51',
           'Unnamed: 52', 'Unnamed: 53', 'Unnamed: 54', 'Unnamed: 55',
           'Unnamed: 56', 'Unnamed: 57', 'Unnamed: 58', 'Unnamed: 59',
           'Unnamed: 60', 'Unnamed: 61', 'Unnamed: 62', 'Unnamed: 63',
           'Unnamed: 64', 'Unnamed: 65', 'Unnamed: 66', 'Unnamed: 67',
           'Unnamed: 68', 'Unnamed: 69', 'Unnamed: 70', 'Unnamed: 71',
           'Unnamed: 72', 'Unnamed: 73', 'Unnamed: 74', 'Unnamed: 75',
           'Unnamed: 76', 'Unnamed: 77', 'Unnamed: 78', 'Unnamed: 79',
           'Unnamed: 80', 'Unnamed: 81', 'Unnamed: 82', 'Unnamed: 83',
           'Unnamed: 84', 'Unnamed: 85', 'Unnamed: 86', 'Unnamed: 87',
           'Unnamed: 88', 'Unnamed: 89', 'Unnamed: 90', 'Unnamed: 91',
           'Unnamed: 92', 'Unnamed: 93', 'Unnamed: 94', 'Unnamed: 95',
           'Unnamed: 96'],
          dtype='object')
    

All the columns seem to be categorized but the target.


```python
df_data["cwe_name"].value_counts()
```




     Improper Neutralization of Input During Web Page Generation ('Cross-site Scripting')           12325
     Improper Restriction of Operations within the Bounds of a Memory Buffer                        12325
     Improper Input Validation                                                                       7869
     Information Exposure                                                                            6592
     Permissions Privileges and Access Controls                                                      5813
     Improper Neutralization of Special Elements used in an SQL Command ('SQL Injection')            5740
     Improper Limitation of a Pathname to a Restricted Directory ('Path Traversal')                  3062
     Resource Management Errors                                                                      2960
     Cryptographic Issues                                                                            2552
     Cross-Site Request Forgery (CSRF)                                                               2415
     Improper Control of Generation of Code ('Code Injection')                                       2412
     Out-of-bounds Read                                                                              2105
     Improper Authentication                                                                         1801
     Improper Access Control                                                                         1658
     Numeric Errors                                                                                  1363
     Use After Free                                                                                  1340
     Integer Overflow or Wraparound                                                                  1134
     NULL Pointer Dereference                                                                         971
     Credentials Management                                                                           967
     Out-of-bounds Write                                                                              853
     Improper Neutralization of Special Elements used in an OS Command ('OS Command Injection')       829
     Concurrent Execution using Shared Resource with Improper Synchronization ('Race Condition')      643
     Uncontrolled Resource Consumption                                                                576
     7PK - Security Features                                                                          568
     Improper Link Resolution Before File Access ('Link Following')                                   541
     Improper Neutralization of Special Elements used in a Command ('Command Injection')              520
     Unrestricted Upload of File with Dangerous Type                                                  447
     Incorrect Permission Assignment for Critical Resource                                            423
     Improper Restriction of XML External Entity Reference                                            420
     Improper Certificate Validation                                                                  391
     Improper Privilege Management                                                                    390
     Neutralization of Special Elements in Output Used by a Downstream Component ('Injection')        366
     Use of Hard-coded Credentials                                                                    342
     Missing Release of Resource after Effective Lifetime                                             324
     Untrusted Search Path                                                                            301
     URL Redirection to Untrusted Site ('Open Redirect')                                              278
     Deserialization of Untrusted Data                                                                277
     Configuration                                                                                    248
     Data Processing Errors                                                                           238
     Use of Externally-Controlled Format String                                                       218
     Loop with Unreachable Exit Condition ('Infinite Loop')                                           217
     Server-Side Request Forgery (SSRF)                                                               207
     Incorrect Type Conversion or Cast                                                                196
     Insufficiently Protected Credentials                                                             186
     Double Free                                                                                      174
     DEPRECATED: Code                                                                                 163
     Inclusion of Sensitive Information in Log Files                                                  162
     Incorrect Authorization                                                                          156
     Permission Issues                                                                                155
     Session Fixation                                                                                 124
     Inadequate Encryption Strength                                                                   121
     Divide By Zero                                                                                   117
     Improper Authorization                                                                           117
     Buffer Copy without Checking Size of Input ('Classic Buffer Overflow')                           109
     Missing Authorization                                                                            106
     Missing Authentication for Critical Function                                                      98
     Incorrect Default Permissions                                                                     93
     Improper Verification of Cryptographic Signature                                                  82
     Allocation of Resources Without Limits or Throttling                                              81
     Cleartext Transmission of Sensitive Information                                                   81
     Reachable Assertion                                                                               80
     Improper Validation of Array Index                                                                78
     Uncontrolled Search Path Element                                                                  76
     Exposure of Resource to Wrong Sphere                                                              72
     Insufficient Verification of Data Authenticity                                                    72
     Use of a Broken or Risky Cryptographic Algorithm                                                  71
     Key Management Errors                                                                             65
     Improper Resource Shutdown or Release                                                             61
     Weak Password Recovery Mechanism for Forgotten Password                                           60
     Missing Encryption of Sensitive Data                                                              58
     Integer Underflow (Wrap or Wraparound)                                                            55
     7PK - Errors                                                                                      50
     Cleartext Storage of Sensitive Information                                                        49
     Improper Handling of Exceptional Conditions                                                       49
     Origin Validation Error                                                                           49
     Use of Insufficiently Random Values                                                               48
     Improper Initialization                                                                           48
     Improper Check for Unusual or Exceptional Conditions                                              42
     Insufficient Session Expiration                                                                   41
     Uncontrolled Recursion                                                                            40
     Unquoted Search Path or Element                                                                   39
     Insecure Default Initialization of Resource                                                       39
     Excessive Iteration                                                                               35
     Improper Neutralization of CRLF Sequences ('CRLF Injection')                                      34
     Use of Cryptographically Weak Pseudo-Random Number Generator (PRNG)                               32
     XML Injection (aka Blind XPath Injection)                                                         32
     Improper Preservation of Permissions                                                              31
     Protection Mechanism Failure                                                                      30
     Weak Password Requirements                                                                        29
     Incorrect Calculation                                                                             27
     Access of Uninitialized Pointer                                                                   27
     Direct Request ('Forced Browsing')                                                                25
     Improper Neutralization of CRLF Sequences in HTTP Headers ('HTTP Response Splitting')             24
     Improper Restriction of Excessive Authentication Attempts                                         24
     Authorization Bypass Through User-Controlled Key                                                  24
     Improperly Implemented Security Check for Standard                                                23
     Inconsistent Interpretation of HTTP Requests ('HTTP Request Smuggling')                           23
     Files or Directories Accessible to External Parties                                               21
     Missing Initialization of Resource                                                                20
     Information Exposure Through Discrepancy                                                          20
     Authentication Bypass by Spoofing                                                                 20
     Improper Neutralization of Argument Delimiters in a Command ('Argument Injection')                20
     Channel and Path Errors                                                                           19
     Externally Controlled Reference to a Resource in Another Sphere                                   18
     Incorrect Regular Expression                                                                      17
     Insufficient Entropy                                                                              17
     Use of Uninitialized Resource                                                                     17
     Information Exposure Through an Error Message                                                     14
     Incorrect Access of Indexable Resource ('Range Error')                                            13
     Improper Encoding or Escaping of Output                                                           12
     Improper Restriction of Rendered UI Layers or Frames                                              12
     Time-of-check Time-of-use (TOCTOU) Race Condition                                                 12
     Inclusion of Functionality from Untrusted Control Sphere                                          12
     Insufficient Entropy in PRNG                                                                      11
     Improper Neutralization of Special Elements used in an LDAP Query ('LDAP Injection')              11
     Download of Code Without Integrity Check                                                          11
     Use of Password Hash With Insufficient Computational Effort                                       10
     Insecure Storage of Sensitive Information                                                         10
     Encoding Error                                                                                     9
     Write-what-where Condition                                                                         8
     Improper Validation of Certificate with Host Mismatch                                              8
     Reliance on Cookies without Validation and Integrity Checking                                      8
     Improper Validation of Integrity Check Value                                                       8
     Authentication Bypass by Capture-replay                                                            8
     Access of Resource Using Incompatible Type ('Type Confusion')                                      8
     File and Directory Information Exposure                                                            8
     Exposed Dangerous Method or Function                                                               8
     DEPRECATED: Source Code                                                                            7
     Incorrect Resource Transfer Between Spheres                                                        7
     7PK - Time and State                                                                               7
     Information Management Errors                                                                      6
     Unintended Proxy or Intermediary ('Confused Deputy')                                               5
     Unchecked Return Value                                                                             5
     Incomplete Cleanup                                                                                 5
     DEPRECATED: Information Exposure Through Debug Log Files                                           5
     Improper Check for Dropped Privileges                                                              5
     Incorrect Conversion between Numeric Types                                                         4
     Off-by-one Error                                                                                   4
     Incorrect Calculation of Buffer Size                                                               4
     Pathname Traversal and Equivalence Errors                                                          4
     DEPRECATED: Uncontrolled File Descriptor Consumption                                               4
     Improper Restriction of Recursive Entity References in DTDs ('XML Entity Expansion')               4
     Inefficient Algorithmic Complexity                                                                 4
     Use of Incorrectly-Resolved Name or Reference                                                      3
     Improper Handling of Case Sensitivity                                                              3
     Improper Enforcement of Message Integrity During Transmission in a Communication Channel           3
     Use of Externally-Controlled Input to Select Classes or Code ('Unsafe Reflection')                 3
     Incomplete Blacklist                                                                               3
     Interpretation Conflict                                                                            3
     Improper Neutralization of Special Elements in Data Query Logic                                    3
     Improper Control of Resource Identifiers ('Resource Injection')                                    3
     Improper Control of Dynamically-Managed Code Resources                                             3
     Incorrect Usage of Seeds in Pseudo-Random Number Generator (PRNG)                                  3
     Improper Enforcement of Message or Data Structure                                                  2
     State Issues                                                                                       2
     Containment Errors (Container Errors)                                                              2
     Improper Control of a Resource Through its Lifetime                                                2
     External Control of Critical State Data                                                            2
     Operation on a Resource after Expiration or Release                                                2
     Release of Invalid Pointer or Reference                                                            2
     Improperly Controlled Modification of Dynamically-Determined Object Attributes                     2
     DEPRECATED: Location                                                                               2
     Improper Restriction of Power Consumption                                                          2
     Allocation of File Descriptors or Handles Without Limits or Throttling                             1
     Asymmetric Resource Consumption (Amplification)                                                    1
     Inappropriate Encoding for Output Context                                                          1
     Always-Incorrect Control Flow Implementation                                                       1
     7PK - Code Quality                                                                                 1
     Modification of Assumed-Immutable Data (MAID)                                                      1
     Missing Release of File Descriptor or Handle after Effective Lifetime                              1
    Name: cwe_name, dtype: int64



We notice that this column contains CWE names, which are therefore categories. There is no need to perform NLP on this column because these are categories that we can encode later.


```python
df_data["cwe_code"].value_counts()
```




    79      12325
    119     12325
    20       7869
    200      6592
    264      5813
    89       5740
    22       3062
    399      2960
    310      2552
    352      2415
    94       2412
    125      2105
    287      1801
    284      1658
    189      1363
    416      1340
    190      1134
    476       971
    255       967
    787       853
    78        829
    362       643
    400       576
    254       568
    59        541
    77        520
    434       447
    732       423
    611       420
    295       391
    269       390
    74        366
    798       342
    772       324
    426       301
    601       278
    502       277
    16        248
    19        238
    134       218
    835       217
    918       207
    704       196
    522       186
    415       174
    17        163
    532       162
    863       156
    275       155
    384       124
    326       121
    369       117
    285       117
    120       109
    862       106
    306        98
    276        93
    347        82
    770        81
    319        81
    617        80
    129        78
    427        76
    668        72
    345        72
    327        71
    320        65
    404        61
    640        60
    311        58
    191        55
    388        50
    312        49
    755        49
    346        49
    330        48
    665        48
    754        42
    613        41
    674        40
    428        39
    1188       39
    834        35
    93         34
    338        32
    91         32
    281        31
    693        30
    521        29
    682        27
    824        27
    425        25
    113        24
    307        24
    639        24
    358        23
    444        23
    552        21
    909        20
    203        20
    290        20
    88         20
    417        19
    610        18
    185        17
    331        17
    1187       17
    209        14
    118        13
    116        12
    1021       12
    367        12
    829        12
    332        11
    90         11
    494        11
    916        10
    922        10
    172         9
    123         8
    297         8
    565         8
    354         8
    294         8
    843         8
    538         8
    749         8
    18          7
    669         7
    361         7
    199         6
    441         5
    252         5
    459         5
    534         5
    273         5
    681         4
    193         4
    131         4
    21          4
    769         4
    776         4
    407         4
    706         3
    178         3
    924         3
    470         3
    184         3
    436         3
    943         3
    99          3
    913         3
    335         3
    707         2
    371         2
    216         2
    664         2
    642         2
    672         2
    763         2
    915         2
    1           2
    920         2
    774         1
    405         1
    838         1
    670         1
    398         1
    471         1
    775         1
    Name: cwe_code, dtype: int64



We notice an interesting detail in this dataset. The column 'cwe_code' is an encoding of the values of 'cwe_name'. So we can delete the non-encoded value.


```python
df_data=df_data.drop(['cwe_name', ],axis=1)
```


```python
cat_features = df_data.select_dtypes(include=['int64','object']).columns
for feature in cat_features:
    print(f'{feature :-<30} {df_data[feature].unique()}')
```

    cve--------------------------- ['CVE-2019-16548' 'CVE-2019-16547' 'CVE-2019-16546' ... 'CVE-2007-6442'
     'CVE-2007-6370' 'CVE-2007-3004']
    mod_date---------------------- ['21/11/2019 15:15' '20/11/2019 21:22' '20/11/2019 20:15' ...
     '11/02/2008 05:00' '21/01/2008 05:00' '10/01/2008 05:00']
    pub_date---------------------- ['21/11/2019 15:15' '20/11/2019 21:15' '20/11/2019 20:15' ...
     '24/07/2007 17:30' '11/04/2003 04:00' '10/01/2008 01:46']
    cwe_code---------------------- [ 352  732  639   79   89  200   20  319  276  269  426   74  362  273
      416   59  611  434  287   22  295  400   78  125  190  476  532  119
      668  120  307  787  522  754  863  755  610  862  772  384  613  354
      327  290  134  617  918  415  345  494  311  502  835  674  798  824
       91  330  347  312 1188  640  916  704  601  264  306  829  129 1187
      521  338  665  331  326  776   94  294  399  404  843  254  428  552
     1021  770  669  367  255  436  191  427  444  565  284  189  310   77
      275  369  924  285  281  682  706   19  470  922  203  693  209  388
      320   99   90  534  441  371  216  332  913   16  116  538  172  417
      118  346  425  185  471  113  358  398  642  123  769   93  459  297
      920  909  838  193  178  834  681   88  915  252  335  763  672  131
      670  749  407  707  199   21  664   17   18  943  361  405  774    1
      184  775]
    summary----------------------- ['A cross-site request forgery vulnerability in Jenkins Google Compute Engine Plugin 4.1.1 and earlier in ComputeEngineCloud#doProvision could be used to provision new agents.'
     'Missing permission checks in various API endpoints in Jenkins Google Compute Engine Plugin 4.1.1 and earlier allow attackers with Overall/Read permission to obtain limited information about the plugin configuration and environment.'
     'Jenkins Google Compute Engine Plugin 4.1.1 and earlier does not verify SSH host keys when connecting agents created by the plugin, enabling man-in-the-middle attacks.'
     ...
     '** REJECT **  DO NOT USE THIS CANDIDATE NUMBER.  ConsultIDs: CVE-2007-6114.  Reason: This candidate is a duplicate of CVE-2007-6114.  Notes: All CVE users should reference CVE-2007-6114 instead of this candidate.  All references and descriptions in this candidate have been removed to prevent accidental usage.'
     '** REJECT **  DO NOT USE THIS CANDIDATE NUMBER.  ConsultIDs: CVE-2007-5583.  Reason: This candidate is a duplicate of CVE-2007-5583.  Notes: All CVE users should reference CVE-2007-5583 instead of this candidate.  All references and descriptions in this candidate have been removed to prevent accidental usage.'
     '** REJECT **  DO NOT USE THIS CANDIDATE NUMBER.  ConsultIDs: CVE-2007-2788.  Reason: This candidate is a duplicate of CVE-2007-2788.  Notes: All CVE users should reference CVE-2007-2788 instead of this candidate.  All references and descriptions in this candidate have been removed to prevent accidental usage.']
    access_authentication--------- [nan 'NONE' 'SINGLE' 'MULTIPLE']
    access_complexity------------- [nan 'LOW' 'MEDIUM' 'HIGH']
    access_vector----------------- [nan 'NETWORK' 'LOCAL' 'ADJACENT_NETWORK']
    impact_availability----------- [nan 'NONE' 'COMPLETE' 'PARTIAL']
    impact_confidentiality-------- [nan 'COMPLETE' 'NONE' 'PARTIAL']
    impact_integrity-------------- [nan 'NONE' 'COMPLETE' 'PARTIAL']
    Unnamed: 24------------------- [nan ' 17.2 versions above and including 17.2R2-S4 prior to 17.2R2-S6'
     ' V200R003C00' 3.61 3.6
     ' (24) Absolute Sound Recorder, Video to Audio Converter, and MP3 Splitter'
     ' V600R006C00' ' V100R001C10SPC300' ' VIE-L09C02B131'
     ' V100R001C10SPC700B010' ' S5700 V200R006C00'
     ' (55) GROUPSONLY parameter in userList.cgi']
    Unnamed: 25------------------- [nan
     ' 17.2X75 versions above and including 17.2X75-D100 prior to X17.2X75-D101, 17.2X75-D110'
     ' V200R003C02' 3.62 3.61 ' (25) Easy Ringtone Maker' ' TE40 V500R002C00'
     ' V100R001C10SPC500' ' VIE-L09C109B181' ' V500R002C00SPC200'
     ' V200R007C00'
     ' or (56) USER parameter in userView.cgi.",NONE,MEDIUM,NETWORK,NONE,NONE,PARTIAL']
    Unnamed: 26------------------- [nan
     ' 17.3 versions above and including 17.3R1-S4 on All non-SRX Series and SRX100, SRX110, SRX210, SRX220, SRX240m, SRX550m SRX650, SRX300, SRX320, SRX340, SRX345, SRX1500, SRX4100, SRX4200, SRX4600 and vSRX'
     ' V200R005C00' ' fixed versions: 3.70.0056 and newer' ' 3.62'
     ' (26) RecordNRip' ' V600R006C00' ' V100R001C10SPC600' ' VIE-L09C113B170'
     ' V500R002C00SPC500' ' V200R008C00']
    Unnamed: 27------------------- [nan
     ' 17.3 versions above and including 17.3R2-S2 prior to 17.3R2-S4 on All non-SRX Series and SRX100, SRX110, SRX210, SRX220, SRX240m, SRX550m SRX650, SRX300, SRX320, SRX340, SRX345, SRX1500, SRX4100, SRX4200, SRX4600 and vSRX'
     ' V200R005C01'
     ' 3.81.0032 and newer), Bosch Video Management System (BVMS) (vulnerable versions: 3.50.00XX'
     ' 3.70'
     ' (27) McFunSoft iPod Audio Studio, Audio Recorder for Free, and others'
     ' TE50 V500R002C00' ' V100R001C10SPC700B010' ' VIE-L09C150B170'
     ' V500R002C00SPC600' ' V200R009C00']
    Unnamed: 28------------------- [nan
     ' 17.3R3 on All non-SRX Series and SRX100, SRX110, SRX210, SRX220, SRX240m, SRX550m SRX650, SRX300, SRX320, SRX340, SRX345, SRX1500, SRX4100, SRX4200, SRX4600 and vSRX'
     ' V200R005C02' ' 3.55.00XX' ' 3.71 before 3.71.0032 '
     ' (28) MP3 WAV Converter' ' V600R006C00' ' V100R001C10SPC800'
     ' VIE-L09C25B120' ' V500R002C00SPC700' ' V200R010C00']
    Unnamed: 29------------------- [nan
     ' 17.4 versions above and including 17.4R1-S3 prior to 17.4R1-S5 on All non-SRX Series and SRX100, SRX110, SRX210, SRX220, SRX240m, SRX550m SRX650, SRX300, SRX320, SRX340, SRX345, SRX1500, SRX4100, SRX4200, SRX4600 and vSRX'
     ' V200R005C03' ' 3.60.00XX' ' fixed versions: 3.71.0032'
     ' (29) BearShare 6.0.2.26789' ' TE60 V100R001C01' ' V500R002C00SPC200'
     ' VIE-L09C40B181' ' V500R002C00SPC900' ' S6700 V200R008C00']
    Unnamed: 30------------------- [nan
     ' 17.4R2 on All non-SRX Series and SRX100, SRX110, SRX210, SRX220, SRX240m, SRX550m SRX650, SRX300, SRX320, SRX340, SRX345, SRX1500, SRX4100, SRX4200, SRX4600 and vSRX'
     ' V200R006C00' ' fixed versions: 7.5'
     ' 3.81.0032 and newer), Bosch Video Management System (BVMS) (vulnerable versions: 3.50.00XX'
     ' and (30) Oracle Siebel SimBuilder and CRM 7.x.",NONE,MEDIUM,NETWORK,COMPLETE,COMPLETE,COMPLETE'
     ' V100R001C10' ' V500R002C00SPC500' ' VIE-L09C432B181'
     ' V500R002C00SPCb00' ' V200R009C00']
    Unnamed: 31------------------- [nan
     ' 18.1 versions above and including 18.1R2 prior to 18.1R2-S3, 18.1R3 on All non-SRX Series and SRX100, SRX110, SRX210, SRX220, SRX240m, SRX550m SRX650, SRX300, SRX320, SRX340, SRX345, SRX1500, SRX4100, SRX4200, SRX4600 and vSRX'
     ' V200R007C00' ' 3.70.0056).",NONE,MEDIUM,NETWORK,NONE,PARTIAL,PARTIAL'
     ' 3.55.00XX' ' V500R002C00' ' V500R002C00SPC600' ' VIE-L09C55B170'
     ' V600R006C00' ' V200R010C00']
    Unnamed: 32------------------- [nan
     ' 18.2 versions above and including 18.2R1 prior to 18.2R1-S2, 18.2R1-S3, 18.2R2 on All non-SRX Series and SRX100, SRX110, SRX210, SRX220, SRX240m, SRX550m SRX650, SRX300, SRX320, SRX340, SRX345, SRX1500, SRX4100, SRX4200, SRX4600 and vSRX'
     ' V200R008C00' ' 3.60.00XX'
     ' V600R006C00 has a buffer overflow vulnerability. An unauthenticated, remote attacker has to control the peer device and send specially crafted message to the affected products. Due to insufficient input validation, successful exploit may cause some services abnormal.",NONE,MEDIUM,NETWORK,PARTIAL,NONE,NONE'
     ' V500R002C00SPC700' ' VIE-L09C605B131' ' V600R006C00SPC200'
     ' S7700 V200R007C00']
    Unnamed: 33------------------- [nan
     ' 18.2X75 versions above and including 18.2X75-D5 prior to 18.2X75-D20.",NONE,MEDIUM,NETWORK,COMPLETE,NONE,NONE'
     ' V200R009C00' ' 3.70.0056' ' V500R002C00SPC900' ' VIE-L09ITAC555B130'
     ' V600R006C00SPC300' ' V200R008C00']
    Unnamed: 34------------------- [nan 'S6700 V200R001C00' ' fixed versions: 7.5' ' V500R002C00SPCb00'
     ' VIE-L29C10B170' ' TE40 V500R002C00SPC600' ' V200R009C00']
    Unnamed: 35------------------- [nan ' V200R001C01' ' 3.71.0032).",SINGLE,LOW,NETWORK,NONE,PARTIAL,NONE'
     ' V600R006C00' ' VIE-L29C185B181' ' V500R002C00SPC700' ' V200R010C00']
    Unnamed: 36------------------- [nan ' V200R002C00' ' TE40 V500R002C00SPC600' ' VIE-L29C605B131'
     ' V500R002C00SPC900' ' S9700 V200R007C00']
    Unnamed: 37------------------- [nan ' V200R003C00' ' V500R002C00SPC700'
     ' VIE-L29C636B202 have a denial of service (DoS) vulnerability. An attacker can trick a user to install a malicious application to exploit this vulnerability. Successful exploitation can cause camera application unusable.,NONE,MEDIUM,NETWORK,PARTIAL,NONE,NONE'
     ' V500R002C00SPCb00' ' V200R007C01']
    Unnamed: 38------------------- [nan ' V200R005C00' ' V500R002C00SPC900' ' V600R006C00' ' V200R008C00']
    Unnamed: 39------------------- [nan ' V200R005C01' ' V500R002C00SPCb00' ' V600R006C00SPC200'
     ' V200R009C00']
    Unnamed: 40------------------- [nan ' V200R005C02' ' V600R006C00' ' V600R006C00SPC300' ' V200R010C00']
    Unnamed: 41------------------- [nan ' V200R008C00' ' V600R006C00SPC200' ' TE50 V500R002C00SPC600'
     ' Secospace USG6300 V500R001C00']
    Unnamed: 42------------------- [nan ' V200R009C00' ' TE50 V500R002C00SPC600' ' V500R002C00SPC700'
     ' V500R001C30']
    Unnamed: 43------------------- [nan 'S7700 V200R001C00' ' V500R002C00SPC700' ' V500R002C00SPCb00'
     ' Secospace USG6500 V500R001C00']
    Unnamed: 44------------------- [nan ' V200R001C01' ' V500R002C00SPCb00' ' V600R006C00' ' V500R001C30']
    Unnamed: 45------------------- [nan ' V200R002C00' ' V600R006C00' ' V600R006C00SPC200'
     ' Secospace USG6600 V500R001C00']
    Unnamed: 46------------------- [nan ' V200R003C00' ' V600R006C00SPC200' ' V600R006C00SPC300'
     ' V500R001C30S']
    Unnamed: 47------------------- [nan ' V200R005C00' ' TE60 V100R001C01SPC100' ' TE60 V100R001C10'
     ' TE30 V100R001C02']
    Unnamed: 48------------------- [nan ' V200R006C00' ' V100R001C01SPC107TB010' ' V100R001C10B001'
     ' V100R001C10']
    Unnamed: 49------------------- [nan ' V200R006C01' ' V100R001C10' ' V100R001C10B002' ' V500R002C00']
    Unnamed: 50------------------- [nan ' V200R007C00' ' V100R001C10SPC300' ' V100R001C10B010' ' V600R006C00']
    Unnamed: 51------------------- [nan ' V200R007C01' ' V100R001C10SPC400' ' V100R001C10B011'
     ' TE40 V500R002C00']
    Unnamed: 52------------------- [nan ' V200R008C00' ' V100R001C10SPC500' ' V100R001C10B012' ' V600R006C00']
    Unnamed: 53------------------- [nan ' V200R008C06' ' V100R001C10SPC600' ' V100R001C10B013'
     ' TE50 V500R002C00']
    Unnamed: 54------------------- [nan ' V200R009C00' ' V100R001C10SPC700' ' V100R001C10B014' ' V600R006C00']
    Unnamed: 55------------------- [nan 'S9700 V200R001C00' ' V100R001C10SPC800' ' V100R001C10B016'
     ' TE60 V100R001C01']
    Unnamed: 56------------------- [nan ' V200R001C01' ' V100R001C10SPC900' ' V100R001C10B017' ' V100R001C10']
    Unnamed: 57------------------- [nan ' V200R002C00' ' V500R002C00' ' V100R001C10B018']
    Unnamed: 58------------------- [nan ' V200R003C00' ' V500R002C00SPC100' ' V100R001C10B019' ' V600R006C00']
    Unnamed: 59------------------- [nan ' V200R005C00' ' V500R002C00SPC200' ' V100R001C10SPC400'
     ' TP3106 V100R002C00']
    Unnamed: 60------------------- [nan ' V200R006C00' ' V500R002C00SPC300' ' V100R001C10SPC500'
     ' TP3206 V100R002C00']
    Unnamed: 61------------------- [nan ' V200R007C00' ' V500R002C00SPC600' ' V100R001C10SPC600'
     ' V100R002C10']
    Unnamed: 62------------------- [nan ' V200R007C01' ' V500R002C00SPC700' ' V100R001C10SPC700'
     ' USG9500 V500R001C00']
    Unnamed: 63------------------- [nan ' V200R008C00' ' V500R002C00SPC800' ' V100R001C10SPC800B011'
     ' V500R001C30']
    Unnamed: 64------------------- [nan
     ' V200R009C00 have a memory leak vulnerability. In some specific conditions, if attackers send specific malformed MPLS Service PING messages to the affected products, products do not release the memory when handling the packets. So successful exploit will result in memory leak of the affected products.",NONE,MEDIUM,NETWORK,PARTIAL,NONE,NONE'
     ' V500R002C00SPC900' ' V100R001C10SPC900' ' ViewPoint 9030 V100R011C02']
    Unnamed: 65------------------- [nan ' V500R002C00SPCa00' ' V500R002C00'
     ' V100R011C03 has an Out-of-Bounds memory access vulnerability due to insufficient verification. An authenticated local attacker can make processing crash by a malicious certificate. The attacker can exploit this vulnerability to cause a denial of service.,NONE,LOW,LOCAL,PARTIAL,NONE,NONE'
     ' V100R011C03 has a DoS vulnerability in PEM module of Huawei products due to insufficient verification. An authenticated local attacker can make processing into deadloop by a malicious certificate. The attacker can exploit this vulnerability to cause a denial of service.,NONE,LOW,LOCAL,PARTIAL,NONE,NONE'
     ' V100R011C03 has a heap overflow vulnerability due to insufficient verification. An authenticated local attacker can make processing crash by a malicious certificate. The attacker can exploit this vulnerability to cause a denial of service.,NONE,LOW,LOCAL,PARTIAL,NONE,NONE'
     ' V100R011C03 has a null pointer reference vulnerability due to insufficient verification. An authenticated local attacker calls PEM decoder with special parameter which could cause a denial of service.,NONE,LOW,LOCAL,PARTIAL,NONE,NONE']
    Unnamed: 66------------------- [nan ' V500R002C00SPCb00' ' V500R002C00B010']
    Unnamed: 67------------------- [nan ' V500R002C00SPCd00' ' V500R002C00B011']
    Unnamed: 68------------------- [nan ' V600R006C00' ' V500R002C00SPC100']
    Unnamed: 69------------------- [nan ' V600R006C00SPC100' ' V500R002C00SPC200']
    Unnamed: 70------------------- [nan ' V600R006C00SPC200' ' V500R002C00SPC300']
    Unnamed: 71------------------- [nan ' V600R006C00SPC300' ' V500R002C00SPC600']
    Unnamed: 72------------------- [nan ' TP3106 V100R002C00' ' V500R002C00SPC700']
    Unnamed: 73------------------- [nan ' V100R002C00SPC200' ' V500R002C00SPC800']
    Unnamed: 74------------------- [nan ' V100R002C00SPC400' ' V500R002C00SPC900']
    Unnamed: 75------------------- [nan ' V100R002C00SPC600' ' V500R002C00SPCa00']
    Unnamed: 76------------------- [nan ' V100R002C00SPC700' ' V500R002C00SPCb00']
    Unnamed: 77------------------- [nan ' V100R002C00SPC800' ' V500R002C00SPCd00']
    Unnamed: 78------------------- [nan ' TP3206 V100R002C00' ' V500R002C00SPCe00']
    Unnamed: 79------------------- [nan ' V100R002C00SPC200' ' V600R006C00']
    Unnamed: 80------------------- [nan ' V100R002C00SPC400' ' V600R006C00SPC100']
    Unnamed: 81------------------- [nan ' V100R002C00SPC600' ' V600R006C00SPC200']
    Unnamed: 82------------------- [nan ' V100R002C00SPC700'
     ' V600R006C00SPC300 use the CIDAM protocol, which contains sensitive information in the message when it is implemented. So these products has an information disclosure vulnerability. An authenticated remote attacker could track and get the message of a target system. Successful exploit could allow the attacker to get the information and cause the sensitive information disclosure.",SINGLE,LOW,NETWORK,NONE,PARTIAL,NONE']
    Unnamed: 83------------------- [nan ' V100R002C10']
    Unnamed: 84------------------- [nan ' ViewPoint 9030 V100R011C02SPC100']
    Unnamed: 85------------------- [nan ' V100R011C03B012SP15']
    Unnamed: 86------------------- [nan ' V100R011C03B012SP16']
    Unnamed: 87------------------- [nan ' V100R011C03B015SP03']
    Unnamed: 88------------------- [nan ' V100R011C03LGWL01SPC100']
    Unnamed: 89------------------- [nan ' V100R011C03SPC100']
    Unnamed: 90------------------- [nan ' V100R011C03SPC200']
    Unnamed: 91------------------- [nan ' V100R011C03SPC300']
    Unnamed: 92------------------- [nan ' V100R011C03SPC400']
    Unnamed: 93------------------- [nan ' V100R011C03SPC500']
    Unnamed: 94------------------- [nan ' eSpace U1960 V200R003C30SPC200']
    Unnamed: 95------------------- [nan ' eSpace U1981 V100R001C20SPC700']
    Unnamed: 96------------------- [nan
     ' V200R003C20SPCa00 has an overflow vulnerability when the module process a specific amount of state. The module cannot handle it causing SIP module DoS.,NONE,LOW,NETWORK,PARTIAL,NONE,NONE'
     ' V200R003C20SPCa00 has an overflow vulnerability that the module cannot parse a malformed SIP message when validating variables. Attacker can exploit it to make one process reboot at random.,NONE,LOW,NETWORK,PARTIAL,NONE,NONE'
     ' V200R003C20SPCa00 has an overflow vulnerability that attacker can exploit by sending a specially crafted SIP message leading to a process reboot at random.,NONE,LOW,NETWORK,PARTIAL,NONE,NONE']
    

We notice here that most of the categorical values are encodable, especially the columns 'access_authentication', 'complexity', 'access_vector', 'impact_availability', 'impact_confidentiality', and 'impact_integrity' which have few categories. On the other hand, the summary column has a lot of different values (more than 30, and almost one different value per row), which would imply the use of NLP (through an adapted neural network), so we will remove it from the dataset for the moment. If no classical regression model gives convincing results, we can wonder about using Tensorflow or a competing library.


```python
df_data=df_data.drop(['summary'],axis=1)
```


```python
dfMissvalues = pd.DataFrame(
                           (round(100* df_data.isnull().sum()/len(df_data), 2)), 
                           columns=['Percentage of missing values']
                           )
dfMissvalues.sort_values(by=['Percentage of missing values'], ascending=False)
```





  <div id="df-a1fc7be3-df93-44c7-9ab2-08404d2c7518">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Percentage of missing values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Unnamed: 96</th>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Unnamed: 81</th>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Unnamed: 66</th>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Unnamed: 67</th>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Unnamed: 68</th>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Unnamed: 69</th>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Unnamed: 70</th>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Unnamed: 71</th>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Unnamed: 72</th>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Unnamed: 73</th>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Unnamed: 74</th>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Unnamed: 75</th>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Unnamed: 76</th>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Unnamed: 78</th>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Unnamed: 79</th>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Unnamed: 80</th>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Unnamed: 77</th>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Unnamed: 82</th>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Unnamed: 89</th>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Unnamed: 95</th>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Unnamed: 94</th>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Unnamed: 83</th>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Unnamed: 92</th>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Unnamed: 91</th>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Unnamed: 90</th>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Unnamed: 93</th>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Unnamed: 88</th>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Unnamed: 87</th>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Unnamed: 86</th>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Unnamed: 85</th>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Unnamed: 84</th>
      <td>100.00</td>
    </tr>
    <tr>
      <th>Unnamed: 56</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 65</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 64</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 63</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 62</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 61</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 60</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 59</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 58</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 57</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 55</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 34</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 43</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 32</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 33</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 54</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 35</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 37</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 38</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 39</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 40</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 41</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 42</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 36</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 44</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 46</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 47</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 48</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 49</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 50</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 51</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 52</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 53</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 45</th>
      <td>99.99</td>
    </tr>
    <tr>
      <th>Unnamed: 27</th>
      <td>99.98</td>
    </tr>
    <tr>
      <th>Unnamed: 24</th>
      <td>99.98</td>
    </tr>
    <tr>
      <th>Unnamed: 25</th>
      <td>99.98</td>
    </tr>
    <tr>
      <th>Unnamed: 26</th>
      <td>99.98</td>
    </tr>
    <tr>
      <th>Unnamed: 29</th>
      <td>99.98</td>
    </tr>
    <tr>
      <th>Unnamed: 28</th>
      <td>99.98</td>
    </tr>
    <tr>
      <th>Unnamed: 30</th>
      <td>99.98</td>
    </tr>
    <tr>
      <th>Unnamed: 31</th>
      <td>99.98</td>
    </tr>
    <tr>
      <th>impact_integrity</th>
      <td>4.61</td>
    </tr>
    <tr>
      <th>impact_confidentiality</th>
      <td>4.61</td>
    </tr>
    <tr>
      <th>impact_availability</th>
      <td>4.61</td>
    </tr>
    <tr>
      <th>access_vector</th>
      <td>4.61</td>
    </tr>
    <tr>
      <th>access_complexity</th>
      <td>4.61</td>
    </tr>
    <tr>
      <th>access_authentication</th>
      <td>4.61</td>
    </tr>
    <tr>
      <th>cvss</th>
      <td>0.00</td>
    </tr>
    <tr>
      <th>pub_date</th>
      <td>0.00</td>
    </tr>
    <tr>
      <th>mod_date</th>
      <td>0.00</td>
    </tr>
    <tr>
      <th>cwe_code</th>
      <td>0.00</td>
    </tr>
    <tr>
      <th>cve</th>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-a1fc7be3-df93-44c7-9ab2-08404d2c7518')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-a1fc7be3-df93-44c7-9ab2-08404d2c7518 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-a1fc7be3-df93-44c7-9ab2-08404d2c7518');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




We notice that some variables have almost 100% missing values. They are therefore not usable. Therefore, we can delete them.


```python
miss_rates = df_data.isna().sum() / len(df_data)
features = df_data.columns[miss_rates  < 0.9]
df_data = df_data[features]                 
```


```python
df_data.shape
```




    (89531, 11)



The dataset is now much smaller, from almost 100 columns to 13. We notice however that some columns can still be deleted, like for example 'cve' which represents only the name of the CVE and thus acts as an ID.


```python
df_data=df_data.drop(['cve'],axis=1)
```

We now need to manage the dates. Indeed, in the given format, the dates are not exploitable by our Machine Learning algorithms, and therefore cannot be taken into account in the modeling. However, we can assume that they play a more or less important role in the evaluation of the CVSS, for example if the vulnerabilities discovered at certain times of the day give a higher score on the danger of the latter.


```python
# Coupage de la colonne 'mod_date'
df_data['mod_minute'] = pd.to_datetime(df_data['mod_date']).dt.minute
df_data['mod_hour'] = pd.to_datetime(df_data['mod_date']).dt.hour
df_data['mod_day'] = pd.to_datetime(df_data['mod_date']).dt.day
df_data['mod_month'] = pd.to_datetime(df_data['mod_date']).dt.month
df_data['mod_year'] = pd.to_datetime(df_data['mod_date']).dt.year
df_data = df_data.drop(['mod_date'], axis=1)

# Coupage de la colonne 'pub_date'
df_data['pub_minute'] = pd.to_datetime(df_data['pub_date']).dt.minute
df_data['pub_hour'] = pd.to_datetime(df_data['pub_date']).dt.hour
df_data['pub_day'] = pd.to_datetime(df_data['pub_date']).dt.day
df_data['pub_month'] = pd.to_datetime(df_data['pub_date']).dt.month
df_data['pub_year'] = pd.to_datetime(df_data['pub_date']).dt.year
df_data = df_data.drop(['pub_date'], axis=1)
```


```python
df_data.head()
```





  <div id="df-9a791bb4-6501-40cb-ae1b-b7ecae7a8419">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cvss</th>
      <th>cwe_code</th>
      <th>access_authentication</th>
      <th>access_complexity</th>
      <th>access_vector</th>
      <th>impact_availability</th>
      <th>impact_confidentiality</th>
      <th>impact_integrity</th>
      <th>mod_minute</th>
      <th>mod_hour</th>
      <th>mod_day</th>
      <th>mod_month</th>
      <th>mod_year</th>
      <th>pub_minute</th>
      <th>pub_hour</th>
      <th>pub_day</th>
      <th>pub_month</th>
      <th>pub_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.8</td>
      <td>352</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>15</td>
      <td>21</td>
      <td>11</td>
      <td>2019</td>
      <td>15</td>
      <td>15</td>
      <td>21</td>
      <td>11</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>732</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>15</td>
      <td>21</td>
      <td>11</td>
      <td>2019</td>
      <td>15</td>
      <td>15</td>
      <td>21</td>
      <td>11</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.3</td>
      <td>639</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>15</td>
      <td>21</td>
      <td>11</td>
      <td>2019</td>
      <td>15</td>
      <td>15</td>
      <td>21</td>
      <td>11</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.3</td>
      <td>79</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22</td>
      <td>21</td>
      <td>20</td>
      <td>11</td>
      <td>2019</td>
      <td>15</td>
      <td>21</td>
      <td>20</td>
      <td>11</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.5</td>
      <td>89</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15</td>
      <td>20</td>
      <td>20</td>
      <td>11</td>
      <td>2019</td>
      <td>15</td>
      <td>20</td>
      <td>20</td>
      <td>11</td>
      <td>2019</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-9a791bb4-6501-40cb-ae1b-b7ecae7a8419')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-9a791bb4-6501-40cb-ae1b-b7ecae7a8419 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-9a791bb4-6501-40cb-ae1b-b7ecae7a8419');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




Breaking the date into multiple columns allows us to use them in our models now, as the format is compatible with modeling approaches.


```python
cat_features = df_data.select_dtypes(include=['int64','object']).columns
```


```python
print("Categorical columns:\n\n", cat_features.values)
```

    Categorical columns:
    
     ['cwe_code' 'access_authentication' 'access_complexity' 'access_vector'
     'impact_availability' 'impact_confidentiality' 'impact_integrity'
     'mod_minute' 'mod_hour' 'mod_day' 'mod_month' 'mod_year' 'pub_minute'
     'pub_hour' 'pub_day' 'pub_month' 'pub_year']
    


```python
# Dataset size
n_samples, n_features = df_data.shape
print("Rows:", n_samples)
print("Columns:", n_features)
```

    Rows: 89531
    Columns: 18
    

Finally, the dataframe is made of 18 columns, so the reduction is impressive compared to the initial dataset, which makes the problem more understandable.

## Missing values


```python
# Afficher tous les na
plt.figure(figsize=(26, 12))
sns.heatmap(df_data.isna(), cbar=False)
plt.show()
```


    
![png](output_53_0.png)
    


For the remaining columns, the missing values are quite small, and all present on 6 columns. It will be necessary to use an imputer during the pre-processing in order to fill these values with the modes or the k nearest neighbors for example.

## Bivariate analysis between the features

### Correlation between the continuous features


```python
corrmat = df_data.corr()
```


```python
plt.figure(figsize=(26, 12))
sns.heatmap(corrmat, annot=True, cbar=True, square=True, vmax=1)
plt.show()
```


    
![png](output_58_0.png)
    



```python
print(corrmat)
```

                    cvss  cwe_code  mod_minute  mod_hour   mod_day  mod_month  \
    cvss        1.000000  0.008936   -0.012434 -0.036548  0.014663   0.061450   
    cwe_code    0.008936  1.000000   -0.090925  0.009906 -0.083563  -0.101448   
    mod_minute -0.012434 -0.090925    1.000000  0.119121  0.035701   0.134383   
    mod_hour   -0.036548  0.009906    0.119121  1.000000 -0.164237   0.165034   
    mod_day     0.014663 -0.083563    0.035701 -0.164237  1.000000   0.092714   
    mod_month   0.061450 -0.101448    0.134383  0.165034  0.092714   1.000000   
    mod_year   -0.002000  0.136709    0.116638  0.266811 -0.071939  -0.011458   
    pub_minute -0.019658 -0.017412   -0.003303 -0.100567  0.003219  -0.043743   
    pub_hour   -0.010110  0.017498    0.020329  0.014949  0.000645  -0.027473   
    pub_day    -0.038241  0.017597    0.000334 -0.041161  0.140743  -0.016951   
    pub_month   0.033502  0.007147   -0.010573  0.030069  0.042998   0.190277   
    pub_year   -0.141871  0.218662   -0.042195  0.350713 -0.173652  -0.222543   
    
                mod_year  pub_minute  pub_hour   pub_day  pub_month  pub_year  
    cvss       -0.002000   -0.019658 -0.010110 -0.038241   0.033502 -0.141871  
    cwe_code    0.136709   -0.017412  0.017498  0.017597   0.007147  0.218662  
    mod_minute  0.116638   -0.003303  0.020329  0.000334  -0.010573 -0.042195  
    mod_hour    0.266811   -0.100567  0.014949 -0.041161   0.030069  0.350713  
    mod_day    -0.071939    0.003219  0.000645  0.140743   0.042998 -0.173652  
    mod_month  -0.011458   -0.043743 -0.027473 -0.016951   0.190277 -0.222543  
    mod_year    1.000000   -0.189302  0.061019 -0.047088  -0.027537  0.478135  
    pub_minute -0.189302    1.000000 -0.071754 -0.038132  -0.052611  0.045312  
    pub_hour    0.061019   -0.071754  1.000000 -0.012011  -0.044346  0.044861  
    pub_day    -0.047088   -0.038132 -0.012011  1.000000  -0.011096 -0.061045  
    pub_month  -0.027537   -0.052611 -0.044346 -0.011096   1.000000 -0.061221  
    pub_year    0.478135    0.045312  0.044861 -0.061045  -0.061221  1.000000  
    

We notice that there is no significant linear correlation between the value 'cve_code' and the target. This makes sense, because the code is only an encoded value and not a quantity. On the other hand, the columns 'mod_year' and 'pub_year' have a fairly strong linear correlation, which is quite logical. However, it is only 0.43, which means that more than half of the time there is a delay of at least one year between the publication and the modification.

### Correlation between the categorical features


```python
cat_features_display = df_data.select_dtypes(include=['object']).columns
for feature in cat_features_display:
    for feature2 in cat_features:
        if feature2 is not feature:
            plt.figure()
            sns.heatmap(pd.crosstab(df_data[feature2], df_data[feature]), annot=True, fmt='d')
            plt.show()
```


    
![png](output_62_0.png)
    



    
![png](output_62_1.png)
    



    
![png](output_62_2.png)
    



    
![png](output_62_3.png)
    



    
![png](output_62_4.png)
    



    
![png](output_62_5.png)
    



    
![png](output_62_6.png)
    



    
![png](output_62_7.png)
    



    
![png](output_62_8.png)
    



    
![png](output_62_9.png)
    



    
![png](output_62_10.png)
    



    
![png](output_62_11.png)
    



    
![png](output_62_12.png)
    



    
![png](output_62_13.png)
    



    
![png](output_62_14.png)
    



    
![png](output_62_15.png)
    



    
![png](output_62_16.png)
    



    
![png](output_62_17.png)
    



    
![png](output_62_18.png)
    



    
![png](output_62_19.png)
    



    
![png](output_62_20.png)
    



    
![png](output_62_21.png)
    



    
![png](output_62_22.png)
    



    
![png](output_62_23.png)
    



    
![png](output_62_24.png)
    



    
![png](output_62_25.png)
    



    
![png](output_62_26.png)
    



    
![png](output_62_27.png)
    



    
![png](output_62_28.png)
    



    
![png](output_62_29.png)
    



    
![png](output_62_30.png)
    



    
![png](output_62_31.png)
    



    
![png](output_62_32.png)
    



    
![png](output_62_33.png)
    



    
![png](output_62_34.png)
    



    
![png](output_62_35.png)
    



    
![png](output_62_36.png)
    



    
![png](output_62_37.png)
    



    
![png](output_62_38.png)
    



    
![png](output_62_39.png)
    



    
![png](output_62_40.png)
    



    
![png](output_62_41.png)
    



    
![png](output_62_42.png)
    



    
![png](output_62_43.png)
    



    
![png](output_62_44.png)
    



    
![png](output_62_45.png)
    



    
![png](output_62_46.png)
    



    
![png](output_62_47.png)
    



    
![png](output_62_48.png)
    



    
![png](output_62_49.png)
    



    
![png](output_62_50.png)
    



    
![png](output_62_51.png)
    



    
![png](output_62_52.png)
    



    
![png](output_62_53.png)
    



    
![png](output_62_54.png)
    



    
![png](output_62_55.png)
    



    
![png](output_62_56.png)
    



    
![png](output_62_57.png)
    



    
![png](output_62_58.png)
    



    
![png](output_62_59.png)
    



    
![png](output_62_60.png)
    



    
![png](output_62_61.png)
    



    
![png](output_62_62.png)
    



    
![png](output_62_63.png)
    



    
![png](output_62_64.png)
    



    
![png](output_62_65.png)
    



    
![png](output_62_66.png)
    



    
![png](output_62_67.png)
    



    
![png](output_62_68.png)
    



    
![png](output_62_69.png)
    



    
![png](output_62_70.png)
    



    
![png](output_62_71.png)
    



    
![png](output_62_72.png)
    



    
![png](output_62_73.png)
    



    
![png](output_62_74.png)
    



    
![png](output_62_75.png)
    



    
![png](output_62_76.png)
    



    
![png](output_62_77.png)
    



    
![png](output_62_78.png)
    



    
![png](output_62_79.png)
    



    
![png](output_62_80.png)
    



    
![png](output_62_81.png)
    



    
![png](output_62_82.png)
    



    
![png](output_62_83.png)
    



    
![png](output_62_84.png)
    



    
![png](output_62_85.png)
    



    
![png](output_62_86.png)
    



    
![png](output_62_87.png)
    



    
![png](output_62_88.png)
    



    
![png](output_62_89.png)
    



    
![png](output_62_90.png)
    



    
![png](output_62_91.png)
    



    
![png](output_62_92.png)
    



    
![png](output_62_93.png)
    



    
![png](output_62_94.png)
    



    
![png](output_62_95.png)
    


We notice here several interesting elements in the bivariate analysis of the categorical variables:
- 'access_complexity' and 'access_authentication' have a fairly high correlation, especially with 'NONE' and 'MEDIUM' values of the former around the 'NONE' value of the latter
- access_complexity' and 'impact_availability' have little correlation
- the 'NONE' and 'SINGLE' values of 'access_authentication' are mostly clustered around the 'NETWORK' value of 'access_vector
- there is a strong correlation between 'impact_confidentiality' and 'impact_availability

Thus, the columns starting with 'impact' seem to have similar distributions, as well as those starting with 'access'.

## Correlation between the features and the target


```python
plt.figure(figsize=(16, 8))
corrmat[targetName].sort_values(ascending=True)[:-1].plot(kind='barh')
plt.title("Corrélation entre les variables explicatives et la cible")
plt.tight_layout()
plt.show()
```


    
![png](output_65_0.png)
    


The variables that seem to have the most linear correlation with the target are 'pub_year' and 'mod_month'. Since the year of publication is not very important because it is not cyclical, unlike the month, the first one can be considered as not very interesting. However, if no linear correlation seems strong at first sight, it is possible that a polynomial correlation for example exists.

# Preprocessing

In order to try to find a chain of treatments that would predict the CVSS, it is appropriate to create several versions of the dataset that would be preprocessed differently, before testing them for each model afterwards.

## Target normalization


```python
df_train, df_test = train_test_split(df_data, test_size=0.2, random_state=_RANDOM_STATE_)
```

The division into two separate sets at the beginning of the pre-processing will allow us to ensure the independence between the training and test sets.


```python
df_train=df_train.reset_index(drop=True)
```


```python
df_test=df_test.reset_index(drop=True)
```

In order to avoid problems with certain pre-processing, reset the indexes of the dataframes to zero.


```python
cat_features_display = df_data.select_dtypes(include=['int64','object']).columns
con_features_display = []
```


```python
# Verifying the shape of the repartition of the target
print("Skewness: %f" %df_train[targetName].skew())
print("Kurtosis: %f" %df_train[targetName].kurt())
```

    Skewness: 0.256461
    Kurtosis: -0.679964
    

We notice that the skewness and kurtosis coefficients confirm that the distribution of the target variable is not normal. It is therefore necessary to perform a log-transformation in order to ensure this necessary condition as explained previously during the exploratory analysis of the data.


```python
# setting up the transformation
loc, scale = norm.fit(df_train[targetName])
n = norm(loc=loc, scale=scale)
print(kstest(df_train[targetName], n.cdf))
# Log transformation
df_train_scaled = df_train.copy()
df_train_scaled[targetName] = np.log1p(df_train[targetName])
df_test_scaled = df_test.copy()
df_test_scaled[targetName] = np.log1p(df_test[targetName])
```

    KstestResult(statistic=0.16435651443511345, pvalue=0.0)
    


```python
sns.distplot(df_train_scaled[targetName], fit=norm);
```


    
![png](output_79_0.png)
    



```python
print("Skewness: %f" %df_train_scaled[targetName].skew())
print("Kurtosis: %f" %df_train_scaled[targetName].kurt())
```

    Skewness: -0.415541
    Kurtosis: 0.010535
    

We notice here that despite a log transformation of the data, the distribution is not quite normal. However, the coefficients seem better.

If this version of the dataset is used in the future, it will be necessary to think of reversing the log-transformation in order to obtain the true values of the CVSS, because the data range of the target dataset used as input will be modified.

## Preprocessing functions


```python
# Can take for instance StandardScaler, MinMaxScaler, RobustScaler
def scaler(scl, df_train, df_test, con_features = []):
    for feature in con_features:
        df_train[feature] = scl.fit_transform(df_train[feature].values.reshape(-1,1)).ravel()
        df_test[feature] = scl.transform(df_test[feature].values.reshape(-1,1)).ravel()
    return df_train, df_test
```

 A scaler function, as described above, can take as parameter a scaler among StandardScaler, MinMaxScaler and RobustScaler, and then perform a scaling operation on the continuous data of the dataset.


```python
#Can take for instance LabelEncoder, OneHotEncoder
def encoder(ecd, df_train, df_test, cat_features_to_encode = []):
    # If we have one column with two values, we can encode it
    if type(ecd).__name__ != "OneHotEncoder":
        for feature in cat_features_to_encode:
        # The OneHotEncoder feature needs a new shape, instead of the other encoders
            df_train[feature] = ecd.fit_transform(df_train[feature].values.reshape(-1,1)).ravel()
            df_test[feature] = ecd.transform(df_test[feature].values.reshape(-1,1)).ravel()
    else:
        for feature in cat_features_to_encode:
            df_train_to_add = pd.DataFrame(ecd.fit_transform(df_train[feature].values.reshape(-1,1)).toarray())
            df_train_to_add.columns = ecd.get_feature_names_out()
            df_train_to_add.columns = [s.replace('x0' , feature) for s in df_train_to_add.columns]
            
            # Adding columns to train
            for serie in df_train_to_add:
                df_train=pd.concat([df_train, df_train_to_add[serie]], axis=1)
            df_train=df_train.drop([feature],axis=1)

            df_test_to_add = pd.DataFrame(ecd.transform(df_test[feature].values.reshape(-1,1)).toarray())
            df_test_to_add.columns = ecd.get_feature_names_out()
            df_test_to_add.columns = [s.replace('x0' , feature) for s in df_test_to_add.columns]
            # Adding columns to test
            for serie in df_test_to_add:
                df_test=pd.concat([df_test, df_test_to_add[serie]], axis=1)
            df_test=df_test.drop([feature],axis=1)
            
    return df_train, df_test
```

An encoder function can take as parameter an encoder among LabelEncoder and OneHotEncoder, to then perform an encoding operation on the categorical data of the dataset in order to allow their reading by the classification algorithms.


```python
 # Can take for instance KNNImputer, SimpleImputer
def imputer(imp, df_train, df_test, cat_features_to_impute = [], cont_features_to_impute = []):
    # Application de l'imputer sur les variables continues
    for feature in cont_features_to_impute :
        df_train[feature] = imp.fit_transform(df_train[feature].values.reshape(-1,1)).ravel()
        df_test[feature] = imp.transform(df_test[feature].values.reshape(-1,1)).ravel()
    # Application de l'imputer sur les variables catégories 
    for feature in cat_features_to_impute :
        df_train[feature] = imp.fit_transform(df_train[feature].values.reshape(-1,1)).ravel()
        df_test[feature] = imp.transform(df_test[feature].values.reshape(-1,1)).ravel()
    return df_train, df_test
```

An impute function will be able to take as parameter one of KNNImputer and SimpleImputer, to replace missing values of continuous variables (KNN and Simple) as well as categorical values (Simple only).


```python
def preprocessing(df_train, df_test, targetName, scl = None, imp = None, ecd = None):
    # Deep Copy of dataframe to keep the original intact
    df_train_tmp = df_train.copy()
    df_test_tmp = df_test.copy()
    # Using the preprocess
    df_train_tmp, df_test_tmp = [df_train_tmp, df_test_tmp] if imp is None else imputer(imp, df_train_tmp, df_test_tmp, cat_features, con_features)
    df_train_tmp, df_test_tmp = [df_train_tmp, df_test_tmp] if scl is None else scaler(scl, df_train_tmp, df_test_tmp, con_features)
    df_train_tmp, df_test_tmp = [df_train_tmp, df_test_tmp] if ecd is None else encoder(ecd, df_train_tmp, df_test_tmp, cat_features)
    return df_train_tmp, df_test_tmp
```

The pre-processing function takes in parameter the different pre-processes to apply, by category, then performs one by one the treatments on a copy of the dataset provided in input before returning a new pre-processed version of the training and testing sets. As seen previously, we must impute the missing values (impute and not delete because the number of missing values is negligible). For this, we will use the KNNImputer.

## Creation of the preprocessing combinations

### Simple process

A first suite of simple pre-processing will be used as a basis for comparing the other strings. It must therefore not increase the complexity of the dataframe, while ensuring the basic processing, particularly encoding.


```python
df_train1, df_test1 = preprocessing(df_train_scaled, df_test, targetName, scl = MinMaxScaler(), imp = SimpleImputer(strategy="most_frequent"), ecd = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
```


```python
df_train1.head()
```





  <div id="df-076bc208-61db-40c7-80f8-a6805d4f7ed7">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cvss</th>
      <th>cwe_code</th>
      <th>access_authentication</th>
      <th>access_complexity</th>
      <th>access_vector</th>
      <th>impact_availability</th>
      <th>impact_confidentiality</th>
      <th>impact_integrity</th>
      <th>mod_minute</th>
      <th>mod_hour</th>
      <th>mod_day</th>
      <th>mod_month</th>
      <th>mod_year</th>
      <th>pub_minute</th>
      <th>pub_hour</th>
      <th>pub_day</th>
      <th>pub_month</th>
      <th>pub_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.747222</td>
      <td>39.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>19.0</td>
      <td>23.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>29.0</td>
      <td>17.0</td>
      <td>18.0</td>
      <td>9.0</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.747222</td>
      <td>39.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>44.0</td>
      <td>13.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>15.0</td>
      <td>19.0</td>
      <td>28.0</td>
      <td>9.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.695488</td>
      <td>12.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>30.0</td>
      <td>17.0</td>
      <td>30.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>30.0</td>
      <td>17.0</td>
      <td>30.0</td>
      <td>2.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.695488</td>
      <td>12.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>28.0</td>
      <td>20.0</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>11.0</td>
      <td>15.0</td>
      <td>14.0</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.695488</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>15.0</td>
      <td>4.0</td>
      <td>24.0</td>
      <td>6.0</td>
      <td>11.0</td>
      <td>15.0</td>
      <td>17.0</td>
      <td>26.0</td>
      <td>5.0</td>
      <td>20.0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-076bc208-61db-40c7-80f8-a6805d4f7ed7')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-076bc208-61db-40c7-80f8-a6805d4f7ed7 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-076bc208-61db-40c7-80f8-a6805d4f7ed7');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




A first pre-processing will be a MinMaxScaler (simple), and an OrdinalEncoder. This encoder will not split the columns into several binary columns unlike OneHotEncoder, and may cause weight concerns when creating models.

### Simple process with columns deletion

As observed during the exploratory analysis of the data, some columns have little importance and may be counterproductive (such as years because they are not cyclical). It would therefore be interesting to remove the least useful columns in order to verify whether they have a positive impact on the metrics.


```python
df_train2 = df_train_scaled.copy()
df_test2 = df_test_scaled.copy()
df_train2 = df_train2.drop(['pub_year', 'mod_year'], axis=1)
df_test2 = df_test2.drop(['pub_year', 'mod_year'], axis=1)
```


```python
df_train2, df_test2 = preprocessing(df_train_scaled, df_test, targetName, scl = StandardScaler(), imp = SimpleImputer(strategy="most_frequent"), ecd = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
```

We can also select the K best features among the remaining ones using the SelectKBest method.


```python
# Cutting the dataset for the selector
df_train_select = df_train2.drop(targetName, axis=1)
target_train = df_train2[targetName]
# Using it and getting n variables
n = 10
select = SelectKBest(score_func=f_regression, k=n)
z = select.fit_transform(df_train_select, target_train)
# Getting the scores
selection_scores = pd.DataFrame(select.scores_)
data_columns = pd.DataFrame(df_train_select.columns)
# Concatenating both dataframes
scores = pd.concat([data_columns,selection_scores],axis=1)
scores.columns=['Feature','Score']
print(scores.nlargest(n,'Score'))
# Showing the result
scores = scores.sort_values(by="Score", ascending=False)
plt.figure(figsize=(20,7), facecolor='w')
sns.barplot(x='Feature',y='Score',data=scores,palette='BuGn_r')
plt.title("Plot showing the best features in descending order", size=20)
plt.show()
```

                       Feature        Score
    6         impact_integrity  8914.226212
    5   impact_confidentiality  6596.936578
    3            access_vector  3870.870270
    4      impact_availability  3595.843911
    1    access_authentication  2160.657191
    16                pub_year  1413.535053
    2        access_complexity   737.473694
    10               mod_month   280.332642
    8                 mod_hour   125.348486
    14                 pub_day    71.231978
    


    
![png](output_102_1.png)
    


We notice here that a minority of columns have an impact on the value of the regression, so we can try to keep only the top 10 impacting values.


```python
to_keep = np.array(scores.nlargest(10,'Score')["Feature"])
to_keep = np.append(to_keep,targetName)
df_train2 = df_train2[to_keep]
df_test2 = df_test2[to_keep]
```


```python
df_train2.head()
```





  <div id="df-a28d5cf8-3e40-41f0-a394-bb0bcbd6661c">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>impact_integrity</th>
      <th>impact_confidentiality</th>
      <th>access_vector</th>
      <th>impact_availability</th>
      <th>access_authentication</th>
      <th>pub_year</th>
      <th>access_complexity</th>
      <th>mod_month</th>
      <th>mod_hour</th>
      <th>pub_day</th>
      <th>cvss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>18.0</td>
      <td>1.0</td>
      <td>9.0</td>
      <td>19.0</td>
      <td>18.0</td>
      <td>-0.382681</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>20.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>28.0</td>
      <td>-0.382681</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>17.0</td>
      <td>30.0</td>
      <td>-0.799447</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>20.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>20.0</td>
      <td>7.0</td>
      <td>-0.799447</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>20.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>26.0</td>
      <td>-0.799447</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-a28d5cf8-3e40-41f0-a394-bb0bcbd6661c')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-a28d5cf8-3e40-41f0-a394-bb0bcbd6661c button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-a28d5cf8-3e40-41f0-a394-bb0bcbd6661c');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




### Complex encoded process

A more complex processing chain will limit the impact of the weights of the first method, without worrying about the complexity and size of the dataframe.


```python
df_train3, df_test3 = preprocessing(df_train_scaled, df_test_scaled, targetName, scl = StandardScaler(), imp = SimpleImputer(strategy="most_frequent"), ecd = OneHotEncoder(handle_unknown='ignore'))
```


```python
df_train3.head()
```





  <div id="df-4c001e85-1c81-4b1e-a30d-38b317cbd029">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cvss</th>
      <th>cwe_code_1</th>
      <th>cwe_code_16</th>
      <th>cwe_code_17</th>
      <th>cwe_code_18</th>
      <th>cwe_code_19</th>
      <th>cwe_code_20</th>
      <th>cwe_code_21</th>
      <th>cwe_code_22</th>
      <th>cwe_code_59</th>
      <th>cwe_code_74</th>
      <th>cwe_code_77</th>
      <th>cwe_code_78</th>
      <th>cwe_code_79</th>
      <th>cwe_code_88</th>
      <th>cwe_code_89</th>
      <th>cwe_code_90</th>
      <th>cwe_code_91</th>
      <th>cwe_code_93</th>
      <th>cwe_code_94</th>
      <th>cwe_code_99</th>
      <th>cwe_code_113</th>
      <th>cwe_code_116</th>
      <th>cwe_code_118</th>
      <th>cwe_code_119</th>
      <th>cwe_code_120</th>
      <th>cwe_code_123</th>
      <th>cwe_code_125</th>
      <th>cwe_code_129</th>
      <th>cwe_code_131</th>
      <th>cwe_code_134</th>
      <th>cwe_code_172</th>
      <th>cwe_code_178</th>
      <th>cwe_code_184</th>
      <th>cwe_code_185</th>
      <th>cwe_code_189</th>
      <th>cwe_code_190</th>
      <th>cwe_code_191</th>
      <th>cwe_code_193</th>
      <th>cwe_code_199</th>
      <th>cwe_code_200</th>
      <th>cwe_code_203</th>
      <th>cwe_code_209</th>
      <th>cwe_code_216</th>
      <th>cwe_code_252</th>
      <th>cwe_code_254</th>
      <th>cwe_code_255</th>
      <th>cwe_code_264</th>
      <th>cwe_code_269</th>
      <th>cwe_code_273</th>
      <th>cwe_code_275</th>
      <th>cwe_code_276</th>
      <th>cwe_code_281</th>
      <th>cwe_code_284</th>
      <th>cwe_code_285</th>
      <th>cwe_code_287</th>
      <th>cwe_code_290</th>
      <th>cwe_code_294</th>
      <th>cwe_code_295</th>
      <th>cwe_code_297</th>
      <th>cwe_code_306</th>
      <th>cwe_code_307</th>
      <th>cwe_code_310</th>
      <th>cwe_code_311</th>
      <th>cwe_code_312</th>
      <th>cwe_code_319</th>
      <th>cwe_code_320</th>
      <th>cwe_code_326</th>
      <th>cwe_code_327</th>
      <th>cwe_code_330</th>
      <th>cwe_code_331</th>
      <th>cwe_code_332</th>
      <th>cwe_code_335</th>
      <th>cwe_code_338</th>
      <th>cwe_code_345</th>
      <th>cwe_code_346</th>
      <th>cwe_code_347</th>
      <th>cwe_code_352</th>
      <th>cwe_code_354</th>
      <th>cwe_code_358</th>
      <th>cwe_code_361</th>
      <th>cwe_code_362</th>
      <th>cwe_code_367</th>
      <th>cwe_code_369</th>
      <th>cwe_code_371</th>
      <th>cwe_code_384</th>
      <th>cwe_code_388</th>
      <th>cwe_code_398</th>
      <th>cwe_code_399</th>
      <th>cwe_code_400</th>
      <th>cwe_code_404</th>
      <th>cwe_code_407</th>
      <th>cwe_code_415</th>
      <th>cwe_code_416</th>
      <th>cwe_code_417</th>
      <th>cwe_code_425</th>
      <th>cwe_code_426</th>
      <th>cwe_code_427</th>
      <th>cwe_code_428</th>
      <th>cwe_code_434</th>
      <th>cwe_code_436</th>
      <th>cwe_code_441</th>
      <th>cwe_code_444</th>
      <th>cwe_code_459</th>
      <th>cwe_code_470</th>
      <th>cwe_code_476</th>
      <th>cwe_code_494</th>
      <th>cwe_code_502</th>
      <th>cwe_code_521</th>
      <th>cwe_code_522</th>
      <th>cwe_code_532</th>
      <th>cwe_code_534</th>
      <th>cwe_code_538</th>
      <th>cwe_code_552</th>
      <th>cwe_code_565</th>
      <th>cwe_code_601</th>
      <th>cwe_code_610</th>
      <th>cwe_code_611</th>
      <th>cwe_code_613</th>
      <th>cwe_code_617</th>
      <th>cwe_code_639</th>
      <th>cwe_code_640</th>
      <th>cwe_code_642</th>
      <th>cwe_code_664</th>
      <th>cwe_code_665</th>
      <th>cwe_code_668</th>
      <th>cwe_code_669</th>
      <th>cwe_code_670</th>
      <th>cwe_code_672</th>
      <th>cwe_code_674</th>
      <th>cwe_code_681</th>
      <th>cwe_code_682</th>
      <th>cwe_code_693</th>
      <th>cwe_code_704</th>
      <th>cwe_code_706</th>
      <th>cwe_code_707</th>
      <th>cwe_code_732</th>
      <th>cwe_code_749</th>
      <th>cwe_code_754</th>
      <th>cwe_code_755</th>
      <th>cwe_code_763</th>
      <th>cwe_code_769</th>
      <th>cwe_code_770</th>
      <th>cwe_code_772</th>
      <th>cwe_code_775</th>
      <th>cwe_code_776</th>
      <th>cwe_code_787</th>
      <th>cwe_code_798</th>
      <th>cwe_code_824</th>
      <th>cwe_code_829</th>
      <th>cwe_code_834</th>
      <th>cwe_code_835</th>
      <th>cwe_code_838</th>
      <th>cwe_code_843</th>
      <th>cwe_code_862</th>
      <th>cwe_code_863</th>
      <th>cwe_code_909</th>
      <th>cwe_code_913</th>
      <th>cwe_code_915</th>
      <th>cwe_code_916</th>
      <th>cwe_code_918</th>
      <th>cwe_code_920</th>
      <th>cwe_code_922</th>
      <th>cwe_code_924</th>
      <th>cwe_code_943</th>
      <th>cwe_code_1021</th>
      <th>cwe_code_1187</th>
      <th>cwe_code_1188</th>
      <th>access_authentication_MULTIPLE</th>
      <th>access_authentication_NONE</th>
      <th>access_authentication_SINGLE</th>
      <th>access_complexity_HIGH</th>
      <th>access_complexity_LOW</th>
      <th>access_complexity_MEDIUM</th>
      <th>access_vector_ADJACENT_NETWORK</th>
      <th>access_vector_LOCAL</th>
      <th>access_vector_NETWORK</th>
      <th>impact_availability_COMPLETE</th>
      <th>impact_availability_NONE</th>
      <th>impact_availability_PARTIAL</th>
      <th>impact_confidentiality_COMPLETE</th>
      <th>impact_confidentiality_NONE</th>
      <th>impact_confidentiality_PARTIAL</th>
      <th>impact_integrity_COMPLETE</th>
      <th>impact_integrity_NONE</th>
      <th>impact_integrity_PARTIAL</th>
      <th>mod_minute_0</th>
      <th>mod_minute_1</th>
      <th>mod_minute_2</th>
      <th>mod_minute_3</th>
      <th>mod_minute_4</th>
      <th>mod_minute_5</th>
      <th>mod_minute_6</th>
      <th>mod_minute_7</th>
      <th>mod_minute_8</th>
      <th>mod_minute_9</th>
      <th>mod_minute_10</th>
      <th>mod_minute_11</th>
      <th>mod_minute_12</th>
      <th>mod_minute_13</th>
      <th>mod_minute_14</th>
      <th>mod_minute_15</th>
      <th>mod_minute_16</th>
      <th>mod_minute_17</th>
      <th>mod_minute_18</th>
      <th>mod_minute_19</th>
      <th>mod_minute_20</th>
      <th>mod_minute_21</th>
      <th>mod_minute_22</th>
      <th>mod_minute_23</th>
      <th>mod_minute_24</th>
      <th>mod_minute_25</th>
      <th>mod_minute_26</th>
      <th>mod_minute_27</th>
      <th>mod_minute_28</th>
      <th>mod_minute_29</th>
      <th>mod_minute_30</th>
      <th>mod_minute_31</th>
      <th>mod_minute_32</th>
      <th>mod_minute_33</th>
      <th>mod_minute_34</th>
      <th>mod_minute_35</th>
      <th>mod_minute_36</th>
      <th>mod_minute_37</th>
      <th>mod_minute_38</th>
      <th>mod_minute_39</th>
      <th>mod_minute_40</th>
      <th>mod_minute_41</th>
      <th>mod_minute_42</th>
      <th>mod_minute_43</th>
      <th>mod_minute_44</th>
      <th>mod_minute_45</th>
      <th>mod_minute_46</th>
      <th>mod_minute_47</th>
      <th>mod_minute_48</th>
      <th>mod_minute_49</th>
      <th>mod_minute_50</th>
      <th>mod_minute_51</th>
      <th>mod_minute_52</th>
      <th>mod_minute_53</th>
      <th>mod_minute_54</th>
      <th>mod_minute_55</th>
      <th>mod_minute_56</th>
      <th>mod_minute_57</th>
      <th>mod_minute_58</th>
      <th>mod_minute_59</th>
      <th>mod_hour_0</th>
      <th>mod_hour_1</th>
      <th>mod_hour_2</th>
      <th>mod_hour_3</th>
      <th>mod_hour_4</th>
      <th>mod_hour_5</th>
      <th>mod_hour_6</th>
      <th>mod_hour_7</th>
      <th>mod_hour_8</th>
      <th>mod_hour_9</th>
      <th>mod_hour_10</th>
      <th>mod_hour_11</th>
      <th>mod_hour_12</th>
      <th>mod_hour_13</th>
      <th>mod_hour_14</th>
      <th>mod_hour_15</th>
      <th>mod_hour_16</th>
      <th>mod_hour_17</th>
      <th>mod_hour_18</th>
      <th>mod_hour_19</th>
      <th>mod_hour_20</th>
      <th>mod_hour_21</th>
      <th>mod_hour_22</th>
      <th>mod_hour_23</th>
      <th>mod_day_1</th>
      <th>mod_day_2</th>
      <th>mod_day_3</th>
      <th>mod_day_4</th>
      <th>mod_day_5</th>
      <th>mod_day_6</th>
      <th>mod_day_7</th>
      <th>mod_day_8</th>
      <th>mod_day_9</th>
      <th>mod_day_10</th>
      <th>mod_day_11</th>
      <th>mod_day_12</th>
      <th>mod_day_13</th>
      <th>mod_day_14</th>
      <th>mod_day_15</th>
      <th>mod_day_16</th>
      <th>mod_day_17</th>
      <th>mod_day_18</th>
      <th>mod_day_19</th>
      <th>mod_day_20</th>
      <th>mod_day_21</th>
      <th>mod_day_22</th>
      <th>mod_day_23</th>
      <th>mod_day_24</th>
      <th>mod_day_25</th>
      <th>mod_day_26</th>
      <th>mod_day_27</th>
      <th>mod_day_28</th>
      <th>mod_day_29</th>
      <th>mod_day_30</th>
      <th>mod_day_31</th>
      <th>mod_month_1</th>
      <th>mod_month_2</th>
      <th>mod_month_3</th>
      <th>mod_month_4</th>
      <th>mod_month_5</th>
      <th>mod_month_6</th>
      <th>mod_month_7</th>
      <th>mod_month_8</th>
      <th>mod_month_9</th>
      <th>mod_month_10</th>
      <th>mod_month_11</th>
      <th>mod_month_12</th>
      <th>mod_year_2008</th>
      <th>mod_year_2009</th>
      <th>mod_year_2010</th>
      <th>mod_year_2011</th>
      <th>mod_year_2012</th>
      <th>mod_year_2013</th>
      <th>mod_year_2014</th>
      <th>mod_year_2015</th>
      <th>mod_year_2016</th>
      <th>mod_year_2017</th>
      <th>mod_year_2018</th>
      <th>mod_year_2019</th>
      <th>pub_minute_0</th>
      <th>pub_minute_1</th>
      <th>pub_minute_2</th>
      <th>pub_minute_3</th>
      <th>pub_minute_4</th>
      <th>pub_minute_5</th>
      <th>pub_minute_6</th>
      <th>pub_minute_7</th>
      <th>pub_minute_8</th>
      <th>pub_minute_9</th>
      <th>pub_minute_10</th>
      <th>pub_minute_11</th>
      <th>pub_minute_12</th>
      <th>pub_minute_13</th>
      <th>pub_minute_14</th>
      <th>pub_minute_15</th>
      <th>pub_minute_16</th>
      <th>pub_minute_17</th>
      <th>pub_minute_18</th>
      <th>pub_minute_19</th>
      <th>pub_minute_20</th>
      <th>pub_minute_21</th>
      <th>pub_minute_22</th>
      <th>pub_minute_23</th>
      <th>pub_minute_24</th>
      <th>pub_minute_25</th>
      <th>pub_minute_26</th>
      <th>pub_minute_27</th>
      <th>pub_minute_28</th>
      <th>pub_minute_29</th>
      <th>pub_minute_30</th>
      <th>pub_minute_31</th>
      <th>pub_minute_32</th>
      <th>pub_minute_33</th>
      <th>pub_minute_34</th>
      <th>pub_minute_35</th>
      <th>pub_minute_36</th>
      <th>pub_minute_37</th>
      <th>pub_minute_38</th>
      <th>pub_minute_39</th>
      <th>pub_minute_40</th>
      <th>pub_minute_41</th>
      <th>pub_minute_42</th>
      <th>pub_minute_43</th>
      <th>pub_minute_44</th>
      <th>pub_minute_45</th>
      <th>pub_minute_46</th>
      <th>pub_minute_47</th>
      <th>pub_minute_48</th>
      <th>pub_minute_49</th>
      <th>pub_minute_50</th>
      <th>pub_minute_51</th>
      <th>pub_minute_52</th>
      <th>pub_minute_53</th>
      <th>pub_minute_54</th>
      <th>pub_minute_55</th>
      <th>pub_minute_56</th>
      <th>pub_minute_57</th>
      <th>pub_minute_58</th>
      <th>pub_minute_59</th>
      <th>pub_hour_0</th>
      <th>pub_hour_1</th>
      <th>pub_hour_2</th>
      <th>pub_hour_3</th>
      <th>pub_hour_4</th>
      <th>pub_hour_5</th>
      <th>pub_hour_6</th>
      <th>pub_hour_7</th>
      <th>pub_hour_8</th>
      <th>pub_hour_9</th>
      <th>pub_hour_10</th>
      <th>pub_hour_11</th>
      <th>pub_hour_12</th>
      <th>pub_hour_13</th>
      <th>pub_hour_14</th>
      <th>pub_hour_15</th>
      <th>pub_hour_16</th>
      <th>pub_hour_17</th>
      <th>pub_hour_18</th>
      <th>pub_hour_19</th>
      <th>pub_hour_20</th>
      <th>pub_hour_21</th>
      <th>pub_hour_22</th>
      <th>pub_hour_23</th>
      <th>pub_day_1</th>
      <th>pub_day_2</th>
      <th>pub_day_3</th>
      <th>pub_day_4</th>
      <th>pub_day_5</th>
      <th>pub_day_6</th>
      <th>pub_day_7</th>
      <th>pub_day_8</th>
      <th>pub_day_9</th>
      <th>pub_day_10</th>
      <th>pub_day_11</th>
      <th>pub_day_12</th>
      <th>pub_day_13</th>
      <th>pub_day_14</th>
      <th>pub_day_15</th>
      <th>pub_day_16</th>
      <th>pub_day_17</th>
      <th>pub_day_18</th>
      <th>pub_day_19</th>
      <th>pub_day_20</th>
      <th>pub_day_21</th>
      <th>pub_day_22</th>
      <th>pub_day_23</th>
      <th>pub_day_24</th>
      <th>pub_day_25</th>
      <th>pub_day_26</th>
      <th>pub_day_27</th>
      <th>pub_day_28</th>
      <th>pub_day_29</th>
      <th>pub_day_30</th>
      <th>pub_day_31</th>
      <th>pub_month_1</th>
      <th>pub_month_2</th>
      <th>pub_month_3</th>
      <th>pub_month_4</th>
      <th>pub_month_5</th>
      <th>pub_month_6</th>
      <th>pub_month_7</th>
      <th>pub_month_8</th>
      <th>pub_month_9</th>
      <th>pub_month_10</th>
      <th>pub_month_11</th>
      <th>pub_month_12</th>
      <th>pub_year_1999</th>
      <th>pub_year_2000</th>
      <th>pub_year_2001</th>
      <th>pub_year_2002</th>
      <th>pub_year_2003</th>
      <th>pub_year_2004</th>
      <th>pub_year_2005</th>
      <th>pub_year_2006</th>
      <th>pub_year_2007</th>
      <th>pub_year_2008</th>
      <th>pub_year_2009</th>
      <th>pub_year_2010</th>
      <th>pub_year_2011</th>
      <th>pub_year_2012</th>
      <th>pub_year_2013</th>
      <th>pub_year_2014</th>
      <th>pub_year_2015</th>
      <th>pub_year_2016</th>
      <th>pub_year_2017</th>
      <th>pub_year_2018</th>
      <th>pub_year_2019</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.382681</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.382681</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.799447</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.799447</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.799447</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-4c001e85-1c81-4b1e-a30d-38b317cbd029')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-4c001e85-1c81-4b1e-a30d-38b317cbd029 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-4c001e85-1c81-4b1e-a30d-38b317cbd029');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
df_train3.shape
```




    (71624, 473)



A third combination of preprocesses introduces the OneHotEncoder. This explodes the categorical columns into binary columns, so that the integer value of the labels does not impact the results of some modeling, where larger integer values than others could have a higher weight, while these integers are only encodings of categories.

# Modelization

Now that the data are pre-processed, they can be used to create models that answer our initial problem. To improve our results as much as possible, we will create several models, then we will determine the performance of each model. We will also have to use each pre-process on each model, in order to determine the best data processing chain.


```python
def testModel(df_train, df_test, targetName, model, showMetrics = False):
    # Separation of the target from the features
    y_train =  df_train[targetName]
    y_test =  df_test[targetName]
    X_train = df_train.drop(targetName, axis=1)
    X_test = df_test.drop(targetName, axis=1)
    # Fitting the model
    model.fit(X_train, y_train)
    # Getting the results on both the training set and the testing set
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    # Calculating the metrics and displaying them
    rmse=np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae=mean_absolute_error(y_test, y_pred_test)
    medae=median_absolute_error(y_test, y_pred_test)
    print("--> rmse =", rmse)
    print("--> mae =", mae)
    print("--> medae =", medae)
    # This representative metric can be changed if needed
    toReturn = medae
    if showMetrics:
        N, train_score, val_score = learning_curve(model, 
                                                  X_train, 
                                                  y_train, 
                                                  cv=5, 
                                                  scoring='neg_mean_squared_error',
                                                  train_sizes=np.linspace(0.1, 1, 10))
        plt.figure(figsize=(14,7))
        plt.title("Learning curve of the model: " + str(model.__class__.__name__))
        plt.plot(N, train_score.mean(axis=1), label='Train Score')
        plt.plot(N, val_score.mean(axis=1), label='Validation Score')
        plt.legend()
        
    return toReturn
```

The testModel function allows to apply an instantiated model passed in parameter to the data of a pre-processed dataset also passed in parameter. It will then be very easy to loop over all the models and pre-processings in order to compare them and thus to obtain the best chain of pre-processings and the best model.

The 3 metrics chosen to evaluate the models are :
- root mean square error, which is the square root of the calculated variance
- mean absolute error, which determines the average distance between the predicted results and the true value
- median absolute error, which determines the median distance separating the predicted results from the true value

In order to avoid outliers, we prefer to use the median here.


```python
 def getBestChain(l_preprocess, l_models):
    chosen = None
    tableResult = []
    print("Training")
    for p in l_preprocess:
        print("> Using preprocess:", p)
        df_train = l_preprocess[p]["train"]
        df_test = l_preprocess[p]["test"]
        targetName = l_preprocess[p]["target"]
        for m in l_models:
            print("-> Training on model:", m)
            result = {"model": m, "preprocess": p}
            result["score"] = testModel(df_train, df_test, targetName, l_models[m], showMetrics = False)
            tableResult.append(result)
    tableResult = sorted(tableResult, key=lambda d: d['score'], reverse=True)
    print(tableResult)
    return tableResult[0]
```

The getBestChain function will iterate over all preprocesses and models to find the best processing chain, with the median absolute error as the reference for the metric.


```python
# Pré-traitements à tester
l_preprocess = {"1": {"train": df_train1, "test": df_test1, "target": targetName},
                "2": {"train": df_train2, "test": df_test2, "target": targetName},
                "3": {"train": df_train3, "test": df_test3, "target": targetName}}
# Modèles à tester
l_models = {"Ridge": Ridge(random_state=_RANDOM_STATE_),
            "Ridge06": Ridge(alpha=0.06, random_state=_RANDOM_STATE_),
            "Ridge10": Ridge(alpha=10, random_state=_RANDOM_STATE_),
            "Lasso": Lasso(),
            "LinearSVR": LinearSVR(random_state=_RANDOM_STATE_),
            # "SVR" : SVR(), # Very long to execute + no impressive results
            "SGDRegressor" : SGDRegressor(random_state=_RANDOM_STATE_),
            "GradientBoostingRegressor" : GradientBoostingRegressor(random_state=_RANDOM_STATE_),
            "RandomForestRegressor": RandomForestRegressor(random_state=_RANDOM_STATE_),
            "LinearRegression": LinearRegression()
            }
bestResult = getBestChain(l_preprocess, l_models)
```

    Training
    > Using preprocess: 1
    -> Training on model: Ridge
    --> rmse = 1.9025306057114988
    --> mae = 1.7242452115649627
    --> medae = 1.662370996227533
    -> Training on model: Ridge06
    --> rmse = 1.9025300813867134
    --> mae = 1.7242451667815806
    --> medae = 1.6623722667608867
    -> Training on model: Ridge10
    --> rmse = 1.90253562200261
    --> mae = 1.7242456400419854
    --> medae = 1.6623588329742605
    -> Training on model: Lasso
    --> rmse = 1.9170782150770287
    --> mae = 1.7256380970084615
    --> medae = 1.6240631159767067
    -> Training on model: LinearSVR
    --> rmse = 2.001655082191505
    --> mae = 1.8302723235274512
    --> medae = 1.8202729692135438
    -> Training on model: SGDRegressor
    --> rmse = 1114475927.8876984
    --> mae = 889625756.8276603
    --> medae = 759826967.6410962
    -> Training on model: GradientBoostingRegressor
    --> rmse = 1.8692201701736617
    --> mae = 1.72372571138184
    --> medae = 1.6631306949110445
    -> Training on model: RandomForestRegressor
    --> rmse = 1.8674680571386209
    --> mae = 1.7241079997086237
    --> medae = 1.6193690491271293
    -> Training on model: LinearRegression
    --> rmse = 1.9025300479165963
    --> mae = 1.7242451639228655
    --> medae = 1.6623723478597114
    > Using preprocess: 2
    -> Training on model: Ridge
    --> rmse = 15.328213298160176
    --> mae = 13.890794575930899
    --> medae = 13.4166958210012
    -> Training on model: Ridge06
    --> rmse = 15.328208992574146
    --> mae = 13.890794199950928
    --> medae = 13.416705650680775
    -> Training on model: Ridge10
    --> rmse = 15.328254490643314
    --> mae = 13.890798173235988
    --> medae = 13.416608528085298
    -> Training on model: Lasso
    --> rmse = 15.443857210026076
    --> mae = 13.901628090489242
    --> medae = 13.083346660535957
    -> Training on model: LinearSVR
    --> rmse = 15.246334037389628
    --> mae = 13.812896575041075
    --> medae = 13.606019919378276
    -> Training on model: SGDRegressor
    --> rmse = 15.100352970043291
    --> mae = 13.635602622688667
    --> medae = 13.129309080504473
    -> Training on model: GradientBoostingRegressor
    --> rmse = 15.062824703420889
    --> mae = 13.886217366437684
    --> medae = 13.34452648458999
    -> Training on model: RandomForestRegressor
    --> rmse = 15.053761246680837
    --> mae = 13.8896713549665
    --> medae = 13.045531563797057
    -> Training on model: LinearRegression
    --> rmse = 15.328208717728554
    --> mae = 13.890794175950527
    --> medae = 13.416706333248278
    > Using preprocess: 3
    -> Training on model: Ridge
    --> rmse = 0.31612469749529926
    --> mae = 0.18523075742964096
    --> medae = 0.11925608462623616
    -> Training on model: Ridge06
    --> rmse = 0.3161154214701427
    --> mae = 0.18522829167045776
    --> medae = 0.11920086457683121
    -> Training on model: Ridge10
    --> rmse = 0.3163786744157716
    --> mae = 0.18536913737599853
    --> medae = 0.11982378765276391
    -> Training on model: Lasso
    --> rmse = 1.0065117224833633
    --> mae = 0.8499308231856115
    --> medae = 0.7874849980032461
    -> Training on model: LinearSVR
    --> rmse = 0.35000674349283967
    --> mae = 0.13508589265909593
    --> medae = 0.012909620712256986
    -> Training on model: SGDRegressor
    --> rmse = 0.31801000526921935
    --> mae = 0.1845104512959773
    --> medae = 0.11773015869140185
    -> Training on model: GradientBoostingRegressor
    --> rmse = 0.22257589500239525
    --> mae = 0.08319819436596887
    --> medae = 0.03252970271660405
    -> Training on model: RandomForestRegressor
    --> rmse = 0.181525163276696
    --> mae = 0.031156834844837483
    --> medae = 8.326672684688674e-15
    -> Training on model: LinearRegression
    --> rmse = 2906851.3339366084
    --> mae = 37624.79709020385
    --> medae = 0.11853968550324556
    [{'model': 'SGDRegressor', 'preprocess': '1', 'score': 759826967.6410962}, {'model': 'LinearSVR', 'preprocess': '2', 'score': 13.606019919378276}, {'model': 'LinearRegression', 'preprocess': '2', 'score': 13.416706333248278}, {'model': 'Ridge06', 'preprocess': '2', 'score': 13.416705650680775}, {'model': 'Ridge', 'preprocess': '2', 'score': 13.4166958210012}, {'model': 'Ridge10', 'preprocess': '2', 'score': 13.416608528085298}, {'model': 'GradientBoostingRegressor', 'preprocess': '2', 'score': 13.34452648458999}, {'model': 'SGDRegressor', 'preprocess': '2', 'score': 13.129309080504473}, {'model': 'Lasso', 'preprocess': '2', 'score': 13.083346660535957}, {'model': 'RandomForestRegressor', 'preprocess': '2', 'score': 13.045531563797057}, {'model': 'LinearSVR', 'preprocess': '1', 'score': 1.8202729692135438}, {'model': 'GradientBoostingRegressor', 'preprocess': '1', 'score': 1.6631306949110445}, {'model': 'LinearRegression', 'preprocess': '1', 'score': 1.6623723478597114}, {'model': 'Ridge06', 'preprocess': '1', 'score': 1.6623722667608867}, {'model': 'Ridge', 'preprocess': '1', 'score': 1.662370996227533}, {'model': 'Ridge10', 'preprocess': '1', 'score': 1.6623588329742605}, {'model': 'Lasso', 'preprocess': '1', 'score': 1.6240631159767067}, {'model': 'RandomForestRegressor', 'preprocess': '1', 'score': 1.6193690491271293}, {'model': 'Lasso', 'preprocess': '3', 'score': 0.7874849980032461}, {'model': 'Ridge10', 'preprocess': '3', 'score': 0.11982378765276391}, {'model': 'Ridge', 'preprocess': '3', 'score': 0.11925608462623616}, {'model': 'Ridge06', 'preprocess': '3', 'score': 0.11920086457683121}, {'model': 'LinearRegression', 'preprocess': '3', 'score': 0.11853968550324556}, {'model': 'SGDRegressor', 'preprocess': '3', 'score': 0.11773015869140185}, {'model': 'GradientBoostingRegressor', 'preprocess': '3', 'score': 0.03252970271660405}, {'model': 'LinearSVR', 'preprocess': '3', 'score': 0.012909620712256986}, {'model': 'RandomForestRegressor', 'preprocess': '3', 'score': 8.326672684688674e-15}]
    

The results vary enormously according to the pre-processing, and a little less according to the models. However, we locate here a processing chain that reduces the absolute median error: random forest with the 3rd pre-processing chain.

We can now try to improve the results obtained by optimizing the parameters of the latter.

# Evaluation

In order to check if the model is not overfitted, we can display its learning curve.


```python
def evaluation(model, X_train, y_train, X_test, y_test):
    N, train_score, val_score = learning_curve(model, 
                                              X_test, 
                                              y_test, 
                                              cv=5, 
                                              scoring='neg_mean_squared_error',
                                              train_sizes=np.linspace(0.1, 1, 5))
    plt.figure(figsize=(14,7))
    plt.title("Learning curve of the model: " + str(model.__class__.__name__))
    plt.plot(N, train_score.mean(axis=1), label='Train Score')
    plt.plot(N, val_score.mean(axis=1), label='Validation Score')
    plt.legend()
    
    
```


```python
X_train = df_train3.drop(targetName, axis = 1)
y_train = df_train3[targetName]

X_test = df_test3.drop(targetName, axis = 1)
y_test = df_test3[targetName]
model = RandomForestRegressor(random_state=_RANDOM_STATE_)
evaluation(model, X_train, y_train, X_test, y_test)
```


    
![png](output_123_0.png)
    


We notice here that the model is not overfitting, which is positive for our problem. Indeed, the two curves seem to converge, and the validation curve has a strictly increasing score, which means that the results on the validation set are improving. However, we can restart this evaluation by passing the testing set, which is 4 times larger than the testing set, as a parameter, if more computational resources are available.

# Optimization

In order to further improve the results, it is possible to perform a parameter optimization on the chosen model. Here, it is possible to optimize various parameters of RandomForestRegressor:
- n_estimators, the number of trees per forest, thus the number of estimators
- max_depth, the maximum depth of each tree (risk of being exponential in complexity)
- min_samples_split, minimum number of samples in the dataset to cut the experiment into a new node

Other even more precise parameters can also be optimized. However, here, we will limit ourselves to the number of estimators, because otherwise the duration of the experiment would be too long (hardware limit).


```python
def optimisation(model, x_train, y_train, param_opti, metric, interv_min, interv_max, pas):
    param_grid={param_opti:  np.arange(interv_min, interv_max, pas)}
    
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring=metric)
    grid.fit(x_train, y_train)
    print("Best score est: ", grid.best_score_)
    print("Best value of the optimized parameter: ", grid.best_params_)
    
    param_range = np.arange(interv_min, interv_max, pas)
    
    train_score, val_score = validation_curve(model, 
                                              x_train,
                                              y_train,
                                              param_name=param_opti, 
                                              param_range=param_range, 
                                              cv=5,
                                             scoring=metric)

    plt.figure(figsize=(12, 4))
    plt.plot(param_range, train_score.mean(axis = 1), label = 'train')
    plt.plot(param_range, val_score.mean(axis = 1), label = 'validation')
    plt.legend()
    plt.title("Validation curve for " + str(model.__class__.__name__))
    plt.ylabel('score')
    plt.xlabel('Regularization parameter: '+param_opti)
    plt.show()
    
    return grid.best_estimator_
```


```python
grid_result = optimisation(model, X_train, y_train, "n_estimators", "r2", 10, 500, 100)
```

Because it is too long to execute on my computer, I haven't executed this part. However, simply uncomment it if time is not a problem.

# Conclusion

In conclusion, the dataset is unfortunately quite noisy and few strong correlations appear with the target value. However, we were able to perform a chain of treatments giving satisfactory results, with a confidence of about 2%. The use of OneHotEncoder increases the complexity, the training times of the models are multiplied, but allow to give very good results for the majority of the models, moreover with RandomForestRegressor.

We can also propose some improvement tracks:
- Add new models
- Add new pre-processing
- Introduce NLP on the summary column with a Tensorflow model
- Try the class approach (classification)
- Improve the quality of the dataset
