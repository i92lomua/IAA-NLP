import config
import pandas as pd
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import StratifiedKFold

nltk.download("stopwords")

##  Funcion para crear los pliegues de la validacion cruzada
def get_folds() -> None:
#   Leemos el train set original
    df = pd.read_csv(config.ORIGINAL_TRAIN)

#   Creamos una nueva columna y la rellenamos con -1
    df[config.FOLD] = -1

#   Mezclamos los datos
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

#   Inicializamos el kfold (validacion cruzada) con 5 pliegues
    kfold = StratifiedKFold(n_splits=5, shuffle=False)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X=df, y=df[config.TARGET])):
        df.loc[val_idx, config.FOLD] = fold

#   Guardamos el nuevo train set
    df.to_csv(config.MODIFIED_TRAIN, index=False)

##  Funcion para separar cada tweet en tokens, eliminando lo que no necesitamos
def clean_tweet(tweet:str) -> str:
    
#   Eliminamos etiquetas del mercado de valores como $GE
    tweet = re.sub(r'\$\w*', '', str(tweet))

#   Eliminamos el antiguo retweet "RT"
    tweet = re.sub(r'^RT[\s]+', '', str(tweet))

#   Eliminamos hypervinculos
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', str(tweet))

#   Eliminamos hashtags
    tweet = re.sub(r'#', '', str(tweet))
    
#   Eliminamos puntuaciones
    punct = set(string.punctuation)
    tweet = "".join(ch for ch in tweet if ch not in punct)
    
#   Eliminamos stopwords
    stop_words = set(stopwords.words("english"))
    tweet = " ".join(word for word in tweet.split() if word not in stop_words)
    
#   Devolvemos el tweet limpio
    return tweet