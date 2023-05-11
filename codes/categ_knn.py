##################################################
## {Description}
##################################################
## {License_info}
##################################################
## Author: Cledson Sousa
## ## License: GNU GPLv3
## Version: 1.3.0
## Mmaintainer: Julia e Marielly
## Email: 
## Status: dev
##################################################
## This code uses KNN model to train file tickets.csv and classify new tickets among 
# those already created and print the description of the ticket and the category it 
# belongs to.
###################################################


import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sys import argv 

# baixa as stopwords em português
#nltk.download('stopwords')

# carrega os dados em um dataframe pandas
data = pd.read_csv('tickets.csv', sep=';', encoding='utf-8')

# pré-processamento dos dados
data['Descricao'] = data['Descricao'].str.replace('[^a-zA-Z0-9 \n\.]', '', regex=True)
data['Descricao'] = data['Descricao'].str.lower()

# remove as stopwords das descrições
stopwords_pt = set(stopwords.words('portuguese'))
data['Descricao'] = data['Descricao'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords_pt]))

# label encoding dos dados de saída
le = LabelEncoder()
data['Servico'] = le.fit_transform(data['Servico'])

# cria um pipeline para a vetorização e classificação das sentenças
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', KNeighborsClassifier(n_neighbors=5))])

# divide os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(data['Descricao'], data['Servico'], test_size=0.30, random_state=42)

# treina o modelo nos dados de treinamento
text_clf.fit(X_train, y_train)

# avalia o desempenho do modelo nos dados de teste
accuracy = text_clf.score(X_test, y_test)
print("Acurácia:", accuracy)

# faz previsões em novos dados usando o modelo treinado
#new_data = ["O meu computador não liga. Preciso de ajuda.", "A minha impressora está com problema. O que devo fazer?"]
new_data = argv[1:]
new_data_pred = text_clf.predict(new_data)
new_data_pred = le.inverse_transform(new_data_pred) # transforma as previsões de volta para as categorias originais
print("Previsões para novos dados:")
for i, desc in enumerate(new_data):
    print(f"O chamado:\"{desc}\" foi incluído na categoria",f"\"{new_data_pred[i]}\"", f"com {accuracy*100:.2f}% de certeza")


import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sys import argv 

# baixa as stopwords em português
#nltk.download('stopwords')

# carrega os dados em um dataframe pandas
data = pd.read_csv('tickets.csv', sep=';', encoding='utf-8')

# pré-processamento dos dados
data['Descricao'] = data['Descricao'].str.replace('[^a-zA-Z0-9 \n\.]', '', regex=True)
data['Descricao'] = data['Descricao'].str.lower()

# remove as stopwords das descrições
stopwords_pt = set(stopwords.words('portuguese'))
data['Descricao'] = data['Descricao'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords_pt]))

# label encoding dos dados de saída
le = LabelEncoder()
data['Servico'] = le.fit_transform(data['Servico'])

# cria um pipeline para a vetorização e classificação das sentenças
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', KNeighborsClassifier(n_neighbors=5))])

# divide os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(data['Descricao'], data['Servico'], test_size=0.30, random_state=42)

# treina o modelo nos dados de treinamento
text_clf.fit(X_train, y_train)

# avalia o desempenho do modelo nos dados de teste
accuracy = text_clf.score(X_test, y_test)
print("Acurácia:", accuracy)

# faz previsões em novos dados usando o modelo treinado,parece errado
#new_data = ["O meu computador não liga. Preciso de ajuda.", "A minha impressora está com problema. O que devo fazer?"]
new_data = argv[1:]
new_data_pred = text_clf.predict(new_data)
new_data_pred = le.inverse_transform(new_data_pred) # transforma as previsões de volta para as categorias originais
print("Previsões para novos dados:")
for i, desc in enumerate(new_data):
    print(f"O chamado:\"{desc}\" foi incluído na categoria",f"\"{new_data_pred[i]}\"", f"com {accuracy*100:.2f}% de certeza")

