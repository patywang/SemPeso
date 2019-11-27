import wisardpkg as wp
from sklearn.datasets import fetch_openml
import copy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np

print("Leitura da base...")
mnist_data = fetch_openml('mnist_784', version=1)
Xaux, yaux = mnist_data['data'], mnist_data['target']
print("OK!")

threshold = 125

Xaux2=Xaux.astype(int)
Xaux2[Xaux2>threshold] = 1
Xaux3 = copy.deepcopy(Xaux2)

for idx1, i in enumerate(Xaux3): # percorrer elementos (sublistas), e respetivos indices, contidos na lista principal
  for idx2, j in enumerate(Xaux3[idx1]): # percorrer elementos e respetivos indices contidos na sublista, aqui enumerate(i) tambem daria
    if(j <= threshold and j>1):
      Xaux3[idx1][idx2] = 0

print("Separando os elementos em conjuntos...")
X_train, X_test, y_train, y_test = Xaux3[:60000].astype(int), Xaux3[60000:].astype(int), yaux[:60000], yaux[60000:]
n_folds = 3
address=30
minScore = 0.1
discriminatorLimit = 5 

print("Número de folds: ", n_folds)
print("Address Size: ", address)
print("MinScore:", minScore)
print("discriminatorLimit:", discriminatorLimit)
clus = wp.ClusWisard(address, minScore, threshold, discriminatorLimit, verbose=False)

kfold = KFold(n_folds, shuffle=True)

acuracia_list = []
acertos_list = []
for train_ix, test_ix in kfold.split(X_train):
  trainX, trainY, testX, testY = X_train[train_ix], y_train[train_ix], X_train[test_ix], y_train[test_ix]
  print("vai treinar")
  clus.train(trainX, trainY)
  print("treinou")
  out = clus.classify(testX)
  print("classificou")
  
  # acertos=0
  # for i, d in enumerate(testY):
  #     if(out[i] == testY[i]):
  #         acertos = acertos + 1
  
  print("Matriz de confusão:")
  print(confusion_matrix(testY, out))
  
  #array = confusion_matrix(testY, out)
#   df_cm = pd.DataFrame(array, index = [i for i in "0123456789"],
#                   columns = [i for i in "0123456789"])
#   plt.figure(figsize = (10,10))
#   sn.heatmap(df_cm, annot=True, cmap="GnBu")
#   plt.show()
  
  #print("Acurácia: ", (acertos/len(out))*100)
  ac_score = accuracy_score(testY,out, normalize=True)
  acerto = accuracy_score(testY,out, normalize=False)
  acuracia_list.append(ac_score)
  acertos_list.append(acerto)
  
  patterns = clus.getMentalImages()
  print("Mapa mental...")
  for key in patterns:
    cluster = patterns[key]
    for index,discriminator in enumerate(cluster):
      pixels = np.array(discriminator)
      pixels = pixels.reshape((28, 28))
      plt.imshow(pixels, cmap='viridis')
      plt.show()

  

  print("Numero de acertos:",acerto)
  print("Porcentagem em acuracy_score:",ac_score )
  print("Porcentagem em acuracy_score com % :",ac_score*100 )
  
print("Media Acurácia:", np.mean(acuracia_list))
print("Média Variância:", np.var(acuracia_list))
print("Média Desvio Padrão:", np.std(acuracia_list))
print("Média de acertos:", np.mean(acertos_list))