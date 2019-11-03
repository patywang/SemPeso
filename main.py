
import wisardpkg as wp
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

# 1 modelagem de entrada (limiar estático e vizinhança) ???
# 2 tamanho da tupla ??
# 3 vários mapeamentos ??
# 4 k-fold (conjunto de k partes + desvio padrao e k partes == tam do conjunto de dados e sem desvio padrao)
#   matriz de confusão ->  from sklearn.metrics import confusion_matrix -> precisa da predição
#   análise qualitativa
#   pedir pra vizualizar a imagem mental ( o wisard tem mas so aparece número ...)

data = pd.read_csv('./train.csv', sep=',')
label = data['label'].apply(str)

del data['label']
# print(data.head(5))

threshold = 100

appliedData = data.apply(lambda x: (x >= threshold).astype(
    int) if x.name != 'label' else x)

# print(appliedData.head(15))

# X = appliedData.values.tolist()
# y = label.values.tolist()

addressSize = 3     # number of addressing bits in the ram
ignoreZero = False  # optional; causes the rams to ignore the address 0

# False by default for performance reasons,
# when True, WiSARD prints the progress of train() and classify()
verbose = False
k_fold = 2
wsd = wp.Wisard(addressSize, ignoreZero=ignoreZero, verbose=verbose)
kf = KFold(n_splits=k_fold, shuffle=True)
acuracia_list = []
for train_index, test_index in kf.split(appliedData):

    train_based_values = appliedData.loc[train_index].values
    train_based_values_list = train_based_values.tolist()

    label_train_based = label.loc[train_index].values
    label_train_based_list = label_train_based.tolist()

    test_based_values = appliedData.loc[test_index].values
    test_based_values_list = test_based_values.tolist()

    label_test_based = label.loc[test_index].values
    label_test_based_list = label_test_based.tolist()

    try:
        wsd.train(train_based_values_list, label_train_based_list)

        out = wsd.classify(train_based_values_list)

        numAcertos = 0
        for i, d in enumerate(train_based_values_list):
            if(out[i] == label_test_based_list[i]):
                numAcertos = numAcertos + 1

        acuracia = numAcertos/len(label_train_based_list)
        print('Acurácia:', acuracia)
        acuracia_list.append(acuracia)
    # apanhando a imagem mental
        patterns = wsd.getMentalImages()
        print("Imagem Mental:")
        for key in patterns:
            print(key, patterns[key])
        print("Matriz de Confusão:")
        print(confusion_matrix(label_test_based_list, out))


#     #apanhando a imagem mental
#     patterns = wsd.getMentalImages()

    except Exception as e:
        print(str(e))
print("K-FOLD: ", k_fold)
print("MÉDIA acuracia: ", np.mean(acuracia_list))


# esse codigo na minah máquina esta funcionando sem erro algum.
# wisard sem kfold e imagem mental
# try:
#     # train using the input data
#     wsd.train(X,y)

#     # classify some data
#     out = wsd.classify(X)
#     print("passou do classify")
#     # the output of classify is a string list in the same sequence as the input
#     for i,d in enumerate(X):
#         print(out[i],d)


#     #apanhando a imagem mental
#     patterns = wsd.getMentalImages()

#     print("Imagem Mental")
#     for key in patterns:
#         print(key, patterns[key])


#     #print("Wisard: ", wsd.json(True,"/home/patricia/Work/PESC/Redes Neurais Sem Peso/"))

#     print("TERMINOU!")
# except Exception as e:
#     print(str(e))
