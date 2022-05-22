import train
import preprocessing as pp

#   Funcion para elegir el metodo de aprendizaje que usaremos para realizar inferencia
def trainSelector():

    sum = 0

    pp.get_folds()
    print("\n")
    for i in range(5):
        sum += train.run_training(model="logistic_regression", fold=i)
    
    avg = sum / 5

    print("\nLogistic Regression Average: ", avg)
    
    print("\n")
    sum = 0
    for i in range(5):
        sum += train.run_training(model="naive_bayes", fold=i)
    
    avg = sum / 5

    print("\nNaive Bayes Average: ", avg)

    print("\n")
    sum = 0
    for i in range(5):
        sum += train.run_training(model="random_forest", fold=i)
    
    avg = sum / 5

    print("\nRandom Forest Average: ", avg)

    print("\n")
    sum = 0
    for i in range(5):
        sum += train.run_training(model="svm", fold=i)
    
    avg = sum / 5

    print("\nSVM Average: ", avg)


        