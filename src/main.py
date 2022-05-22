from matplotlib.pyplot import switch_backend
import train as t
import predict as p
import pandas as pd
import trainSelector as ts
import config

if __name__ == "__main__":
    
#   Muestra la medida F1 para los 4 modelos disponibles y los entrena, guardando el modelo (Para cada fold)
    ts.trainSelector()
    
    print("\nMedida con la que entrenar el modelo: ")
    print("1.- Regresion Logistica")
    print("2.- Naive Bayes")
    print("3.- Random Forest")
    print("4.- SVM")

    option = input("Opcion -- ")
    
    if option == "1":
        model = "logistic_regression"
    elif option == "2":
        model = "naive_bayes"
    elif option == "3":
        model = "random_forest"
    elif option == "4":
        model = "svm"
    
#   Hacemos las predicciones y generamos el csv final
    submission = p.predict(model_type=model)
    sample_sub = pd.read_csv(config.SUBMISSION)
    sample_sub.loc[:, config.TARGET] = submission
    sample_sub.to_csv(f"{config.MODEL_DIR}{model}.csv", index=False)

    print("Prediccion realizada con exito!")