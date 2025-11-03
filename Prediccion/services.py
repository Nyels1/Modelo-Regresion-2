import joblib
import numpy as np

class SistemaRecomendacion:
    def __init__(self, modelo_path, scaler_path):
        self.modelo = joblib.load(modelo_path)
        self.scaler = joblib.load(scaler_path)
        self.umbral_confianza = 0.9

    def analizar_cliente(self, edad, salario, nombre=""):
        datos_cliente = np.array([[edad, salario]])
        datos_escalados = self.scaler.transform(datos_cliente)

        prediccion = self.modelo.predict(datos_escalados)[0]
        probabilidades = self.modelo.predict_proba(datos_escalados)[0]
        confianza = max(probabilidades)

        if prediccion == 1 and confianza >= self.umbral_confianza:
            recomendacion = "CLIENTE PROMETEDOR - Invertir en marketing"
            accion = "Contactar con ofertas personalizadas"
        elif prediccion == 1 and confianza < self.umbral_confianza:
            recomendacion = "CLIENTE POTENCIAL - Monitorear"
            accion = "Enviar comunicaciones generales"
        else:
            recomendacion = "CLIENTE ESTÁNDAR - Mantener contacto básico"
            accion = "Incluir en mailing list general"

        return {
            'nombre': nombre,
            'edad': edad,
            'salario': salario,
            'prediccion': prediccion,
            'probabilidad_compra': probabilidades[1],
            'confianza': confianza,
            'recomendacion': recomendacion,
            'accion': accion
        }
