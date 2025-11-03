from django.shortcuts import render
from .services import SistemaRecomendacion
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELO_PATH = os.path.join(BASE_DIR, 'models', 'modelo_regresion_logistica.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

def index(request):
    return render(request, 'index.html')

def predecir_view(request):
    resultado = None
    if request.method == 'POST':
            nombre = request.POST.get('nombre', '')
            edad = request.POST.get('edad')
            salario = request.POST.get('salario')
            if edad is None or salario is None:
                raise ValueError("Edad o salario vacíos")

            edad = int(edad)
            salario = float(salario)
            sistema = SistemaRecomendacion(MODELO_PATH, SCALER_PATH)
            resultado = sistema.analizar_cliente(edad, salario, nombre)

    return render(request, 'index.html', {'resultado': resultado,})
