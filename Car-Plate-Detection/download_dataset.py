from roboflow import Roboflow
import os

# Inicializa o Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")  # Você precisará fornecer sua API key
project = rf.workspace("trabalho-jnal6").project("placa-de-carro-oz0eg")
dataset = project.version(6).download("yolov8")

print("Dataset baixado com sucesso!")
