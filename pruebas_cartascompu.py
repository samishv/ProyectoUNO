import carta
from skimage import io
import numpy as np

#Variable para controlar comprobaciones 
a=0

#Seccion donde se lee la carta en la mesa o se modifica el color del juego (ej. Cambiar color)
mesa = carta.Carta("Azul","1")
ima = io.imread(r"F:\ARCHIVOS HDD\Materias\7mo Semestre\Reconocimiento de Patrones\proyectoUNO\fotos_uno\verde_0.jpg")
# mesa.identificar_color(ima)
# mesa.atributos()

#Seccion que lee las cartas de la maquina
compu1 = carta.Carta("Azul","1")
compu2 = carta.Carta("Verde","3")

compuT = [compu1,compu2]

# numcartas = compu.shape[1]

compuT[1].atributos()