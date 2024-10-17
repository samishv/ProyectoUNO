
#Primero observamos si no existe una carta cancelar/cambio/añadir cartas
def check_especial(mesa,compu):
    if mesa.atributos=="Cancelar":
        print("No se puede jugar en este turno")
        a=1
    elif mesa.atributos=="Cambio":
        print("Se cambio el sentido de juego")
        a=1
    elif mesa.atributos=="mas1":
        print("Se añade 1 carta al mazo")
        compu = agregar(compu,add=1)
        a=1
    elif mesa.atributo=="mas2":
        print("Se añaden 2 cartas al mazo")
        compu = agregar(compu,add=2)
        a=1
    
    return a

#Se observa si se puede jugar carta de color
def check_color(mesa,compu,a):
    if a==0:
        if mesa.color == compu.color:
            print("Maquina juega: ")
            compu.atributos()
            a=1
        else:
            print("No se puede jugar la carta")
            a=0
        
    return a

#Se observa si se puede jugar carta de numero
def check_symbol(mesa,compu,a):
    if a == 0:
        if mesa.contenido == compu.contenido:
            print("Maquina juega: ")
            compu.atributos()
            a=1
        else:
            print("No se puede jugar la carta")
            a=0
    
    return a

#Se observa si se puede jugar carta negra (Cambiar color o Mas 2)
def check_negro(compu,a):
    if compu.color == "Negro":
        print("Maquina juega:")
        compu.atributos()
        a=1
    else:
        #Para este punto la ultima opcion es tomar una carta
        a=0

    return a

#Funcion agregar cartas
def agregar(compu,add):
    for i in range(add):
        nueva = ["Color","numero"]
        compu.extend(compu(nueva))
    
    return compu

# #Funcion quitar carta jugada
# def delete(compu,):
#     for i in range(add):
#         nueva = ["Color","numero"]
#         compu.extend(compu(nueva))
    
#     return compu

import carta
from skimage import io
import numpy as np

#Variable para controlar comprobaciones 
a=0

#Seccion donde se lee la carta en la mesa o se modifica el color del juego (ej. Cambiar color)
mesa = carta.Carta("Azul","1")
ima = io.imread(r"F:\ARCHIVOS HDD\Materias\7mo Semestre\Reconocimiento de Patrones\proyectoUNO\fotos_uno\verde_0.jpg")
mesa.identificar_color(ima)
mesa.atributos()

#Seccion que lee las cartas de la maquina
compu = np.zeros(5)
compu[0] = carta.Carta("Azul","1")
compu[1] = carta.Carta["Verde","3"]

numcartas = compu.shape[1]

for i in range(numcartas):
   a = check_especial(mesa)
   a = check_color(mesa,compu[i],a)
   a = check_symbol(mesa, compu[i], a)
   a = check_negro(compu[i],a)
   
   