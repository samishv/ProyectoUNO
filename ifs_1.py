
import carta
from skimage import io

#Variable para controlar comprobaciones 
a=0

#Seccion donde se lee la carta en la mesa
mesa = carta.Carta("Azul","1")
ima = io.imread(r"F:\ARCHIVOS HDD\Materias\7mo Semestre\Reconocimiento de Patrones\proyectoUNO\fotos_uno\verde_0.jpg")
mesa.identificar_color(ima)
mesa.atributos()

#Seccion que lee 1 carta de la maquina
compu = carta.Carta("Azul","1")
        

def check_color(mesa,compu):
    if mesa.color == compu.color:
        print("Maquina juega: ")
        mesa.atributos()
        a=1
    else:
        print("No se puede jugar la carta")
        a=0
        
    return a

def check_symbol(mesa,compu,a):
    if a == 0:
        if mesa.contenido == compu.contenido:
            print("Maquina juega: ")
            mesa.atributos()
            a=1
        else:
            print("No se puede jugar la carta")
            a=0
    
    return a

