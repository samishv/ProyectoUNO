# Código principal del juego

import carta # Es el archivo o modulo que contiene la clase carta
from skimage import io

# Crea una carta random de prueba
mi_carta = carta.Carta("azul", "uno")
mi_carta.atributos()

# Prueba actualizar el atributo color después de leer una imagen
imagen_leida = io.imread(r"C:\Users\ikerf\Desktop\Upiita\7mo semestre\Patrones\fotos_uno\fotos_uno\verde_1.jpg")
mi_carta.identificar_color(imagen_leida)
mi_carta.atributos() 