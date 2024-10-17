# Código principal del juego

import carta # Es el archivo o modulo que contiene la clase carta
from skimage import io

# Crea una carta random de prueba
mi_carta = carta.Carta("azul", "uno")
mi_carta.atributos()

# Prueba actualizar el atributo color después de leer una imagen
imagen_leida = io.imread(r"F:\ARCHIVOS HDD\Materias\7mo Semestre\Reconocimiento de Patrones\proyectoUNO\fotos_uno\amarillo_0.jpg")
mi_carta.identificar_color(imagen_leida)
mi_carta.atributos() 