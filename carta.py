# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:18:42 2024

@author: samarav
"""
from skimage import data,io
import numpy as np
import matplotlib.pyplot as plt
from statistics import mode
from collections import Counter


class Carta:
    def __init__(self,color,contenido):
        self.color = color
        self.contenido = contenido
        
    def atributos(self):
        print("- Color: ", self.color)
        print("- Contenido: ",self.contenido)
          
        
    def identificar_color(self,imagen):
        clase_rojo = [107.9, 9.4, 3.3]
        clase_amarillo = [174.9, 162.8, 40.7]
        clase_verde = [63.9, 110, 28.5]
        clase_azul = [3.3, 67.2, 135.1]

        #imagen = io.imread(r"C:\Users\ikerf\Desktop\Upiita\7mo semestre\Patrones\fotos_uno\fotos_uno\verde_1.jpg")
        imagen = imagen[:, 310:1050]
        plt.figure(0)
        plt.imshow(imagen)
        n = 0
        pix_asignados = np.zeros(imagen.shape[0]*imagen.shape[1])

        for i in range(imagen.shape[0]):
            for j in range(imagen.shape[1]):
                dato = imagen[i,j,:]
                dist1 = np.sqrt((dato[0]-clase_rojo[0])**2 + (dato[1]-clase_rojo[1])**2 + (dato[2]-clase_rojo[2])**2)
                dist2 = np.sqrt((dato[0]-clase_amarillo[0])**2 + (dato[1]-clase_amarillo[1])**2 + (dato[2]-clase_amarillo[2])**2)
                dist3 = np.sqrt((dato[0]-clase_verde[0])**2 + (dato[1]-clase_verde[1])**2 + (dato[2]-clase_verde[2])**2)
                dist4 = np.sqrt((dato[0]-clase_azul[0])**2 + (dato[1]-clase_azul[1])**2 + (dato[2]-clase_azul[2])**2)
                
                distancias = np.array([dist1,dist2,dist3,dist4])
                minimo = np.argmin(distancias)
                
                if minimo == 0 and distancias[0] < 50:
                    pix_asignados[n] = 1
                elif minimo == 1 and distancias[1] < 50:
                    pix_asignados[n] = 2
                elif minimo == 2 and distancias[2] < 50:
                    pix_asignados[n] = 3
                elif minimo == 3 and distancias[3] < 50:
                    pix_asignados[n] = 4
                else:
                    pix_asignados[n] = 5
                
                n += 1

        clases_mapeo = {1: 'Rojo',
                        2: 'Amarillo',
                        3: 'Verde',
                        4: 'Azul',
                        5: 'Desconocido'}  
        
        conteo = Counter(pix_asignados)
        frecuencias_ordenadas = conteo.most_common()
        color = clases_mapeo[frecuencias_ordenadas[1][0]] # devuelve el segundo más frecuente  
        print("Color identificado: ", color)
        self.color = color
#----------------------------------------------------------------------
        
# mi_carta = Carta("azul", "uno")
# mi_carta.atributos()
# mi_carta.identificar_color()
# mi_carta.atributos()       
    
