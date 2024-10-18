#from skimage import io, transform
import matplotlib.pyplot as plt
#import numpy as np
#import os
 
import os
import numpy as np
from skimage import io, transform

def recortar_imagen(imagen, x_inicio, y_inicio, ancho, alto):
    return imagen[y_inicio:y_inicio+alto, x_inicio:x_inicio+ancho]

# Definir el tamaño de la imagen que se utilizará (28x28)
image_size = (28, 28)

# Función para leer y procesar las imágenes
def process_images(input_folder):
    patterns_list = []
    labels_list = []
    print(f"Procesando imágenes desde: {input_folder}")
    
    for filename in os.listdir(input_folder):
        print(f"Archivo encontrado: {filename}")  # Verificar si encuentra archivos
        if filename.endswith('.jpg'):  # Asegúrate de que las imágenes son .png
            try:
                # Leer la imagen
                image_path = os.path.join(input_folder, filename)
                image = io.imread(image_path, as_gray=True)
                # Redimensionar la imagen a 28x28 píxeles
                x_inicio, y_inicio, ancho, alto = 405, 100, 500, 560
                imagen_recortada = recortar_imagen(image, x_inicio, y_inicio, ancho, alto)
                resized_image = transform.resize(imagen_recortada, image_size, mode='reflect', anti_aliasing=True)
                # Convertir la imagen en un vector unidimensional
                image_vector = resized_image.flatten()
                # Normalizar los valores de los píxeles (entre 0 y 1)
                image_vector = image_vector / np.max(image_vector)
                # Añadir el vector a la lista de patrones
                patterns_list.append(image_vector)
                
                # Extraer la etiqueta del nombre del archivo
                try:
                    label = int(filename.split('_')[-1].split('.')[0])
                    labels_list.append(label)
                except ValueError:
                    print(f"No se pudo extraer la etiqueta de {filename}")
                    continue
            except Exception as e:
                print(f"Error al procesar la imagen {filename}: {e}")
                continue
    
    # Imprimir la cantidad de patrones y etiquetas procesados
    print(f"Total de patrones procesados: {len(patterns_list)}")
    print(f"Total de etiquetas procesadas: {len(labels_list)}")
    
    # Si no se ha procesado ningún patrón, devuelve un error
    if len(patterns_list) == 0 or len(labels_list) == 0:
        raise ValueError("No se procesaron patrones o etiquetas")
    
    # Retornar ambas listas convertidas a arrays de numpy
    return np.array(patterns_list), np.array(labels_list)

# Directorio de entrada
input_dir = r"C:\Users\980032406\Documents\7 semestre\Reconocimiento de patrones\1 Parcial\Proyecto 1\Proyecto1_UNO\Train3"

# Procesar imágenes
patterns, labels = process_images(input_dir)


 
# Procesar las imágenes en la carpeta
process_images(input_dir)

# Verificar el tamaño de las listas
print(f"Total de patrones procesados: {len(patterns)}")
print(f"Total de etiquetas procesadas: {len(labels)}")
 
# Convertir las listas a arrays de numpy para ser usados en la red neuronal
patterns   = np.array(patterns)
labels     = np.array(labels)
labels[0]  = 0
labels[1]  = 1
labels[2]  = 2
labels[3]  = 3
labels[4]  = 4
labels[5]  = 0
labels[6]  = 1
labels[7]  = 2
labels[8]  = 3
labels[9]  = 4
labels[10] = 5
labels[11] = 6
labels[12] = 7
labels[13] = 8
labels[14] = 9
labels[15] = 5
labels[16] = 6
labels[17] = 7
labels[18] = 8
labels[19] = 9
labels[20] = 0
labels[21] = 1
labels[22] = 2
labels[23] = 3
labels[24] = 4
labels[25] = 5
labels[26] = 6
labels[27] = 7
labels[28] = 8
labels[29] = 9
labels[30] = 0
labels[31] = 1
labels[32] = 2
labels[33] = 3
labels[34] = 4
labels[35] = 5
labels[36] = 6
labels[37] = 7
labels[38] = 8
labels[39] = 9
labels[40] = 0
labels[41] = 1
labels[42] = 2
labels[43] = 3
labels[44] = 4
labels[45] = 5
labels[46] = 6
labels[47] = 7
labels[48] = 8
labels[49] = 9
 
# Mostrar información sobre los datos procesados
# print(f"Total de patrones extraídos: {len(patterns)}")
# print(f"Tamaño de cada patrón: {patterns.shape[1]} (corresponde a la imagen redimensionada a 28x28 = 784 píxeles)")
# print(f"Etiquetas de los primeros 10 patrones: {labels[:10]}")
 
# # Extraer la primera fila de la variable 'patterns'
# first_pattern = patterns[1]
 
# # # Darle forma a la imagen (reshape) de 28x28 píxeles
# reconstructed_image = first_pattern.reshape((28, 28))
 
# # # Mostrar la imagen reconstruida
# plt.close('all')
# plt.imshow(reconstructed_image, cmap='gray')
# plt.title(f'Número representado: {labels[1]}')  # Mostrar la etiqueta correspondiente
# plt.axis('off')
# plt.show()
 
#=================================================================#
#<================ Funciones de Activación ======================>#
def hardlim(n):
    if (n > 0):
        a = 1
    else:
        a = 0
    return a
 
def purelin(n):
    a = 1 * n
    return a
 
# Derivada de la Funcion purelin
def purelin_derivada(n):
    f_punto = 1
    return f_punto
 
def sigmoid(n):
    f = 1 / (1 + np.exp(-n))
    return f
 
# Derivada de la Funcion sigmoid
def sigmoid_derivada(n):
    f_punto = n * (1 - n)
    return f_punto
 
# Función de activación
def softmax(n):
    exps   = np.exp(n - np.max(n, axis=1, keepdims=True))
    output = exps / np.sum(exps, axis=1, keepdims=True)
    return output
 
# Inicialización de los pesos y sesgos
def initialize_weights(input_size, hidden1_size, hidden2_size, output_size):
    np.random.seed(42)
    weights = {
        'W1': np.random.randn(input_size,   hidden1_size) * 0.01,
        'b1': np.zeros((1, hidden1_size)),
        'W2': np.random.randn(hidden1_size, hidden2_size) * 0.01,
        'b2': np.zeros((1, hidden2_size)),
        'W3': np.random.randn(hidden2_size, output_size)  * 0.01,
        'b3': np.zeros((1, output_size)),
    }
    return weights
 
# Función de forward propagation
def forward_propagation(X, weights):
    n1 = np.dot(X, weights['W1']) + weights['b1']
    a1 = sigmoid(n1)
    n2 = np.dot(a1, weights['W2']) + weights['b2']
    a2 = sigmoid(n2)
    n3 = np.dot(a2, weights['W3']) + weights['b3']
    a3 = softmax(n3)
    cache = {'a1': a1, 'a2': a2, 'a3': a3, 'n1': n1, 'n2': n2, 'n3': n3}
    return a3, cache
 
# Función de cálculo de pérdida (cross-entropy)
def compute_loss(Y, a3):
    m = Y.shape[0]
    log_probs = -np.log(a3[range(m), Y.argmax(axis=1)])
    loss = np.sum(log_probs) / m
    return loss
 
# Función de backpropagation
def back_propagation(X, Y, cache, weights):
    m = X.shape[0]
    # Cálculo de los gradientes
    dn3 = cache['a3'] - Y
    dW3 = np.dot(cache['a2'].T, dn3) / m
    db3 = np.sum(dn3, axis=0, keepdims=True) / m
 
    dn2 = np.dot(dn3, weights['W3'].T) * sigmoid_derivada(cache['a2'])
    dW2 = np.dot(cache['a1'].T, dn2) / m
    db2 = np.sum(dn2, axis=0, keepdims=True) / m
 
    dn1 = np.dot(dn2, weights['W2'].T) * sigmoid_derivada(cache['a1'])
    dW1 = np.dot(X.T, dn1) / m
    db1 = np.sum(dn1, axis=0, keepdims=True) / m
 
    # Actualización de los pesos y sesgos
    gradients = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'dW3': dW3, 'db3': db3}
    return gradients
 
# Actualización de los pesos usando los gradientes calculados
def update_weights(weights, gradients, learning_rate):
    weights['W1'] -= learning_rate * gradients['dW1']
    weights['b1'] -= learning_rate * gradients['db1']
    weights['W2'] -= learning_rate * gradients['dW2']
    weights['b2'] -= learning_rate * gradients['db2']
    weights['W3'] -= learning_rate * gradients['dW3']
    weights['b3'] -= learning_rate * gradients['db3']
    return weights
 
# Función de entrenamiento
def train(X, Y, input_size, hidden1_size, hidden2_size, output_size, learning_rate, epochs):
    weights = initialize_weights(input_size, hidden1_size, hidden2_size, output_size)
    for epoch in range(epochs):
        # Forward propagation
        a3, cache = forward_propagation(X, weights)
 
        # Compute loss
        loss = compute_loss(Y, a3)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
 
        # Backpropagation
        gradients = back_propagation(X, Y, cache, weights)
 
        # Update weights
        weights = update_weights(weights, gradients, learning_rate)
 
    return weights
 
# Predicción
def predict(X, weights):
    a3, _ = forward_propagation(X, weights)
    return np.argmax(a3, axis=1)
 
# Parámetros de la red
input_size    = 784
hidden1_size  = 128
hidden2_size  = 64
output_size   = 10  # Cambiado a 10 para los dígitos 0-9
learning_rate = 0.01
epochs        = 35000
 
# Convertir las etiquetas a one-hot encoding
Y_one_hot = np.zeros((labels.size, output_size))
Y_one_hot[np.arange(labels.size), labels] = 1
 
# Entrenar la red neuronal
weights = train(patterns, Y_one_hot, input_size, hidden1_size, hidden2_size, output_size, learning_rate, epochs)
 
# Evaluar el modelo en los datos de entrenamiento
predictions = predict(patterns, weights)
accuracy    = np.mean(predictions == labels)
print(f'Precisión en el conjunto de entrenamiento: {accuracy:.4f}')
 
#=================================================================#
#--------------------------AQUIIIIII-------------------------------
#<===============================================================>#

def recortar_imagen(imagen, x_inicio, y_inicio, ancho, alto):
    return imagen[y_inicio:y_inicio+alto, x_inicio:x_inicio+ancho]

def mostrar_imagen(imagen, titulo="Imagen"):
    plt.figure(figsize=(6, 6))
    plt.imshow(imagen, cmap='gray')
    plt.title(titulo)
    plt.axis('on')  # Mostrar ejes para ver las coordenadas
    plt.show()
    
def process_imagesT(input_folder):
    patterns_test = []
    labels_test = []
    #print(f"Procesando imágenes desde: {input_folder}")
    
    for filename in os.listdir(input_folder):
        #print(f"Archivo encontrado: {filename}")  # Verificar si encuentra archivos
        if filename.endswith('.jpg'):  # Asegúrate de que las imágenes son .png
            try:
                # Leer la imagen
                image_path = os.path.join(input_folder, filename)
                image = io.imread(image_path, as_gray=True)
                # Redimensionar la imagen a 28x28 píxeles
                x_inicio, y_inicio, ancho, alto = 405, 100, 500, 560
                imagen_recortada = recortar_imagen(image, x_inicio, y_inicio, ancho, alto)
                image_size = (28, 28)
                resized_image = transform.resize(imagen_recortada, image_size, mode='reflect', anti_aliasing=True)
                # Convertir la imagen en un vector unidimensional
                image_vector = resized_image.flatten()
                # Normalizar los valores de los píxeles (entre 0 y 1)
                image_vector = image_vector / np.max(image_vector)
                # Añadir el vector a la lista de patrones
                patterns_test.append(image_vector)
                
                # Extraer la etiqueta del nombre del archivo
                try:
                    label = int(filename.split('_')[-1].split('.')[0])
                    labels_test.append(label)
                except ValueError:
                    print(f"No se pudo extraer la etiqueta de {filename}")
                    continue
            except Exception as e:
                print(f"Error al procesar la imagen {filename}: {e}")
                continue
    
    # Imprimir la cantidad de patrones y etiquetas procesados
    print(f"Total de patrones procesados: {len(patterns_test)}")
    print(f"Total de etiquetas procesadas: {len(labels_test)}")
    
    # Si no se ha procesado ningún patrón, devuelve un error
    if len(patterns_test) == 0 or len(labels_test) == 0:
        raise ValueError("No se procesaron patrones o etiquetas")
    
    # Retornar ambas listas convertidas a arrays de numpy
    return np.array(patterns_test), np.array(labels_test)

nueva_imagen = r"C:\Users\980032406\Documents\7 semestre\Reconocimiento de patrones\1 Parcial\Proyecto 1\Proyecto1_UNO\Cartas"
#mostrar_imagen(nueva_imagen, titulo="Imagen Original")

# Procesar las imágenes de prueba
patterns_test, labels_test = process_imagesT(nueva_imagen)
 
# Establecer las etiquetas manualmente si es necesario (usando las etiquetas que se proporcionaron)
labels_test[0]  = 6
labels_test[1]  = 2
labels_test[2]  = 1
labels_test[3]  = 9
labels_test[4]  = 4
labels_test[5]  = 4
labels_test[6]  = 0
labels_test[7]  = 6
labels_test[8]  = 3
labels_test[9]  = 7

# Predicción en el conjunto de prueba
predictions_test = predict(patterns_test, weights)
 
# Calcular la precisión en el conjunto de prueba
accuracy_test = np.mean(predictions_test == labels_test)
print(f'Precisión en el conjunto de prueba: {accuracy_test:.4f}')
 
# Mostrar algunas imágenes del conjunto de prueba junto con las etiquetas predichas
num_images_to_show = 10
plt.figure(figsize=(15, 6))  # Ajustar el tamaño de la figura para acomodar dos filas
 
for i in range(num_images_to_show):
    # Para la primera fila (imágenes 0 a 9)
    if i < 10:
        plt.subplot(2, 10, i + 1)  # 2 filas, 10 columnas
    # Para la segunda fila (imágenes 10 a 19)
    else:
        plt.subplot(2, 10, i + 1)
    # Mostrar la imagen
    image = patterns_test[i].reshape((28, 28))
    plt.imshow(image, cmap='gray')
    plt.title(f'Verdadero: {labels_test[i]}\nPredicho: {predictions_test[i]}')
    plt.axis('off')
 
plt.tight_layout()
plt.show()


# def get_predictions():
#     # Aquí carga o calcula predictions_test y labels_test
#     # En este caso, puedes usar los mismos procesos de cálculo o cargar de un archivo
#     predictions_test = np.load('predictions_test.npy')  # Carga desde el archivo
#     return predictions_test

#--------------------VALORES PARA LLAMAR DESDE OTRO GRUPO------------------------------------
np.save('valores_predichos.npy', predictions_test)

#CODIGO DESDE EL OTRO PROGRAMA---------------------------------------------------------------
#********************************************************************************************
# import numpy as np
# valores_predichos = np.load('valores_predichos.npy')
# print(valores_predichos) 
# for i in range(len(valores_predichos)):
#     valor_individual = valores_predichos[i]
#     print(f'Valor en el índice {i}: {valor_individual}')