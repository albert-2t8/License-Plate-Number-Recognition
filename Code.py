from skimage import io, transform, color
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.morphology import opening, disk
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
 


input_dir = r"Imagenes_binarizadas"
 

image_size = (28, 28)
 

patterns = []
labels = []

 
# Función para leer y procesar las imágenes
def process_images(input_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
            
            image_path = os.path.join(input_folder, filename)
            
            image = io.imread(image_path, as_gray=True)
          
            resized_image = transform.resize(image, image_size, mode='reflect', anti_aliasing=True)
            
            image_vector = resized_image.flatten()
           
            image_vector = image_vector / np.max(image_vector)
           
            patterns.append(image_vector)
           
            label = int(filename.split('_')[-1].split('.')[0])
            labels.append(label)
 
    
 

process_images(input_dir)


 

patterns   = np.array(patterns)
labels     = np.array(labels)
labels[0]  = 0
labels[1]  = 1
labels[2]  = 2
labels[3]  = 3
labels[4]  = 4
labels[5]  = 5
labels[6]  = 6
labels[7]  = 7
labels[8]  = 8
labels[9]  = 9
labels[10]  = 8
labels[11]  = 0
labels[12]  = 7
labels[13]  = 9
labels[14]  = 4
labels[15]  = 9
labels[16]  = 4
labels[17]  = 8
labels[18]  = 6
labels[19]  = 5
labels[20]  = 6
labels[21]  = 3
labels[22]  = 4
labels[23]  = 0
labels[24]  = 9
labels[25]  = 7
labels[26]  = 8
labels[27]  = 0
labels[28]  = 2
labels[29]  = 3
labels[30]  = 6
labels[31]  = 1
labels[32]  = 5
labels[33]  = 2



print(f"Total de patrones extraídos: {len(patterns)}")
print(f"Tamaño de cada patrón: {patterns.shape[1]} (corresponde a la imagen redimensionada a 28x28 = 784 píxeles)")
print(f"Etiquetas de los primeros 10 patrones: {labels[:10]}")
 


first_pattern = patterns[8]
 

reconstructed_image = first_pattern.reshape((28, 28))
 



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
 

def purelin_derivada(n):
    f_punto = 1
    return f_punto
 
def ReLu(n):
    
    return np.maximum(0,n)



 

def sigmoid_derivada(n):

    return np.where(n > 0, 1, 0)


 

def softmax(n):
    exps   = np.exp(n - np.max(n, axis=1, keepdims=True))
    output = exps / np.sum(exps, axis=1, keepdims=True)
    return output



 
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
 


def forward_propagation(X, weights):
    n1 = np.dot(X, weights['W1']) + weights['b1']
    a1 = ReLu(n1)
    n2 = np.dot(a1, weights['W2']) + weights['b2']
    a2 = ReLu(n2)
    n3 = np.dot(a2, weights['W3']) + weights['b3']
    a3 = softmax(n3)
    cache = {'a1': a1, 'a2': a2, 'a3': a3, 'n1': n1, 'n2': n2, 'n3': n3}
    return a3, cache
 


def compute_loss(Y, a3):
    m = Y.shape[0]
    log_probs = -np.log(a3[range(m), Y.argmax(axis=1)])
    loss = np.sum(log_probs) / m
    return loss
 


def back_propagation(X, Y, cache, weights):
    m = X.shape[0]

    dn3 = cache['a3'] - Y
    dW3 = np.dot(cache['a2'].T, dn3) / m
    db3 = np.sum(dn3, axis=0, keepdims=True) / m
 
    dn2 = np.dot(dn3, weights['W3'].T) * sigmoid_derivada(cache['a2'])
    dW2 = np.dot(cache['a1'].T, dn2) / m
    db2 = np.sum(dn2, axis=0, keepdims=True) / m
 
    dn1 = np.dot(dn2, weights['W2'].T) * sigmoid_derivada(cache['a1'])
    dW1 = np.dot(X.T, dn1) / m
    db1 = np.sum(dn1, axis=0, keepdims=True) / m
 
   
    gradients = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'dW3': dW3, 'db3': db3}
    return gradients
 


def update_weights(weights, gradients, learning_rate):
    weights['W1'] -= learning_rate * gradients['dW1']
    weights['b1'] -= learning_rate * gradients['db1']
    weights['W2'] -= learning_rate * gradients['dW2']
    weights['b2'] -= learning_rate * gradients['db2']
    weights['W3'] -= learning_rate * gradients['dW3']
    weights['b3'] -= learning_rate * gradients['db3']
    return weights
 


def train(X, Y, input_size, hidden1_size, hidden2_size, output_size, learning_rate, epochs):
    weights = initialize_weights(input_size, hidden1_size, hidden2_size, output_size)
    for epoch in range(epochs):
      
        a3, cache = forward_propagation(X, weights)
 
        
        loss = compute_loss(Y, a3)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
 
        
        gradients = back_propagation(X, Y, cache, weights)
 
    
        weights = update_weights(weights, gradients, learning_rate)
 
    return weights



def predict(X, weights):
    a3, _ = forward_propagation(X, weights)
    return np.argmax(a3, axis=1)



input_size    = 784
hidden1_size  = 200  
hidden2_size  = 100   
output_size   = 10  
learning_rate = 0.01
epochs        = 35000
 


Y_one_hot = np.zeros((labels.size, output_size))
Y_one_hot[np.arange(labels.size), labels] = 1
 

weights = train(patterns, Y_one_hot, input_size, hidden1_size, hidden2_size, output_size, learning_rate, epochs)
 


predictions = predict(patterns, weights)
accuracy    = np.mean(predictions == labels)
print(f'Precisión en el conjunto de entrenamiento: {accuracy:.4f}')



###################################################################################################
##############################Cortar placa#########################################################

def procesar_digito(digito):
    estructura = disk(5)
    dil = opening(digito, estructura)  
    return dil


def normalizar_histograma(suma_columna, longitud_deseada):
    if len(suma_columna) == 0:
        return np.zeros(longitud_deseada)
    
    x_original = np.linspace(0, len(suma_columna) - 1, num=len(suma_columna))
    f = interp1d(x_original, suma_columna, kind='linear', fill_value="extrapolate")
    x_nueva = np.linspace(0, len(suma_columna) - 1, num=longitud_deseada)
    suma_normalizada = f(x_nueva)
    return suma_normalizada




def guardar_imagen(imagen, nombre_archivo, carpeta_destino):
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)  
    ruta_completa = os.path.join(carpeta_destino, nombre_archivo)
    io.imsave(ruta_completa, imagen)  




ruta_imagen_principal = r'im4.jpeg'
imagen_principal = io.imread(ruta_imagen_principal)
imagen_redimensionada = transform.resize(imagen_principal, (500, 1000))



gris = color.rgb2gray(imagen_redimensionada)
binario = gris < 0.5
binario_derecha = binario[:, binario.shape[1] // 3:]




suma_col = np.sum(binario_derecha, axis=0)
longitud_histograma_principal = 700
suma_col_normalizada_principal = normalizar_histograma(suma_col, longitud_histograma_principal)



valor_maximo = np.max(suma_col_normalizada_principal)
umbral_relativo = 0.47 * valor_maximo
picos, _ = find_peaks(suma_col_normalizada_principal, height=umbral_relativo)



distancia_minima_entre_picos = 100
picos_filtrados = [picos[0]]



for i in range(1, len(picos)):
    if picos[i] - picos_filtrados[-1] > distancia_minima_entre_picos:
        picos_filtrados.append(picos[i])

picos_finales = picos_filtrados[:4]




plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(binario_derecha, cmap='gray')
plt.title('Mitad derecha de la imagen')
plt.axis('off')



plt.subplot(1, 2, 2)
plt.plot(suma_col_normalizada_principal)
plt.plot(picos_finales, suma_col_normalizada_principal[picos_finales], "x")
plt.title('Picos filtrados en la suma de columnas')
plt.tight_layout()
plt.show()




numeros_detectados = []
longitud_histograma = 100

digitos_procesados = []
carpeta_destino = r'Digitos Procesados'



for i, pico in enumerate(picos_finales):
    ancho_digito = 120
    digito = binario_derecha[:, max(0, pico - ancho_digito // 2): min(pico + ancho_digito // 2, binario_derecha.shape[1])]

    if digito.size == 0:
        continue
    
    digito_procesado = procesar_digito(digito)
    digitos_procesados.append(digito_procesado)


    # Guardar la imagen procesada
    nombre_archivo = f'digito_{i + 1}.jpg'  # Nombre del archivo
    guardar_imagen(digito_procesado, nombre_archivo, carpeta_destino)
    
    



plt.figure(figsize=(12, 6))
for idx, digito in enumerate(digitos_procesados):
    plt.subplot(1, len(digitos_procesados), idx + 1)
    plt.imshow(digito, cmap='gray')
    plt.axis('off')
    plt.title(f'Dígito {idx + 1}')

plt.tight_layout()
plt.show()






input_dir_test = carpeta_destino
 

image_size_test = (28, 28)
 

patterns_test = []
labels_test   = []
 


def process_images(input_folder):
    patterns_list = []
    labels_list = []
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
        
            image_path = os.path.join(input_folder, filename)
            
            image = io.imread(image_path, as_gray=True)
            
            resized_image = transform.resize(image, image_size_test, mode='reflect', anti_aliasing=True)
            
            image_vector = resized_image.flatten()
            
            image_vector = image_vector / np.max(image_vector)
          
            patterns_list.append(image_vector)
            
            label = int(filename.split('_')[-1].split('.')[0])
            
            labels_list.append(label)
 
    return np.array(patterns_list), np.array(labels_list)


 

patterns_test, labels_test = process_images(input_dir_test)


labels_test[0]  = 2
labels_test[1]  = 2
labels_test[2]  = 0
labels_test[3]  = 8



predictions_test = predict(patterns_test, weights)
 


accuracy_test = np.mean(predictions_test == labels_test)


print(f'Precisión en el conjunto de prueba: {accuracy_test:.4f}')
 

num_images_to_show = 4
plt.figure(figsize=(15, 6))



for i in range(num_images_to_show):
    if i < 4:
        plt.subplot(1, 4, i + 1)  
    else:
        plt.subplot(1, 4, i + 1)

    image = patterns_test[i].reshape((28, 28))
    plt.imshow(image, cmap='gray')
    plt.title(f'Verdadero: {labels_test[i]}\nPredicho: {predictions_test[i]}')
    plt.axis('off')
 
plt.tight_layout()
plt.show()