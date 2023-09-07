from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
import IDmyPillQRC

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import cv2
from cv2 import cvtColor
import skimage
import SimpleITK as sitk
from scipy import signal
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops

import os
from openpyxl import load_workbook
import pyqtgraph as pg

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

######################################################################################################################

# FUNCIONES 

def funcImagenBN(path, plot=True):
  imagenBN = cv2.imread(path,0)
  return imagenBN

def funcImagenRGB(path, plot=True):
    imagenBGR = cv2.imread(path,1)
    imagenRGB = cv2.cvtColor(imagenBGR, cv2.COLOR_BGR2RGB)
    return imagenRGB

def funcBinarizarImagen(imagen, plot=True):
  #Binariza con Otsu
  umbral_Otsu, imagen_Otsu = cv2.threshold(imagen,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  return umbral_Otsu, imagen_Otsu

def funcApertura(imagen, plot=True):
  kernel = np.ones((5, 5), 'uint8')
  img_erosion = cv2.erode(imagen, kernel, iterations=3)
  img_dilatacion = cv2.dilate(img_erosion, kernel, iterations=3)
  return img_dilatacion

def funcCierre(imagen, plot=True):
  kernel = np.ones((5, 5), 'uint8')
  img_dilatacion = cv2.dilate(imagen, kernel, iterations=3)
  img_erosion = cv2.erode(img_dilatacion, kernel, iterations=3)
  return img_erosion

def funcCortarCuadrado(imagen, plot=True):
  Canny = cv2.Canny(imagen, 70, 120)
  cnts = cv2.findContours(Canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  xmax, ymax, wmax, hmax = cv2.boundingRect(cnts[0])
  for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    if w*h > wmax*hmax:
      xmax, ymax, wmax, hmax = x,y,w,h
  cuadrado = imagen[ymax:ymax+hmax, xmax:xmax+wmax]
  n_filas = np.shape(cuadrado)[0]
  margen = int(n_filas*0.1)
  cuadrado[0:margen,:] *= 0
  cuadrado[-margen:,:] *= 0
  cuadrado[:,0:margen] *= 0
  cuadrado[:,-margen:] *= 0
  coordenadas = [ymax, hmax, xmax, wmax]
  return cuadrado, coordenadas

def funcDimensionCuadrado(cuadrado):
  alto_pixel_mm = 70/np.shape(cuadrado)[0] #en mm
  ancho_pixel_mm = 70/np.shape(cuadrado)[1] #en mm
  area_pixel_mm = alto_pixel_mm*ancho_pixel_mm
  return alto_pixel_mm, ancho_pixel_mm, area_pixel_mm

def funcSegmentarPastillaBinaria(cuadrado, plot=True):
  Canny = cv2.Canny(cuadrado, 200, 230)
  cnts = cv2.findContours(Canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  xmax, ymax, wmax, hmax = cv2.boundingRect(cnts[0])
  for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    if w*h > wmax*hmax:
      xmax, ymax, wmax, hmax = x,y,w,h
  cv2.rectangle(Canny, (xmax, ymax), (xmax + wmax, ymax + hmax), (36,255,12), 2)
  pastilla = cuadrado[ymax:ymax+hmax, xmax:xmax+wmax]
  coordenadas = [ymax, hmax, xmax, wmax]
  return pastilla, coordenadas

def funcSegmentarPastilla(cuadrado, coordenadas_pastilla, pastilla_binaria, title, plot=True):
  ymax_pastilla = coordenadas_pastilla[0]
  hmax_pastilla = coordenadas_pastilla[1]
  xmax_pastilla = coordenadas_pastilla[2]
  wmax_pastilla = coordenadas_pastilla[3]
  pastilla = cuadrado[ymax_pastilla:ymax_pastilla+hmax_pastilla, xmax_pastilla:xmax_pastilla+wmax_pastilla]
  pastilla *= pastilla_binaria
  return pastilla

def funcSegmentarPastillaRGB(cuadrado, coordenadas_pastilla, title, plot=True):
  ymax_pastilla = coordenadas_pastilla[0]
  hmax_pastilla = coordenadas_pastilla[1]
  xmax_pastilla = coordenadas_pastilla[2]
  wmax_pastilla = coordenadas_pastilla[3]
  pastilla = cuadrado[ymax_pastilla:ymax_pastilla+hmax_pastilla, xmax_pastilla:xmax_pastilla+wmax_pastilla]
  return pastilla

def funcBinarizar01(imagen_binaria255):
  pastilla_binaria01 = imagen_binaria255.copy()
  for fila in range(np.shape(imagen_binaria255)[0]):
    for col in range(np.shape(imagen_binaria255)[1]):
      if imagen_binaria255[fila,col] == 255:
        pastilla_binaria01[fila,col] = 1
  return pastilla_binaria01

def funcDimensionPastilla(pastilla_binaria1, alto_pixel_mm, ancho_pixel_mm, area_pixel_mm):
  #pixeles_pastilla = np.sum(pastilla_binaria1)
  alto_pastilla = np.round((np.shape(pastilla_binaria1)[0])*alto_pixel_mm, 2)
  ancho_pastilla = np.round((np.shape(pastilla_binaria1)[1])*ancho_pixel_mm, 2)
  area_pastilla = np.round((alto_pastilla*ancho_pastilla), 2)
  return alto_pastilla, ancho_pastilla, area_pastilla

def funcColorPastilla(imagenRGB):
  # Obtener el color dominante de la imagen segmentada
  # Convertir la imagen segmentada a espacio de color HSV
  imagen_hsv = cv2.cvtColor(imagenRGB, cv2.COLOR_BGR2HSV)
  # Calcular el histograma de tonos (Hue)
  hist = cv2.calcHist([imagen_hsv], [0], None, [180], [0, 180])
  # Encontrar el índice del tono más dominante (valor máximo en el histograma)
  indice_tono_dominante = np.argmax(hist)
  # Calcular el color dominante correspondiente al tono encontrado
  color_dominante = int(180 * indice_tono_dominante / 180)
  return color_dominante

def MatrizCoocurrencia(imagen, direccion):
  n_filas = imagen.shape[0]
  n_cols = imagen.shape[1]
  max = np.max(imagen)
  min = np.min(imagen)
  if max-min == 0:
    matriz = np.ones((1,1))
  else:
    matriz = np.zeros((max-min+1,max-min+1))
    if direccion==0:
      for i in range(n_filas):
        for j in range(n_cols-1):
          a = imagen[i,j]
          b = imagen[i,j+1]
          indice_a = a-min
          indice_b = b-min
          matriz[indice_a,indice_b]+=1
          matriz[indice_b,indice_a]+=1
    elif direccion==45:
      for i in range(1, n_filas):
        for j in range(n_cols-1):
          a = imagen[i,j]
          b = imagen[i-1,j+1]
          indice_a = a-min
          indice_b = b-min
          matriz[indice_a,indice_b]+=1
          matriz[indice_b,indice_a]+=1
    elif direccion==90:
      for i in range(n_filas-1):
        for j in range(n_cols):
          a = imagen[i,j]
          b = imagen[i+1,j]
          indice_a = a-min
          indice_b = b-min
          matriz[indice_a,indice_b]+=1
          matriz[indice_b,indice_a]+=1
    elif direccion==135:
      for i in range(n_filas-1):
        for j in range(n_cols-1):
          a = imagen[i,j]
          b = imagen[i+1,j+1]
          indice_a = a-min
          indice_b = b-min
          matriz[indice_a,indice_b]+=1
          matriz[indice_b,indice_a]+=1
    matriz = matriz/2*((max-min+1)**2)
  return matriz, min

def Media(matriz, min):
  n_filas = matriz.shape[0]
  n_cols = matriz.shape[1]
  media = 0
  for i in range(n_filas):
    for j in range(n_cols):
      media += (i+min)*matriz[i,j]
  return media

def Desvio(matriz, media, min):
  n_filas = matriz.shape[0]
  n_cols = matriz.shape[1]
  desvio = 0
  for i in range(n_filas):
    for j in range(n_cols):
      desvio += ((i+min-media)**2)*matriz[i,j]
  return desvio

def Energia(matriz):
  n_filas = matriz.shape[0]
  n_cols = matriz.shape[1]
  energia = 0
  for i in range(n_filas):
    for j in range(n_cols):
      energia += matriz[i,j]**2
  return energia

def Entropia(matriz):
  n_filas = matriz.shape[0]
  n_cols = matriz.shape[1]
  entropia = 0
  for i in range(n_filas):
    for j in range(n_cols):
      if (matriz[i,j] != 0):
        entropia += matriz[i,j]*np.log2(matriz[i,j])
  return entropia

def Correlacion(matriz, media, desvio, min):
  n_filas = matriz.shape[0]
  n_cols = matriz.shape[1]
  correlacion = 0
  for i in range(n_filas):
    for j in range(n_cols):
      correlacion += ((i+min-media)*(j+min-media)*matriz[i,j])/(desvio**2)
  return correlacion

def IDM(matriz):
  n_filas = matriz.shape[0]
  n_cols = matriz.shape[1]
  idm = 0
  for i in range(n_filas):
    for j in range(n_cols):
      idm += matriz[i,j]/(1+(i-j)**2)
  return idm

def Contraste(matriz):
  n_filas = matriz.shape[0]
  n_cols = matriz.shape[1]
  contraste = 0
  for i in range(n_filas):
    for j in range(n_cols):
      contraste += ((i-j)**2)*matriz[i,j]
  return contraste

def ClusterShade(matriz, min):
  n_filas = matriz.shape[0]
  n_cols = matriz.shape[1]
  cluster = 0
  for i in range(n_filas):
    for j in range(n_cols):
      cluster += matriz[i,j]*((i+min+j+min)**3)
  return cluster

def ClusterProminence(matriz,min):
  n_filas = matriz.shape[0]
  n_cols = matriz.shape[1]
  cluster_prom = 0
  for i in range(n_filas):
    for j in range(n_cols):
      cluster_prom += matriz[i,j]*((i+min+j+min)**4)
  return cluster_prom

def MetricasTextura(imagen):
  # Calcula la matriz y las métricas para cada dirección y el promedio de las métricas en las 4 direcciones
  # Creo un diccionario para almacenar todos los valores
  dic_metricas = {}
  sum_energia = 0
  sum_entropia = 0
  sum_correlacion = 0
  sum_idm = 0
  sum_contraste = 0
  sum_cluster_shade = 0
  sum_cluster_prominence = 0
  # Calculo la matriz de coocurrencia y las métricas para cada dirección
  dirs = [0,45,90,135]
  for i in range (len(dirs)):
    matriz, min = MatrizCoocurrencia(imagen, dirs[i])
    media = Media(matriz,min)
    desvio = Desvio(matriz,media,min)
    energia = Energia(matriz)
    entropia = Entropia(matriz)
    correlacion = Correlacion(matriz,media,desvio,min)
    idm = IDM(matriz)
    contraste = Contraste(matriz)
    cluster_shade = ClusterShade(matriz,min)
    cluster_prominence = ClusterProminence(matriz,min)
    # Sumo los valores para el cálculo de los promedios
    sum_energia += energia
    sum_entropia += entropia
    sum_correlacion += correlacion
    sum_idm += idm
    sum_contraste += contraste
    sum_cluster_shade += cluster_shade
    sum_cluster_prominence += cluster_prominence
    # Guardo cada métrica de dicha dirección en el diccionario
    dic_metricas[f'Dirección {dirs[i]} °'] = {'media':media, 'desvio':desvio, 'energia':energia, 'entropia':entropia, 'correlacion':correlacion, 'idm':idm, 'contraste':contraste, 'cluster shade':cluster_shade, 'cluster prominence':cluster_prominence}
  #Guardo los promedios
  dic_metricas[f'Promedio'] = {'media':0, 'desvio':0, 'energia':sum_energia/4, 'entropia':sum_entropia/4, 'correlacion':sum_correlacion/4, 'idm':sum_idm/4, 'contraste':sum_contraste/4, 'cluster shade':sum_cluster_shade/4, 'cluster prominence':sum_cluster_prominence/4}
  df_metricas = DICaDF(dic_metricas)
  return df_metricas

def funcTexturasPastilla(pastilla):
  metricas_texturas_pastilla = MetricasTextura(pastilla)
  return metricas_texturas_pastilla

def DICaDF(dic):
  df = pd.DataFrame([key for key in dic.keys()], columns=['Dirección'])
  df['Media'] = [value['media'] for value in dic.values()]
  df['Desvio'] = [value['desvio'] for value in dic.values()]
  df['Energía'] = [value['energia'] for value in dic.values()]
  df['Entropia'] = [value['entropia'] for value in dic.values()]
  df['Correlación'] = [value['correlacion'] for value in dic.values()]
  df['IDM'] = [value['idm'] for value in dic.values()]
  df['Contraste'] = [value['contraste'] for value in dic.values()]
  df['Cluster Shade'] = [value['cluster shade'] for value in dic.values()]
  df['Cluster Prominence'] = [value['cluster prominence'] for value in dic.values()]
  return df

def funcNuevaPastilla(path):
    caracteristicas = []
    imagen_BN = funcImagenBN(path)
    imagen_RGB = funcImagenRGB(path)
    umbral_Otsu, imagen_Otsu = funcBinarizarImagen(imagen_BN)
    imagen_cierre = funcCierre(imagen_Otsu)
    imagen_apertura = funcApertura(imagen_cierre)
    cuadrado, coordenadas_cuadrado = funcCortarCuadrado(imagen_apertura, plot=True)
    ymax_cuadrado = coordenadas_cuadrado[0]
    hmax_cuadrado = coordenadas_cuadrado[1]
    xmax_cuadrado = coordenadas_cuadrado[2]
    wmax_cuadrado = coordenadas_cuadrado[3]
    cuadrado_original_BN = imagen_BN[ymax_cuadrado:ymax_cuadrado+hmax_cuadrado, xmax_cuadrado:xmax_cuadrado+wmax_cuadrado]
    cuadrado_original_RGB = imagen_RGB[ymax_cuadrado:ymax_cuadrado+hmax_cuadrado, xmax_cuadrado:xmax_cuadrado+wmax_cuadrado]
    alto_pixel_mm, ancho_pixel_mm, area_pixel_mm = funcDimensionCuadrado(cuadrado)
    pastilla_binaria255, coordenadas_pastilla = funcSegmentarPastillaBinaria(cuadrado, plot=True)
    pastilla_binaria01 = funcBinarizar01(pastilla_binaria255)
    ancho_pastilla_mm, alto_pastilla_mm, area_pastilla_mm = funcDimensionPastilla(pastilla_binaria01, alto_pixel_mm, ancho_pixel_mm, area_pixel_mm)
    caracteristicas.append(ancho_pastilla_mm)
    caracteristicas.append(alto_pastilla_mm)
    caracteristicas.append(area_pastilla_mm)
    pastilla_BN = funcSegmentarPastilla(cuadrado_original_BN, coordenadas_pastilla, pastilla_binaria01, "Pastilla BN" , plot=True)
    pastilla_RGB = funcSegmentarPastillaRGB(cuadrado_original_RGB, coordenadas_pastilla, "Pastilla RGB" , plot=True)
    color = funcColorPastilla(pastilla_RGB)
    caracteristicas.append(color)
    metricasTexturas = funcTexturasPastilla(pastilla_BN)
    caracteristicas.append(metricasTexturas["Media"].tolist())
    caracteristicas.append(metricasTexturas["Desvio"].tolist())
    caracteristicas.append(metricasTexturas["Energía"].tolist())
    caracteristicas.append(metricasTexturas["Entropia"].tolist())
    caracteristicas.append(metricasTexturas["Correlación"].tolist())
    caracteristicas.append(metricasTexturas["IDM"].tolist())
    caracteristicas.append(metricasTexturas["Contraste"].tolist())
    caracteristicas.append(metricasTexturas["Cluster Shade"].tolist())
    caracteristicas.append(metricasTexturas["Cluster Prominence"].tolist())
    cuadrado_original_R = cuadrado_original_RGB[:,:,0]
    cuadrado_original_G = cuadrado_original_RGB[:,:,1]
    cuadrado_original_B = cuadrado_original_RGB[:,:,2]
    pastilla_R = funcSegmentarPastilla(cuadrado_original_R, coordenadas_pastilla, pastilla_binaria01, "Pastilla R" , plot=True)
    pastilla_G = funcSegmentarPastilla(cuadrado_original_G, coordenadas_pastilla, pastilla_binaria01, "Pastilla G" ,plot=True)
    pastilla_B = funcSegmentarPastilla(cuadrado_original_B, coordenadas_pastilla, pastilla_binaria01, "Pastilla B" ,plot=True)
    # Obtener promedio de los primeros 4 elementos para Media y Desvio
    caracteristicas[4] = np.mean(caracteristicas[4][:4])
    caracteristicas[5] = np.mean(caracteristicas[5][:4])
    # Obtener promedio de toda la lista para las demás características
    columnas_promedio = [6, 7, 8, 9, 10, 11, 12]
    for columna in columnas_promedio:
        caracteristicas[columna] = (caracteristicas[columna][4])
    return caracteristicas

def normalizarCaracteristicas(caracteristicas):
  # Leer el archivo Excel
  mediasyDesvios = [[12.65637931, 5.54617574], [13.18461207, 5.475959594], [183.1752155, 144.3184592], [94.31034483, 25.79374884],
  [298608490266, 441824660680],[2.90078E+33, 1.072E+34], [1.87828E+18, 7.69256E+18], [50480204239, 84192672366],
  [2.08976234547338E-30, 6,99225085825557E-30], [1160879895, 2146904239], [628233010408, 443484141861], [1.00778E+17, 1.64061E+17], [4.25803E+19, 7.263E+19]]
  for i in range(len(caracteristicas)):
    caracteristicas[i] = (caracteristicas[i] - mediasyDesvios[i][0])/mediasyDesvios[i][1]
  return caracteristicas

def LISTtoDF(lista):
  dic = {'Ancho': lista[0],
          'Alto': lista[1],
          'Area': lista[2],
          'Color': lista[3],
          'Media': lista[4],
          'Desvio': lista[5],
          'Energia': lista[6],
          'Entropia': lista[7],
          'Correlacion': lista[8],
          'IDM': lista[9],
          'Contraste': lista[10],
          'Cluster shade': lista[11],
          'Cluster prominence': lista[12],
          }
  # Creates pandas DataFrame
  df = pd.DataFrame(dic, index = ['Nueva pastilla'])
  return df

def funcClasificarPastilla(df_caracteristicas_nueva_pastilla_norm):
    dataset = pd.read_excel('D:\Desktop\IDmyPill\Datos normalizados y promediados.xlsx', index_col=0, squeeze=True)
    dataset = dataset.dropna()
    X_train = dataset.iloc[:,4:]
    y_train = dataset.iloc[:,0]
    feat_list = list(X_train.keys())

    # KNN
    clf = KNeighborsClassifier(n_neighbors=1, weights='distance', metric = 'euclidean')
    
    # RF
    #clf = RandomForestClassifier(n_estimators = 200, max_depth = 13, random_state = 0, criterion='entropy', class_weight='balanced')
    
    std_anova_svm = make_pipeline(clf)
    std_anova_svm.fit(X_train, y_train)

    X_test = df_caracteristicas_nueva_pastilla_norm.iloc[:, :]
    y_pred = std_anova_svm.predict(X_test)
    return y_pred

def funcObtener_info_pastilla(nombre_pastilla):
    # Leer el archivo Excel
    path_base_datos = 'D:\Desktop\IDmyPill\Datos normalizados y promediados.xlsx'
    df = pd.read_excel(path_base_datos, engine='openpyxl')
    # Buscar la fila que coincide con el nombre de la pastilla en la columna "Imagen Pastilla"
    fila = df[df["Pastilla"] == nombre_pastilla]
    if len(fila) == 0:
        print("No se encontró la pastilla en el archivo.")
        return None
    # Obtener los valores de las columnas "Descripcion", "Segmentada" y "Caracteristicas" de la fila encontrada
    descripcion = fila["Descripcion"].values[0]
    segmentada = fila["Segmentada"].values[0]
    caja = fila["Caja"].values[0]
    return descripcion, segmentada, caja


#####################################################################################################################################

class Ui_InterfazIDmyPill(object):
    def setupUi(self, InterfazIDmyPill):
        InterfazIDmyPill.setObjectName("InterfazIDmyPill")
        InterfazIDmyPill.resize(844, 668)

        self.centralwidget = QtWidgets.QWidget(InterfazIDmyPill)
        self.centralwidget.setObjectName("centralwidget")
 
        self.groupBox_Todo = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_Todo.setGeometry(QtCore.QRect(0, 0, 841, 661))
        self.groupBox_Todo.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.groupBox_Todo.setTitle("")
        self.groupBox_Todo.setObjectName("groupBox_Todo")
 
        self.lineEdit_Ruta = QtWidgets.QLineEdit(self.groupBox_Todo)
        self.lineEdit_Ruta.setGeometry(QtCore.QRect(140, 170, 671, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.lineEdit_Ruta.setFont(font)
        self.lineEdit_Ruta.setStyleSheet("background-color: rgb(220, 220, 220);\n""border-color: rgb(255, 0, 0);")
        self.lineEdit_Ruta.setFrame(False)
        self.lineEdit_Ruta.setObjectName("lineEdit_Ruta")

        self.textEdit_Pastilla = QtWidgets.QTextEdit(self.groupBox_Todo)
        self.textEdit_Pastilla.setGeometry(QtCore.QRect(140, 270, 671, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.textEdit_Pastilla.setFont(font)
        self.textEdit_Pastilla.setStyleSheet("background-color: rgb(220, 220, 220);\n""border-color: rgb(255, 0, 0);")
        self.textEdit_Pastilla.setFrameShape(QtWidgets.QFrame.Panel)
        self.textEdit_Pastilla.setObjectName("textEdit_Pastilla")
        self.textEdit_Pastilla.setReadOnly(True)

        self.textEdit_Descripcion = QtWidgets.QTextEdit(self.groupBox_Todo)
        self.textEdit_Descripcion.setGeometry(QtCore.QRect(140, 310, 671, 51))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.textEdit_Descripcion.setFont(font)
        self.textEdit_Descripcion.setStyleSheet("background-color: rgb(220, 220, 220);\n""border-color: rgb(255, 0, 0);")
        self.textEdit_Descripcion.setFrameShape(QtWidgets.QFrame.Panel)
        self.textEdit_Descripcion.setObjectName("textEdit_Descripcion")
        self.textEdit_Descripcion.setReadOnly(True)

        self.label_Ruta = QtWidgets.QLabel(self.groupBox_Todo)
        self.label_Ruta.setGeometry(QtCore.QRect(30, 160, 91, 41))
        self.label_Ruta.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_Ruta.setObjectName("label_Ruta")

        self.label_Pastilla = QtWidgets.QLabel(self.groupBox_Todo)
        self.label_Pastilla.setGeometry(QtCore.QRect(30, 260, 101, 41))
        self.label_Pastilla.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_Pastilla.setObjectName("label_Pastilla")

        self.label_Descripcion = QtWidgets.QLabel(self.groupBox_Todo)
        self.label_Descripcion.setGeometry(QtCore.QRect(30, 300, 91, 41))
        self.label_Descripcion.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_Descripcion.setObjectName("label_Descripcion")

        self.pushButton_Buscar = QtWidgets.QPushButton(self.groupBox_Todo)
        self.pushButton_Buscar.setGeometry(QtCore.QRect(730, 210, 81, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.pushButton_Buscar.setFont(font)
        self.pushButton_Buscar.setStyleSheet("background-color: rgb(226, 226, 226);")
        self.pushButton_Buscar.setObjectName("pushButton_Buscar")
        self.pushButton_Buscar.clicked.connect(self.funcBuscar)

        self.label_Resultado = QtWidgets.QLabel(self.groupBox_Todo)
        self.label_Resultado.setGeometry(QtCore.QRect(30, 220, 91, 41))
        self.label_Resultado.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_Resultado.setObjectName("label_Resultado")

        self.verticalLayoutWidget_Pastilla = QtWidgets.QWidget(self.groupBox_Todo)
        self.verticalLayoutWidget_Pastilla.setGeometry(QtCore.QRect(140, 390, 321, 251))
        self.verticalLayoutWidget_Pastilla.setObjectName("verticalLayoutWidget_Pastilla")

        self.verticalLayout_Pastilla = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_Pastilla)
        self.verticalLayout_Pastilla.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_Pastilla.setObjectName("verticalLayout_Pastilla")

        self.plt_Pastilla = pg.PlotWidget()
        self.plt_Pastilla.getPlotItem().hideAxis('bottom')
        self.plt_Pastilla.getPlotItem().hideAxis('left')
        self.plt_Pastilla.setBackground('#F0F0F0')
        self.plt_Pastilla.getPlotItem().setAspectLocked(True)
        self.verticalLayout_Pastilla.addWidget(self.plt_Pastilla)

        self.verticalLayoutWidget_Caja = QtWidgets.QWidget(self.groupBox_Todo)
        self.verticalLayoutWidget_Caja.setGeometry(QtCore.QRect(490, 390, 321, 251))
        self.verticalLayoutWidget_Caja.setObjectName("verticalLayoutWidget_Caja")

        self.verticalLayout_Caja = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_Caja)
        self.verticalLayout_Caja.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_Caja.setObjectName("verticalLayout_Caja")

        self.plt_Caja = pg.PlotWidget()
        self.plt_Caja.getPlotItem().hideAxis('bottom')
        self.plt_Caja.getPlotItem().hideAxis('left')
        self.plt_Caja.setBackground('#F0F0F0')
        self.plt_Caja.getPlotItem().setAspectLocked(True)
        self.verticalLayout_Caja.addWidget(self.plt_Caja)

        self.label_Logo = QtWidgets.QLabel(self.groupBox_Todo)
        self.label_Logo.setGeometry(QtCore.QRect(10, 10, 331, 141))
        self.label_Logo.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_Logo.setObjectName("label_Logo")


        InterfazIDmyPill.setCentralWidget(self.centralwidget)
        self.retranslateUi(InterfazIDmyPill)
        QtCore.QMetaObject.connectSlotsByName(InterfazIDmyPill)



    ################################################################################################################
    # FUNCIONES

    def funcBuscar(self):
        #self.pushButton_Buscar.setDisabled(True)
 
        # Primero chequea que la ruta ingresada exista
        try:
            ruta_ingresada = self.lineEdit_Ruta.text()

            ruta_ingresada = ruta_ingresada[1:-1]
            
            # Despues levanta la imagen ingresada y ejecuta la funcion de procesamiento con la imagen
            caracteristicas_nueva_pastilla = funcNuevaPastilla(ruta_ingresada)
        
            # Se obtienen las caracteristicas
            lista_caracteristicas_nueva_pastilla_norm = normalizarCaracteristicas(caracteristicas_nueva_pastilla)
            
            # Se normalizan las caracteristicas 
            df_caracteristicas_nueva_pastilla_norm = LISTtoDF(lista_caracteristicas_nueva_pastilla_norm)
            
            # Se clasifica la pastilla
            nombre_pastilla_clasificacion = funcClasificarPastilla(df_caracteristicas_nueva_pastilla_norm)

            # Del nombre de la pastilla reconocida se obtiene la descripción, el path de la imagen de la pastilla segmentada y de la caja comercial        
            descripcion, imagen_segmentada, imagen_caja = funcObtener_info_pastilla(nombre_pastilla_clasificacion[0])

            # Se imprime el nombre, y la descripcion de la pastilla en los textEdits
            self.textEdit_Pastilla.setPlainText(str(nombre_pastilla_clasificacion[0]))
            self.textEdit_Descripcion.setPlainText(str(descripcion))

            # Se plotea la imagen de la pastilla segmentada y de la caja en los vertical layouts
            nombre_imagen_segmentada = imagen_segmentada + '.png'
            nombre_imagen_caja = imagen_caja + '.png'
            path_imagenes_segmentadas = 'D:\Desktop\IDmyPill\Fotos Segmentadas'
            path_imagenes_cajas = 'D:\Desktop\IDmyPill\Fotos Comerciales'
            path_imagen_segmentada = os.path.join(path_imagenes_segmentadas, nombre_imagen_segmentada)
            path_imagen_caja = os.path.join(path_imagenes_cajas, nombre_imagen_caja)

            graf_segmentada = funcImagenRGB(path_imagen_segmentada)
            graf_caja = funcImagenRGB(path_imagen_caja)

            # Se muestra la imagen de la pastilla en el widget de gráfico de imagen "plt_Pastilla"
            self.plt_Pastilla.clear()
            image_item_pastilla = pg.ImageItem()
            image_item_pastilla.setImage(graf_segmentada)
            image_item_pastilla.setTransform(QtGui.QTransform().rotate(-90))  # Corregir la rotación
            self.plt_Pastilla.addItem(image_item_pastilla)

            # Se muestra la imagen de la caja en el widget de gráfico de imagen "plt_Caja"
            self.plt_Caja.clear()
            image_item_caja = pg.ImageItem()
            image_item_caja.setImage(graf_caja)
            image_item_caja.setTransform(QtGui.QTransform().rotate(-90))  # Corregir la rotación
            self.plt_Caja.addItem(image_item_caja)


        except:
            pop_up = QMessageBox()
            pop_up.setIcon(QMessageBox.Information)
            pop_up.setWindowTitle("Atención")
            pop_up.setText("No se ha encontrado ninguna imagen con la ruta ingresada. Verifique que la ruta ingresada sea correcta.")
            pop_up.setStandardButtons(QMessageBox.Ok)
            pop_up.exec_()


    ################################################################################################################

    def retranslateUi(self, InterfazIDmyPill):
        _translate = QtCore.QCoreApplication.translate
        InterfazIDmyPill.setWindowTitle(_translate("InterfazIDmyPill", "MainWindow"))
        self.lineEdit_Ruta.setPlaceholderText(_translate("InterfazIDmyPill", "Ingrese la ruta de acceso a la imagen de la pastilla que desea identificar."))
        self.textEdit_Pastilla.setHtml(_translate("InterfazIDmyPill", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n""<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n""p, li { white-space: pre-wrap; }\n""</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:11pt; font-weight:400; font-style:normal;\">\n""<p align=\"left\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.textEdit_Descripcion.setHtml(_translate("InterfazIDmyPill", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n""<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n""p, li { white-space: pre-wrap; }\n""</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:11pt; font-weight:400; font-style:normal;\">\n""<p align=\"left\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.label_Ruta.setText(_translate("InterfazIDmyPill", "<html><head/><body><p><span style=\" font-size:10pt;\">Ruta de acceso:</span></p></body></html>"))
        self.label_Pastilla.setText(_translate("InterfazIDmyPill", "<html><head/><body><p><span style=\" font-size:10pt;\">Nombre pastilla:</span></p></body></html>"))
        self.label_Descripcion.setText(_translate("InterfazIDmyPill", "<html><head/><body><p><span style=\" font-size:10pt;\">Descripción:</span></p></body></html>"))
        self.pushButton_Buscar.setText(_translate("InterfazIDmyPill", "Buscar"))
        self.label_Resultado.setText(_translate("InterfazIDmyPill", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600;\">Resultado:</span></p></body></html>"))
        self.label_Logo.setText(_translate("InterfazIDmyPill", "<html><head/><body><p><img src=\":/IDmyPillLogo/IDmyPillLogo.png\"/></p></body></html>"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    InterfazIDmyPill = QtWidgets.QMainWindow()
    ui = Ui_InterfazIDmyPill()
    ui.setupUi(InterfazIDmyPill)
    InterfazIDmyPill.show()
    sys.exit(app.exec_())
