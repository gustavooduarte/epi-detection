'''
Para realizar o treinamento da rede Yolo é necessário realizar a conversão dos rótulos da Dataset de Origem,
pois as anotações origienais estão em um formato que não é reconhecido pela rede. Portanto, as fuções nesse
código realizam a conversão para formato compatível com Yolo e já salva na pasta adequada.

Na anotação original do dataset o formato da coordenda o formato é:
    <class-name> <xmin> <ymin> <xmax> <ymax>
    - <class-name>: Nome da classe
    - <xmin>: Posição x inicial do bouding box
    - <ymin>: Posição y inicial do bouding box
    - <xmax>: Posição x final do bouding box
    - <ymax>: Posição y final do bouding box

Porém para o treinamento da rede Yolo o formato acima não é reconhecido. Assim a funcção abaixo realiza a conversão para:
    <class-id> <x> <y> <width> <height>
    - <class-id>: Número que representa o id da classe
    - <x>: Ponto central do bouding box no eixo x (float que varia entre 0.0 a 1.0)
    - <y>: Ponto central do bouding box no eixo y (float que varia entre 0.0 a 1.0)
    - <width>: Valor relativo da largura do bouding box (float que varia entre 0.0 a 1.0)
    - <height>: Valor relativo da altura do bouding box (float que varia entre 0.0 a 1.0)
'''

import numpy as np
from pathlib import Path
from xml.dom.minidom import parse
import os

classes = ['person_with_helmet', 'person_no_helmet']

### Função para coletar dados do XML pertinentes e retornar lista
def get_xml_info(annotacion_path):
    dom = parse(annotacion_path)
    root = dom.documentElement

    # Coletando tamanho da imagem
    img_size = root.getElementsByTagName("size")[0]
    img_width = img_size.getElementsByTagName("width")[0].childNodes[0].data
    img_height = img_size.getElementsByTagName("height")[0].childNodes[0].data
    img_size = [img_width, img_height]
    
    objs = root.getElementsByTagName("object")
    bounding_boxes = []

    for box in objs:
        cls_name = box.getElementsByTagName("name")[0].childNodes[0].data
        print("{} : {}".format(annotacion_path, cls_name))
        
        if cls_name != 'person_with_helmet'and cls_name != 'person_no_helmet':
            continue
        
        xmin = int(box.getElementsByTagName("xmin")[0].childNodes[0].data)
        ymin = int(box.getElementsByTagName("ymin")[0].childNodes[0].data)
        xmax = int(box.getElementsByTagName("xmax")[0].childNodes[0].data)
        ymax = int(box.getElementsByTagName("ymax")[0].childNodes[0].data)
        
        bounding_boxes.append([cls_name, xmin, ymin, xmax, ymax])

    return  [img_size, bounding_boxes]

### Função que irá normalizar o bounding_box para as coordenadas do Yolo
def converter_annotations(img_width, img_height , bounding_box):
    #Separando informações do bouding box
    class_id = classes.index(bounding_box[0])
    xmin = int(bounding_box[1])
    ymin = int(bounding_box[2])
    xmax = int(bounding_box[3])
    ymax = int(bounding_box[4])

    #Pegando medidas para no formato Yolo
    width = xmax - xmin
    height = ymax - ymin
    x = xmin + (width / 2)
    y = ymin + (height / 2)

    #Convertendo para valores relativos entre 0.0 e 1.0
    x = np.float32(x / int(img_width))
    y = np.float32(y / int(img_height))
    width = np.float32(width / int(img_width))
    height = np.float32(height / int(img_height))

    return [class_id, x, y, width, height]

def main():
    # Chamada para percorrar todas as anotações da lista e chamar a função repsonsável pela conversão dos rótulos
    files = os.listdir('./dataset/annotations')
    for file in files:
        file_name = file.split(".")[0]
        img_size, bounding_boxes = get_xml_info('./dataset/annotations/' + file_name + '.xml')

        # Salvando arquivo na pasta adequada
        file_path = './dataset/labels/' +  file_name + '.txt'
        with open(file_path ,'a+') as file:
            for box in bounding_boxes:
                #Realizando a conversão
                box = converter_annotations(img_size[0], img_size[1], box)

                file.write(f"{box[0]} {box[1]} {box[2]} {box[3]} {box[4]}\n")
            file.flush()
            file.close()

    print('Processo de conversão finalizado')

if __name__ == '__main__':
    main()