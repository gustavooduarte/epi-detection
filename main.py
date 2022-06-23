# Pacotes necessários
import cv2
import imutils
import numpy as np
import time
import argparse

############# Constantes #################
THRESHOLD = 0.5       #Valor constante para limiar da detecção
THRESHOLD_NMS = 0.3   #Valor constante para Non-Maximum Suppression

# Valores dos caminhos mais utilizados
names_path = 'data/obj.names'       # Path com os nomes das classes treinadas
cfg_path = 'cfg/yolov4_custom.cfg'  # Path para o arquivo com as configurações da rede Yolo
weights_path = 'cfg/yolov4_custom_final.weights'    # Path para os pesos da rede neural


############# Leitura dos argumentos #################
parser = argparse.ArgumentParser(description="Detecção de EPI")
parser.add_argument('-m', '--mode', help='video ou webcam', required=True)
parser.add_argument('-v', '--video', help='Path do vídeo', required=False)
args = vars(parser.parse_args())


############ Configurando rede neural e opencv ##############
# Carregando configuraçoes e pesos da rede no opencv
print("Carregando o detector de EPI...")
names_list = open(names_path).read().strip().split('\n')
detector = cv2.dnn.readNet(cfg_path, weights_path)
# Aprimorando para utilizar o Movidius da Intel (descomentar quando utilizar o Movidius)
#detector.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# Montando lista de cores para cada tipo de objeto detectado na imagem
#    Colors[0] - Pessoa com capacete
#    Colors[1] - Pessoa sem capacete
COLORS = np.array([[0, 255, 0], [0, 0, 255]])

# Como o reconhecimento será realizado em real time, utilizando captura das telas
print("Começando reconhecimento em tempo real...")

# Atribuindo modo de captura webcam ou video
if args['mode'] == 'webcam':
    videoStream_file = cv2.VideoCapture(0) # Quando valor igual a 0, é utilizado imagem da câmera
    time.sleep(3.0)
elif args['mode'] == 'video':
    videoStream_file = cv2.VideoCapture(args['video'])
else:
    print('Operação incorreta, tente novamente.')
    exit()

ln = detector.getLayerNames() # Coletando quantidade de variáveis da rede
ln = [ln[i[0] - 1] for i in detector.getUnconnectedOutLayers()] # Pegando informação da última camada da rede


############# Realizando a detecção #################
start = time.time()
total_frames = 0		  #Contador de frames para calcular Frame rate no final
start_frame_time = 0
end_frame_time = 0
while(True):
    #Resetando valores da caixa delimitadora, confianças e ID das classes
    bounding_boxes = []   #Lista para para armazenar medidas do bounding boxes
    confidences = []      #Lista para armazenar confiança de cada classe
    IDclasses = []        #Lista para armazenar precição feita
    
   
    (conectado, frame) = videoStream_file.read()
    

    # Redimensionando o frame para 600 pixels (mantendo o aspect ratio)
    try:
        frame = imutils.resize(frame, width=600)
        (H, W) = frame.shape[:2]
    except:
        break
 
    # Fazendo o blob da imagem
    img_blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (64, 64), swapRB=True, crop=False)

    # Fazendo a detecção
    detector.setInput(img_blob)
    layers_outputs = detector.forward(ln)

    for output in layers_outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > THRESHOLD:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype('int')

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                bounding_boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                IDclasses.append(classID)

    # Aplicando Non-Max-Suppression
    objs = cv2.dnn.NMSBoxes(bounding_boxes, confidences, THRESHOLD, THRESHOLD_NMS)

    if len(objs) > 0:
        for i in objs.flatten():
            (x, y) = (bounding_boxes[i][0], bounding_boxes[i][1])
            (w, h) = (bounding_boxes[i][2], bounding_boxes[i][3])

            cor = [int(c) for c in COLORS[IDclasses[i]]]

            cv2.rectangle(frame, (x, y), (x + w, y + h), cor, 2)
            text = '{}: {:.2f}'.format(names_list[IDclasses[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

    # Calculando e printando os FPS da aplicação executando
    end_frame_time = time.time()
    fps = 1/(end_frame_time - start_frame_time)
    fps_text = 'FPS: {:.1f}'.format(fps)
    cv2.putText(frame, fps_text, (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    start_frame_time = time.time()
    total_frames += 1

    # Apresentando a imagem
    cv2.imshow("Video", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

end = time.time()

frame_rate = total_frames / (end-start)
print("FPS médio: {:.2f}".format(frame_rate))

videoStream_file.release()
cv2.destroyAllWindows()

