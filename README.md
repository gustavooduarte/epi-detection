# Detecção de EPI com visão computacional e hardware de baixo custo

## Resumo
O setor de construção civil possui uma quantidade muito grande de pessoas empregadas e é um setor que possui um alto índice de acidentes de trabalho, os quais alguns deixam sequelas irreparáveis. Logo, é muito importante que algumas medidas de prevenção sejam tomadas para evitar os acidentes e, caso ocorram, os danos sejam menores. Uma das medidas a serem tomadas é a utilização de equipamentos de proteção individual (EPI), assim é necessário ter uma pessoa para verificar se os trabalhadores estão ou não utilizando esses equipamentos. Com o objetivo de facilitar a fiscalização da utilização desses equipamentos, neste trabalho é proposto um sistema de detecção de utilização de EPI de funcionários que atuam na construção civil a partir de técnicas de visão computacional e deep learning (aprendizado profundo) com equipamentos de baixo custo. O sistema proposto utilizou a arquitetura YOLOv4 e obteve um mAp de 89,5\% com uma média de 7,5 FPS em um computador com Core I5 e sem uso de GPU.


## Dependências
```
sudo apt-get install python-opencv
pip install argparse
pip install numpy
pip install imutils
```

## Execução do programa
```
prog -m modo -v video
```
### Operações possiveis:
**1. Execução em tempo real**
```
python main.py -m webcam
```
**2. Executando com vídeo**
```
python main.py -m video -v video/video_teste.mp4
```

## Dataset
O conjunto de dados utilizado para o treinamento foi o *Safety Helmet Wearing Dataset* ([download](https://data.mendeley.com/datasets/9rcv8mm682/1)). O *dataset* consiste em uma amostra de 5000 imagens, onde cada imagem do conjunto possui anotações rotuladas em seis classes diferentes (*helmet, head with helmet, person with helmet, head, person no helmet e face*) com um total de 75578 rótulos.