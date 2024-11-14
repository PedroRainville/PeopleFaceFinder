import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def detectar_pessoas(imagem_path, confidence_threshold=0.9):
    img = cv2.cvtColor(cv2.imread(imagem_path), cv2.COLOR_BGR2RGB)

    detector = MTCNN()
    
    results = detector.detect_faces(img)
    
    results_filtrados = [p for p in results if p['confidence'] >= confidence_threshold]
    
    total_pessoas = len(results_filtrados)
    print(f'Total de pessoas detectadas: {total_pessoas} (com limiar de confiança >= {confidence_threshold})')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img)
    
    for result in results_filtrados:
        x1, y1, width, height = result['box']
        ax.add_patch(Rectangle((x1, y1), width, height, fill=False, ec='r', lw=2))
    
    ax.axis('off')

    plt.savefig('mostrar_rosto.png', pad_inches=0.1)
    plt.show()

def testar_imagens():
    confidence_threshold = float(input("Digite o limiar de confiança (entre 0.1 e 0.99 , ex: 0.9): "))
    
    while True:
        imagem_path = input("Digite o caminho da imagem para testar ou 'sair' para finalizar: ")
        if imagem_path.lower() == 'sair':
            break
        try:
            detectar_pessoas(imagem_path, confidence_threshold)
        except Exception as e:
            print(f"Erro ao carregar a imagem: {e}")

testar_imagens()
