# Object Detection and Tracking with YOLOv4

Este projeto implementa um sistema de detecção e rastreamento de objetos em vídeos utilizando YOLOv4 e OpenCV. Ele inclui um script principal (`main.py`) que permite configurar os caminhos dos modelos YOLOv4 e o vídeo de entrada através de argumentos da linha de comando.

## Instalação

Para executar este projeto, siga os passos abaixo:

1. **Clone o repositório**

   Clone este repositório para o seu ambiente local:

   ```bash
   git clone https://github.com/BFostek/yolov4-people-counting
   cd yolov4-people-counting
   ```

2. **Instale as dependências**

   Certifique-se de ter o Python instalado (versão 3.6 ou superior) e instale as dependências necessárias listadas no arquivo `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

   Este projeto utiliza bibliotecas como `opencv-python` para processamento de imagem.

---

## Download do Modelo YOLOv4

Para utilizar este projeto, é necessário baixar os modelos YOLOv4 pré-treinados. Siga os passos abaixo para obter os arquivos necessários:

1. Acesse o repositório do modelo YOLOv4 no GitHub: [kiyoshiiriemon/yolov4_darknet](https://github.com/kiyoshiiriemon/yolov4_darknet).

2. Baixe os seguintes arquivos:
   - **yolov4.cfg**: Arquivo de configuração do YOLOv4.
   - **yolov4.weights**: Pesos treinados para o YOLOv4.
   - **coco.names**: Arquivo que lista as classes de objetos que o modelo pode detectar.

3. Coloque os arquivos baixados na pasta `models/` do seu projeto.

Certifique-se de ajustar os caminhos nos argumentos da linha de comando no script `main.py` para refletir os diretórios onde você baixou os modelos.

---
## Uso

### Executando o script principal (`main.py`)

O script `main.py` permite a detecção e rastreamento de objetos em um vídeo usando YOLOv4. Ele aceita argumentos da linha de comando para configurar os caminhos dos modelos YOLOv4 e o vídeo de entrada.

Para executar o script com os valores padrão dos modelos e do vídeo:

```bash
python main.py
```

#### Argumentos da Linha de Comando

Você pode especificar os seguintes argumentos da linha de comando para personalizar a execução do script:

- `--cfg`: Caminho para o arquivo de configuração do YOLOv4 (padrão: `models/yolov4.cfg`).
- `--weights`: Caminho para o arquivo de pesos do YOLOv4 (padrão: `models/yolov4.weights`).
- `--names`: Caminho para o arquivo de nomes das classes do YOLOv4 (padrão: `models/coco.names`).
- `--video`: Caminho para o arquivo de vídeo de entrada (padrão: `videos/01.mp4`).

Exemplo de uso com argumentos personalizados:

```bash
python main.py --cfg path/to/custom/yolov4.cfg --weights path/to/custom/yolov4.weights --names path/to/custom/coco.names --video path/to/custom/video.mp4
```

### Funcionalidades Adicionais

- **Detecção de Objetos:** Utiliza YOLOv4 para detectar objetos em cada frame do vídeo.
- **Rastreamento de Objetos:** É utilizado o algoritmo [DeepSort](https://github.com/nwojke/deep_sort), para fazer o tracking de cada pessoa detectada para evitar ser contada a mesma pessoa mais de uma vez.
- **Visualização:** Exibe cada frame processado com caixas delimitadoras (bounding boxes) ao redor dos objetos detectados.
