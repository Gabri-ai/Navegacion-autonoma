from controller import Display, Keyboard, Robot, Camera # API de Webots para interactuar con los nodos
from vehicle import Car, Driver                         # Nodos específicos para controlar vehículos en Webots
import numpy as np                                      # Para cálculos matemáticos y manejo de matrices (imágenes)
import cv2                                              # OpenCV para visión artificial y procesamiento de imágenes
import time                                             # Para medir el tiempo
import math                                             # Funciones matemáticas estándar
import os                                               # Para construir rutas de archivos
import json                                             # Para leer parámetros HOG
import joblib                                           # Para cargar modelos de scikit-learn serializados
from skimage.feature import hog as skimage_hog         # Extracción de características HOG


# --- CONSTANTES DE CONFIGURACIÓN DEL VEHÍCULO ---
# Estas variables definen los límites físicos y de control del auto.
DEBOUNCE_TIME = 0.1 # Tiempo (segundos) de espera entre lecturas de teclado para evitar múltiples registros (antirrebote).
MAX_ANGLE = 0.5     # Ángulo máximo de dirección permitido para las llantas (en radianes).
MAX_SPEED = 180     # Velocidad máxima permitida (km/h).
CRUISE_SPEED = 40   # Velocidad crucero
SPEED_INCR = 5      # Cuánto aumenta/disminuye la velocidad ccrucero.
ANGLE_INCR = 0.05   # Incremento manual del ángulo (no se usa activamente porque el PID toma el control del volante).
MAX_ANGLE_RATE = 0.02 # Máximo cambio de ángulo por frame (evita pánico del PID en recuperación)

# --- CONSTANTES DEL LIDAR ---
LIDAR_ANGLE_DEG = 30        # Ángulo total de lectura en grados (15° a cada lado del centro)
LIDAR_MAX_RANGE = 20.0      # Distancia máxima de detección en metros
LIDAR_MIN_RANGE = 0.1       # Distancia mínima (evitar falsos positivos cercanos)
SVM_INTERVAL    = 0.5       # Segundos entre cada predicción SVM (throttle)
SVM_PERSIST     = 0         # Segundos que persiste el mensaje PEATON (0 = sin persistencia)
BRAKE_DISTANCE        = 18.0  # Distancia (m) a la que se activa el freno de emergencia
BRAKE_INTENSITY       = 1.00  # Intensidad máxima del freno de emergencia (0.0 – 1.0)
BRAKE_POWER           = 0.4   # Curvatura del frenado: <1 = agresivo desde lejos, >1 = suave lejos
SVM_ROI_TOP     = 0.10      # Fracción superior a ignorar en el ROI (cielo)
SVM_ROI_BOTTOM  = 0.80      # Fracción inferior a ignorar en el ROI (asfalto cercano)
HUD_DELAY_TO_LIDAR = 0.0    # Segundos antes de cambiar a HUD lidar
HUD_DELAY_TO_LINEA = 0.45   # Segundos antes de volver a HUD linea

lidar_name = "Sick LMS 291"

# --- CONSTANTES DEL CONTROLADOR PID ---

KP = 0.003  # Proporcional: Reacciona al error actual (qué tan lejos está de la línea).
KI = 0.000  # Integral: Reacciona a la acumulación de errores pasados (memoria).
KD = 0.001  # Derivativo: Predice el error futuro amortiguando el movimiento para evitar zig-zag.

# --- VARIABLES GLOBALES PARA EL PID ---
integral = 0.0   # Acumulador del error en el tiempo.
prev_error = 0.0 # Guarda el error del ciclo anterior para calcular la velocidad de cambio (derivada).

# --- RUTAS DE MODELOS ML ---
_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(_BASE_DIR, "Modelos", "pedestrian_svm.pkl")
SCALER_PATH     = os.path.join(_BASE_DIR, "Modelos", "pedestrian_scaler.pkl")
HOG_PARAMS_PATH = os.path.join(_BASE_DIR, "Modelos", "hog_params.json")


# --- CARGA DE MODELOS ML ---

def load_model(model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"Modelo cargado: {model_path}")
    print(f"Scaler cargado: {scaler_path}")
    return model, scaler


def load_hog_params(path):
    with open(path, "r") as f:
        return json.load(f)


def compute_hog_features(window, hog_params):
    gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY) if len(window.shape) == 3 else window
    gray = cv2.resize(gray, (32, 64))
    return skimage_hog(
        gray,
        orientations=hog_params["orientations"],
        pixels_per_cell=tuple(hog_params["pixels_per_cell"]),
        cells_per_block=tuple(hog_params["cells_per_block"]),
        block_norm=hog_params["block_norm"],
    )

def sliding_window_predict(image, model, scaler, hog_params, step_size=16):
    
    # Aplica el modelo SVM sobre la imagen completa usando una ventana deslizante de 32×64 px.
    # En lugar de clasificar la imagen completa, recorre la imagen en pasos de step_size píxeles
    # tanto horizontal como verticalmente, extrayendo y clasificando cada subventana por separado.
    # Si cualquier ventana es clasificada como peatón, retorna True inmediatamente.
    #
    # Parámetros
    # ----------
    # image      : imagen BGR capturada por la cámara del simulador
    # model      : modelo SVM entrenado (pedestrian_svm)
    # scaler     : StandardScaler ajustado sobre el entrenamiento (pedestrian_scaler)
    # hog_params : configuración HOG — debe coincidir exactamente con el entrenamiento
    # step_size  : desplazamiento en píxeles entre ventanas consecutivas (default: 16)
    #              valores menores aumentan la cobertura pero incrementan el tiempo de cómputo
    #
    # Retorna
    # -------
    # True  — se detectó al menos un peatón en alguna ventana
    # False — ninguna ventana clasificada como peatón
    
    win_w, win_h = 32, 64
    img = image[:, :, :3] if image.shape[2] == 4 else image  # descartar canal alfa
    img_h = img.shape[0]

    # ROI: excluye el cielo (top) y el asfalto cercano (bottom) para cámara 256×128
    y_start = int(img_h * SVM_ROI_TOP)
    y_end   = int(img_h * SVM_ROI_BOTTOM)
    roi     = img[y_start:y_end, :]       # recorte vertical de la imagen
    roi_h, roi_w = roi.shape[:2]

    for y in range(0, roi_h - win_h + 1, step_size):
        for x in range(0, roi_w - win_w + 1, step_size):
            window   = roi[y:y + win_h, x:x + win_w]
            features = compute_hog_features(window, hog_params)
            pred     = model.predict(scaler.transform([features]))[0]
            if pred == 1:
                return True  # peatón encontrado en alguna ventana
    return False


# --- FUNCIONES DE VISIÓN ARTIFICIAL ---

# 1. Extraer imagen bruta de la cámara de Webots
def get_image(camera):
    raw_image = camera.getImage()  
    # Convierte la imagen binaria de Webots en una matriz de píxeles
    # que OpenCV puede entender (Alto x Ancho x 4 canales RGBA)
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image

# 2. Convertir imagen a Escala de Grises
def greyscale_cv2(image):
    # Convierte la imagen a escala de grises, utilizando un único canal en lugar de tres.
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img


# 3. Detectar bordes en la imagen (Algoritmo de Canny)
def detect_edges(gray_image):
    # Aplica un filtro gaussiano (desenfoque) para eliminar el ruido o texturas irrelevantes
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    ''' El algoritmo Canny detecta bordes resaltando cambios bruscos de contraste en la imagen. 
        Los valores 50 y 150 corresponden a los umbrales de intensidad: el inferior (50) marca 
        el nivel mínimo para considerar un posible borde, mientras que el superior (150) establece 
        el nivel a partir del cual un borde se confirma con alta certeza. Los píxeles entre ambos 
        valores se clasifican como bordes solo si están conectados a regiones que superan el umbral alto.'''
    edges = cv2.Canny(blurred, 50, 150)
    return edges

# 4. Aislar la Región de Interés (ROI)
def apply_roi(edges_image):
    height, width = edges_image.shape
    # Crea una imagen completamente negra del mismo tamaño que la de la cámara
    mask = np.zeros_like(edges_image)
    
    # Coordenadas que dibujan un polígono (trapecio) en la mitad inferior de la pantalla.
    # Sirve para ignorar el cielo, árboles y otros carriles.
    polygon = np.array([[
        (int(width * 0.10), height),            # Inferior izquierdo
        (int(width * 0.90), height),            # Inferior derecho
        (int(width * 0.70), int(height * 0.60)),# Superior derecho
        (int(width * 0.20), int(height * 0.60)) # Superior izquierdo
    ]], np.int32)
    
    # Rellena el polígono de color blanco (255)
    cv2.fillPoly(mask, polygon, 255)
    # Superpone la máscara con la imagen de Canny. Solo sobreviven los bordes que caigan dentro del trapecio blanco.
    cropped_edges = cv2.bitwise_and(edges_image, mask)

    # Crea una copia a color para poder dibujar líneas verdes encima (HUD visual)
    display_edges = cv2.cvtColor(cropped_edges, cv2.COLOR_GRAY2RGB)
    
    # Se dibuja la frontera verde del trapecio para identificar dónde está mirando el algoritmo
    cv2.polylines(display_edges, [polygon], isClosed=True, color=(0, 255, 0), thickness=1)

    return cropped_edges, display_edges

# 5. Transformar píxeles en líneas matemáticas
def get_line_center(cropped_edges):
    ''' El algoritmo HoughLinesP (Transformada de Hough Probabilística) detecta líneas rectas en una 
        imagen analizando los píxeles de bordes y acumulando “votos” en un espacio de parámetros.
        Devuelve segmentos finitos (x1, y1, x2, y2) en lugar de líneas infinitas, 
        lo que lo hace más práctico para aplicaciones reales. '''
    lines = cv2.HoughLinesP(
        cropped_edges, 
        rho=1,              # Resolución en píxeles del acumulador (distancia mínima entre líneas detectadas).
        theta=np.pi/180,    # Resolución angular en radianes (1°).
        threshold=20,       # Votos mínimos para considerarse línea
        minLineLength=15,   # Longitud mínima de la línea detectada en píxeles
        maxLineGap=40       # Brecha máxima permitida entre dos píxeles para conectarlos como la misma línea
    )

    cantidad_lineas = 0
    if lines is not None:
        cantidad_lineas = len(lines)
    
    if lines is None:
        return None, cantidad_lineas # Retorna vacío si no hay líneas dentro del trapecio
        
    center_x_coords = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Filtro: Desecha cualquier línea que sea demasiado horizontal (ej. sombras largas, paso peatonal parcial)
        if x2 != x1: # Evita error matemático de dividir entre cero si la línea es 100% vertical
            slope = float(y2 - y1) / float(x2 - x1) # Cálculo de pendiente (m = y2-y1 / x2-x1)
            if abs(slope) < 0.5: # Si la pendiente es casi plana, ignórala y pasa a la siguiente
                continue
        
        # Encuentra el centro en el eje X de esta recta
        mid_x = (x1 + x2) / 2.0
        center_x_coords.append(mid_x)
        
    # Si sobrevivieron líneas válidas, promedia todos sus centros X para obtener un solo punto medio global
    if len(center_x_coords) > 0:
        return np.mean(center_x_coords), cantidad_lineas
    return None, cantidad_lineas

# 6. Detección de paso de cebras (cruce peatonal)
def detect_zebra_crossing(gray_image, display_image):
    # Binarización: Todo lo oscuro (asfalto, menor a 200) se hace negro. Lo brillante (blanco/amarillo) se hace blanco puro.
    _, thresh = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
    
    # Busca contornos (perímetros de manchas blancas)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    zebra_detected = False
    height, width = gray_image.shape
    lineas_cebra_encontradas = 0 # Contador de rectángulos válidos
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Filtro: Ignora manchas blancas diminutas (ruido, basura en la cámara)
        if area > 80: 
            # Encierra la mancha en un rectángulo geométrico (bounding box)
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Evaluación Lógica: 
            # 1. ¿Está en la mitad inferior de la imagen? (más cerca del auto)
            # 2. ¿Es mucho más ancho (w) que alto (h)? -> Define la forma clásica de una cebra
            if y > (height * 0.5) and w > (h * 1.5):
                # Si cumple, dibújale un cuadro de advertencia (Azul en formato RGB)
                cv2.rectangle(display_image, (x, y), (x+w, y+h), (0, 0, 255), 1)
                lineas_cebra_encontradas += 1
                
    # Se confirma que hay un cruce peatonal si el algoritmo ve al menos 2 rectángulos horizontales agrupados
    if lineas_cebra_encontradas >= 2:
        zebra_detected = True
                
    return zebra_detected

# 7. Lectura del sensor LIDAR
def get_lidar_data(lidar, max_range=LIDAR_MAX_RANGE):
    # Consulta el LIDAR y devuelve la distancia al objeto más cercano que esté
    # justo al frente del vehículo (dentro de los 30° centrales del sensor).
    #
    # El LIDAR escanea en abanico y entrega una lista de distancias, una por cada
    # dirección que mide. Esta función solo mira las lecturas del centro del abanico
    # y descarta las que no son confiables:
    #   - Lecturas corruptas (valores imposibles como NaN o infinito)
    #   - Objetos demasiado cerca (< 0.1 m): suelen ser partes del propio auto
    #   - Objetos fuera del rango útil (> max_range): no nos interesan
    #
    # Parámetros
    # ----------
    # lidar     : el sensor LIDAR del simulador
    # max_range : distancia máxima a considerar (por defecto 20 m)
    #
    # Retorna
    # -------
    # float — distancia en metros al objeto más cercano detectado al frente
    # None  — si no hay nada en frente o el sensor no entrega datos

    range_image = lidar.getRangeImage()
    if not range_image:
        return None

    n_points = len(range_image)
    fov_deg  = math.degrees(lidar.getFov())
    half_pts = int(round((LIDAR_ANGLE_DEG / 2.0) / fov_deg * n_points))

    center = n_points // 2
    start  = max(0, center - half_pts)
    end    = min(n_points, center + half_pts + 1)

    valid = [
        d for d in range_image[start:end]
        if not math.isnan(d) and not math.isinf(d)
        and LIDAR_MIN_RANGE <= d <= max_range
    ]

    return min(valid) if valid else None


def lidar_color(distancia):
    # Convierte la distancia LiDAR en un color para retroalimentación visual en pantalla.
    # Cian → sin detección | Rojo → peligro | Naranja → alerta | Amarillo → precaución | Verde → despejado
    if distancia is None:
        return (0, 255, 255)   # cian → sin detección
    ratio = distancia / LIDAR_MAX_RANGE
    if ratio < 0.25:
        return (220, 0, 0)     # rojo  → peligro  (< 5 m)
    if ratio < 0.50:
        return (255, 120, 0)   # naranja → alerta (5–10 m)
    if ratio < 0.75:
        return (220, 220, 0)   # amarillo → precaución (10–15 m)
    return (0, 200, 0)         # verde → despejado (> 15 m)


# --- FRENO DE EMERGENCIA ---

def emergency_brake(driver, distancia):
    # Frena el vehículo de forma progresiva según qué tan cerca está el obstáculo.
    # Cuanto más cerca esté el objeto, más fuerte frena:
    #   - A 18 m (BRAKE_DISTANCE): empieza a frenar suavemente
    #   - A 0 m                  : frena al 100% (BRAKE_INTENSITY)
    # Si no hay nada detectado o el objeto está lejos, suelta el freno por completo.
    # Retorna True si está frenando, False si no.
    if distancia is not None and distancia <= BRAKE_DISTANCE:
        ratio = (1.0 - (distancia / BRAKE_DISTANCE)) ** BRAKE_POWER
        driver.setBrakeIntensity(ratio * BRAKE_INTENSITY)
        return True
    driver.setBrakeIntensity(0.0)
    return False


# --- LÓGICA DE CONTROL ---

# Controlador PID: ajusta la salida (ej. giro del volante) en función de la desviación respecto al objetivo (setpoint).
def pid_controller(setpoint, current_value, dt):
    global integral, prev_error  # Variables globales para acumular el error y recordar el error previo
    
    # 1. Cálculo del error actual
    error = setpoint - current_value          
    ''' Representa la diferencia entre el valor deseado (setpoint) y el valor medido (current_value).
        Si es positivo, estamos por debajo del objetivo; si es negativo, por encima.'''
    
    # 2. Componente integral
    integral += error * dt                    
    ''' Acumula el error a lo largo del tiempo. Sirve para corregir desviaciones persistentes
        que el término proporcional por sí solo no logra eliminar (offset o bias).'''
    
    # 3. Componente derivativo
    derivative = (error - prev_error) / dt    
    ''' Mide la velocidad de cambio del error. Si el error está creciendo rápido,
        este término actúa como freno para evitar oscilaciones o sobrecorrecciones.'''
    
    # 4. Fórmula del PID
    output = (KP * error) + (KI * integral) + (KD * derivative)
    ''' Cada término se multiplica por su ganancia (KP, KI, KD):
            - KP (Proporcional): responde directamente a la magnitud del error.
            - KI (Integral): responde a la acumulación del error en el tiempo.
            - KD (Derivativo): responde a la tendencia del error (su velocidad de cambio).
        La suma de los tres genera la señal de control final. '''
    
    # 5. Actualización del error previo
    prev_error = error 
    # Se guarda el error actual para calcular la derivada en el siguiente ciclo.
    
    return output


# Función auxiliar para convertir la imagen de OpenCV (cv2) a un formato que el display de Webots pueda mostrar
def display_image(display, image):
    if len(image.shape) == 2:
        image_rgb = np.dstack((image, image, image))
    else:
        image_rgb = image
        
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)


# --- BUCLE PRINCIPAL (MAIN) ---
def main():
    global integral, prev_error
    speed = CRUISE_SPEED  # Velocidad crucero
    angle = 0.0 # Volante centrado al iniciar
    prev_angle = 0.0 # Ángulo del frame anterior (para limitador de tasa)
    last_press = {} # Diccionario para registrar las teclas presionadas
    display_mode = 0    # 0 = HUD linea, 1 = HUD lidar
    auto_hud        = True  # True = cambia de HUD automáticamente según el LIDAR
    hud_switch_time = 0.0  # Marca de tiempo en que inició la transición pendiente
    last_peaton_time  = 0.0   # Marca de tiempo de la última detección de peatón 
                              # y se usa para persistir el mensaje de peaton
    last_svm_time     = 0.0   # Tiempo de la última ejecución del SVM 
                              # evita correrlo en cada frame para que no se alente la ejecucion del simulador
    clasificacion     = None  # Último resultado SVM cacheado



    # Inicialización del Robot y Driver (motores)
    robot = Car()
    driver = Driver()
    timestep = int(robot.getBasicTimeStep()) # Obtiene la velocidad base de la simulación
    dt = timestep / 1000.0                   # Delta Time: milisegundos convertidos a segundos (fundamental para PID)


    # Encendido de dispositivos periféricos
    camera = robot.getDevice("camera")
    camera.enable(timestep)
    display_img = Display("display_image")
    keyboard = Keyboard()
    keyboard.enable(timestep)

    lidar = robot.getDevice(lidar_name)
    lidar.enable(timestep)

    # Carga del modelo SVM, el scaler y los parámetros HOG al inicio
    model, scaler = load_model(MODEL_PATH, SCALER_PATH)
    hog_params    = load_hog_params(HOG_PARAMS_PATH)

    # --- RESUMEN DE PARÁMETROS AL INICIO ---
    print("=" * 45)
    print("       PARÁMETROS DEL CONTROLADOR")
    print("=" * 45)
    print(f"  [Vehículo]")
    print(f"    Velocidad inicial : {speed} km/h")
    print(f"    Velocidad máxima  : {MAX_SPEED} km/h")
    print(f"    Incremento vel.   : {SPEED_INCR} km/h")
    print(f"    Ángulo máximo     : {MAX_ANGLE} rad")
    print(f"    Distancia freno   : {BRAKE_DISTANCE} m")
    print(f"    Intensidad freno  : {BRAKE_INTENSITY}")
    print(f"    Potencia freno    : {BRAKE_POWER} (curva {'lineal' if BRAKE_POWER==1 else 'cuadrática' if BRAKE_POWER==2 else 'personalizada'})")
    print(f"  [PID]")
    print(f"    Kp = {KP}  |  Ki = {KI}  |  Kd = {KD}")
    print(f"  [LIDAR]")
    print(f"    Ángulo de lectura : {LIDAR_ANGLE_DEG}° ({LIDAR_ANGLE_DEG//2}° c/lado)")
    print(f"    Rango             : {LIDAR_MIN_RANGE} m – {LIDAR_MAX_RANGE} m")
    print(f"    Intervalo SVM     : {SVM_INTERVAL} s")
    print(f"    Persistencia SVM  : {SVM_PERSIST} s {'(desactivada)' if SVM_PERSIST == 0 else ''}")
    print(f"  [Cámara]")
    print(f"    Resolución        : {camera.getWidth()}x{camera.getHeight()} px")
    print(f"    Timestep          : {timestep} ms  (dt={dt:.4f} s)")
    print(f"  [Modelos SVM]")
    print(f"    Archivo SVM       : {os.path.basename(MODEL_PATH)}")
    print(f"    Archivo Scaler    : {os.path.basename(SCALER_PATH)}")
    print(f"    Features esperadas: {scaler.n_features_in_}")
    print(f"    Ventana HOG       : 32x64 px (ancho x alto)")
    print(f"    HOG orientaciones : {hog_params['orientations']}")
    print(f"    HOG px/celda      : {hog_params['pixels_per_cell']}")
    print(f"    HOG celdas/bloque : {hog_params['cells_per_block']}")
    cam_w, cam_h = camera.getWidth(), camera.getHeight()
    roi_h = int(cam_h * SVM_ROI_BOTTOM) - int(cam_h * SVM_ROI_TOP)
    n_x   = len(range(0, cam_w - 32 + 1, 16))
    n_y   = len(range(0, roi_h - 64 + 1, 16))
    print(f"  [ROI SVM  ({cam_w}x{cam_h} cámara)]")
    print(f"    Zona Y            : {int(cam_h*SVM_ROI_TOP)}–{int(cam_h*SVM_ROI_BOTTOM)} px  ({roi_h} px alto)")
    print(f"    Ventanas/frame    : {n_x} x {n_y} = {n_x * n_y}")
    print(f"  [HUD automático]")
    print(f"    Retardo → lidar   : {HUD_DELAY_TO_LIDAR} s")
    print(f"    Retardo → linea   : {HUD_DELAY_TO_LINEA} s")
    print("=" * 45)

    # Inicio del bucle infinito. Se ejecuta una vez por cada "paso" o fotograma de la simulación.
    while robot.step() != -1:
        
        # --- PIPELINE DE VISIÓN ---
        ''' 1. Captura de la imagen desde la cámara del robot.
        Se obtiene el fotograma actual en formato crudo, que servirá como entrada
        para todo el procesamiento posterior.'''
        image = get_image(camera)

        ''' 2. Conversión a escala de grises.
        Se reduce la imagen a un solo canal de intensidad (0-255), eliminando la
        información de color. Esto simplifica los cálculos y acelera los algoritmos
        de visión, ya que muchos detectores de bordes y formas trabajan sobre
        contrastes de luminancia, no de color.'''
        grey_image = greyscale_cv2(image)
        
        ''' 3. Detección de bordes con el algoritmo de Canny.
        Canny identifica transiciones bruscas de intensidad en la imagen.
        Utiliza dos umbrales (bajo y alto) para decidir qué píxeles son bordes
        seguros y cuáles son bordes potenciales, aplicando además un criterio de
        conectividad. El resultado es una imagen binaria donde los bordes aparecen
        resaltados en blanco sobre fondo negro. '''
        edges = detect_edges(grey_image)
        
        ''' 4. Definición y aplicación de la Región de Interés (ROI).
        Se selecciona un área específica de la imagen (ej. la parte inferior central
        donde se espera encontrar la carretera o carriles). Para este ejercicio se utiliza
        un trapecio con las dimensiones:

            - Base inferior (ancho abajo): 0.90 width - 0.10 width = 0.80 width
              La base inferior ocurpa el 80% del ancho total de la imagen

            - Base superior (ancho arriba): 0.70 width - 0.20 width = 0.50 width
              La base superior ocurpa el 50% del ancho total de la imagen

            - Altura: height - 0.60 height = 0.40 height
              La altura es el 40% de la altura total de la imagen
              
        
        Esto descarta ruido en zonas irrelevantes y concentra el análisis en la región útil.'''
        roi_edges, display_edges = apply_roi(edges)

        ''' 4.1. Detección de paso de cebra (cruce peatonal).
        Se analiza la imagen en escala de grises junto con los bordes del ROI
        para identificar patrones característicos de franjas blancas paralelas.
        Si se detecta este patrón, la variable 'hay_cebra' se activa como indicador
        de la presencia de un cruce peatonal.'''
        hay_cebra = detect_zebra_crossing(grey_image, display_edges)

        ''' 5. Detección de líneas rectas y validación.
        Se aplica la Transformada de Hough probabilística sobre los bordes del ROI
        para extraer segmentos de línea. Posteriormente se filtran las líneas
        horizontales (irrelevantes para la navegación) y se calcula la posición
        central de la línea principal detectada. Este valor se usa como referencia
        para el control del vehículo (ej. PID).'''
        line_center_x, total_lineas = get_line_center(roi_edges)
        
        # --- LECTURA DEL LIDAR ---
        distancia_frontal = None
        if lidar is not None:
            distancia_frontal = get_lidar_data(lidar)

        # --- CLASIFICACIÓN CON VENTANAS DESLIZANTES (throttleado) ---
        t_svm = time.time()
        if distancia_frontal is None:
            clasificacion = None
        elif t_svm - last_svm_time >= SVM_INTERVAL:
            hay_peaton    = sliding_window_predict(image, model, scaler, hog_params)
            clasificacion = "PEATON" if hay_peaton else "OBSTACULO"
            last_svm_time = t_svm
            if hay_peaton:
                last_peaton_time = t_svm

        # --- DETECCIÓN DE PEATÓN (con persistencia) ---
        # Se consiero la persistencia de este mensaje ya que al 
        # inicio de las simulaciones desaparecia muy pronto.
        t_ahora = time.time()
        peaton_persistido = SVM_PERSIST > 0 and (t_ahora - last_peaton_time) <= SVM_PERSIST
        mostrar_peaton = distancia_frontal is not None and (
            peaton_persistido or clasificacion == "PEATON"
        )

        # Freno de emergencia (se calcula antes del HUD para poder mostrarlo)
        frenando = emergency_brake(driver, distancia_frontal)
        if frenando and distancia_frontal is not None:
            brake_pct = ((1.0 - distancia_frontal / BRAKE_DISTANCE) ** BRAKE_POWER) * BRAKE_INTENSITY * 100
        else:
            brake_pct = 0.0

        # Variable auxiliar para mostrar mensajes de error o estado en pantalla.
        texto_error = "---"

        # --- LÓGICA DE NAVEGACIÓN ---
        # 6. Setpoint y Controlador PID
        if line_center_x is not None:
            # Setpoint: El objetivo ideal. Queremos la línea exactamente en la mitad del ancho de la cámara.
            setpoint = camera.getWidth() / 2.0
            error_actual = setpoint - line_center_x
            texto_error = f"{error_actual:.2f}"

            # El PID nos entrega el ángulo necesario para regresar el auto a la línea
            steering_adjust = pid_controller(setpoint, line_center_x, dt)
            ''' Se invierte la señal debido a la polaridad de los ejes del vehículo. En las pruebas iniciales,
            el control hacía que el auto se alejara de la línea amarilla en lugar de corregir su trayectoria
            hacia ella, por lo que fue necesario invertir la dirección del ajuste.'''
            angle = -steering_adjust

            # Restricción física: No podemos girar el volante más allá del máximo del coche
            angle = max(min(angle, MAX_ANGLE), -MAX_ANGLE)

        else:
            '''Si no se detecta la línea en la imagen, el vehículo mantiene la dirección recta (ángulo = 0.0)
            para evitar giros erráticos. Además, se reinician las variables del controlador PID (integral y prev_error)
            para borrar la memoria acumulada y evitar que errores pasados influyan en el siguiente ciclo de detección.'''
            angle = 0.0
            integral = 0.0
            prev_error = 0.0

        # Limitador de tasa: evita saltos bruscos del volante en recuperación del PID
        angle = max(min(angle, prev_angle + MAX_ANGLE_RATE), prev_angle - MAX_ANGLE_RATE)
        prev_angle = angle

        # --- INTERFAZ VISUAL (HUD) ---
        if auto_hud:
            target_mode = 1 if distancia_frontal is not None else 0
            if target_mode != display_mode:
                delay = HUD_DELAY_TO_LIDAR if target_mode == 1 else HUD_DELAY_TO_LINEA
                if hud_switch_time == 0.0:
                    hud_switch_time = time.time()
                if time.time() - hud_switch_time >= delay:
                    display_mode = target_mode
                    hud_switch_time = 0.0
            else:
                hud_switch_time = 0.0

        fuente = cv2.FONT_HERSHEY_SIMPLEX
        alto_pantalla = display_edges.shape[0]
        setpoint_x = int(camera.getWidth() / 2.0)

        # HUD 0 (linea): velocidad, error PID y ángulo de giro (Actividad seguimiento de línea)
        # HUD 1 (lidar): distancia al obstáculo, clasificación SVM y porcentaje de freno (Actividad detección con SVM)
        if display_mode == 0:

            cv2.putText(display_edges, f"Vel: {driver.getCurrentSpeed():.1f} km/h", (5, 15), fuente, 0.4, (255, 255, 255), 1)
            cv2.putText(display_edges, f"Err: {texto_error}", (5, 30), fuente, 0.4, (255, 255, 255), 1)
            cv2.putText(display_edges, f"Giro: {angle:.2f}", (5, 45), fuente, 0.4, (255, 255, 255), 1)
            cv2.putText(display_edges, f"Lineas: {total_lineas}", (128, 15), fuente, 0.4, (0, 255, 0), 1)
            cv2.putText(display_edges, f"Ancho: {camera.getWidth()}", (128, 30), fuente, 0.4, (0, 255, 0), 1)
            cv2.putText(display_edges, f"Alto: {camera.getHeight()}", (128, 45), fuente, 0.4, (0, 255, 0), 1)

            tope_trapecio_0 = int(alto_pantalla * 0.60)
            cv2.line(display_edges, (setpoint_x, tope_trapecio_0), (setpoint_x, alto_pantalla), (173, 216, 230), 1)
            if line_center_x is not None:
                centro_detectado_x = int(line_center_x)
                cv2.line(display_edges, (centro_detectado_x, tope_trapecio_0), (centro_detectado_x, alto_pantalla), (255, 0, 255), 1)

            # Alertas
            if hay_cebra:
                cv2.putText(display_edges, "ALERTA: CEBRA", (5, 60), fuente, 0.4, (255, 255, 0), 1)
            if line_center_x is None:
                cv2.putText(display_edges, "ALERTA: LINEA PERDIDA", (5, 75), fuente, 0.4, (255, 0, 0), 1)

            display_image(display_img, display_edges)

        else:

            ancho_pantalla = display_edges.shape[1]
            vel_actual = driver.getCurrentSpeed()
            cv2.putText(display_edges, f"Vel: {vel_actual:.1f} km/h", (5, 15), fuente, 0.4, (255, 255, 255), 1)
            lidar_txt = f"{distancia_frontal:.2f} m" if distancia_frontal is not None else "---"
            cv2.putText(display_edges, f"Lidar: {lidar_txt}", (5, 30), fuente, 0.4, lidar_color(distancia_frontal), 1)


            if mostrar_peaton:
                cv2.putText(display_edges, "SVM: PEATON", (5, 45), fuente, 0.4, (0, 255, 0), 1)
            elif clasificacion == "OBSTACULO":
                cv2.putText(display_edges, "SVM: OBSTACULO", (5, 45), fuente, 0.4, (0, 165, 255), 1)

            if frenando:
                freno_txt = f"FRENO {brake_pct:.0f}%"
                (tw, _), _ = cv2.getTextSize(freno_txt, fuente, 0.45, 1)
                fx = (ancho_pantalla - tw) // 2
                fy = int(alto_pantalla * 0.60) - 5
                cv2.putText(display_edges, freno_txt, (fx, fy), fuente, 0.45, lidar_color(distancia_frontal), 1)

            if hay_cebra:
                cv2.putText(display_edges, "CEBRA", (ancho_pantalla - 105, 15), fuente, 0.4, (255, 255, 0), 1)
            if line_center_x is None:
                cv2.putText(display_edges, "LINEA PERDIDA", (5, 60), fuente, 0.4, (255, 0, 0), 1)

            tope_trapecio = int(alto_pantalla * 0.60)
            cv2.line(display_edges, (setpoint_x, tope_trapecio), (setpoint_x, alto_pantalla), (173, 216, 230), 1)
            if line_center_x is not None:
                centro_detectado_x = int(line_center_x)
                cv2.line(display_edges, (centro_detectado_x, tope_trapecio), (centro_detectado_x, alto_pantalla), (255, 0, 255), 1)

            display_image(display_img, display_edges)
        
        # --- ENTRADA DEL USUARIO (TECLADO) ---
        current_time = time.time()
        key = keyboard.getKey()

        if key not in last_press or (current_time - last_press.get(key, 0) >= DEBOUNCE_TIME):
            if key == keyboard.UP:
                if speed < MAX_SPEED: speed += SPEED_INCR
                last_press[key] = current_time
            elif key == keyboard.DOWN:
                if speed >= SPEED_INCR: speed -= SPEED_INCR
                last_press[key] = current_time
            elif key == ord('A'):
                auto_hud = not auto_hud
                modo_nombre = 'lidar' if display_mode == 1 else 'linea'
                print(f"HUD automático: {'activado' if auto_hud else f'desactivado (HUD {modo_nombre})'}")
                last_press[key] = current_time

        # --- EJECUCIÓN FÍSICA ---
        # Aplica el ángulo calculado por el PID al volante y decide la velocidad según el estado:
        #   1. Obstáculo a menos de 18 m → freno de emergencia, velocidad objetivo = 0
        #   2. Cualquier detección (18–20 m) → reduce velocidad proporcionalmente a la distancia
        #   3. Sin detección             → mantiene la velocidad crucero configurada
        driver.setSteeringAngle(angle)
        vel_actual = driver.getCurrentSpeed()
        if frenando:
            # Freno de emergencia: cortar motor y dejar que setBrakeIntensity pare el auto
            driver.setCruisingSpeed(0)
        elif distancia_frontal is not None:
            # Obstáculo detectado dentro del rango: reducir velocidad gradualmente
            ratio_peaton = distancia_frontal / LIDAR_MAX_RANGE
            target = speed * ratio_peaton
            driver.setCruisingSpeed(target)
            if vel_actual > target + 1.0:
                driver.setBrakeIntensity(min((vel_actual - target) / target, 0.3))
            else:
                driver.setBrakeIntensity(0.0)
        else:
            driver.setCruisingSpeed(speed)
            if vel_actual > speed + 1.0:
                driver.setBrakeIntensity(min((vel_actual - speed) / speed, 0.3))
            else:
                driver.setBrakeIntensity(0.0)

if __name__ == "__main__":
    main()