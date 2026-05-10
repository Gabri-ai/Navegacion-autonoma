from controller import Display, Keyboard, Robot, Camera # API de Webots para interactuar con los nodos
from vehicle import Car, Driver                         # Nodos específicos para controlar vehículos en Webots
import numpy as np                                      # Para cálculos matemáticos y manejo de matrices (imágenes)
import cv2                                              # OpenCV para visión artificial y procesamiento de imágenes
from datetime import datetime                           # Para obtener la hora exacta
import os                                               # Para interactuar con los archivos del sistema operativo
import time                                             # Para medir el tiempo
import math                                             # Funciones matemáticas estándar


# --- CONSTANTES DE CONFIGURACIÓN DEL VEHÍCULO ---
# Estas variables definen los límites físicos y de control del auto.
DEBOUNCE_TIME = 0.1 # Tiempo (segundos) de espera entre lecturas de teclado para evitar múltiples registros (antirrebote).
MAX_ANGLE = 0.5     # Ángulo máximo de dirección permitido para las llantas (en radianes).
MAX_SPEED = 180     # Velocidad máxima permitida (km/h).
SPEED_INCR = 5      # Cuánto aumenta/disminuye la velocidad con cada pulsación de las flechas del teclado.
ANGLE_INCR = 0.05   # Incremento manual del ángulo (no se usa activamente porque el PID toma el control del volante).

# --- CONSTANTES DEL CONTROLADOR PID ---
# Estas "ganancias" deciden qué tan fuerte y rápido reacciona el volante ante un desvío.
KP = 0.003  # Proporcional: Reacciona al error actual (qué tan lejos está de la línea).
KI = 0.000  # Integral: Reacciona a la acumulación de errores pasados (memoria).
KD = 0.001  # Derivativo: Predice el error futuro amortiguando el movimiento para evitar zig-zag.

# --- VARIABLES GLOBALES PARA EL PID ---
# Se inician en 0 y cambian con cada ciclo de simulación.
integral = 0.0   # Acumulador del error en el tiempo.
prev_error = 0.0 # Guarda el error del ciclo anterior para calcular la velocidad de cambio (derivada).

# --- FUNCIONES DE VISIÓN ARTIFICIAL ---

# 1. Extraer imagen bruta de la cámara de Webots
def get_image(camera):
    raw_image = camera.getImage()  
    ''' Convierte el formato binario de Webots a una matriz tridimensional de píxeles inteligible
        por OpenCV (Alto x Ancho x 4 canales RGBA). '''
    
# El ancho y alto de la imagen de la camara se ajusto a 256 y 128 en el mundo de Webots. 
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
    speed = 60 # Velocidad inicial de arranque
    angle = 0.0 # Volante centrado al iniciar
    last_press = {} # Diccionario para registrar las teclas presionadas


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
            # Obtener el error más pequeño calculando la distancia entre el punto medio horizontal de cada línea y el setpoint.
            angle = max(min(angle, MAX_ANGLE), -MAX_ANGLE)
        
        else:
            '''Si no se detecta la línea en la imagen,# el vehículo mantiene la dirección recta (ángulo = 0.0)
            para evitar giros erráticos. Además, se reinician las variables del controlador PID (integral y prev_error)
            para borrar la memoria acumulada y evitar que errores pasados influyan en el siguiente ciclo de detección.'''
            angle = 0.0
            integral = 0.0
            prev_error = 0.0


        # --- INTERFAZ VISUAL (HUD) ---
        fuente = cv2.FONT_HERSHEY_SIMPLEX

        # Imprime telemetría en la pantalla
        cv2.putText(display_edges, f"Vel: {speed} km/h", (5, 15), fuente, 0.4, (255, 255, 255), 1)
        cv2.putText(display_edges, f"Err: {texto_error}", (5, 30), fuente, 0.4, (255, 255, 255), 1)
        cv2.putText(display_edges, f"Giro: {angle:.2f}", (5, 45), fuente, 0.4, (255, 255, 255), 1)
        cv2.putText(display_edges, f"Lineas: {total_lineas}", (128, 15), fuente, 0.4, (0, 255, 0), 1)
        cv2.putText(display_edges, f"Ancho: {camera.getWidth()}", (128, 30), fuente, 0.4, (0, 255, 0), 1)
        cv2.putText(display_edges, f"Alto: {camera.getHeight()}", (128, 45), fuente, 0.4, (0, 255, 0), 1)
        
        alto_pantalla = display_edges.shape[0]

        # Línea Guía AZUL: Representa el Setpoint estático (mitad de la pantalla)
        setpoint_x = int(camera.getWidth() / 2.0)
        cv2.line(display_edges, (setpoint_x, 0), (setpoint_x, alto_pantalla), (173, 216, 230), 1)

        # Línea Guía MAGENTA: Representa dónde está viendo la línea amarilla en la vida real
        if line_center_x is not None:
            centro_detectado_x = int(line_center_x)
            cv2.line(display_edges, (centro_detectado_x, 0), (centro_detectado_x, alto_pantalla), (255, 0, 255), 1)

        # Alertas
        if hay_cebra:
            cv2.putText(display_edges, "ALERTA: CEBRA", (5, 60), fuente, 0.4, (255, 255, 0), 1)
        
        if line_center_x is None:
            cv2.putText(display_edges, "ALERTA: LINEA PERDIDA", (5, 75), fuente, 0.4, (255, 0, 0), 1)

        # Envía toda la salida gráfica finalizada al panel de control (dashboard) de Webots
        display_image(display_img, display_edges)
        
        # --- ENTRADA DEL USUARIO (TECLADO) ---
        current_time = time.time()
        key = keyboard.getKey()

        # Comprueba si pasó el tiempo suficiente desde la última tecla presionada (antirrebote)
        if key not in last_press or (current_time - last_press.get(key, 0) >= DEBOUNCE_TIME):
            if key == keyboard.UP:
                if speed < MAX_SPEED: speed += SPEED_INCR
                last_press[key] = current_time
            elif key == keyboard.DOWN:
                if speed >= SPEED_INCR: speed -= SPEED_INCR
                last_press[key] = current_time
            elif key == ord('A'):
                # Captura de pantalla instantánea
                current_datetime = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
                file_name = current_datetime + ".png"
                print("Imagen guardada:", file_name)
                camera.saveImage(os.getcwd() + "/" + file_name, 1)
                last_press[key] = current_time

        # --- EJECUCIÓN FÍSICA ---
        # Finalmente, manda las instrucciones matemáticas a los motores del simulador
        driver.setSteeringAngle(angle)
        driver.setCruisingSpeed(speed)

if __name__ == "__main__":
    main()