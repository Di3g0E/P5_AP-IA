# P5_AP-IA: Sistema de Autenticación Biométrica Segura

Este es el repositorio correspondiente al Proyecto 5 de Aplicaciones de Inteligencia Artificial enfocado en la construcción de un sistema robusto de reconocimiento facial para entornos de producción, con un fuerte enfoque en la ciberseguridad.

## Estructura del Proyecto

```text
├── config/                 # Archivos de configuración (YAML, JSON, .env)
├── data/
│   ├── raw/                # Datos originales e inmutables (ignorados en git)
│   ├── processed/          # Datos limpios y transformados
│   └── external/           # Datos obtenidos de fuentes externas o APIs
├── doc/                    # Documentación técnica, memoria y diagramas
├── logs/                   # Archivos de registro (.log)
├── models/                 # Binarios de modelos entrenados (ignorados en git)
├── playground/             # Área extra para pruebas rápidas y notebooks
├── references/             # Papers, artículos y manuales de referencia
├── src/                    # Código fuente modular y productivo
│   ├── data/               # Scripts de carga, limpieza y validación (Facial y Financiero)
│   ├── features/           # Ingeniería de variables (CLAHE, MTCNN)
│   ├── models/             # Embeddings, Liveness y Detección de Anomalías
│   ├── evaluation/         # Métricas, validación cruzada y reportes
│   └── utils/              # Funciones de seguridad y soporte
├── tests/                  # Pruebas unitarias de integración
├── main.py                 # Punto de entrada principal
└── requirements.txt        # Dependencias fijas para el entorno de producción
```

## Requisitos e Instalación

Para instalar las dependencias y ejecutar el proyecto, usar un entorno virtual con `uv` o `pip`:

```bash
# Crear el entorno virtual (usando uv)
uv venv

# Activar el entorno virtual (en Windows)
.venv\Scripts\activate

# Instalar dependencias
uv pip install -r requirements.txt
```

## Ejecución (CLI)

El archivo principal `main.py` actúa como orquestador del sistema. Se proporcionan diferentes comandos de ejecución:

### 1. Gestión de Usuarios
- **Registrar un usuario**: Extrae y guarda el embedding a partir de una o varias fotos.
  ```bash
  python main.py register alice data/raw/alice/
  ```
- **Autenticar (Login)**: Verifica la identidad contra la base de datos (pasando por el sistema antispoofing).
  ```bash
  python main.py login alice data/raw/probe.jpg
  ```
- **Listar usuarios**: Muestra todos los usuarios inscritos.
  ```bash
  python main.py list
  ```
- **Eliminar usuario**: Borra un usuario de la base de datos segura.
  ```bash
  python main.py remove alice
  ```

### 2. Evaluación y Calibración
- **Evaluar métricas**: Calcula EER, ROC-AUC, F1 y TAR@FAR usando un dataset de pares.
  ```bash
  python main.py evaluate data/processed/pairs.csv
  ```
- **Benchmark**: Compara la latencia, rendimiento y separación coseno entre diferentes backends (ArcFace vs FaceNet).
  ```bash
  python main.py benchmark data/raw/test_faces/
  ```
- **Tuning (Ajuste óptimo)**: Encuentra automáticamente los umbrales óptimos de similitud y liveness para tus datos y actualiza `config.yaml`.
  ```bash
  python main.py tune data/processed/pairs.csv --apply
  ```

### 3. Control Financiero (Anomalías)
- **Añadir transacción**: Introduce un nuevo gasto/ingreso con validación en tiempo real.
  ```bash
  python main.py finance-add
  ```
  *Nota: El sistema utiliza un modelo híbrido (Isolation Forest + 3-Sigma) para detectar errores de entrada o gastos fuera de lo común antes de guardarlos en el histórico. Añade una explicación de la inconsitencia detectada para mayor explicabilidad para el usuario*

  Ejemplos:
  - Un registro normal: Amount=25, Area=Food, Type=Expenses -> Debe guardarse automáticamente.
  - Un registro anómalo por cantidad y tipo: Amount=5000, Area=Leisure, Type=Expenses -> Debe pedir confirmación.

## Implementaciones de Seguridad (Ciberseguridad)

Al ser un sistema biométrico, el enfoque de seguridad *Security by Design* es central en el proyecto. Se han integrado las siguientes capas de protección:

1. **Confidencialidad de Plantillas (AES-128-CBC)**: Los vectores de características (embeddings) nunca se almacenan en texto plano. Se cifran usando Fernet (AES en modo CBC con relleno PKCS7).
2. **Derivación Robusta de Claves (PBKDF2)**: La clave criptográfica no se almacena, se deriva dinámicamente de una contraseña maestra (passphrase) aplicando PBKDF2-HMAC-SHA256 con 310.000 iteraciones y un *salt* criptográficamente seguro (según estándares NIST SP 800-132).
3. **Integridad de Datos (HMAC-SHA256)**: Cada vector cifrado se acompaña de un hash SHA-256 de su valor en plano. Antes de dar por válido un embedding almacenado, el sistema comprueba la firma para detectar modificaciones maliciosas en el fichero de base de datos.
4. **Prevención de Ataques de Presentación (PAD)**: Se incorpora un modelo `DenseNet201` como *Liveness Detector* para rechazar fotografías impresas o reproducciones en pantallas. Este modelo actúa como una "puerta dura" (hard gate) al principio de la pipeline.
5. **Aislamiento Fotométrico**: El pipeline de preprocesamiento separa la imagen en crudo (enviada al detector de Liveness para que analice artefactos de falsificación como píxeles de pantalla) y la imagen ecualizada mediante CLAHE (enviada al embedder para maximizar la tasa de acierto).
6. **Bloqueo contra Fuerza Bruta (Access Control)**: Se cuenta el número de intentos de acceso fallidos por usuario, implementando un sistema de bloqueo temporal (*Account Lockout*) para mitigar los intentos de ataques reiterados.

## Detección de Anomalías Financieras (Módulo ML)

Se ha integrado un sistema de protección contra errores de entrada de datos y detección de gastos/ingresos inusuales utilizando un enfoque híbrido de Machine Learning:

1. **Regla Estadística de las 3-Sigmas (Z-Score)**: Proporciona una base robusta e interpretable para detectar anomalías univariantes (ej. un importe extremadamente alto para una categoría específica).
2. **Isolation Forest (iForest)**: Un algoritmo de aprendizaje no supervisado que detecta anomalías multivariantes, identificando patrones inusuales en la combinación de `Cantidad`, `Área`, `Tipo` y variables temporales (`Mes`, `Día de la semana`).

El sistema es **autoevolutivo**: el modelo se reentrena automáticamente con cada nueva entrada confirmada, adaptándose a la variabilidad natural del gasto del usuario sin generar falsos positivos excesivos.
