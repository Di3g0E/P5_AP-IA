# P5_AP-IA: Proyecto 5 Aplicaciones IA

Este es el repositorio correspondiente al Proyecto 5 de aprendizaje y planificación en Inteligencia Artificial adaptado para producción.

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
│   ├── data/               # Scripts de carga, limpieza y validación
│   ├── features/           # Ingeniería de variables
│   ├── models/             # Definición de arquitectura y scripts de entrenamiento
│   ├── evaluation/         # Métricas, validación cruzada y reportes
│   └── utils/              # Funciones de soporte
├── tests/                  # Pruebas unitarias de integración (Pytest)
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

## Uso

El archivo base de arranque es `main.py`, que por convención debería orquestar todo el flujo productivo o abrir acceso a funcionalidades modulares:

```bash
python main.py
```
