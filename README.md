# 🏀 NBA Analyzer - Sistema Completo

Herramienta completa para análisis estadístico y predicción del rendimiento de jugadores NBA mediante técnicas de machine learning.

<img alt="NBA Analyzer" src="https://img.shields.io/badge/NBA-Analyzer-orange">
<img alt="Python" src="https://img.shields.io/badge/Python-3.8+-blue">
<img alt="Scikit-learn" src="https://img.shields.io/badge/Scikit--learn-Modelos ML-green">
<img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-Visualización-red">
## 📋 Descripción del Proyecto

**NBA Analyzer** es una aplicación de análisis de datos deportivos que permite explorar estadísticas detalladas de jugadores de la NBA, generar predicciones sobre su rendimiento en diferentes situaciones de juego y realizar comparativas entre ellos. El proyecto combina análisis de datos con modelos predictivos de machine learning para ofrecer insights valiosos sobre el desempeño de los jugadores en la cancha.

### 🔧 Características principales
   **📊 Análisis estadístico detallado**: Exploración completa del historial de tiros, puntos por partido, eficiencia, zonas de tiro favoritas y más.
   **🔮 Predicciones basadas en ML**: Utiliza modelos de Random Forest para predecir probabilidades de éxito y tipos de tiro en diferentes contextos de juego.
   **🆚 Comparación entre jugadores**: Visualización comparativa de rendimiento entre múltiples jugadores.
   **🎮 Predicciones interactivas**: Permite al usuario definir contextos específicos (período, tiempo restante, distancia) para analizar el rendimiento esperado.
   **📈 Visualizaciones avanzadas**: Gráficos detallados de zonas de tiro, eficiencia por período, mapa de tiros y más.
   **💾 Exportación de visualizaciones**: Capacidad para guardar todos los gráficos generados.

## 🚀 Instalación Rápida

### 1. Requisitos del Sistema
🛠️ Requisitos
**Python 3.8+**
**pandas**
**numpy**
**scikit-learn**
**matplotlib**
**seaborn**

⚙️ Instalación
   1. Clonar este repositorio:
```bash
git clone https://github.com/usuario/nba-analyzer.git
cd nba-analyzer
```
   2. Instalar las dependencias:
```bash
pip install -r requirements.txt
```
   3. (Opcional) Configurar acceso a Kaggle API para obtención de datos adicionales.

### 2. Estructura de Archivos
```
tu_proyecto_tfg/
├── data/                        # Datos CSV locales
│   ├── pbp1999.csv             # Datos NBA play-by-play
│   └── ...
├── nba_analyzer.py       # Código principal ⭐
├── demo_nba_analyzer.py        # Script de demostración
├── requirements.txt            # Dependencias
├── README.md                   # Este archivo
```

### 3. Estructura de datos
   El analizador espera archivos en formato CSV con datos de jugadores NBA, con al menos los siguientes campos:

   **playerid**: Identificador único del jugador
   **player_name**: Nombre del jugador
   **gameid**: Identificador único del partido
   **period**: Período del juego (1-4, y posibles prórrogas)
   **clock**: Tiempo en el momento de la jugada
   **type**: Tipo de jugada ('Made Shot', 'Missed Shot', 'Free Throw')
   **subtype**: Categoría del tiro ('Jump Shot', 'Layup', 'Dunk', etc.)
   **x, y**: Coordenadas en la cancha
   **dist**: Distancia al aro
   **result**: Indicador de éxito (Made, Missed)

### 4. Datos Necesarios

> ⚠️ **IMPORTANTE**: Este proyecto ha sido entrenado en conjuntos de datos específicos de Kaggle y no se si funcionará correctamente con otros.

#### Conjuntos de datos requeridos:
- [NBA Play By Play Data](https://www.kaggle.com/datasets/szymonjwiak/nba-play-by-play-data-1997-2023)

#### Pasos para obtener los datos:
1. **Crear cuenta en Kaggle** si aún no tienes una
2. **Descargar los conjuntos de datos** desde los enlaces anteriores
3. **Extraer los archivos** y colocarlos en la carpeta `data/` del proyecto


Aunque el sistema puede generar datos sintéticos para demostración, es **altamente recomendable** descargar estos conjuntos de datos reales antes de utilizar el analizador para obtener resultados precisos y significativos.

> 💡 **Tip**: Para análisis más rápidos durante pruebas, puedes usar inicialmente solo un subconjunto de los años disponibles (ejemplo: 2020-2023).

## 📊 Uso del Sistema

### Uso Básico
Modo demostración completa:
```bash
python demo_nba_analyzer.py
```
Al ejecutar, selecciona la opción 1 para una demostración completa que guiará a través de todas las funcionalidades.

Modo interactivo
```bash
python demo_nba_analyzer.py
```
Selecciona la opción 2 para acceder al menú interactivo, donde podrás:

   - Explorar jugadores disponibles
   - Realizar análisis detallados
   - Generar predicciones
   - Comparar jugadores
   - Crear reportes comprensivos
### Ejemplos de Uso
```python
# Analisis detallado de un jugador
analyzer = NBAAnalyzer()
analyzer.load_data_with_priority()
analyzer.clean_and_process_data()
analyzer.get_detailed_player_stats('Stephen Curry')

# Predicción en contexto específico
context = {
    'period': 4,
    'time_remaining': 30,
    'score_diff': -2,
    'dist': 22,
    'x': -100,
    'y': 200
}
analyzer.predict_player_performance("M. Jordan", context)

# Comparacion entre jugadores
analyzer.compare_players(['N. Jokic', 'Joel Embiid'])
```

## 🔧 Funcionalidades Detalladas

### 1. Sistema de Carga de Datos
- **Prioridad 1**: Datos locales en carpeta `data/`
- **Prioridad 2**: Conexión a Kaggle API (opcional)
- **Prioridad 3**: Generación de datos sintéticos para testing

### 2. Análisis Estadístico
- Estadísticas generales (jugadas, tiros, puntos)
- Análisis por tipo de tiro (3PT, Dunk, Layup, etc.)
- Rendimiento por período
- Distribución de distancias de tiro

### 3. Machine Learning
- **Modelo 1**: Predicción de éxito de tiro: Estima la probabilidad de que un tiro sea exitoso basado en el contexto.
- **Modelo 2**: Predicción de tipo de tiro: Predice qué tipo de tiro realizará un jugador en determinada situación.
- **Modelo 3**: Predicción de puntos esperados: Combina los modelos anteriores para calcular los puntos esperados.
- Entrenamiento automático con RandomForest



- Los modelos utilizan caracteristicas como:
- **Período del juego**
- **Tiempo restante**
- **Diferencia de puntos**
- **Distancia al aro**
- **Coordenadas en la cancha**
- **Zonas específicas de la cancha (pintada, esquinas, etc.)**

### 4. Visualizaciones (9 gráficos automáticos)
1. Distribución de tipos de tiro
2. Porcentaje de éxito por tipo
3. Puntos por período
4. Evolución de efectividad por juego
5. Distribución de distancias
6. Mapa de calor de efectividad
7. Tipos de jugadas más frecuentes
8. Mapa de tiros en la cancha
9. Puntos acumulados por temporada

### 5. Comparación de Jugadores
- Métricas comparativas
- Gráficos de barras múltiples
- Gráfico radar normalizado
- Eficiencia de tiro

## Escalabilidad
El proyecto es parte de un Trabajo Final de Grado pero está abierto a mejoras:

- Áreas de oportunidad:
 - Integración con APIs en tiempo real
 - Modelos más sofisticados (Deep Learning)
 - Interfaz gráfica independiente
 - Análisis de video con Computer Vision

## Autor
Antonio Luis Godino - Proyecto de Trabajo Final de Grado (TFG)
