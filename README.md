# 🏀 NBA Analyzer - Sistema Completo

Herramienta completa para análisis estadístico y predicción del rendimiento de jugadores NBA mediante técnicas de machine learning.

<img alt="NBA Analyzer" src="https://img.shields.io/badge/NBA-Analyzer-orange">
<img alt="Python" src="https://img.shields.io/badge/Python-3.8+-blue">
<img alt="Scikit-learn" src="https://img.shields.io/badge/Scikit--learn-Modelos ML-green">
<img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-Visualización-red">
## 📋 Descripción del Proyecto

**NBA Analyzer** es una aplicación de análisis de datos deportivos que permite explorar estadísticas detalladas de jugadores de la NBA, generar predicciones sobre su rendimiento en diferentes situaciones de juego y realizar comparativas entre ellos. El proyecto combina análisis de datos con modelos predictivos de machine learning para ofrecer insights valiosos sobre el desempeño de los jugadores en la cancha.

### 🔧 Características principales
   📊 Análisis estadístico detallado: Exploración completa del historial de tiros, puntos por partido, eficiencia, zonas de tiro favoritas y más.
   🔮 Predicciones basadas en ML: Utiliza modelos de Random Forest para predecir probabilidades de éxito y tipos de tiro en diferentes contextos de juego.
   🆚 Comparación entre jugadores: Visualización comparativa de rendimiento entre múltiples jugadores.
   🎮 Predicciones interactivas: Permite al usuario definir contextos específicos (período, tiempo restante, distancia) para analizar el rendimiento esperado.
   📈 Visualizaciones avanzadas: Gráficos detallados de zonas de tiro, eficiencia por período, mapa de tiros y más.
   💾 Exportación de visualizaciones: Capacidad para guardar todos los gráficos generados.

## 🚀 Instalación Rápida

### 1. Requisitos del Sistema
🛠️ Requisitos
Python 3.8+
pandas
numpy
scikit-learn
matplotlib
seaborn

⚙️ Instalación
   1.Clonar este repositorio:

git clone https://github.com/usuario/nba-analyzer.git
cd nba-analyzer

   2.Instalar las dependencias:
pip install -r requirements.txt

3.(Opcional) Configurar acceso a Kaggle API para obtención de datos adicionales.

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

   playerid: Identificador único del jugador
   player_name: Nombre del jugador
   gameid: Identificador único del partido
   period: Período del juego (1-4, y posibles prórrogas)
   clock: Tiempo en el momento de la jugada
   type: Tipo de jugada ('Made Shot', 'Missed Shot', 'Free Throw')
   subtype: Categoría del tiro ('Jump Shot', 'Layup', 'Dunk', etc.)
   x, y: Coordenadas en la cancha
   dist: Distancia al aro
   result: Indicador de éxito (Made, Missed)
🚀 Uso

## 📊 Uso del Sistema

### Uso Básico
Modo demostración completa:

python demo_nba_analyzer.py

Al ejecutar, selecciona la opción 1 para una demostración completa que guiará a través de todas las funcionalidades.

Modo interactivo

python demo_nba_analyzer.py

Selecciona la opción 2 para acceder al menú interactivo, donde podrás:

   Explorar jugadores disponibles
   Realizar análisis detallados
   Generar predicciones
   Comparar jugadores
   Crear reportes comprensivos
### Ejemplos de Uso
```python
# Reporte comprensivo completo
analyzer.generate_comprehensive_report("K. Bryant")

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

# Obtener jugadores disponibles
players = analyzer.get_available_players(season=2023)
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
- Análisis situacional (clutch vs normal)
- Distribución de distancias de tiro

### 3. Machine Learning
- **Modelo 1**: Predicción de éxito de tiro
- **Modelo 2**: Predicción de tipo de jugada
- **Modelo 3**: Predicción de puntos esperados
- Entrenamiento automático con RandomForest

### 4. Visualizaciones (9 gráficos automáticos)
1. Distribución de tipos de tiro
2. Porcentaje de éxito por tipo
3. Puntos por período
4. Evolución de efectividad por juego
5. Distribución de distancias
6. Rendimiento clutch vs normal
7. Tipos de jugadas más frecuentes
8. Mapa de calor de efectividad
9. Puntos acumulados por temporada

### 5. Comparación de Jugadores
- Métricas comparativas
- Gráficos de barras múltiples
- Gráfico radar normalizado
- Análisis de rendimiento clutch

## 📁 Archivos del Sistema

### `nba_analyzer_final.py`
Archivo principal con la clase `NBAAnalyzerComplete` que contiene toda la funcionalidad.

**Métodos principales:**
- `load_data_with_priority()`: Carga de datos inteligente
- `clean_and_process_data()`: Limpieza y procesamiento
- `get_detailed_player_stats()`: Análisis detallado
- `predict_player_performance()`: Predicciones ML
- `compare_players()`: Comparación entre jugadores
- `generate_comprehensive_report()`: Reporte completo

### `demo_nba_analyzer.py`
Script de demostración con tres modos:
- **Demostración completa**: Para presentaciones TFG
- **Demostración rápida**: Validación rápida
- **Menú interactivo**: Exploración manual

## 🎓 Aplicaciones para TFG

### Análisis Implementados
1. **Análisis descriptivo**: Estadísticas históricas detalladas
2. **Análisis predictivo**: Machine Learning para predecir rendimiento
3. **Análisis comparativo**: Comparación entre múltiples jugadores
4. **Análisis situacional**: Rendimiento en diferentes contextos de juego

### Casos de Uso TFG
- 📚 **Análisis de datos deportivos**
- 🤖 **Machine Learning aplicado al deporte**
- 📊 **Visualización de datos masivos**
- 🏀 **Optimización de estrategias deportivas**
- 🎯 **Predicción de rendimiento atlético**

## 🔍 Ejemplos de Resultados

### Análisis de Jugador
```
📈 ESTADÍSTICAS DETALLADAS: L. James
============================================================
🏀 ESTADÍSTICAS GENERALES:
   • Total de jugadas: 15,432
   • Total de tiros: 8,765
   • Tiros anotados: 4,123
   • Porcentaje de tiro: 47.0%
   • Total de puntos: 12,845
   • Puntos por jugada: 0.83

🎯 ANÁLISIS POR TIPO DE TIRO:
   • Jump Shot: 45.2% (2,345/5,187)
   • Layup: 62.1% (856/1,378)
   • 3-Point: 35.4% (234/661)
   • Dunk: 89.2% (456/511)
```

### Predicciones ML
```
🔮 PREDICCIONES PARA: S. Curry
==================================================
📊 ESTADÍSTICAS HISTÓRICAS:
   • Porcentaje de éxito histórico: 43.2%

🎮 CONTEXTO DEL JUEGO:
   • period: 4
   • time_remaining: 45
   • score_diff: -2
   • dist: 25

🔮 PREDICCIONES:
   • Probabilidad de éxito: 67.3%
   • Tipo de tiro probable: 3-Point
   • Puntos esperados: 2.1

💡 ANÁLISIS:
   • RECOMENDACIÓN: Situación favorable para tomar el tiro
```

## ⚙️ Configuración Avanzada

### Datos Propios
1. Coloca tus archivos CSV en la carpeta `data/`
2. El sistema detectará automáticamente los archivos
3. Formatos soportados: CSV con columnas estándar NBA

### Kaggle API (Opcional)
```bash
# Instalar kaggle
pip install kaggle

# Configurar credenciales
mkdir ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Personalización
- Modificar `prediction_features` para diferentes modelos
- Ajustar parámetros de RandomForest
- Personalizar visualizaciones en `_generate_comprehensive_charts()`

## 🐛 Solución de Problemas

### Error: "No module named 'kaggle'"
```bash
pip install kaggle
# O usar solo datos locales/sintéticos
```

### Error: "No se encontraron datos"
- Verificar archivos en carpeta `data/`
- El sistema generará datos sintéticos automáticamente

### Error: "Datos insuficientes para ML"
- Necesitas al menos 100 registros para entrenar modelos
- Usar datos sintéticos para testing

## 📞 Soporte

### Para TFG
- Sistema completamente funcional y documentado
- Código comentado y estructurado
- Ejemplos de uso incluidos
- Análisis estadísticos robustos

### Extensiones Posibles
- Integración con APIs en tiempo real
- Modelos ML más avanzados (Deep Learning)
- Análisis de video/imágenes
- Interfaces web con Flask/Django

## 📜 Licencia

Proyecto desarrollado para TFG. Uso académico permitido.

---

**🏀 NBA Analyzer** - Sistema completo de análisis NBA con Machine Learning  
*Desarrollado para Trabajo de Fin de Grado*

**Características destacadas para TFG:**
- ✅ Sistema robusto y escalable
- ✅ Machine Learning integrado
- ✅ Visualizaciones automáticas
- ✅ Análisis estadísticos avanzados
- ✅ Documentación completa
- ✅ Casos de uso reales
