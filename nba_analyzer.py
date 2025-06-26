import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings
import re
import os
from collections import defaultdict

warnings.filterwarnings('ignore')

class NBAAnalyzer:
    """    NBA ANALYZER COMPLETO    """

    def __init__(self, data_folder='data'):
        self.data_folder = data_folder
        self.data = None
        self.processed_data = None
        self.models = {}
        self.label_encoders = {}
        self.available_players = {}
        self.available_seasons = []
        self.prediction_features = ['period', 'time_remaining', 'score_diff', 'dist', 'x', 'y']

        # Crear carpeta data si no existe
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
            print(f"📁 Creada carpeta: {self.data_folder}")

    # [Métodos de carga de datos]
    def load_data_with_priority(self, filename=None, selected_seasons=None):
        """
        Sistema de carga con prioridades y filtrado por temporadas:
        1. Datos locales (carpeta data) filtrados por temporadas seleccionadas
        2. Kaggle API (opcional - requiere instalación)
        3. Datos sintéticos para testing
        """
        print("🔄 Iniciando carga de datos con sistema de prioridades...")

        # PRIORIDAD 1: Datos locales con filtrado de temporadas
        if self._try_load_local_data(filename, selected_seasons):
            return True

        # PRIORIDAD 2: Kaggle API (opcional)
        if self._try_load_kaggle_data():
            # Aplicamos el filtro de temporadas después
            if selected_seasons and self.data is not None:
                self._filter_data_by_seasons(selected_seasons)
            return True

        # PRIORIDAD 3: Datos sintéticos
        print("⚠️  No se pudieron cargar datos reales. Generando datos sintéticos para testing...")
        success = self._generate_synthetic_data()
        if success and selected_seasons:
            self._filter_data_by_seasons(selected_seasons)
        return success

    def _try_load_local_data(self, filename=None, selected_seasons=None):
        """Intenta cargar datos desde carpeta local con filtrado por temporadas"""
        try:
            # Verificar si la carpeta de datos existe
            if not os.path.exists(self.data_folder):
                print(f"❌ No se encontró la carpeta de datos: {self.data_folder}")
                return False
            
            # Buscar todos los archivos CSV en la carpeta
            csv_files = [f for f in os.listdir(self.data_folder) if f.endswith('.csv')]
            
            # Si se especifica un archivo específico, priorizarlo
            if filename and filename in csv_files:
                csv_files.insert(0, csv_files.pop(csv_files.index(filename)))
            
            if not csv_files:
                print(f"❌ No se encontraron archivos CSV en {self.data_folder}")
                return False
            
            print(f"✅ Encontrados {len(csv_files)} archivos CSV en {self.data_folder}")
            
            # Análisis preliminar de temporadas disponibles en archivos
            seasons_in_files = self._scan_seasons_in_csv_files(csv_files)
            
            # Filtrar archivos por temporadas seleccionadas
            if selected_seasons:
                filtered_csv_files = []
                for file in csv_files:
                    file_seasons = seasons_in_files.get(file, [])
                    # Incluir el archivo si contiene alguna de las temporadas seleccionadas
                    if any(season in selected_seasons for season in file_seasons):
                        filtered_csv_files.append(file)
                
                # Actualizar la lista de archivos CSV
                csv_files = filtered_csv_files
                
                print(f"🔍 Filtrado: {len(csv_files)} archivos CSV contienen las temporadas seleccionadas")
            
            # Si no hay archivos después del filtrado
            if not csv_files:
                print("❌ No se encontraron archivos para las temporadas seleccionadas")
                return False
            
            # Cargar y combinar archivos CSV filtrados
            all_data = []
            for csv_file in csv_files:
                file_path = os.path.join(self.data_folder, csv_file)
                try:
                    print(f"   📂 Cargando {csv_file}...")
                    df = pd.read_csv(file_path)
                    
                    # Aplicar filtro por temporada al dataframe si es necesario
                    if selected_seasons and 'season' in df.columns:
                        original_len = len(df)
                        df = df[df['season'].isin(selected_seasons)]
                        print(f"      ✅ {len(df)} registros para temporadas seleccionadas (filtrado de {original_len})")
                    else:
                        print(f"      ✅ {len(df)} registros")
                    
                    all_data.append(df)
                except Exception as file_error:
                    print(f"   ❌ Error al cargar {csv_file}: {file_error}")
            
            if not all_data:
                print("❌ No se pudieron cargar datos de ningún archivo CSV")
                return False
            
            # Concatenar todos los DataFrames
            self.data = pd.concat(all_data, ignore_index=True).drop_duplicates()
            print(f"\n📊 TOTAL: {len(self.data)} registros cargados de {len(all_data)} archivos")
            
            # Mostrar información sobre las temporadas cargadas
            if 'season' in self.data.columns:
                seasons = sorted(self.data['season'].unique())
                print(f"📅 Temporadas incluidas: {seasons}")
            
            # Eliminar duplicados (originalmente contaban tiros libres consecutivos como duplicados)            
            # Identificar tiros libres
            self.data['is_ft'] = self.data['type'] == 'Free Throw'

            # Agrupar por clave básica y añadir contador para tiros libres
            self.data['ft_counter'] = self.data.groupby(['gameid', 'period', 'clock', 'player', 'is_ft']).cumcount()

            # Crear clave única que incluya el contador para tiros libres
            self.data['play_key'] = self.data.apply(
                lambda row: f"{row['gameid']}_{row['period']}_{row['clock']}_{row['player']}_{row['ft_counter']}" 
                            if row['is_ft'] 
                            else f"{row['gameid']}_{row['period']}_{row['clock']}_{row['player']}", 
                axis=1
            )
            
            # Eliminar duplicados por esta clave única
            self.data = self.data.drop_duplicates(subset=['play_key'])
            
            print(f"\n📊 TOTAL después de eliminar eventos duplicados: {len(self.data)} registros")
            
            return True
        except Exception as e:
            print(f"❌ Error cargando datos locales: {e}")
            return False

    def _scan_seasons_in_csv_files(self, csv_files):
        """Escanea los archivos CSV para identificar qué temporadas contienen sin cargarlos completos"""
        seasons_in_files = {}
        
        for csv_file in csv_files:
            try:
                file_path = os.path.join(self.data_folder, csv_file)
                # Leer solo las primeras filas para detectar temporadas
                sample_df = pd.read_csv(file_path, nrows=1000)
                
                if 'season' in sample_df.columns:
                    seasons = sorted(sample_df['season'].unique())
                    seasons_in_files[csv_file] = seasons
                    print(f"   📅 {csv_file}: Contiene temporadas {seasons}")
                else:
                    seasons_in_files[csv_file] = []
                    print(f"   ⚠️ {csv_file}: No se encontró columna 'season'")
            except Exception as e:
                print(f"   ❌ Error escaneando {csv_file}: {e}")
                seasons_in_files[csv_file] = []
                
        return seasons_in_files

    def _filter_data_by_seasons(self, selected_seasons):
        """Filtra los datos cargados por temporadas seleccionadas"""
        if self.data is not None and 'season' in self.data.columns:
            original_len = len(self.data)
            self.data = self.data[self.data['season'].isin(selected_seasons)]
            print(f"🔍 Datos filtrados por temporadas {selected_seasons}: {len(self.data)} de {original_len} registros")

    def get_available_seasons_from_files(self):
        """Obtiene las temporadas disponibles en los archivos sin cargarlos completamente"""
        seasons = set()
        
        if not os.path.exists(self.data_folder):
            return list(seasons)
        
        csv_files = [f for f in os.listdir(self.data_folder) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            try:
                file_path = os.path.join(self.data_folder, csv_file)
                # Leer solo las primeras filas para detectar temporadas
                sample_df = pd.read_csv(file_path, nrows=1000)
                
                if 'season' in sample_df.columns:
                    file_seasons = sample_df['season'].unique()
                    for season in file_seasons:
                        seasons.add(season)
            except Exception as e:
                print(f"⚠️ Error analizando {csv_file}: {e}")
        
        return sorted(list(seasons))
            
    def _try_load_kaggle_data(self, dataset_name=None):
        """Intenta cargar datos desde Kaggle API"""
        try:
            print("🌐 Intentando conectar con Kaggle API...")
            
            # Usar dataset por defecto o el especificado
            if dataset_name is None:
                dataset_name = "szymonjwiak/nba-play-by-play-data-1997-2023"
            
            # Verificar si kaggle está instalado
            try:
                import kaggle
                print("✅ Módulo Kaggle encontrado")
            except ImportError:
                print("❌ Kaggle no instalado.")
                
                return False

            # Verificar configuración de Kaggle
            try:
                kaggle.api.authenticate()
                print("✅ Autenticación Kaggle exitosa")
            except Exception as auth_error:
                print(f"❌ Error de autenticación Kaggle: {auth_error}")
                return False
            
            # Descargar dataset
            print(f"📥 Descargando dataset: {dataset_name}...")
            download_path = self.data_folder
            
            try:
                kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)
                print(f"✅ Dataset descargado y descomprimido en: {download_path}")
            except Exception as download_error:
                print(f"❌ Error descargando dataset: {download_error}")
                print(f"⚠️ Verifique que el dataset '{dataset_name}' existe y es público")
                return False
            
            # Buscar archivos CSV descargados
            csv_files = [f for f in os.listdir(download_path) if f.endswith('.csv')]
            if csv_files:
                print(f"📊 Archivos CSV encontrados: {csv_files}")
                
                # Cargar el archivo más grande (suponiendo que tiene más datos)
                csv_paths = [os.path.join(download_path, csv) for csv in csv_files]
                largest_file = max(csv_paths, key=os.path.getsize)
                
                self.data = pd.read_csv(largest_file)
                print(f"✅ Datos cargados desde Kaggle: {len(self.data)} registros del archivo {os.path.basename(largest_file)}")
                return True
            else:
                print("❌ No se encontraron archivos CSV en el dataset descargado")
                return False
            
        except Exception as e:
            print(f"❌ Error conectando con Kaggle: {e}")
            return False

    def _generate_synthetic_data(self):
        """Genera datos sintéticos realistas para testing"""
        try:
            print("🔧 Generando datos sintéticos para testing...")

            # Jugadores sintéticos basados en estrellas reales
            players = [
                'L. James', 'S. Curry', 'K. Durant', 'K. Bryant', 'M. Jordan', 
                'K. Leonard', 'G. Antetokounmpo', 'J. Harden', 'A. Davis', 'D. Lillard',
                'P. George', 'J. Tatum', 'L. Doncic', 'T. Young', 'Z. Williamson',
                'K. Thompson', 'C. Paul', 'R. Westbrook', 'J. Butler', 'B. Adebayo'
            ]

            teams = ['LAL', 'GSW', 'BOS', 'MIA', 'CHI', 'SAS', 'MIL', 'HOU', 'TOR', 'POR',
                    'NYK', 'BRK', 'PHI', 'ATL', 'ORL', 'WAS', 'CLE', 'DET', 'IND', 'CHA']

            # Tipos de jugadas realistas
            shot_types = ['Jump Shot', 'Layup Shot', 'Dunk Shot', '3PT Field Goal', 'Hook Shot']
            play_types = ['Made Shot', 'Missed Shot', 'Free Throw', 'Foul', 'Turnover', 'Rebound']

            # Generar datos sintéticos
            n_records = 75000  # Más datos para mejor entrenamiento
            np.random.seed(42)

            synthetic_data = []
            gameid = 1
            current_h_pts = 0
            current_a_pts = 0

            for i in range(n_records):
                if i % 600 == 0:  # Nuevo juego cada 600 registros
                    gameid += 1
                    current_h_pts = 0
                    current_a_pts = 0

                # Generar jugada
                period = np.random.randint(1, 5)
                time_remaining = np.random.randint(0, 720)  # 12 minutos = 720 segundos
                player = np.random.choice(players)
                team = np.random.choice(teams)
                play_type = np.random.choice(play_types, p=[0.25, 0.25, 0.15, 0.15, 0.1, 0.1])

                # Coordenadas realistas de cancha NBA
                x = np.random.randint(-250, 251)  # Ancho de cancha
                y = np.random.randint(-47, 423)   # Largo de cancha
                dist = np.sqrt(x**2 + y**2) / 10  # Convertir a pies

                # Determinar éxito basado en distancia y tipo
                base_success_rate = 0.45  # Porcentaje base

                # Ajustar por distancia
                if dist < 3:  # Cerca del aro
                    success_rate = 0.65
                elif dist < 15:  # Media distancia
                    success_rate = 0.45
                elif dist > 23:  # Tres puntos
                    success_rate = 0.35
                else:
                    success_rate = 0.42

                # Ajustar por tipo de jugada
                if 'Dunk' in str(np.random.choice(shot_types)):
                    success_rate = 0.95
                elif 'Layup' in str(np.random.choice(shot_types)):
                    success_rate = 0.60

                # Determinar resultado
                if play_type in ['Made Shot']:
                    success = 1 if np.random.random() < success_rate else 0
                    result = 'Made' if success else 'Missed'
                    play_type = 'Made Shot' if success else 'Missed Shot'
                elif play_type == 'Free Throw':
                    success = 1 if np.random.random() < 0.75 else 0  # 75% FT
                    result = 'Made' if success else 'Missed'
                elif play_type == 'Missed Shot':
                    success = 0
                    result = 'Missed'
                else:
                    success = 0
                    result = 'nan'

                # Calcular puntos
                points = 0
                if success == 1:
                    if play_type == 'Free Throw':
                        points = 1
                    elif dist > 23:  # Tres puntos
                        points = 3
                    else:
                        points = 2

                    # Actualizar marcador
                    if np.random.random() > 0.5:  # 50% home team
                        current_h_pts += points
                    else:
                        current_a_pts += points

                # Seleccionar subtipo
                if 'Shot' in play_type:
                    if points == 3:
                        subtype = '3PT Field Goal'
                    elif dist < 3:
                        subtype = np.random.choice(['Dunk Shot', 'Layup Shot'])
                    else:
                        subtype = 'Jump Shot'
                else:
                    subtype = 'nan'

                # Crear descripción 
                if points > 0:
                    if points == 3:
                        desc = f"{player} {dist:.0f}' 3PT Jump Shot ({points} PTS)"
                    else:
                        desc = f"{player} {dist:.0f}' {subtype} ({points} PTS)"
                elif play_type == 'Missed Shot':
                    if dist > 23:
                        desc = f"MISS {player} {dist:.0f}' 3PT Jump Shot"
                    else:
                        desc = f"MISS {player} {dist:.0f}' {subtype}"
                elif play_type == 'Free Throw':
                    desc = f"{player} Free Throw {'Made' if success else 'Missed'}"
                else:
                    desc = f"{player} {play_type}"

                record = {
                    'gameid': gameid,
                    'period': period,
                    'clock': f"PT{time_remaining//60}M{time_remaining%60:02d}.00S",
                    'h_pts': current_h_pts,
                    'a_pts': current_a_pts,
                    'team': team,
                    'playerid': hash(player) % 10000,
                    'player': player,
                    'type': play_type,
                    'subtype': subtype,
                    'result': result,
                    'x': x,
                    'y': y,
                    'dist': dist,
                    'desc': desc,
                    'season': 2023
                }

                synthetic_data.append(record)

            self.data = pd.DataFrame(synthetic_data)
            print(f"✅ Datos sintéticos generados: {len(self.data)} registros")

            # Guardar datos sintéticos
            synthetic_path = os.path.join(self.data_folder, 'synthetic_nba_data.csv')
            self.data.to_csv(synthetic_path, index=False)
            print(f"💾 Datos sintéticos guardados en: {synthetic_path}")

            return True

        except Exception as e:
            print(f"❌ Error generando datos sintéticos: {e}")
            return False

    def clean_and_process_data(self):
        """Limpia y procesa los datos eliminando valores vacíos"""
        if self.data is None:
            print("❌ No hay datos cargados")
            return False

        print("🔄 Procesando y limpiando datos...")
        df = self.data.copy()

        # FILTRAR REGISTROS VÁLIDOS - ELIMINAR TÉRMINOS VACÍOS
        valid_mask = (
            (df['player'].notna()) & 
            (df['player'] != 'nan') & 
            (df['player'].astype(str).str.strip() != '') &
            (df['type'].notna()) & 
            (df['type'] != 'nan') &
            (df['team'].notna()) & 
            (df['team'] != 'nan')
        )

        df = df[valid_mask].copy()
        print(f"✅ Registros válidos después de filtrar: {len(df)}")

        # CREACIÓN DE UN MAPEO DE PLAYERID -> NOMBRE COMPLETO
        # 1. Identificar playerids únicos y sus nombres asociados
        player_id_names = df.groupby('playerid')['player'].unique().reset_index()
        
        # 2. Crear un diccionario de mapeo playerid -> nombre completo
        self.player_id_mapping = {}
        for _, row in player_id_names.iterrows():
            player_id = row['playerid']
            # Usar el nombre más largo disponible como nombre completo
            full_name = max(row['player'], key=len) if isinstance(row['player'], (list, np.ndarray)) else row['player']
            self.player_id_mapping[player_id] = str(full_name)
        
        # 3. Añadir columna con nombre completo basado en el ID
        df['full_player_name'] = df['playerid'].map(self.player_id_mapping)
        
        df['player_name'] = df['player'].astype(str).str.strip()
        df['player_id'] = df['playerid']
        
        print(f"✅ {len(self.player_id_mapping)} jugadores identificados por ID")
        
         # Procesar tiempo
        df['time_remaining'] = df['clock'].apply(self._parse_time)
        
        # Calcular distancia desde el aro para determinar linea de 3
        df['is_behind_3pt_line'] = df.apply(
            lambda row: self._calculate_is_behind_3pt_line(row['x'], row['y']), 
            axis=1
        )
        
        ## Identificar equipos local y visitante para cada partido
        #print("🏠 Identificando equipos local/visitante en cada partido...")
        #
        ## Paso 1: Crear un mapeo de partidos a equipos local/visitante
        #home_away_teams = {}
#
        ## Agrupar por ID de partido
        #for game_id in df['gameid'].unique():
        #    game_data = df[df['gameid'] == game_id].copy()
        #    
        #    # MÉTODO 1: Buscar quién anota en cada canasta (enfoque directo)
        #    teams = game_data['team'].dropna().unique()
        #    if len(teams) < 2:
        #        continue
        #    
        #    # Contar cuántas veces cada equipo aumenta los puntos locales vs visitantes
        #    team_h_points = {}
        #    team_a_points = {}
        #    
        #    for team in teams:
        #        team_data = game_data[game_data['team'] == team].copy()
        #        if len(team_data) < 5:  # Ignorar equipos con pocas jugadas
        #            continue
        #            
        #        team_data = team_data.sort_values(['period', 'time_remaining'], ascending=[True, False])
        #        
        #        # Contar incrementos en h_pts y a_pts
        #        h_increases = 0
        #        a_increases = 0
        #        
        #        for i in range(1, len(team_data)):
        #            if team_data.iloc[i]['h_pts'] > team_data.iloc[i-1]['h_pts']:
        #                h_increases += 1
        #            if team_data.iloc[i]['a_pts'] > team_data.iloc[i-1]['a_pts']:
        #                a_increases += 1
        #                
        #        team_h_points[team] = h_increases
        #        team_a_points[team] = a_increases
        #    
        #    # Determinar equipos locales y visitantes basado en quién anota más en cada canasta
        #    if team_h_points and team_a_points:
        #        # Equipo que aporta más puntos a h_pts es probablemente local
        #        home_team_candidate = max(team_h_points.items(), key=lambda x: x[1])[0]
        #        # Equipo que aporta más puntos a a_pts es probablemente visitante
        #        away_team_candidate = max(team_a_points.items(), key=lambda x: x[1])[0]
        #        
        #        # Solo si son equipos diferentes
        #        if home_team_candidate != away_team_candidate:
        #            home_away_teams[game_id] = {'home': home_team_candidate, 'away': away_team_candidate}
        #            continue
        #    
        #    # MÉTODO 2: Usar análisis estadístico (si el método 1 falló)
        #    team_counts = game_data['team'].value_counts()
        #    if len(team_counts) >= 2:
        #        # El equipo con más jugadas suele ser el local
        #        home_team = team_counts.index[0]
        #        away_team = team_counts.index[1]
        #        
        #        if home_team != away_team:
        #            home_away_teams[game_id] = {'home': home_team, 'away': away_team}
#
        #print(f"✅ Identificados equipos local/visitante para {len(home_away_teams)} partidos")
        #
        ## Paso 2: Calcular diferencia de puntos desde la perspectiva del jugador
        #df['player_team_position'] = df.apply(
        #    lambda row: 'home' if row['team'] == home_away_teams.get(row['gameid'], {}).get('home') else
        #            ('away' if row['team'] == home_away_teams.get(row['gameid'], {}).get('away') else None), 
        #    axis=1
        #)
        #
        ## Calcular diferencia de puntos desde la perspectiva del jugador
        #df['player_score_diff'] = df.apply(
        #    lambda row: row['h_pts'] - row['a_pts'] if row['player_team_position'] == 'home' else
        #            row['a_pts'] - row['h_pts'] if row['player_team_position'] == 'away' else
        #            None, 
        #    axis=1
        #)

       

        # Extraer puntos
        df['points_scored'] = df.apply(self._extract_points_from_desc, axis=1)

        # Calcular diferencia de puntos
        df['score_diff'] = df['h_pts'] - df['a_pts']

        # Determinar éxito
        df['success'] = df['result'].apply(lambda x: 1 if str(x).lower() == 'made' else 0)

        # Categorizar tipos de tiro
        df['shot_category'] = df.apply(self._categorize_shot, axis=1)

        ## Situaciones clutch
        #df['is_clutch'] = ((df['period'] >= 4) & 
        #             (df['time_remaining'] <= 300) & 
        #             (df['player_score_diff'].notna()) &  
        #             (df['player_score_diff'] <= 0) &     
        #             (df['player_score_diff'] >= -5) &    
        #             (df['type'].isin(['Made Shot', 'Missed Shot', 'Free Throw']))).astype(int)
        

        # Obtener jugadores y temporadas usando nombres completos e IDs
        # Esta estructura almacenará: {temporada: [(player_id, full_player_name), ...]}
        players_by_season = {}
        for season, group in df.groupby('season'):
            players_in_season = group[['playerid', 'full_player_name']].drop_duplicates()
            players_by_season[season] = list(zip(players_in_season['playerid'], players_in_season['full_player_name']))
        
        self.available_players = players_by_season
        self.available_seasons = sorted([int(s) for s in df['season'].unique()])

        self.processed_data = df
        print(f"✅ Datos procesados: {len(self.processed_data)} registros válidos")
        print(f"📅 Temporadas disponibles: {self.available_seasons}")

        return True
    
    def _get_player_data_by_identifier(self, player_identifier):
        """
        Método unificado para obtener datos de un jugador por ID o nombre.
        
        Args:
            player_identifier: ID (int/str), nombre (str) o tupla (id, nombre)
        
        Returns:
            tuple: (player_data, player_id, player_name) o None si no se encuentra
        """
        player_data = None
        player_id = None
        player_name = None
        
        # Caso 1: Es una tupla (id, nombre)
        if isinstance(player_identifier, tuple):
            player_id, player_name = player_identifier
            player_data = self.processed_data[self.processed_data['playerid'] == player_id]
            if player_data.empty:
                print(f"⚠️ No se encontraron datos para ID: {player_id} (Nombre: {player_name})")
                return None
        
        # Caso 2: Es un ID numérico
        elif isinstance(player_identifier, (int, float)) or (isinstance(player_identifier, str) and player_identifier.isdigit()):
            player_id = int(player_identifier)
            player_data = self.processed_data[self.processed_data['playerid'] == player_id]
            if not player_data.empty:
                player_name = player_data['full_player_name'].iloc[0]
            else:
                print(f"⚠️ No se encontró jugador con ID {player_id}")
                return None
        
        # Caso 3: Es un nombre de jugador
        else:
            player_name = player_identifier
            # Buscar por nombre completo
            player_data = self.processed_data[self.processed_data['full_player_name'] == player_name]
            
            # Si no se encuentra, probar con nombre original
            if player_data.empty:
                player_data = self.processed_data[self.processed_data['player_name'] == player_name]
            
            # Si sigue sin encontrarse, buscar coincidencias parciales
            if player_data.empty:
                matches = self.processed_data[
                    self.processed_data['full_player_name'].str.contains(player_name, case=False) |
                    self.processed_data['player_name'].str.contains(player_name, case=False)
                ][['playerid', 'full_player_name']].drop_duplicates().values.tolist()
                
                if matches:
                    print(f"⚠️ No se encontró coincidencia exacta para '{player_name}'. Opciones disponibles:")
                    for i, (pid, name) in enumerate(matches[:10], 1):
                        print(f"   {i}. ID: {pid} - {name}")
                    return None
                else:
                    print(f"⚠️ No se encontró ningún jugador que coincida con '{player_name}'")
                    return None
            else:
                # Obtener playerid para futuras referencias
                player_id = player_data['playerid'].iloc[0]
        
        return player_data, player_id, player_name
    
    
    def _calculate_is_behind_3pt_line(self, x, y):
        """Determina si un punto (x,y) está detrás de la línea de 3 puntos"""
        # Simplificación: distancia al aro > 23.75 pies (línea de 3pt NBA)
        # Coordenada 0,0 representa el aro
        distance = np.sqrt(x**2 + y**2)
        return 1 if distance > 23.75 else 0
    
    def _determine_court_zone(self, x, y):
        """Clasifica un tiro en zonas específicas de la cancha"""
        distance = np.sqrt(x**2 + y**2)
        angle = np.arctan2(y, x) * 180 / np.pi
        
        if distance < 4:
            return "Restricted Area"
        elif distance < 8:
            return "Paint"
        elif distance < 16:
            return "Mid-Range"
        elif distance < 23.75:
            return "Long 2"
        else:  # 3pt territory
            if abs(angle) < 45:
                return "Corner 3"
            else:
                return "Arc 3"

    def _parse_time(self, time_str):
        """Convierte tiempo a segundos restantes"""
        try:
            if pd.isna(time_str) or str(time_str) == 'nan':
                return 0
            time_str = str(time_str)
            if 'PT' in time_str and 'M' in time_str and 'S' in time_str:
                time_part = time_str.replace('PT', '').replace('S', '')
                if 'M' in time_part:
                    minutes, seconds = time_part.split('M')
                    return int(minutes) * 60 + float(seconds)
            return 0
        except:
            return 0
          
    def _categorize_shot(self, row):
        """Categoriza el tipo de tiro"""
        try:
            subtype = str(row['subtype']).lower()
            desc = str(row['desc']).lower()

            if 'free throw' in subtype or 'free throw' in desc:
                return 'Free Throw'
            elif '3pt' in desc or 'three' in desc:
                return '3-Point'
            elif 'dunk' in subtype or 'dunk' in desc:
                return 'Dunk'
            elif 'layup' in subtype or 'layup' in desc:
                return 'Layup'
            elif 'jump shot' in subtype or 'jump shot' in desc:
                return 'Jump Shot'
            elif 'hook' in subtype or 'hook' in desc:
                return 'Hook Shot'
            else:
                return 'Other'
        except:
            return 'Other'
          
    def _extract_points_from_desc(self, row):
        """Extrae puntos de la descripción con validación"""
        try:
            desc = str(row['desc']).lower()
            shot_type = str(row.get('subtype', '')).lower()
            result = str(row.get('result', '')).lower()
            play_type = str(row.get('type', '')).lower()
            
            # Primera prioridad: patrón explícito de puntos
            point_pattern = r'\((\d+)\s+PTS\)'
            match = re.search(point_pattern, desc)
            if match:
                return int(match.group(1))
            
            # Segunda prioridad: para tiros libres específicamente
            if 'free throw' in play_type or 'free throw' in desc:
                if result == 'made' or 'made free throw' in desc:
                    return 1
                return 0    
            
            # Tercera prioridad: combinación de tipo y éxito
            if result == 'made' or row.get('success', 0) == 1:
                if '3pt' in desc or '3-point' in shot_type or 'three point' in desc:
                    return 3
                else:
                    return 2
                    
            return 0
        except Exception as e:
            print(f"⚠️ Error calculando puntos: {e}")
            return 0
        
    # Preparacion de los modelos de predicción
    def _prepare_prediction_models(self):
        """Prepara modelos de Machine Learning para predicciones"""
        try:
            print("🤖 Preparando modelos de predicción...")

            # Filtrar datos para entrenamiento
            shooting_data = self.processed_data[
                self.processed_data['type'].isin(['Made Shot', 'Missed Shot']) &
                (self.processed_data['type'] != 'Free Throw')  # Excluir explícitamente
            ].copy()
            
            # Crear un modelo separado solo para tiros libres
            ft_data = self.processed_data[
                self.processed_data['type'] == 'Free Throw'
            ].copy()
            
            if len(ft_data) >= 100:
                # Features específicos para tiros libres (más simples)
                ft_features = ['period', 'time_remaining', 'score_diff']
                X_ft = ft_data[ft_features].fillna(0)
                y_ft = ft_data['success']
                
                self.models['ft_predictor'] = RandomForestClassifier(n_estimators=100, random_state=42)
                self.models['ft_predictor'].fit(X_ft, y_ft)

            if len(shooting_data) < 100:
                print("⚠️  Datos insuficientes para entrenar modelos")
                return

            # Preparar features
            features = []
            for feature in self.prediction_features:
                if feature in shooting_data.columns:
                    features.append(feature)

            X = shooting_data[features].fillna(0)

            # Modelo 1: Predicción de éxito de tiro
            y_success = shooting_data['success']
            if len(y_success.unique()) > 1:
                X_train, X_test, y_train, y_test = train_test_split(X, y_success, test_size=0.2, random_state=42)

                self.models['success_predictor'] = RandomForestClassifier(n_estimators=100, random_state=42)
                self.models['success_predictor'].fit(X_train, y_train)

                y_pred = self.models['success_predictor'].predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"✅ Modelo de éxito entrenado - Precisión: {accuracy:.3f}")

            # Modelo 2: Predicción de tipo de tiro
            y_shot_type = shooting_data['shot_category']
            if len(y_shot_type.unique()) > 1:
                self.label_encoders['shot_type'] = LabelEncoder()
                y_shot_encoded = self.label_encoders['shot_type'].fit_transform(y_shot_type)

                X_train, X_test, y_train, y_test = train_test_split(X, y_shot_encoded, test_size=0.2, random_state=42)

                self.models['shot_type_predictor'] = RandomForestClassifier(n_estimators=100, random_state=42)
                self.models['shot_type_predictor'].fit(X_train, y_train)

                y_pred = self.models['shot_type_predictor'].predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"✅ Modelo de tipo de tiro entrenado - Precisión: {accuracy:.3f}")

            # Modelo 3: Predicción de puntos
            y_points = shooting_data['points_scored']
            if len(y_points.unique()) > 1:
                X_train, X_test, y_train, y_test = train_test_split(X, y_points, test_size=0.2, random_state=42)

                self.models['points_predictor'] = RandomForestRegressor(n_estimators=100, random_state=42)
                self.models['points_predictor'].fit(X_train, y_train)

                print(f"✅ Modelo de puntos entrenado")

        except Exception as e:
            print(f"❌ Error preparando modelos: {e}")
            
    def show_available_players(analyzer, page_size=20):
        """Muestra jugadores disponibles con sus IDs"""
        all_players = []
        # Reunir jugadores de todas las temporadas disponibles con sus IDs
        for season, players_info in analyzer.available_players.items():
            for player_id, player_name in players_info:
                if (player_id, player_name) not in all_players:
                    all_players.append((player_id, player_name))
                    
        # Ordenar por nombre
        players_sorted = sorted(all_players, key=lambda x: x[1])
        
        print(f"📊 JUGADORES DISPONIBLES: {len(players_sorted)}")
        
        # Mostrar jugadores en páginas
        current_page = 0
        total_pages = (len(players_sorted) + page_size - 1) // page_size
        
        while True:
            # Mostrar la página actual
            start_idx = current_page * page_size
            end_idx = min(start_idx + page_size, len(players_sorted))
            
            print(f"\nPágina {current_page + 1}/{total_pages} - Jugadores {start_idx + 1}-{end_idx} de {len(players_sorted)}")
            
            for i, (player_id, player_name) in enumerate(players_sorted[start_idx:end_idx], start_idx + 1):
                print(f"   {i:3d}. ID: {player_id:<7} - {player_name}")
            
            print("\n🔍 OPCIONES:")
            print("   • S: Siguiente página")
            print("   • A: Anterior página") 
            print("   • B: Buscar jugador")
            print("   • C: Continuar")
            
            choice = input("\n🎯 Tu elección: ").strip().upper()
            
            if choice == 'S':
                if current_page < total_pages - 1:
                    current_page += 1
                else:
                    print("Ya estás en la última página")
            elif choice == 'A':
                if current_page > 0:
                    current_page -= 1
                else:
                    print("Ya estás en la primera página")
            elif choice == 'B':
                search_term = input("🔍 Buscar jugador (nombre parcial): ").strip().lower()
                if search_term:
                    matches = [(pid, name) for pid, name in players_sorted if search_term in name.lower()]
                    if matches:
                        print(f"\nJugadores que coinciden con '{search_term}':")
                        for i, (pid, name) in enumerate(matches[:20], 1):
                            print(f"   {i:3d}. ID: {pid:<7} - {name}")
                    else:
                        print(f"❌ No se encontraron jugadores que coincidan con '{search_term}'")
            elif choice == 'C':
                break
    
    # MÉTODO OPTIMIZADO PARA CÁLCULO CORRECTO DE ESTADÍSTICAS
    def get_detailed_player_stats(self, player_identifier, season=None):
        """Obtiene estadísticas detalladas de un jugador"""
        if self.processed_data is None:
            print("❌ No hay datos procesados disponibles")
            return None

        result = self._get_player_data_by_identifier(player_identifier)
        if result is None:
            return None
        
        player_data, player_id, player_name = result

        # Filtrar por temporada si se especifica
        if season:
            player_data = player_data[player_data['season'] == season]
            if player_data.empty:
                print(f"❌ No hay datos para {player_name} (ID: {player_id}) en la temporada {season}")
                return None

        # Obtener partidos únicos para el jugador
        games_played = player_data['gameid'].unique()
        
        print(f"\n🏀 ESTADÍSTICAS DETALLADAS: {player_name}")
        print(f"📊 Partidos analizados: {len(games_played)}")
        
        # Calcular estadísticas partido por partido para evitar duplicación
        game_stats = []
        total_points = 0
        total_made_2pt = 0
        total_made_3pt = 0
        total_made_ft = 0
        total_attempts_2pt = 0
        total_attempts_3pt = 0
        total_attempts_ft = 0
        
                
        
        for game_id in games_played:
            game_data = player_data[player_data['gameid'] == game_id]
            
            # Contar tiros por tipo
            shots_2pt = game_data[(game_data['type'].isin(['Made Shot', 'Missed Shot'])) & 
                                 (~game_data['desc'].str.contains('3PT', na=False))]
            shots_3pt = game_data[(game_data['type'].isin(['Made Shot', 'Missed Shot'])) & 
                                 (game_data['desc'].str.contains('3PT', na=False))]
            free_throws = game_data[game_data['type'] == 'Free Throw']
            
            # Contar aciertos
            made_2pt = shots_2pt[shots_2pt['success'] == 1]
            made_3pt = shots_3pt[shots_3pt['success'] == 1]
            made_ft = free_throws[~free_throws['desc'].str.contains('MISS', na=False, case=True)]
            
            # Calcular puntos del partido
            points_2pt = len(made_2pt) * 2
            points_3pt = len(made_3pt) * 3
            points_ft = len(made_ft)
            game_points = points_2pt + points_3pt + points_ft
            
            # Acumular estadísticas
            total_points += game_points
            total_made_2pt += len(made_2pt)
            total_made_3pt += len(made_3pt)
            total_made_ft += len(made_ft)
            total_attempts_2pt += len(shots_2pt)
            total_attempts_3pt += len(shots_3pt)
            total_attempts_ft += len(free_throws)
            
            # Guardar estadísticas del partido
            game_stats.append({
                'game_id': game_id,
                'points': game_points,
                'fg_made': len(made_2pt) + len(made_3pt),
                'fg_attempts': len(shots_2pt) + len(shots_3pt),
                'ft_made': len(made_ft),
                'ft_attempts': len(free_throws),
                'fg_pct': (len(made_2pt) + len(made_3pt)) / (len(shots_2pt) + len(shots_3pt)) if (len(shots_2pt) + len(shots_3pt)) > 0 else 0
            })
        
        # Calcular porcentajes globales
        fg_pct_2pt = total_made_2pt / total_attempts_2pt if total_attempts_2pt > 0 else 0
        fg_pct_3pt = total_made_3pt / total_attempts_3pt if total_attempts_3pt > 0 else 0
        ft_pct = total_made_ft / total_attempts_ft if total_attempts_ft > 0 else 0
        
        # Mostrar estadísticas detalladas
        print("\n📊 PUNTOS TOTALES:")
        print(f"   • Tiros de 2PT: {total_made_2pt}/{total_attempts_2pt} ({fg_pct_2pt:.1%}) = {total_made_2pt * 2} pts")
        print(f"   • Tiros de 3PT: {total_made_3pt}/{total_attempts_3pt} ({fg_pct_3pt:.1%}) = {total_made_3pt * 3} pts")
        print(f"   • Tiros libres: {total_made_ft}/{total_attempts_ft} ({ft_pct:.1%}) = {total_made_ft} pts")
        print(f"   • TOTAL PUNTOS: {total_points}")
        
        # Calcular estadísticas generales
        total_plays = len(player_data)
        shooting_plays = player_data[player_data['type'].isin(['Made Shot', 'Missed Shot'])]
        total_shots = len(shooting_plays)
        made_shots = len(shooting_plays[shooting_plays['success'] == 1])
        
        print(f"\n🏀 ESTADÍSTICAS GENERALES:")
        print(f"   • Total de jugadas: {total_plays:,}")
        print(f"   • Total de tiros: {total_shots:,}")
        print(f"   • Tiros anotados: {made_shots:,}")
        print(f"   • Porcentaje de tiro: {made_shots/total_shots*100:.1f}%" if total_shots > 0 else "   • Porcentaje de tiro: N/A")
        print(f"   • Puntos por partido: {total_points/len(games_played):.1f}")
        
        # Análisis por tipo de tiro - Rehecho correctamente
        print(f"\n🎯 ANÁLISIS POR TIPO DE TIRO:")
        shot_types = player_data[player_data['type'].isin(['Made Shot', 'Missed Shot'])].groupby('shot_category').agg({
            'success': ['count', 'sum']
        })
        
        shot_types.columns = ['Intentos', 'Anotados']
        shot_types['Porcentaje'] = shot_types['Anotados'] / shot_types['Intentos']
        
        # Calcular puntos por tipo de tiro correctamente
        shot_types['Puntos'] = 0
        
        for shot_type in shot_types.index:
            points_value = 3 if '3-Point' in shot_type else 2
            shot_types.loc[shot_type, 'Puntos'] = shot_types.loc[shot_type, 'Anotados'] * points_value
        
        # Añadir tiros libres
        ft_data = player_data[player_data['type'] == 'Free Throw']
        ft_attempts = len(ft_data)
        ft_made = len(ft_data[~ft_data['desc'].str.contains('MISS', na=False, case=True)])
        
        if ft_attempts > 0:
            new_row = pd.DataFrame({
                'Intentos': [ft_attempts],
                'Anotados': [ft_made],
                'Porcentaje': [ft_made/ft_attempts],
                'Puntos': [ft_made]
            }, index=['Free Throw'])
            shot_types = pd.concat([shot_types, new_row])
        
        for shot_type in shot_types.index:
            row = shot_types.loc[shot_type]
            print(f"   • {shot_type}:")
            print(f"     - Intentos: {int(row['Intentos'])}")
            print(f"     - Anotados: {int(row['Anotados'])}")
            print(f"     - Porcentaje: {row['Porcentaje']*100:.1f}%")
            print(f"     - Puntos totales: {int(row['Puntos'])}")
        
        # Análisis situacional
        #print(f"\n⚡ ANÁLISIS SITUACIONAL:")
        #
        #clutch_data = shooting_plays[shooting_plays['is_clutch'] == 1]
        #normal_data = shooting_plays[shooting_plays['is_clutch'] == 0]
        #
        #clutch_attempts = len(clutch_data)
        #normal_attempts = len(normal_data)
        #
        ## Contar explícitamente éxitos y calcular porcentaje
        #clutch_success_count = sum(1 for _, row in clutch_data.iterrows() 
        #                    if (~pd.isna(row['desc']) and 
        #                    'MISS' not in str(row['desc']).upper()))
        #normal_success_count = sum(1 for _, row in normal_data.iterrows() 
        #                    if (~pd.isna(row['desc']) and 
        #                    'MISS' not in str(row['desc']).upper()))
#
        #clutch_success_rate = clutch_success_count / clutch_attempts if clutch_attempts > 0 else 0
        #normal_success_rate = normal_success_count / normal_attempts if normal_attempts > 0 else 0
#
        #print(f"   • Situaciones normales: {normal_success_rate*100:.1f}% ({normal_attempts} tiros, {normal_success_count} aciertos)")
        #print(f"   • Situaciones clutch: {clutch_success_rate*100:.1f}% ({clutch_attempts} tiros, {clutch_success_count} aciertos)")
        
        save_path = self._get_save_path(f"{player_name}_stats_{season if season else 'all'}")
        
        # Generar visualizaciones
        self._generate_corrected_charts(player_data, game_stats, player_name, season, save_path)
        
        return {
            'player_name': player_name,
            'games_played': len(games_played),
            'total_points': total_points,
            'shot_types': shot_types,
            'game_stats': game_stats
        }
    
    def _generate_corrected_charts(self, player_data, game_stats, player_name, season, save_path=None):
        """Genera gráficos para análisis del jugador"""
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Distribución de tipos de tiro
        ax1 = plt.subplot(3, 3, 1)
        shooting_data = player_data[player_data['type'].isin(['Made Shot', 'Missed Shot'])]
        shot_counts = shooting_data['shot_category'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(shot_counts)))
        wedges, texts, autotexts = ax1.pie(shot_counts.values, labels=shot_counts.index, 
                                          autopct='%1.1f%%', colors=colors)
        ax1.set_title('Distribución de Tipos de Tiro')
        
        # 2. Porcentaje de éxito por tipo
        ax2 = plt.subplot(3, 3, 2)
        success_by_type = shooting_data.groupby('shot_category')['success'].mean()
        bars = ax2.bar(success_by_type.index, success_by_type.values, 
                      color='skyblue', edgecolor='navy')
        ax2.set_title('Porcentaje de Éxito por Tipo')
        ax2.set_ylabel('Porcentaje')
        ax2.tick_params(axis='x', rotation=45)
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}', ha='center', va='bottom')
        
        # 3. Puntos por partido - CORREGIDO
        ax3 = plt.subplot(3, 3, 3)
        game_df = pd.DataFrame(game_stats)
        if not game_df.empty:
            ax3.bar(range(len(game_df)), game_df['points'], color='lightgreen', edgecolor='darkgreen')
            ax3.set_title('Puntos por Partido')
            ax3.set_xlabel('Número de Partido')
            ax3.set_ylabel('Puntos')
            # Línea de promedio
            avg_pts = game_df['points'].mean()
            ax3.axhline(avg_pts, color='red', linestyle='--', 
                       label=f'Promedio: {avg_pts:.1f}')
            ax3.legend()
        
        # 4. Evolución de efectividad por partido - CORREGIDO
        ax4 = plt.subplot(3, 3, 4)
        if not game_df.empty and 'fg_pct' in game_df.columns:
            ax4.plot(range(len(game_df)), game_df['fg_pct'], 
                    marker='o', color='red', alpha=0.7)
            ax4.set_title('Efectividad por Partido')
            ax4.set_xlabel('Número de Partido')
            ax4.set_ylabel('Porcentaje de Éxito')
            ax4.grid(True, alpha=0.3)
        
        # 5. Distribución de distancias
        ax5 = plt.subplot(3, 3, 5)
        distances = shooting_data['dist'][shooting_data['dist'] > 0]
        if not distances.empty:
            ax5.hist(distances, bins=15, color='orange', alpha=0.7, edgecolor='black')
            ax5.set_title('Distribución de Distancias')
            ax5.set_xlabel('Distancia (pies)')
            ax5.set_ylabel('Frecuencia')
        
        ## 6. Clutch vs Normal
        #ax6 = plt.subplot(3, 3, 6)
        #clutch_perf = shooting_data.groupby('is_clutch')['success'].mean()
        #labels = ['Normal', 'Clutch']
        #bars = ax6.bar(labels, [clutch_perf.get(0, 0), clutch_perf.get(1, 0)], 
        #              color=['lightblue', 'red'], alpha=0.8)
        #ax6.set_title('Rendimiento: Normal vs Clutch')
        #ax6.set_ylabel('Porcentaje de Éxito')
        #for bar in bars:
        #    height = bar.get_height()
        #    if height > 0:
        #        ax6.text(bar.get_x() + bar.get_width()/2., height,
        #                f'{height:.1%}', ha='center', va='bottom')
        
        # 6. Mapa de calor de efectividad por período y tiempo (reemplazando la gráfica de clutch)
        ax6 = plt.subplot(3, 3, 6)
        if len(shooting_data) > 0:  # Incluso con pocos datos
            # Crear bins de tiempo
            shooting_data_copy = shooting_data.copy()
            shooting_data_copy['time_bin'] = pd.cut(shooting_data_copy['time_remaining'], 
                                                bins=4, labels=['0-3min', '3-6min', '6-9min', '9-12min'])
            
            # Crear un DataFrame con todas las combinaciones posibles
            all_periods = list(range(1, 7))  # Del 1 al 6 (incluyendo prórrogas)
            time_bins = ['0-3min', '3-6min', '6-9min', '9-12min']
            
            # Crear índice con todas las combinaciones
            full_index = pd.MultiIndex.from_product([all_periods, time_bins], 
                                                names=['period', 'time_bin'])
            
            # Calcular la media de éxito por período y tiempo
            success_means = shooting_data_copy.groupby(['period', 'time_bin'])['success'].mean()
            
            # Crear un DataFrame completo con todas las combinaciones
            heatmap_data = success_means.reindex(full_index).unstack()
            
            # Crear máscara para valores NaN (sin datos)
            mask = pd.isna(heatmap_data)
            
            # Crear heatmap con máscara para valores sin datos
            sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                    vmin=0, vmax=1, ax=ax6, cbar_kws={'label': 'Efectividad'},
                    mask=mask)
            
            ax6.set_title('Efectividad por Período y Tiempo (incluye prórrogas)')
            ax6.set_ylabel('Período (1-6)')
        
        # 7. Tipos de jugadas
        ax7 = plt.subplot(3, 3, 7)
        play_types = player_data['type'].value_counts().head(6)
        ax7.barh(play_types.index, play_types.values, color='lightcoral')
        ax7.set_title('Tipos de Jugadas Más Frecuentes')
        ax7.set_xlabel('Frecuencia')
        
        # 8. Mapa de tiros por zonas de la cancha 
        ax8 = plt.subplot(3, 3, 8)
        # Filtrar tiros con coordenadas válidas
        shots_with_coords = shooting_data[(shooting_data['x'].notna()) & (shooting_data['y'].notna())]

        if len(shots_with_coords) > 10:
            # Crear un scatter plot coloreado por éxito
            made_shots = shots_with_coords[shots_with_coords['success'] == 1]
            missed_shots = shots_with_coords[shots_with_coords['success'] == 0]
            
            # Dibujar tiros fallados en rojo y acertados en verde
            ax8.scatter(missed_shots['x'], missed_shots['y'], color='red', alpha=0.5, 
                    marker='x', s=30, label='Fallados')
            ax8.scatter(made_shots['x'], made_shots['y'], color='green', alpha=0.7, 
                    marker='o', s=30, label='Anotados')
            
            # Configurar el gráfico
            ax8.set_title('Mapa de Tiros')
            ax8.set_xlabel('Coordenada X')
            ax8.set_ylabel('Coordenada Y')
            ax8.legend(loc='upper right')
            ax8.grid(True, alpha=0.3)
            
            # Mantener relación de aspecto
            ax8.set_aspect('equal')
        else:
            ax8.text(0.5, 0.5, 'Datos insuficientes para\nmapa de tiros', 
                    ha='center', va='center', fontsize=12)
            ax8.axis('off')
        
        # 9. Puntos acumulados por partido
        ax9 = plt.subplot(3, 3, 9)
        if not game_df.empty:
            cumulative_points = game_df['points'].cumsum()
            ax9.plot(range(len(cumulative_points)), cumulative_points, 
                    marker='o', linewidth=2, markersize=4, color='purple')
            ax9.fill_between(range(len(cumulative_points)), cumulative_points, 
                           alpha=0.3, color='purple')
            ax9.set_title('Puntos Acumulados')
            ax9.set_xlabel('Número de Partido')
            ax9.set_ylabel('Puntos Acumulados')
            ax9.grid(True, alpha=0.3)
        
        plt.suptitle(f'Análisis Comprensivo: {player_name}' + 
                    (f' - Temporada {season}' if season else ''), 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Guardar el gráfico si se proporciona una ruta
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ Gráfico guardado en: {save_path}")
            except Exception as e:
                print(f"❌ Error al guardar el gráfico: {e}")
        
        plt.show()
        
        
    def _generate_comparison_charts(self, comparison_df, save_path=None):
        """Genera visualizaciones comparativas entre jugadores"""
        if comparison_df.empty:
            print("❌ No hay datos para generar comparaciones")
            return

        # Asegurar que los nombres de jugadores sean strings para visualización
        comparison_df['Jugador'] = comparison_df['Jugador'].astype(str)
        
        plt.figure(figsize=(16, 12))  
        bar_width = 0.6  
        
        # 1. Gráfico de barras para Puntos Totales
        plt.subplot(2, 2, 1)
        x = np.arange(len(comparison_df))  
        barras = plt.bar(x, comparison_df['Total_Puntos'], width=bar_width, color='royalblue')
        plt.title('Total de Puntos', fontsize=14)
        plt.ylabel('Puntos', fontsize=12)
        
        # Configuración correcta para los nombres del eje X
        plt.xticks(x, comparison_df['Jugador'], rotation=45, ha='right', fontsize=11)
        plt.tight_layout()
        
        # Añadir valores encima de las barras
        for barra in barras:
            altura = barra.get_height()
            plt.text(barra.get_x() + barra.get_width()/2., altura + 0.1,
                    f'{int(altura)}', ha='center', va='bottom', fontsize=11)

        # 2. Gráfico de barras para Puntos por Partido
        plt.subplot(2, 2, 2)
        barras = plt.bar(x, comparison_df['Puntos_Por_Partido'], width=bar_width, color='seagreen')
        plt.title('Puntos por Partido', fontsize=14)
        plt.ylabel('PPP', fontsize=12)
        plt.xticks(x, comparison_df['Jugador'], rotation=45, ha='right', fontsize=11)
        
        # Añadir valores encima de las barras
        for barra in barras:
            altura = barra.get_height()
            plt.text(barra.get_x() + barra.get_width()/2., altura + 0.1,
                    f'{altura:.1f}', ha='center', va='bottom', fontsize=11)

        # 3. Gráfico de barras para Porcentaje de Tiro
        plt.subplot(2, 2, 3)
        barras = plt.bar(x, comparison_df['Porcentaje_Tiro'], width=bar_width, color='coral')
        plt.title('Porcentaje de Tiro', fontsize=14)
        plt.ylabel('Porcentaje (%)', fontsize=12)
        plt.xticks(x, comparison_df['Jugador'], rotation=45, ha='right', fontsize=11)
        
        # Añadir valores encima de las barras
        for barra in barras:
            altura = barra.get_height()
            plt.text(barra.get_x() + barra.get_width()/2., altura + 0.1,
                    f'{altura:.1f}%', ha='center', va='bottom', fontsize=11)

        # 4. NUEVA GRÁFICA: Eficiencia de Tiro (Puntos por Intento)
        plt.subplot(2, 2, 4)
        # Calcular puntos por tiro intentado
        comparison_df['Puntos_Por_Tiro'] = comparison_df['Total_Puntos'] / comparison_df['Total_Tiros'].where(comparison_df['Total_Tiros'] > 0, 1)
        barras = plt.bar(x, comparison_df['Puntos_Por_Tiro'], width=bar_width, color='purple')
        plt.title('Eficiencia de Tiro', fontsize=14)
        plt.ylabel('Puntos por Tiro', fontsize=12)
        plt.xticks(x, comparison_df['Jugador'], rotation=45, ha='right', fontsize=11)
        
        # Añadir valores encima de las barras
        for barra in barras:
            altura = barra.get_height()
            plt.text(barra.get_x() + barra.get_width()/2., altura + 0.1,
                    f'{altura:.2f}', ha='center', va='bottom', fontsize=11)

        plt.subplots_adjust(wspace=0.3, hspace=0.3, top=0.9)  # Más espacio entre gráficos
        plt.suptitle('Comparativa de Rendimiento entre Jugadores', fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Guardar el gráfico si se proporciona una ruta
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ Gráfico guardado en: {save_path}")
            except Exception as e:
                print(f"❌ Error al guardar el gráfico: {e}")
        
        plt.show()    

      ## 4. Gráfico de barras para Porcentaje en Clutch
      #plt.subplot(2, 2, 4)
      #barras = plt.bar(comparison_df['Jugador'], comparison_df['Clutch_Porcentaje'], color='purple')
      #plt.title('Rendimiento en Situaciones Clutch')
      #plt.ylabel('Porcentaje (%)')
      #plt.xticks(rotation=45, ha='right')
      
      # Añadir valores encima de las barras
      #for barra in barras:
      #    altura = barra.get_height()
      #    plt.text(barra.get_x() + barra.get_width()/2., altura + 0.1,
      #            f'{altura:.1f}%', ha='center', va='bottom')
    
    def _get_save_path(self, default_filename):
        """Obtiene la ruta de guardado para un gráfico"""
        while True:
            try:
                save_option = input("¿Desea guardar el gráfico? (s/n): ").strip().lower()
                if save_option in ['n', 'no']:
                    return None
                
                if save_option in ['s', 'si', 'sí', 'y', 'yes']:
                    # Solicitar nombre de archivo o usar el predeterminado
                    filename = input(f"Nombre de archivo ({default_filename}): ").strip()
                    if not filename:
                        filename = default_filename
                    
                    # Asegurar que tiene extensión
                    if not any(filename.endswith(ext) for ext in ['.png', '.jpg', '.pdf', '.svg']):
                        filename += '.png'
                    
                    # Crear carpeta 'plots' si no existe
                    save_dir = 'plots'
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    
                    # Construir ruta completa
                    save_path = os.path.join(save_dir, filename)
                    
                    print(f"✅ El gráfico se guardará en: {save_path}")
                    return save_path
                    
                print("Opción inválida. Por favor responda 's' o 'n'.")
            
            except Exception as e:
                print(f"❌ Error: {e}")
                return None
    
    def predict_interactive(self, player_identifier, user_context=None):
        """Genera predicciones interactivas basadas en parámetros definidos por el usuario"""
        print(f"\n🔍 PREDICCIÓN INTERACTIVA DE JUGADAS")
        
        if self.processed_data is None:
            print("❌ No hay datos procesados disponibles")
            return None
        
        result = self._get_player_data_by_identifier(player_identifier)
        if result is None:
            return None
        
        player_data, player_id, player_name = result
    
        print(f"🏀 Jugador: {player_name}")
        
        # Si no hay contexto definido por el usuario, solicitarlo interactivamente
        if user_context is None:
            user_context = self._get_interactive_context()
        
        # Mostrar el contexto definido
        print(f"\n📊 CONTEXTO DE PREDICCIÓN:")
        print(f"   • Periodo: {user_context['period']}")
        print(f"   • Tiempo restante: {user_context['time_remaining']} segundos")
        print(f"   • Diferencia de puntos: {user_context['score_diff']}")
        print(f"   • Distancia a canasta: {user_context['dist']} pies")
        
        # Generar predicciones para diferentes tipos de jugadas
        shot_types = ['3-Point', 'Jump Shot', 'Layup', 'Hook Shot', 'Dunk', 'Free Throw']
        
        print(f"\n🎮 PREDICCIONES DE JUGADAS Y PROBABILIDADES:")
        
        play_predictions = []
        
        # Entrenar modelos si es necesario
        if not self.models.get('success_predictor') or not self.models.get('shot_type_predictor'):
            print("🔄 Entrenando modelos predictivos... (esto puede tomar un momento)")
            self._prepare_prediction_models()
        else:
            print("✅ Usando modelos ya entrenados")
        
        # 1. Predicción de tipo de jugada
        play_probs = self._predict_play_probabilities(player_data, user_context, shot_types)
        
        # 2. Para cada tipo de jugada, predecir probabilidad de éxito
        for shot_type in shot_types:
            # Copiar el contexto y agregar tipo de tiro
            shot_context = user_context.copy()
            shot_context['shot_type'] = shot_type
            
            # Probabilidad de que se realice este tipo de jugada
            play_probability = play_probs.get(shot_type, 0.05)
            
            # Predecir probabilidad de éxito para este tipo de jugada
            success_probability = self._predict_shot_success(player_data, shot_context)
            
            # Calcular puntos esperados
            if shot_type == '3-Point':
                points = 3
            elif shot_type == 'Free Throw':
                points = 1
            else:
                points = 2
                
            expected_points = points * success_probability
            
            play_predictions.append({
                'shot_type': shot_type,
                'play_probability': play_probability,
                'success_probability': success_probability,
                'points': points,
                'expected_points': expected_points
            })
        
        # Ordenar por probabilidad de jugada (más probable primero)
        play_predictions = sorted(play_predictions, key=lambda x: x['play_probability'], reverse=True)
        
        # Mostrar resultados en formato de tabla
        print("\n┌─────────────────────────────────┬───────────────┬───────────────┬───────────┬─────────────────┐")
        print("│ TIPO DE JUGADA                  │ PROB. JUGADA  │ PROB. ÉXITO   │ PUNTOS    │ PUNTOS ESPERADOS │")
        print("├─────────────────────────────────┼───────────────┼───────────────┼───────────┼─────────────────┤")
        
        for pred in play_predictions:
            print(f"│ {pred['shot_type']:<30}  │ {pred['play_probability']*100:6.1f}%      │ {pred['success_probability']*100:6.1f}%      │ {pred['points']:<9} │ {pred['expected_points']:<15.2f} │")
        
        print("└─────────────────────────────────┴───────────────┴───────────────┴───────────┴─────────────────┘")
        
        # Encontrar la mejor jugada (mayor puntos esperados)
        best_play = max(play_predictions, key=lambda x: x['expected_points'])
        
        print(f"\n🏆 MEJOR JUGADA RECOMENDADA: {best_play['shot_type']}")
        print(f"   • Probabilidad de éxito: {best_play['success_probability']*100:.1f}%")
        print(f"   • Puntos esperados: {best_play['expected_points']:.2f}")
        
        show_viz = input("\n¿Desea ver visualización gráfica? (s/n): ").strip().lower()
        if show_viz == 's' or show_viz == 'si' or show_viz == 'sí' or show_viz == 'y' or show_viz == 'yes':
            # Preguntar si también desea guardar la visualización
            save_path = self._get_save_path(f"{player_name}_prediction")
            self.visualize_interactive_prediction(player_identifier, play_predictions, save_path)
        
        return play_predictions
        

    def _get_interactive_context(self):
        """Solicita al usuario los parámetros para la predicción"""
        print("\n🎯 CONFIGURACIÓN DE PARÁMETROS DE PREDICCIÓN")
        
        try:
            # Periodo (cuarto)
            while True:
                period = input("📝 Periodo [1-4, defecto=4]: ").strip()
                if not period:
                    period = 4
                    break
                if period.isdigit() and 1 <= int(period) <= 4:
                    period = int(period)
                    break
                print("❌ Valor inválido. Ingrese un número entre 1 y 4.")
            
            # Tiempo restante
            while True:
                time_remaining = input("📝 Tiempo restante en segundos [0-720, defecto=60]: ").strip()
                if not time_remaining:
                    time_remaining = 60
                    break
                if time_remaining.isdigit() and 0 <= int(time_remaining) <= 720:
                    time_remaining = int(time_remaining)
                    break
                print("❌ Valor inválido. Ingrese un número entre 0 y 720.")
            
            # Diferencia de puntos
            while True:
                score_diff = input("📝 Diferencia de puntos [-30 a 30, defecto=-2]: ").strip()
                if not score_diff:
                    score_diff = -2
                    break
                try:
                    score_diff = int(score_diff)
                    if -30 <= score_diff <= 30:
                        break
                    print("❌ Valor inválido. Ingrese un número entre -30 y 30.")
                except:
                    print("❌ Valor inválido. Ingrese un número entero.")
            
            # Distancia a canasta
            while True:
                dist = input("📝 Distancia a canasta en pies [0-35, defecto=15]: ").strip()
                if not dist:
                    dist = 15
                    break
                try:
                    dist = float(dist)
                    if 0 <= dist <= 35:
                        break
                    print("❌ Valor inválido. Ingrese un número entre 0 y 35.")
                except:
                    print("❌ Valor inválido. Ingrese un número.")
            
            # Posición en cancha (x, y)
            print("📝 Posición en cancha:")
            x = input("   • Coordenada X [-250 a 250, defecto=-100]: ").strip()
            x = int(x) if x and x.lstrip('-').isdigit() and -250 <= int(x) <= 250 else -100
            
            y = input("   • Coordenada Y [0 a 400, defecto=150]: ").strip()
            y = int(y) if y and y.isdigit() and 0 <= int(y) <= 400 else 150
            
        except Exception as e:
            print(f"❌ Error en la entrada: {e}")
            print("⚠️ Usando valores predeterminados")
            return {'period': 4, 'time_remaining': 60, 'score_diff': -2, 'dist': 15, 'x': -100, 'y': 150}
        
        return {
            'period': period,
            'time_remaining': time_remaining,
            'score_diff': score_diff,
            'dist': dist,
            'x': x,
            'y': y
        }

    def _predict_play_probabilities(self, player_data, context, shot_types):
        """Predice la probabilidad de cada tipo de jugada basado en el contexto"""
        # Extraer jugadas previas para contextos similares
        similar_contexts = player_data[
            (abs(player_data['period'] - context['period']) <= 1) &
            (abs(player_data['time_remaining'] - context['time_remaining']) <= 120) &
            (abs(player_data['score_diff'] - context['score_diff']) <= 10)
        ]
        
        if len(similar_contexts) < 5:
            similar_contexts = player_data  # Si hay pocos datos similares, usar todos
            
        # Determinar la zona basada en x, y del contexto
        court_zone = self._determine_court_zone(context['x'], context['y'])
        is_behind_3pt = context['dist'] > 23.75
        
        # Ajustar probabilidades según geometría de la cancha
        shot_probs = {}
        
        # Primera pasada: contar ocurrencias históricas
        shot_counts = {}
        total_shots = 0
        
        for shot_type in shot_types:
            # Contar ocurrencias históricas
            if shot_type == '3-Point':
                count = len(similar_contexts[similar_contexts['desc'].str.contains('3PT', na=False)])
            elif shot_type == 'Free Throw':
                count = len(similar_contexts[similar_contexts['type'] == 'Free Throw'])
            else:
                count = len(similar_contexts[
                    (similar_contexts['shot_category'] == shot_type) & 
                    (~similar_contexts['desc'].str.contains('3PT', na=False, case=False))
                ])
            
            shot_counts[shot_type] = count
            total_shots += count
                
        # Segunda pasada: aplicar reglas de geometría y calcular probabilidades
        for shot_type in shot_types:
            if shot_type == '3-Point' and not is_behind_3pt:
                shot_probs[shot_type] = 0.01  # Prácticamente imposible
            elif shot_type == 'Free Throw' and court_zone != "Free Throw Line":
                shot_probs[shot_type] = 0.0  # Imposible
            elif shot_type == 'Dunk' and context['dist'] > 5:
                shot_probs[shot_type] = 0.01  # Muy improbable
            else:
                # Usar conteos históricos con suavizado
                smoothing = 2
                if total_shots > 0:
                    shot_probs[shot_type] = (shot_counts[shot_type] + smoothing) / (total_shots + smoothing * len(shot_types))
                else:
                    shot_probs[shot_type] = 1.0 / len(shot_types)  # Distribución uniforme
            
        
        # Normalizar las probabilidades                
        total_prob = sum(shot_probs.values())
        if total_prob > 0:
            for shot_type in shot_probs:
                shot_probs[shot_type] /= total_prob
        
        return shot_probs
        
    
    def _get_player_shooting_profile(self, player_data):
        """Determina el perfil de tiro de un jugador basado en sus datos históricos"""
        shooting_profile = {}
        
        # 1. Proporción de tipos de tiro
        shot_counts = player_data.groupby('shot_category')['success'].count()
        total_shots = shot_counts.sum() if len(shot_counts) > 0 else 1
        shooting_profile['type_ratios'] = {shot_type: count/total_shots for shot_type, count in shot_counts.items()}
        
        # 2. Posiciones favoritas (zonas de cancha)
        player_data['court_zone'] = player_data.apply(
            lambda row: self._determine_court_zone(row.get('x', 0), row.get('y', 0)), axis=1
        )
        zone_counts = player_data.groupby('court_zone')['success'].count()
        shooting_profile['zone_ratios'] = {zone: count/total_shots for zone, count in zone_counts.items()}
        
        # 3. Eficiencia por distancia
        dist_bins = [0, 5, 10, 15, 20, 25, 30, 100]
        player_data['distance_bin'] = pd.cut(player_data['dist'], bins=dist_bins)
        efficiency_by_dist = player_data.groupby('distance_bin')['success'].mean()
        shooting_profile['dist_efficiency'] = {str(bin): eff for bin, eff in zip(efficiency_by_dist.index, efficiency_by_dist.values)}
        
        return shooting_profile

    def _predict_shot_success(self, player_data, context):
        """Predice la probabilidad de éxito de un tiro específico"""
        shot_type = context.get('shot_type', None)
        
        # Obtener perfil de tiro del jugador
        player_profile = self._get_player_shooting_profile(player_data)
        
        # Determinar la zona actual
        court_zone = self._determine_court_zone(context.get('x', 0), context.get('y', 0))
        
        # Ajustar por distancia según perfil
        dist_bin = pd.cut([context['dist']], [0, 5, 10, 15, 20, 25, 30, 100])[0]
        if str(dist_bin) in player_profile['dist_efficiency']:
            success_rate = player_profile['dist_efficiency'][str(dist_bin)]
        
        # Filtrar por tipo de tiro si está especificado
        if shot_type:
            if shot_type == '3-Point':
                filtered_data = player_data[player_data['desc'].str.contains('3PT', na=False)]
            elif shot_type == 'Free Throw':
                filtered_data = player_data[player_data['type'] == 'Free Throw']
            else:
                filtered_data = player_data[
                    (player_data['shot_category'] == shot_type) &
                    (~player_data['desc'].str.contains('3PT', na=False))
                ]
        else:
            filtered_data = player_data
        
        if len(filtered_data) < 5:
            filtered_data = player_data  # Si hay pocos datos, usar todos
        
        # Calcular probabilidad de éxito basado en datos históricos
        success_rate = 0.5  # Valor predeterminado
        
        if len(filtered_data) > 0:
            # Ver directamente el éxito histórico
            if shot_type == 'Free Throw':
                success_count = len(filtered_data[~filtered_data['desc'].str.contains('MISS', na=False, case=True)])
                success_rate = success_count / len(filtered_data) if len(filtered_data) > 0 else 0.75
            else:
                success_count = len(filtered_data[filtered_data['result'] == 'Made'])
                success_rate = success_count / len(filtered_data) if len(filtered_data) > 0 else 0.45
        
        # Factores contextuales que ajustan la probabilidad
        
        # Factor de presión por tiempo
        time_factor = 1.0
        if context['time_remaining'] <= 24:  # Shot clock final
            time_factor = 0.9
        elif context['time_remaining'] <= 5:  # Últimos segundos
            time_factor = 0.8
        
        # Factor de presión por marcador
        score_factor = 1.0
        if abs(context['score_diff']) <= 3 and context['period'] >= 4 and context['time_remaining'] <= 60:
            score_factor = 0.95  # Momento clutch
        
        # Factor de distancia
        distance_factor = 1.0
        if shot_type != 'Free Throw':
            # A mayor distancia, menor probabilidad (excepto para tiros libres)
            distance_factor = max(0.7, 1.0 - (context['dist'] / 100))
        
        # Aplicar todos los factores
        adjusted_success_rate = success_rate * time_factor * score_factor * distance_factor
        
        # Limitar a rangos realistas
        if shot_type == 'Free Throw':
            adjusted_success_rate = min(0.95, max(0.65, adjusted_success_rate))  # Entre 65% y 95%
        elif shot_type == 'Dunk':
            adjusted_success_rate = min(0.98, max(0.85, adjusted_success_rate))  # Entre 85% y 98%
        elif shot_type == 'Layup':
            adjusted_success_rate = min(0.85, max(0.55, adjusted_success_rate))  # Entre 55% y 85%
        elif shot_type == '3-Point':
            adjusted_success_rate = min(0.65, max(0.30, adjusted_success_rate))  # Entre 30% y 65%
        else:
            adjusted_success_rate = min(0.80, max(0.35, adjusted_success_rate))  # Entre 35% y 80%
        
        return adjusted_success_rate
    
    def visualize_interactive_prediction(self, player_identifier, predictions, save_path=None):
        """Genera gráfico visual para las predicciones"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        result = self._get_player_data_by_identifier(player_identifier)
        if result is None:
            player_name = "Jugador desconocido"
        else:
            player_name = result
        
        # Crear dos subplots: probabilidades de jugada y probabilidades de éxito
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'Predicciones para {player_name}', fontsize=16)
        
        # Datos para gráficas
        shot_types = [p['shot_type'] for p in predictions]
        play_probs = [p['play_probability'] * 100 for p in predictions]
        success_probs = [p['success_probability'] * 100 for p in predictions]
        expected_points = [p['expected_points'] for p in predictions]
        
        # Gráfico 1: Probabilidades de jugada
        bars1 = ax1.bar(shot_types, play_probs, color='skyblue')
        ax1.set_title('Probabilidad de Cada Tipo de Jugada')
        ax1.set_ylabel('Probabilidad (%)')
        ax1.set_ylim(0, 100)
        
        # Etiquetar barras
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # Gráfico 2: Probabilidades de éxito y puntos esperados
        x = np.arange(len(shot_types))
        width = 0.35
        
        bars2 = ax2.bar(x - width/2, success_probs, width, label='Prob. Éxito (%)', color='green')
        ax2_twin = ax2.twinx()
        bars3 = ax2_twin.bar(x + width/2, expected_points, width, label='Puntos Esperados', color='orange')
        
        # Configurar ejes y leyendas
        ax2.set_title('Probabilidad de Éxito y Puntos Esperados')
        ax2.set_xlabel('Tipo de Jugada')
        ax2.set_ylabel('Probabilidad (%)')
        ax2.set_ylim(0, 100)
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        for bar in bars3:
            height = bar.get_height()
            ax2_twin.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.2f}', ha='center', va='bottom')
        
        # Combinar leyendas
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Guardar el gráfico si se proporciona una ruta
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ Gráfico guardado en: {save_path}")
            except Exception as e:
                print(f"❌ Error al guardar el gráfico: {e}")
        
        plt.show()
        
        return fig
    
    # MÉTODO PARA COMPARACIÓN DE JUGADORES
    def compare_players(self, players_list, season=None):
        """Compara múltiples jugadores"""
        if self.processed_data is None:
            print("❌ No hay datos procesados disponibles")
            return None
        
        comparison_data = []
    
        for player in players_list:
            result = self._get_player_data_by_identifier(player)
            if result is None:
                continue
            
            player_data, player_id, player_name = result
            
            if season:
                player_data = player_data[player_data['season'] == season]
                if player_data.empty:
                    print(f"⚠️ No hay datos para {player_name} en temporada {season}")
                    continue
                
            if not player_data.empty:
                # Usar el nombre correcto para la visualización
                if player_name is None:
                    player_name = str(player)
                # Calcular puntos por partido
                games_played = player_data['gameid'].unique()
                total_points = 0
                
                for game_id in games_played:
                    game_data = player_data[player_data['gameid'] == game_id]
                    
                    # Contar tiros por tipo
                    shots_2pt = game_data[(game_data['type'].isin(['Made Shot', 'Missed Shot'])) & 
                                         (~game_data['desc'].str.contains('3PT', na=False))]
                    shots_3pt = game_data[(game_data['type'].isin(['Made Shot', 'Missed Shot'])) & 
                                         (game_data['desc'].str.contains('3PT', na=False))]
                    free_throws = game_data[game_data['type'] == 'Free Throw']
                    
                    # Contar aciertos
                    made_2pt = shots_2pt[shots_2pt['success'] == 1]
                    made_3pt = shots_3pt[shots_3pt['success'] == 1]
                    made_ft = free_throws[free_throws['success'] == 1]
                    
                    # Calcular puntos del partido
                    game_points = len(made_2pt) * 2 + len(made_3pt) * 3 + len(made_ft)
                    total_points += game_points
                
                shooting_plays = player_data[player_data['type'].isin(['Made Shot', 'Missed Shot'])]
                #clutch_plays = shooting_plays[shooting_plays['is_clutch'] == 1]
                
                comparison_data.append({
                    'Jugador': player_name,
                    'Partidos': len(games_played),
                    'Total_Jugadas': len(player_data),
                    'Total_Tiros': len(shooting_plays),
                    'Tiros_Anotados': len(shooting_plays[shooting_plays['success'] == 1]),
                    'Porcentaje_Tiro': len(shooting_plays[shooting_plays['success'] == 1]) / len(shooting_plays) * 100 if len(shooting_plays) > 0 else 0,
                    'Total_Puntos': total_points,
                    'Puntos_Por_Partido': total_points / len(games_played) if len(games_played) > 0 else 0,
                    #'Clutch_Porcentaje': clutch_plays['success'].mean() * 100 if len(clutch_plays) > 0 else 0
                })
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            print("\n🆚 COMPARACIÓN DE JUGADORES")
            print("="*60)
            print(df_comparison.round(2))
            
             # Preguntar si desea guardar el gráfico
            players_str = '_'.join([p.split(' ')[0] for p in df_comparison['Jugador'].astype(str).tolist()])
            save_path = self._get_save_path(f"comparison_{players_str}")
            
            # Generar gráficos
            self._generate_comparison_charts(df_comparison, save_path)
            
            return df_comparison
        else:
            print("❌ No se encontraron datos para ningún jugador")
            return None
    
    def predict_player_performance(self, player_identifier, game_context=None):
        """Predice el rendimiento de un jugador en una situación específica"""
        # Verificar si los modelos están entrenados, entrenarlos si no
        if not self.models.get('success_predictor'):
            print("🔄 Entrenando modelos predictivos (primera vez)... (esto puede tomar un momento)")
            self._prepare_prediction_models()
        
        result = self._get_player_data_by_identifier(player_identifier)
        if result is None:
            return None
        
        player_data, player_id, player_name = result
    
        print(f"🔮 PREDICCIÓN PARA: {player_name}")
        print("="*50)

        # Estadísticas históricas
        shooting_data = player_data[player_data['type'].isin(['Made Shot', 'Missed Shot'])]
        historical_success_rate = shooting_data['success'].mean() if len(shooting_data) > 0 else 0

        print(f"📊 ESTADÍSTICAS HISTÓRICAS:")
        print(f"   • Porcentaje de éxito histórico: {historical_success_rate:.1%}")
        print(f"   • Total de tiros en datos: {len(shooting_data)}")

        # Contexto del juego
        if game_context is None:
            game_context = {
                'period': 4,
                'time_remaining': 120,
                'score_diff': -3,
                'dist': 18,
                'x': -100,
                'y': 150
            }

        print(f"🎮 CONTEXTO DEL JUEGO:")
        for key, value in game_context.items():
            print(f"   • {key}: {value}")

        # Realizar predicciones
        feature_values = []
        for feature in self.prediction_features:
            if feature in game_context:
                feature_values.append(game_context[feature])
            else:
                avg_value = player_data[feature].mean() if feature in player_data.columns else 0
                feature_values.append(avg_value)

        prediction_input = np.array(feature_values).reshape(1, -1)

        # Realizar predicciones
        predictions = {}

        if 'success_predictor' in self.models:
            success_prob = self.models['success_predictor'].predict_proba(prediction_input)[0]
            predictions['success_probability'] = success_prob[1] if len(success_prob) > 1 else success_prob[0]
            predictions['predicted_success'] = self.models['success_predictor'].predict(prediction_input)[0]

        if 'shot_type_predictor' in self.models:
            shot_type_pred = self.models['shot_type_predictor'].predict(prediction_input)[0]
            if 'shot_type' in self.label_encoders:
                predictions['predicted_shot_type'] = self.label_encoders['shot_type'].inverse_transform([shot_type_pred])[0]

        if 'points_predictor' in self.models:
            predictions['predicted_points'] = self.models['points_predictor'].predict(prediction_input)[0]

        # Mostrar predicciones
        print(f"\n🔮 PREDICCIONES:")
        if 'success_probability' in predictions:
            print(f"   • Probabilidad de éxito: {predictions['success_probability']:.1%}")
        if 'predicted_shot_type' in predictions:
            print(f"   • Tipo de tiro probable: {predictions['predicted_shot_type']}")
        if 'predicted_points' in predictions:
            print(f"   • Puntos esperados: {predictions['predicted_points']:.1f}")

        # Recomendaciones
        print(f"\n💡 ANÁLISIS:")
        if 'success_probability' in predictions:
            if predictions['success_probability'] > 0.5:
                print("   • RECOMENDACIÓN: Situación favorable para tomar el tiro")
            else:
                print("   • RECOMENDACIÓN: Considerar buscar mejor posición o pase")
        
        return predictions
    
    # MÉTODO DE REPORTE COMPRENSIVO
    def generate_comprehensive_report(self, player_identifier, season=None):
        """Genera un reporte comprensivo combinando todas las funcionalidades"""
        
        result = self._get_player_data_by_identifier(player_identifier)
        if result is None:
            return None
        
        player_data, player_id, player_name = result
        
        print(f"\n🏀 REPORTE COMPRENSIVO NBA - {player_name}")
        print("="*80)

        # 1. Estadísticas detalladas
        detailed_stats = self.get_detailed_player_stats(player_identifier, season)

        if detailed_stats is None:
            return None

        # 2. Predicciones en diferentes contextos
        print(f"\n🔮 PREDICCIONES EN DIFERENTES CONTEXTOS:")
        print("-"*50)

        contexts = [
            {'name': 'Final cerrado', 'period': 4, 'time_remaining': 30, 'score_diff': -1, 'dist': 20, 'x': -50, 'y': 200},
            {'name': 'Inicio de juego', 'period': 1, 'time_remaining': 600, 'score_diff': 0, 'dist': 15, 'x': 0, 'y': 150},
            {'name': 'Ventaja cómoda', 'period': 3, 'time_remaining': 300, 'score_diff': 15, 'dist': 25, 'x': -200, 'y': 250}
        ]

        predictions_results = []
        for context in contexts:
            print(f"\n📍 {context['name'].upper()}:")
            context_copy = context.copy()
            del context_copy['name']
            predictions = self.predict_player_performance(player_identifier, context_copy)
            predictions_results.append(predictions)

        return {
            'detailed_stats': detailed_stats,
            'predictions': predictions_results
        }

# FUNCIONES DE UTILIDAD PARA DEMOSTRACIÓN RÁPIDA
def quick_demo():
    """Demostración rápida del sistema"""
    print("🚀 DEMOSTRACIÓN RÁPIDA DEL NBA ANALYZER")
    print("="*50)

    # Crear analizador
    analyzer = NBAAnalyzer()

    # Cargar datos
    if analyzer.load_data_with_priority():
        if analyzer.clean_and_process_data():
            # Obtener jugadores disponibles
            players = analyzer.get_available_players()
            print(f"\n👥 Jugadores disponibles: {len(players)}")
            print(f"Ejemplos: {players[:5]}")

            # Analizar un jugador
            sample_player = players[0] if players else "L. James"
            print(f"\n📊 Analizando: {sample_player}")
            analyzer.get_detailed_player_stats(sample_player)

            # Hacer predicción
            print(f"\n🔮 Predicción para {sample_player}:")
            analyzer.predict_player_performance(sample_player)

            return analyzer

    return None