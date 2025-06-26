# -*- coding: utf-8 -*-
"""
🏀 NBA ANALYZER - SCRIPT DE DEMOSTRACIÓN PARA TFG
==================================================

Este script demuestra todas las funcionalidades del NBA Analyzer Completo.

Autor: Antonio Luis Godino
Proyecto: TFG - Análisis predictivo de datos NBA con Machine Learning
"""

from nba_analyzer import NBAAnalyzer
import time

def print_separator(title=""):
    """Imprime un separador visual"""
    print("\n" + "="*70)
    if title:
        print(f"🏀 {title}")
        print("="*70)

def demo_completa():
    """Demostración completa del sistema"""
    try:
        print_separator("DEMOSTRACIÓN COMPLETA NBA ANALYZER")
        print("Sistema completo de análisis NBA con Machine Learning")

        # 1. Inicialización 
        analyzer = None
        try:
            analyzer = NBAAnalyzer()
        except Exception as e:
            print(f"❌ Error en inicialización: {e}")
            return None

        # Exploración y selección de temporadas
        print("\n🔍 EXPLORANDO TEMPORADAS DISPONIBLES...")
        available_seasons = analyzer.get_available_seasons_from_files()
        
        if available_seasons:
            print(f"📅 Temporadas disponibles en archivos: {available_seasons}")
            
            # Selector de temporadas
            print("\n📅 SELECCIÓN DE TEMPORADAS:")
            print("Selecciona qué temporadas deseas cargar:")
            print("0. Todas las temporadas")
            for i, season in enumerate(available_seasons, 1):
                print(f"{i}. Temporada {season}")
                
            choice = input("\n🎯 Ingresa los números separados por coma (ej: 1,3,4) o 0 para todas: ")
            
            if choice.strip() == "0":
                print("✅ Seleccionadas todas las temporadas")
                selected_seasons = None
            else:
                try:
                    selected_indices = [int(idx.strip()) for idx in choice.split(',') if idx.strip()]
                    selected_seasons = [available_seasons[idx-1] for idx in selected_indices if 1 <= idx <= len(available_seasons)]
                    
                    if selected_seasons:
                        print(f"✅ Temporadas seleccionadas: {selected_seasons}")
                    else:
                        print("⚠️ No se seleccionaron temporadas válidas. Usando todas las temporadas.")
                        selected_seasons = None
                except Exception as e:
                    print(f"⚠️ Error en selección: {e}. Usando todas las temporadas.")
                    selected_seasons = None
        else:
            print("⚠️ No se pudieron detectar temporadas en los archivos")
            selected_seasons = None
        
        # 2. Carga de datos con validación y filtro por temporadas
        print_separator("2. CARGA DE DATOS")
        print("🔄 Intentando cargar datos...")
        print("   1° Prioridad: Datos locales (carpeta 'data')")
        print("   2° Prioridad: Kaggle API")
        print("   3° Prioridad: Datos sintéticos para testing")

        if selected_seasons:
            print(f"   🔍 Filtro aplicado: Solo temporadas {selected_seasons}")

        if not analyzer.load_data_with_priority(selected_seasons=selected_seasons):
            print("❌ Error: No se pudieron cargar los datos")
            return None

        # 3. Procesamiento y limpieza
        print_separator("3. PROCESAMIENTO Y LIMPIEZA DE DATOS")
        if not analyzer.clean_and_process_data():
            print("❌ Error en procesamiento")
            return None

        # 4. Exploración de datos disponibles
        print_separator("4. EXPLORACIÓN DE DATOS DISPONIBLES")
        all_players = []
        # Reunir jugadores de todas las temporadas disponibles
        for season, players_info in analyzer.available_players.items():
            for player_info in players_info:
                # Asegurarse de manejar tanto tuplas como strings
                if isinstance(player_info, tuple):
                    player_id, player_name = player_info  # Desempaquetar tupla
                    if (player_id, player_name) not in all_players:
                        all_players.append((player_id, player_name))
                else:
                    if player_info not in all_players:
                        all_players.append(player_info)
                        
        # Ordenar por nombre (segunda posición si es tupla)
        players = sorted(all_players, key=lambda x: x[1] if isinstance(x, tuple) else x)
        seasons = analyzer.available_seasons

        print(f"📊 RESUMEN DE DATOS:")
        print(f"   • Total de jugadores: {len(players)}")
        print(f"   • Temporadas disponibles: {seasons}")
        print(f"   • Registros procesados: {len(analyzer.processed_data):,}")

        # Mostrar lista paginada de jugadores disponibles
        print_separator("SELECCIÓN DE JUGADORES PARA ANÁLISIS")
        print("👥 JUGADORES DISPONIBLES:")
        
        # Mostrar jugadores en páginas de 20
        page_size = 20
        current_page = 0
        total_pages = (len(players) + page_size - 1) // page_size
        
        demo_players = []
        selected_players = []  # Para compatibilidad con la visualización
        
        while True:
            # Mostrar la página actual
            start_idx = current_page * page_size
            end_idx = min(start_idx + page_size, len(players))
            
            print(f"\nPágina {current_page + 1}/{total_pages} - Jugadores {start_idx + 1}-{end_idx} de {len(players)}")
            
            for i, player_info in enumerate(players[start_idx:end_idx], start_idx + 1):
                if isinstance(player_info, tuple):
                    player_id, player_name = player_info
                    selected_mark = "✓" if player_info in demo_players else " "
                    print(f"   {i:3d}. [{selected_mark}] ID: {player_id} - {player_name}")
                else:
                    selected_mark = "✓" if player_info in demo_players else " "
                    print(f"   {i:3d}. [{selected_mark}] {player_info}")
            
            print("\n🔍 OPCIONES:")
            print("   • Número del jugador: Seleccionar ese jugador")
            print("   • S: Siguiente página")
            print("   • A: Anterior página")
            print("   • B: Buscar jugador")
            print("   • C: Continuar con selección actual")
            print("   • Q: Salir de la selección")
            
            if demo_players:
                # Versión compatible con tuplas para mostrar jugadores seleccionados
                display_names = []
                for player in demo_players:
                    if isinstance(player, tuple):
                        player_id, player_name = player
                        display_names.append(f"{player_name} (ID: {player_id})")
                    else:
                        display_names.append(str(player))
                print(f"\n🏀 Jugadores seleccionados ({len(demo_players)}): {', '.join(display_names)}")
            
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
                    # Búsqueda adaptada a tuplas o strings
                    matches = []
                    for p in players:
                        if isinstance(p, tuple):
                            _, player_name = p
                            if search_term in player_name.lower():
                                matches.append(p)
                        else:
                            if search_term in p.lower():
                                matches.append(p)
                    
                    if matches:
                        print(f"\nJugadores que coinciden con '{search_term}':")
                        for i, player_info in enumerate(matches[:20], 1):
                            if isinstance(player_info, tuple):
                                player_id, player_name = player_info
                                selected_mark = "✓" if player_info in demo_players else " "
                                print(f"   {i:3d}. [{selected_mark}] ID: {player_id} - {player_name}")
                            else:
                                selected_mark = "✓" if player_info in demo_players else " "
                                print(f"   {i:3d}. [{selected_mark}] {player_info}")
                        
                        select = input("\nSeleccionar número o Enter para volver: ").strip()
                        if select.isdigit() and 1 <= int(select) <= len(matches[:20]):
                            selected_player = matches[int(select) - 1]
                            demo_players.append(selected_player)
                            
                            # Mostrar mensaje legible según el tipo
                            if isinstance(selected_player, tuple):
                                player_id, player_name = selected_player
                                print(f"✅ Agregado: {player_name} (ID: {player_id})")
                            else:
                                print(f"✅ Agregado: {selected_player}")
                    else:
                        print(f"❌ No se encontraron jugadores que coincidan con '{search_term}'")
            
            elif choice == 'C':
                if demo_players:
                    break
                else:
                    print("❌ Debes seleccionar al menos un jugador para continuar")
            
            elif choice == 'Q':
                if not demo_players and len(players) > 0:
                    print("⚠️ No has seleccionado jugadores. Seleccionando el primer jugador por defecto.")
                    demo_players = [players[0]]
                break
            
            elif choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(players):
                    selected_player = players[idx - 1]
                    if selected_player not in demo_players:
                        demo_players.append(selected_player)
                        
                        # Mostrar mensaje legible según el tipo
                        if isinstance(selected_player, tuple):
                            player_id, player_name = selected_player
                            print(f"✅ Agregado: {player_name} (ID: {player_id})")
                        else:
                            print(f"✅ Agregado: {selected_player}")
                    else:
                        demo_players.remove(selected_player)
                        
                        # Mostrar mensaje legible según el tipo
                        if isinstance(selected_player, tuple):
                            player_id, player_name = selected_player
                            print(f"❌ Eliminado: {player_name} (ID: {player_id})")
                        else:
                            print(f"❌ Eliminado: {selected_player}")
                else:
                    print("❌ Número fuera de rango")
            else:
                print("❌ Opción no válida")
        
        if not demo_players:
            print("❌ No se seleccionaron jugadores. Finalizando demostración.")
            return None
        
        # 5. Análisis detallado de jugador
        print_separator("5. ANÁLISIS DETALLADO DE JUGADOR")
        sample_player = demo_players[0]
        
        # Mostrar información legible del jugador
        if isinstance(sample_player, tuple):
            player_id, player_name = sample_player
            print(f"🔍 Analizando en detalle: {player_name} (ID: {player_id})")
        else:
            print(f"🔍 Analizando en detalle: {sample_player}")            

        # Usar ID para análisis cuando está disponible
        if isinstance(sample_player, tuple):
            detailed_stats = analyzer.get_detailed_player_stats(sample_player[0])
        else:
            detailed_stats = analyzer.get_detailed_player_stats(sample_player)
        input("\n📌 Presiona Enter para continuar a predicciones...")

        # 6. Predicciones de rendimiento
        print_separator("6. PREDICCIONES DE RENDIMIENTO CON ML")
        
        # Mostrar información legible del jugador
        if isinstance(sample_player, tuple):
            player_id, player_name = sample_player
            print(f"🤖 Generando predicciones para: {player_name} (ID: {player_id})")
        else:
            print(f"🤖 Generando predicciones para: {sample_player}")
            
        print("📝 Los modelos se entrenan bajo demanda")

        # Escenarios de predicción
        scenarios = [
            {
                'name': '🔥 FINAL CERRADO',
                'context': {'period': 4, 'time_remaining': 45, 'score_diff': -2, 'dist': 22, 'x': -100, 'y': 200}
            },
            {
                'name': '⭐ TIRO LIBRE',
                'context': {'period': 2, 'time_remaining': 300, 'score_diff': 5, 'dist': 15, 'x': 0, 'y': 150}
            },
            {
                'name': '🎯 TRIPLE DECISIVO',
                'context': {'period': 4, 'time_remaining': 15, 'score_diff': -3, 'dist': 25, 'x': -200, 'y': 240}
            }
        ]

        predictions_results = []
        for scenario in scenarios:
            print(f"\n{scenario['name']}:")
            
            # Usar ID para predicciones cuando está disponible
            if isinstance(sample_player, tuple):
                prediction = analyzer.predict_player_performance(sample_player[0], scenario['context'])
            else:
                prediction = analyzer.predict_player_performance(sample_player, scenario['context'])
                
            predictions_results.append(prediction)
            time.sleep(1)  # Pausa para mejor visualización
            input("\n📌 Presiona Enter para continuar...")

        # 7. Comparación entre jugadores
        print_separator("7. COMPARACIÓN ENTRE JUGADORES")
        if len(demo_players) >= 2:
            comparison_players = demo_players[:3] if len(demo_players) >= 3 else demo_players
            
            # Mostrar jugadores a comparar de forma legible
            display_names = []
            for player in comparison_players:
                if isinstance(player, tuple):
                    player_id, player_name = player
                    display_names.append(f"{player_name} (ID: {player_id})")
                else:
                    display_names.append(str(player))
                    
            print(f"🆚 Comparando jugadores: {', '.join(display_names)}")

            # Para comparar, usar IDs directamente si están disponibles
            comparison_ids = []
            for player in comparison_players:
                if isinstance(player, tuple):
                    comparison_ids.append(player[0])  # Usar ID
                else:
                    comparison_ids.append(player)
                    
            comparison_result = analyzer.compare_players(comparison_ids)
        else:
            print("⚠️  Insuficientes jugadores para comparación, necesitas al menos 2")
            if len(demo_players) == 1:
                # Mostrar de forma legible
                if isinstance(sample_player, tuple):
                    player_id, player_name = sample_player
                    print(f"💡 Comparando {player_name} (ID: {player_id}) con otro jugador disponible")
                else:
                    print(f"💡 Comparando {sample_player} con otro jugador disponible")
                    
                # Buscar otro jugador para comparar
                other_player = next((p for p in players if p != sample_player), None)
                if other_player:
                    # Crear array de IDs cuando es posible
                    if isinstance(sample_player, tuple) and isinstance(other_player, tuple):
                        comparison_result = analyzer.compare_players([sample_player[0], other_player[0]])
                    elif isinstance(sample_player, tuple):
                        comparison_result = analyzer.compare_players([sample_player[0], other_player])
                    elif isinstance(other_player, tuple):
                        comparison_result = analyzer.compare_players([sample_player, other_player[0]])
                    else:
                        comparison_result = analyzer.compare_players([sample_player, other_player])
                        
        if comparison_result is not None:  # Añadir esta condición de verificación
            input("\n📌 Presiona Enter para continuar al reporte...")

        # 8. Reporte comprensivo
        print_separator("8. REPORTE COMPRENSIVO")
        
        # Mostrar información legible del jugador
        if isinstance(sample_player, tuple):
            player_id, player_name = sample_player
            print(f"📋 Generando reporte completo para: {player_name} (ID: {player_id})")
            comprehensive_report = analyzer.generate_comprehensive_report(player_id)
        else:
            print(f"📋 Generando reporte completo para: {sample_player}")
            comprehensive_report = analyzer.generate_comprehensive_report(sample_player)
        
        input("\n📌 Presiona Enter para finalizar la demostración...")
            

        print_separator("¡DEMOSTRACIÓN COMPLETADA EXITOSAMENTE!")
        print("🎉 Sistema NBA Analyzer validado y funcional")
        print("📁 Todos los archivos generados están listos")

        return analyzer

    except Exception as e:
        print(f"❌ Error general en la demostración: {e}")
        return None


def menu_interactivo():
    """Menú interactivo mejorado con explorador avanzado como default"""
    print_separator("MENÚ INTERACTIVO NBA ANALYZER - VERSIÓN CORREGIDA")

    try:
        analyzer = NBAAnalyzer()
        print("📊 Usando versión optimizada con cálculo correcto de estadísticas")
        
        # NUEVO - Análisis preliminar de temporadas disponibles
        print("\n🔍 EXPLORANDO TEMPORADAS DISPONIBLES...")
        available_seasons = analyzer.get_available_seasons_from_files()
        
        if not available_seasons:
            print("⚠️ No se pudieron detectar temporadas en los archivos")
            print("🔄 Cargando datos sin filtrar por temporada...")
            selected_seasons = None
        else:
            print(f"📅 Temporadas disponibles en archivos: {available_seasons}")
            
            # Selector de temporadas
            print("\n📅 SELECCIÓN DE TEMPORADAS:")
            print("Selecciona qué temporadas deseas cargar:")
            print("0. Todas las temporadas")
            for i, season in enumerate(available_seasons, 1):
                print(f"{i}. Temporada {season}")
                
            choice = input("\n🎯 Ingresa los números separados por coma (ej: 1,3,4) o 0 para todas: ")
            
            if choice.strip() == "0":
                print("✅ Seleccionadas todas las temporadas")
                selected_seasons = None
            else:
                try:
                    selected_indices = [int(idx.strip()) for idx in choice.split(',') if idx.strip()]
                    selected_seasons = [available_seasons[idx-1] for idx in selected_indices if 1 <= idx <= len(available_seasons)]
                    
                    if selected_seasons:
                        print(f"✅ Temporadas seleccionadas: {selected_seasons}")
                    else:
                        print("⚠️ No se seleccionaron temporadas válidas. Usando todas las temporadas.")
                        selected_seasons = None
                except Exception as e:
                    print(f"⚠️ Error en selección: {e}. Usando todas las temporadas.")
                    selected_seasons = None
            
        if not analyzer.load_data_with_priority(selected_seasons=selected_seasons):
            raise Exception("Error cargando datos")
            
        if not analyzer.clean_and_process_data():
            raise Exception("Error procesando datos")
            
        # Reunir jugadores de todas las temporadas
        all_players = []
        for season, players_info in analyzer.available_players.items():
            for player_info in players_info:
                # Asegurarse de manejar tanto tuplas como strings
                if isinstance(player_info, tuple):
                    player_id, player_name = player_info  # Desempaquetar tupla
                    if (player_id, player_name) not in all_players:
                        all_players.append((player_id, player_name))
                else:
                    # Compatibilidad con versión anterior
                    if player_info not in all_players:
                        all_players.append(player_info)
                        
        # Ordenar por nombre (segunda posición si es tupla)
        players = sorted(all_players, key=lambda x: x[1] if isinstance(x, tuple) else x)
        
        if not players:
            raise Exception("No se encontraron jugadores en los datos")
            
        print(f"✅ {len(players)} jugadores disponibles")

        while True:
            # INICIAR DIRECTAMENTE CON EL EXPLORADOR DE JUGADORES
            print_separator("EXPLORADOR AVANZADO DE JUGADORES")
            print("👥 SELECCIONA JUGADORES PARA ANALIZAR:")
            
            # Mostrar jugadores en páginas
            page_size = 20
            current_page = 0
            total_pages = (len(players) + page_size - 1) // page_size
            selected_players = []
            
            # Loop del explorador
            while True:
                # Mostrar la página actual
                start_idx = current_page * page_size
                end_idx = min(start_idx + page_size, len(players))
                
                print(f"\nPágina {current_page + 1}/{total_pages} - Jugadores {start_idx + 1}-{end_idx} de {len(players)}")
                
                for i, player_info in enumerate(players[start_idx:end_idx], start_idx + 1):
                    if isinstance(player_info, tuple):
                        player_id, player_name = player_info
                        selected_mark = "✓" if player_info in selected_players else " "
                        print(f"   {i:3d}. [{selected_mark}] ID: {player_id} - {player_name}")
                    else:
                        selected_mark = "✓" if player_info in selected_players else " "
                        print(f"   {i:3d}. [{selected_mark}] {player_info}")

                
                print("\n🔍 OPCIONES:")
                print("   • Número: Seleccionar/deseleccionar jugador")
                print("   • S: Siguiente página")
                print("   • A: Anterior página")
                print("   • B: Buscar jugador")
                print("   • C: Continuar con los jugadores seleccionados")
                print("   • Q: Salir al menú principal")
                
                if selected_players:
                    print(f"\n🏀 JUGADORES SELECCIONADOS ({len(selected_players)}):")
                    for i, player in enumerate(selected_players, 1):
                        if isinstance(player, tuple):
                            player_id, player_name = player
                            print(f"   {i}. {player_name} (ID: {player_id})")
                        else:
                            print(f"   {i}. {player}")
                
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
                        # Búsqueda adaptada a tuplas o strings
                        matches = []
                        for p in players:
                            if isinstance(p, tuple):
                                _, player_name = p
                                if search_term in player_name.lower():
                                    matches.append(p)
                            else:
                                if search_term in p.lower():
                                    matches.append(p)
                        
                        if matches:
                            print(f"\nJugadores que coinciden con '{search_term}':")
                            for i, player_info in enumerate(matches[:20], 1):
                                if isinstance(player_info, tuple):
                                    player_id, player_name = player_info
                                    selected_mark = "✓" if player_info in selected_players else " "
                                    print(f"   {i:3d}. [{selected_mark}] ID: {player_id} - {player_name}")
                                else:
                                    selected_mark = "✓" if player_info in selected_players else " "
                                    print(f"   {i:3d}. [{selected_mark}] {player_info}")
                            
                            select = input("\nSeleccionar número o Enter para volver: ").strip()
                            if select.isdigit() and 1 <= int(select) <= len(matches[:20]):
                                player = matches[int(select) - 1]
                                if player in selected_players:
                                    selected_players.remove(player)
                                    if isinstance(player, tuple):
                                        print(f"❌ Eliminado: {player[1]} (ID: {player[0]})")
                                    else:
                                        print(f"❌ Eliminado: {player}")
                                else:
                                    selected_players.append(player)
                                    if isinstance(player, tuple):
                                        print(f"✅ Agregado: {player[1]} (ID: {player[0]})")
                                    else:
                                        print(f"✅ Agregado: {player}")
                        else:
                            print(f"❌ No se encontraron jugadores que coincidan con '{search_term}'")
                
                elif choice == 'C':
                    if selected_players:
                        break
                    else:
                        print("❌ Debes seleccionar al menos un jugador para continuar")
                
                elif choice == 'Q':
                    return
                    
                elif choice.isdigit():
                    idx = int(choice)
                    if 1 <= idx <= len(players):
                        player = players[idx - 1]
                        if player in selected_players:
                            selected_players.remove(player)
                            print(f"❌ Eliminado: {player}")
                        else:
                            selected_players.append(player)
                            print(f"✅ Agregado: {player}")
                    else:
                        print("❌ Número fuera de rango")
                else:
                    print("❌ Opción no válida")
            
            # Menú de acciones para los jugadores seleccionados
            print_separator("ACCIONES PARA JUGADORES SELECCIONADOS")
            print("🏀 Selecciona qué deseas hacer con estos jugadores:")
            print("1. 📊 Análisis detallado")
            print("2. 🔮 Predicción de rendimiento")
            print("3. 🎮 Predicción interactiva personalizada")
            print("4. 🆚 Comparación entre jugadores")
            print("5. 📋 Reporte comprensivo")
            print("6. ↩️ Volver al selector de jugadores")
            
            action = input("\n🎯 Selecciona acción (1-6): ").strip()
            
            if action == '1':
                # Análisis detallado para cada jugador
                for player in selected_players:
                    if isinstance(player, tuple):
                        player_id, player_name = player
                        print(f"\n📊 ANÁLISIS DETALLADO PARA: {player_name} (ID: {player_id})")
                        # Usar directamente el ID numérico que es más preciso
                        analyzer.get_detailed_player_stats(player_id)
                    else:
                        print(f"\n📊 ANÁLISIS DETALLADO PARA: {player}")
                        analyzer.get_detailed_player_stats(player)
                    input("\nPresiona Enter para continuar...")

            elif action == '2':
                # Predicción de rendimiento para cada jugador
                for player in selected_players:
                    if isinstance(player, tuple):
                        player_id, player_name = player
                        print(f"\n🔮 PREDICCIÓN DE RENDIMIENTO PARA: {player_name} (ID: {player_id})")
                        analyzer.predict_player_performance(player_id)
                    else:
                        print(f"\n🔮 PREDICCIÓN DE RENDIMIENTO PARA: {player}")
                        analyzer.predict_player_performance(player)
                    input("\nPresiona Enter para continuar...")

            elif action == '3':
                # Predicción interactiva personalizada
                for player in selected_players:
                    if isinstance(player, tuple):
                        player_id, player_name = player
                        print(f"\n🎮 PREDICCIÓN INTERACTIVA PARA: {player_name} (ID: {player_id})")
                        analyzer.predict_interactive(player_id)
                    else:
                        print(f"\n🎮 PREDICCIÓN INTERACTIVA PARA: {player}")
                        analyzer.predict_interactive(player)
                    input("\nPresiona Enter para continuar...")

            elif action == '4':
                # Comparación entre jugadores
                if len(selected_players) < 2:
                    print("⚠️ Se requieren al menos 2 jugadores para la comparación")
                else:
                    # Extraer IDs para comparación
                    comparison_ids = []
                    for player in selected_players:
                        if isinstance(player, tuple):
                            player_id, player_name = player
                            comparison_ids.append(player_id)
                        else:
                            comparison_ids.append(player)
                    
                    # Mostrar jugadores que serán comparados
                    display_names = []
                    for player in selected_players:
                        if isinstance(player, tuple):
                            display_names.append(f"{player[1]} (ID: {player[0]})")
                        else:
                            display_names.append(str(player))
                            
                    print(f"\n🆚 COMPARANDO JUGADORES: {', '.join(display_names)}")
                    analyzer.compare_players(comparison_ids)

            elif action == '5':
                # Reporte comprensivo para cada jugador
                for player in selected_players:
                    if isinstance(player, tuple):
                        player_id, player_name = player
                        print(f"\n📋 REPORTE COMPRENSIVO PARA: {player_name} (ID: {player_id})")
                        analyzer.generate_comprehensive_report(player_id)
                    else:
                        print(f"\n📋 REPORTE COMPRENSIVO PARA: {player}")
                        analyzer.generate_comprehensive_report(player)
                    input("\nPresiona Enter para continuar...")
            
            elif action == '6':
                # Volver al selector de jugadores
                continue
            
            else:
                print("❌ Opción no válida")
            
            # Preguntar si desea continuar
            if input("\n¿Deseas realizar otra acción? (s/n): ").lower() != 's':
                break

    except Exception as e:
        print(f"❌ Error inicializando el sistema: {e}")

if __name__ == "__main__":
    try:
        print("🏀 NBA ANALYZER - SISTEMA DE DEMOSTRACIÓN")
        print("="*50)
        print("Selecciona el tipo de demostración:")
        print("1. 🎬 Demostración completa")
        print("2. 🎮 Menú interactivo")

        choice = input("\n🎯 Tu elección (1-2): ").strip()
        if choice not in ['1', '2']:
            print("⚠️ Opción no válida. Ejecutando demostración completa por defecto...")
            choice = '1'

        demos = {
            '1': demo_completa,
            '2': menu_interactivo
        }

        demos[choice]()

    except KeyboardInterrupt:
        print("\n👋 Programa interrumpido por el usuario")
    except Exception as e:
        print(f"❌ Error fatal: {e}")
    finally:
        print("\n✨ Programa finalizado")
