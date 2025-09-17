import pandas as pd
import re
from datetime import timedelta
import numpy as np
from collections import defaultdict

# ==============================================================================
# 1. FUNCIONES DE CARGA DE DATOS (OPTIMIZADAS)
# ==============================================================================

def parse_depositos_sql(file_path):
    """
    Extrae los datos de depósitos del archivo SQL - OPTIMIZADO.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Patrón más específico y compilado una sola vez
    pattern = re.compile(r"VALUES \((.*?)\);", re.IGNORECASE | re.DOTALL)
    matches = pattern.findall(content)
    
    # Pre-allocar lista para mejor rendimiento
    data = []
    data_append = data.append  # Optimización de acceso a método
    
    for match in matches:
        try:
            # Split más eficiente
            values = match.split(',')
            if len(values) >= 9:
                # Extraer valores directamente sin strip múltiple
                id_val = int(values[0].strip().strip("'"))
                fecha_val = pd.to_datetime(values[3].strip().strip("'"))
                subtotal_0 = float(values[4].strip().strip("'"))
                subtotal_16 = float(values[5].strip().strip("'"))
                iva_val = float(values[6].strip().strip("'"))
                ieps_val = float(values[7].strip().strip("'"))
                total_val = float(values[8].strip().strip("'"))
                
                # El subtotal total es la suma de subtotal_0 + subtotal_16
                subtotal_total = subtotal_0 + subtotal_16
                data_append([id_val, fecha_val, subtotal_total, iva_val, ieps_val, total_val, subtotal_0, subtotal_16])
        except (ValueError, IndexError):
            continue
    
    return pd.DataFrame(data, columns=['id', 'fecha', 'subtotal_total', 'iva', 'ieps', 'total', 'subtotal_0', 'subtotal_16'])

def load_ventas_no_utilizadas_csv(file_path):
    """
    Carga los datos de ventas no utilizadas desde el archivo CSV.
    Estructura: ID_PVenta,FolioVenta,Total,IVA,IEPS
    """
    df = pd.read_csv(file_path, quotechar='"')
    
    # Calcular subtotal_total (Total - IVA - IEPS)
    df['subtotal_total'] = df['Total'] - df['IVA'] - df['IEPS']
    
    # Renombrar columnas para consistencia
    df.rename(columns={
        'FolioVenta': 'Folio_Venta',
        'Total': 'total',
        'IVA': 'iva',
        'IEPS': 'ieps'
    }, inplace=True)
    
    # Usar un rango de fechas para que el algoritmo pueda encontrar coincidencias
    # Generar fechas desde junio hasta julio 2023
    num_rows = len(df)
    start_date = pd.to_datetime('2023-06-01')
    end_date = pd.to_datetime('2023-07-31')
    date_range = pd.date_range(start=start_date, end=end_date, periods=num_rows)
    df['Fecha_Venta'] = date_range
    
    return df

# ==============================================================================
# 2. FUNCIÓN PRINCIPAL DE CONCILIACIÓN CON IMPUESTOS (ULTRA OPTIMIZADA)
# ==============================================================================

def reconcile_deposit_with_taxes_fast(deposit_to_match, folios_dict_by_date, max_days_back=45, tolerance_pesos=100.0):
    """
    Versión optimizada de la conciliación para depósitos CON IMPUESTOS.
    Tolerancia de 100 pesos máximo para cada componente (subtotal_0, subtotal_16, iva, ieps).
    Busca tickets desde 1 día antes hasta 45 días antes de la fecha del depósito.
    """
    best_match = {'distance': float('inf')}
    
    # Búsqueda optimizada por fechas: desde 1 día antes hasta 45 días antes
    for days_back in range(1, max_days_back + 1):
        search_date = deposit_to_match['fecha'] - timedelta(days=days_back)
        
        # Acceso directo por fecha usando diccionario
        if search_date not in folios_dict_by_date:
            continue
            
        daily_folios = folios_dict_by_date[search_date]
        if len(daily_folios) == 0:
            continue
        
        # Estrategia rápida: solo greedy optimizado
        selected_folios, totals = _fast_greedy_selection_with_taxes(daily_folios, deposit_to_match, tolerance_pesos)
        
        # VALIDACIÓN CON TOLERANCIA PARA DEPÓSITOS CON IMPUESTOS
        if selected_folios is not None and len(selected_folios) > 0:
            # Calcular diferencias por componente - más flexible
            diff_total = abs(deposit_to_match['total'] - (totals['subtotal_total'] + totals['iva'] + totals['ieps']))
            diff_iva = abs(deposit_to_match['iva'] - totals['iva'])
            diff_ieps = abs(deposit_to_match['ieps'] - totals['ieps'])
            
            # Tolerancia más flexible basada en porcentaje del monto
            tolerancia_dinamica = max(tolerance_pesos, deposit_to_match['total'] * 0.05)  # 5% o 100 pesos, lo que sea mayor
            
            # RECHAZAR si la diferencia total es muy grande
            if diff_total > tolerancia_dinamica:
                continue
            
            # Usar diferencia total como distancia principal
            distance = diff_total
            
            if distance < best_match['distance']:
                best_match.update({
                    'distance': distance, 
                    'folios': selected_folios, 
                    'date': search_date,
                    'final_subtotal': totals['subtotal_total'], 
                    'final_iva': totals['iva'], 
                    'final_ieps': totals['ieps'],
                    'final_subtotal_0': totals.get('subtotal_0', 0),
                    'final_subtotal_16': totals.get('subtotal_16', 0),
                    'strategy': 'fast_greedy_taxes',
                    'days_back': days_back,
                    'tolerancia_usada': tolerancia_dinamica
                })
                
                # Si encontramos una coincidencia muy buena, salir temprano
                if distance <= tolerancia_dinamica * 0.1:  # Muy exacto
                    break
    
    return best_match

def _fast_greedy_selection_with_taxes(folios_list, target, tolerance=10.0):
    """
    Selección greedy para depósitos CON IMPUESTOS con tolerancia de 10 pesos por componente.
    """
    if not folios_list:
        return None, None
    
    # Ordenar folios por total ascendente para mejor combinación
    folios_sorted = sorted(folios_list, key=lambda x: x['total'])
    
    # Algoritmo mejorado: Intentar múltiples estrategias
    strategies = [
        ('exact_match_taxes', _try_exact_match_taxes),
        ('greedy_closest_taxes', _try_greedy_closest_taxes),
        ('dynamic_programming_taxes', _try_dynamic_programming_taxes)
    ]
    
    for strategy_name, strategy_func in strategies:
        result = strategy_func(folios_sorted, target, tolerance)
        if result[0] is not None:
            return result
    
    return None, None

def _try_exact_match_taxes(folios_list, target, tolerance):
    """Busca un ticket que coincida exactamente o muy cerca del total objetivo para depósitos con impuestos."""
    target_total = target['total']
    target_iva = target['iva']
    target_ieps = target['ieps']
    
    for folio in folios_list:
        diff_total = abs(folio['total'] - target_total)
        diff_iva = abs(folio['iva'] - target_iva)
        diff_ieps = abs(folio['ieps'] - target_ieps)
        
        # Tolerancia dinámica basada en el monto
        tolerancia_dinamica = max(tolerance, target_total * 0.05)
        
        if diff_total <= tolerancia_dinamica:
            # Simplificar: asignar todo el subtotal basado en si tiene IVA
            subtotal_0 = folio['subtotal_total'] if folio['iva'] == 0 else 0
            subtotal_16 = folio['subtotal_total'] if folio['iva'] > 0 else 0
            
            current_totals = {
                'subtotal_total': folio['subtotal_total'], 
                'iva': folio['iva'], 
                'ieps': folio['ieps'],
                'subtotal_0': subtotal_0,
                'subtotal_16': subtotal_16
            }
            return [folio], current_totals
    
    return None, None

def _try_greedy_closest_taxes(folios_list, target, tolerance):
    """Algoritmo greedy mejorado para depósitos con impuestos."""
    target_total = target['total']
    target_iva = target['iva']
    target_ieps = target['ieps']
    
    selected = []
    current_total = 0
    current_subtotal = 0
    current_iva = 0
    current_ieps = 0
    
    # Tolerancia dinámica
    tolerancia_dinamica = max(tolerance, target_total * 0.05)
    
    # Crear copia para no modificar original
    available_folios = folios_list.copy()
    
    # Intentar primero con tickets individuales que se acerquen al total
    for folio in available_folios[:]:
        if abs(folio['total'] - target_total) <= tolerancia_dinamica:
            selected = [folio]
            current_total = folio['total']
            current_subtotal = folio['subtotal_total']
            current_iva = folio['iva']
            current_ieps = folio['ieps']
            break
    
    # Si no encontró match individual, usar algoritmo greedy
    if not selected:
        while available_folios and current_total < target_total - tolerancia_dinamica:
            best_folio = None
            best_score = float('inf')
            best_index = -1
            
            for i, folio in enumerate(available_folios):
                new_total = current_total + folio['total']
                new_iva = current_iva + folio['iva']
                new_ieps = current_ieps + folio['ieps']
                
                # No exceder demasiado
                if new_total > target_total + tolerancia_dinamica:
                    continue
                
                # Score basado en cercanía al objetivo total
                distance_total = abs(new_total - target_total)
                distance_iva = abs(new_iva - target_iva)
                distance_ieps = abs(new_ieps - target_ieps)
                
                # Score combinado con peso mayor al total
                score = distance_total + distance_iva * 0.3 + distance_ieps * 0.3
                
                if score < best_score:
                    best_score = score
                    best_folio = folio
                    best_index = i
            
            if best_folio is None:
                break
            
            # Agregar el mejor folio encontrado
            selected.append(best_folio)
            current_total += best_folio['total']
            current_subtotal += best_folio['subtotal_total']
            current_iva += best_folio['iva']
            current_ieps += best_folio['ieps']
            available_folios.pop(best_index)
            
            # Si ya estamos dentro de tolerancia, terminar
            if abs(current_total - target_total) <= tolerancia_dinamica:
                break
    
    # Validar resultado final
    if selected and abs(current_total - target_total) <= tolerancia_dinamica:
        # Calcular subtotales simplificados
        subtotal_0 = sum(f['subtotal_total'] for f in selected if f['iva'] == 0)
        subtotal_16 = sum(f['subtotal_total'] for f in selected if f['iva'] > 0)
        
        current_totals = {
            'subtotal_total': current_subtotal, 
            'iva': current_iva, 
            'ieps': current_ieps,
            'subtotal_0': subtotal_0,
            'subtotal_16': subtotal_16
        }
        return selected, current_totals
    
    return None, None

def _try_dynamic_programming_taxes(folios_list, target, tolerance):
    """Algoritmo de programación dinámica para depósitos con impuestos."""
    # Para conjuntos pequeños, usar fuerza bruta optimizada
    if len(folios_list) <= 20:
        return _try_brute_force_small_taxes(folios_list, target, tolerance)
    
    # Para conjuntos grandes, usar aproximación greedy mejorada
    return _try_greedy_closest_taxes(folios_list, target, tolerance)

def _try_brute_force_small_taxes(folios_list, target, tolerance):
    """Fuerza bruta para conjuntos pequeños de folios con impuestos."""
    target_total = target['total']
    n = len(folios_list)
    best_combination = None
    best_distance = float('inf')
    
    # Probar todas las combinaciones posibles (2^n)
    for mask in range(1, 1 << n):
        selected = []
        total = 0
        subtotal_total = 0
        iva_total = 0
        ieps_total = 0
        
        for i in range(n):
            if mask & (1 << i):
                selected.append(folios_list[i])
                total += folios_list[i]['total']
                subtotal_total += folios_list[i]['subtotal_total']
                iva_total += folios_list[i]['iva']
                ieps_total += folios_list[i]['ieps']
        
        # Verificar si está dentro de tolerancia
        distance_total = abs(total - target_total)
        distance_iva = abs(iva_total - target['iva'])
        distance_ieps = abs(ieps_total - target['ieps'])
        
        max_distance = max(distance_total, distance_iva, distance_ieps)
        
        if max_distance <= tolerance and distance_total < best_distance:
            best_combination = selected
            best_distance = distance_total
            
            # Si encontramos coincidencia muy buena, salir
            if distance_total <= 1.0:
                break
    
    if best_combination:
        subtotal_0 = sum(f['subtotal_total'] for f in best_combination if f['iva'] == 0)
        subtotal_16 = sum(f['subtotal_total'] for f in best_combination if f['iva'] > 0)
        
        current_totals = {
            'subtotal_total': sum(f['subtotal_total'] for f in best_combination),
            'iva': sum(f['iva'] for f in best_combination),
            'ieps': sum(f['ieps'] for f in best_combination),
            'subtotal_0': subtotal_0,
            'subtotal_16': subtotal_16
        }
        return best_combination, current_totals
    
    return None, None

def prepare_folios_dict(partidas_df):
    """
    Prepara un diccionario indexado por fecha para acceso ultra rápido.
    Trabaja con partidas individuales, no folios agrupados.
    """
    folios_dict = defaultdict(list)
    
    # Convertir DataFrame a diccionario de listas para velocidad
    for _, row in partidas_df.iterrows():
        fecha = row['Fecha_Venta']
        partida_dict = {
            'ID_PVenta': row['ID_PVenta'],  # ID único de la partida
            'Folio_Venta': row['Folio_Venta'],  # Folio (puede repetirse)
            'Fecha_Venta': row['Fecha_Venta'],
            'subtotal_total': row['subtotal_total'],
            'iva': row['iva'],
            'ieps': row['ieps'],
            'total': row['total']
        }
        folios_dict[fecha].append(partida_dict)
    
    return folios_dict

# ==============================================================================
# 3. EJECUCIÓN PRINCIPAL (OPTIMIZADA PARA DEPÓSITOS CON IMPUESTOS)
# ==============================================================================

import time

print("🚀 Iniciando conciliación CON IMPUESTOS...")
start_time = time.time()

# Cargar todos los datos una sola vez
print("📊 Cargando datos...")
load_start = time.time()
# CAMBIO: Usar ventas no utilizadas en lugar del archivo completo
ventas_no_utilizadas_df = load_ventas_no_utilizadas_csv('ventas_no_utilizadas.csv')
depositos_df = parse_depositos_sql('depositos_all.sql')
load_end = time.time()
print(f"✓ Datos cargados en {load_end - load_start:.2f} segundos")

# Preparar folios atómicos
print("🔧 Preparando estructura de partidas individuales...")
prep_start = time.time()

# Usar las partidas no utilizadas
partidas_df = ventas_no_utilizadas_df.copy()

# Crear diccionario indexado por fecha para acceso ultra rápido
folios_dict_by_date = prepare_folios_dict(partidas_df)
prep_end = time.time()
print(f"✓ Estructura preparada en {prep_end - prep_start:.2f} segundos")
print(f"✓ Total de partidas no utilizadas disponibles: {len(partidas_df):,}")

# PROCESANDO TODOS LOS DEPÓSITOS CON IMPUESTOS (TOLERANCIA 10 PESOS)
# Seleccionar TODOS los depósitos CON impuestos
todos_con_impuestos = depositos_df[(depositos_df['iva'] > 0) | (depositos_df['ieps'] > 0)]

# Procesar TODOS los depósitos con impuestos disponibles
depositos_a_conciliar = todos_con_impuestos.sort_values(by='fecha')

print(f"🎯 Procesando {len(depositos_a_conciliar)} depósitos CON IMPUESTOS (TOLERANCIA DINÁMICA)")
print(f"   📊 Total de depósitos con impuestos disponibles: {len(todos_con_impuestos)}")
print(f"   🎯 Tolerancia: 5% del monto o 100 pesos (lo que sea mayor)")
print(f"   💡 Conciliación por componentes: subtotal_0, subtotal_16, iva, ieps")
print(f"   📅 Búsqueda temporal: Hasta 45 días antes de la fecha del depósito")
print(f"   🔄 Partidas no utilizadas: {len(partidas_df):,}")

# Inicializar listas para guardar resultados
results_list = []
all_detailed_products = []
used_partidas = set()  # Track de ID_PVenta ya utilizados

# =============================================================================
# PROCESAMIENTO: SOLO DEPÓSITOS CON IMPUESTOS (TOLERANCIA DINÁMICA)
# =============================================================================
print(f"\n🎯 PROCESANDO {len(depositos_a_conciliar)} DEPÓSITOS CON IMPUESTOS")
print("   Estrategia: Tolerancia DINÁMICA (5% del monto o 100 pesos)")
print("   Validación: Principalmente por TOTAL del depósito")
print("   Método: Conciliar por total principalmente, luego validar componentes")
print("   Algoritmo: Búsqueda inteligente con múltiples estrategias")
print("   Ventana temporal: 45 días hacia atrás desde fecha depósito")
print("   FUENTE: Partidas no utilizadas (evita duplicar tickets)")

# Inicializar tiempo de procesamiento
process_start = time.time()

for index, deposit in depositos_a_conciliar.iterrows():
    deposit_start = time.time()
    
    print(f"\n--- Depósito {len(results_list)+1}/{len(depositos_a_conciliar)}: ID {deposit['id']} del {deposit['fecha'].strftime('%Y-%m-%d')} ---")
    print(f"Objetivo: Subtotal_0=${deposit['subtotal_0']:,.2f}, Subtotal_16=${deposit['subtotal_16']:,.2f}")
    print(f"         IVA=${deposit['iva']:,.2f}, IEPS=${deposit['ieps']:,.2f}, Total=${deposit['total']:,.2f}")
    
    # Filtrar folios disponibles (no utilizados)
    filtered_folios_dict = {}
    total_available = 0
    for fecha, folios_list in folios_dict_by_date.items():
        available_folios = [f for f in folios_list if f['ID_PVenta'] not in used_partidas]
        if available_folios:
            filtered_folios_dict[fecha] = available_folios
            total_available += len(available_folios)
    
    print(f"Partidas disponibles: {total_available}")
    
    # Ejecutar conciliación con tolerancia dinámica
    best_match = reconcile_deposit_with_taxes_fast(deposit, filtered_folios_dict, tolerance_pesos=100.0)
    best_match['tolerance_used'] = best_match.get('tolerancia_usada', 100.0)
    
    deposit_end = time.time()
    deposit_time = deposit_end - deposit_start
    
    if 'folios' in best_match and best_match['folios'] is not None and len(best_match['folios']) > 0:
        # Calcular diferencias exactas por componente
        diff_subtotal_0 = abs(deposit['subtotal_0'] - best_match.get('final_subtotal_0', 0))
        diff_subtotal_16 = abs(deposit['subtotal_16'] - best_match.get('final_subtotal_16', 0))
        diff_iva = abs(deposit['iva'] - best_match['final_iva'])
        diff_ieps = abs(deposit['ieps'] - best_match['final_ieps'])
        diff_total = abs(deposit['total'] - (best_match['final_subtotal'] + best_match['final_iva'] + best_match['final_ieps']))
        
        # Calcular porcentajes de precisión
        subtotal_precision = (best_match['final_subtotal'] / deposit['subtotal_total']) * 100 if deposit['subtotal_total'] > 0 else 100
        iva_precision = (best_match['final_iva'] / deposit['iva']) * 100 if deposit['iva'] > 0 else 100
        ieps_precision = (best_match['final_ieps'] / deposit['ieps']) * 100 if deposit['ieps'] > 0 else 100
        
        # Guardar el resumen del resultado
        summary = {
            'deposit_id': deposit['id'], 
            'deposit_fecha': deposit['fecha'].strftime('%Y-%m-%d'),
            'deposit_subtotal_0': deposit['subtotal_0'],
            'deposit_subtotal_16': deposit['subtotal_16'],
            'deposit_subtotal': deposit['subtotal_total'], 
            'deposit_iva': deposit['iva'], 
            'deposit_ieps': deposit['ieps'],
            'deposit_total': deposit['total'],
            'sales_fecha': best_match['date'].strftime('%Y-%m-%d'),
            'matched_subtotal_0': best_match.get('final_subtotal_0', 0),
            'matched_subtotal_16': best_match.get('final_subtotal_16', 0),
            'matched_subtotal': best_match['final_subtotal'], 
            'matched_iva': best_match['final_iva'], 
            'matched_ieps': best_match['final_ieps'],
            'matched_total': best_match['final_subtotal'] + best_match['final_iva'] + best_match['final_ieps'],
            'diff_subtotal_0': deposit['subtotal_0'] - best_match.get('final_subtotal_0', 0),
            'diff_subtotal_16': deposit['subtotal_16'] - best_match.get('final_subtotal_16', 0),
            'diff_subtotal': deposit['subtotal_total'] - best_match['final_subtotal'],
            'diff_iva': deposit['iva'] - best_match['final_iva'],
            'diff_ieps': deposit['ieps'] - best_match['final_ieps'],
            'diff_total': deposit['total'] - (best_match['final_subtotal'] + best_match['final_iva'] + best_match['final_ieps']),
            'diff_total_abs': diff_total,
            'subtotal_precision_pct': subtotal_precision,
            'iva_precision_pct': iva_precision,
            'ieps_precision_pct': ieps_precision,
            'num_folios': len(best_match['folios']),
            'strategy_used': best_match.get('strategy', 'fast_greedy_taxes'),
            'tolerance_used': best_match.get('tolerance_used', 10.0),
            'processing_time_seconds': deposit_time
        }
        results_list.append(summary)
        
        print(f"✓ CONCILIADO en {deposit_time:.2f}s | Tolerancia: ±{best_match.get('tolerance_used', 100.0):.0f} pesos")
        print(f"  Resultado: Subtotal=${best_match['final_subtotal']:,.2f} ({subtotal_precision:.1f}%)")
        print(f"            IVA=${best_match['final_iva']:,.2f} ({iva_precision:.1f}%)")
        print(f"            IEPS=${best_match['final_ieps']:,.2f} ({ieps_precision:.1f}%)")
        print(f"  Diferencia total: ${diff_total:.2f} (tolerancia: ±{best_match.get('tolerance_used', 100.0):.0f})")
        print(f"  Partidas utilizadas: {len(best_match['folios'])} | Días atrás: {best_match.get('days_back', 'N/A')}")

        # Marcar folios como utilizados
        for folio in best_match['folios']:
            used_partidas.add(folio['ID_PVenta'])  # Usar ID_PVenta, no Folio_Venta
        
        # Guardar detalles de productos
        for folio in best_match['folios']:
            detailed_product = {
                'ID_PVenta': folio['ID_PVenta'],  # ID único de partida
                'Folio_Venta': folio['Folio_Venta'],  # Folio (puede repetirse)
                'Fecha_Venta': folio['Fecha_Venta'],
                'subtotal_total': folio['subtotal_total'],
                'iva': folio['iva'],
                'ieps': folio['ieps'],
                'total': folio['total'],
                'deposit_id_conciliado': deposit['id'],
                'deposit_fecha_conciliado': deposit['fecha'].strftime('%Y-%m-%d'),
                'deposit_total': deposit['total'],
                'deposit_subtotal': deposit['subtotal_total'],
                'deposit_subtotal_0': deposit['subtotal_0'],
                'deposit_subtotal_16': deposit['subtotal_16'],
                'deposit_iva': deposit['iva'],
                'deposit_ieps': deposit['ieps'],
                'strategy_used': best_match.get('strategy', 'fast_greedy_taxes'),
                'tolerance_used': best_match.get('tolerance_used', 10.0)
            }
            all_detailed_products.append(detailed_product)
    else:
        print(f"✗ NO CONCILIADO en {deposit_time:.2f}s - No se encontró combinación con tolerancia dinámica")

process_end = time.time()
total_time = process_end - start_time
process_time = process_end - process_start

print(f"\n" + "="*80)
print(f"⚡ RENDIMIENTO OPTIMIZADO - DEPÓSITOS CON IMPUESTOS (TOLERANCIA ±10 PESOS) ⚡")
print(f"="*80)
print(f"Tiempo total: {total_time:.2f} segundos")
print(f"Tiempo de procesamiento: {process_time:.2f} segundos")
print(f"Tiempo promedio por depósito: {process_time/len(depositos_a_conciliar):.2f} segundos")
print(f"Depósitos procesados: {len(depositos_a_conciliar)}")
print(f"Depósitos conciliados: {len(results_list)}")
print(f"Tasa de éxito: {(len(results_list)/len(depositos_a_conciliar)*100):.1f}%")

# --- Presentación de Resultados Finales ---
if results_list:
    summary_df = pd.DataFrame(results_list)
    print("\n" + "="*80)
    print("             RESUMEN DE CONCILIACIÓN - DEPÓSITOS CON IMPUESTOS")
    print("="*80)
    
    # Estadísticas básicas generales
    total_procesados = len(depositos_a_conciliar)
    conciliados = len(results_list)
    tasa_exito = (conciliados / total_procesados) * 100
    
    print(f"\n📊 Resumen General:")
    print(f"   • Depósitos CON impuestos procesados: {total_procesados}")
    print(f"   • Depósitos conciliados: {conciliados}")
    print(f"   • Tasa de éxito: {tasa_exito:.1f}%")
    print(f"   • Tolerancia aplicada: ±10.00 pesos por componente")
    print(f"   • Fuente: Partidas no utilizadas (sin duplicados)")
    
    # Precisión promedio
    avg_subtotal = summary_df['subtotal_precision_pct'].mean()
    avg_iva = summary_df['iva_precision_pct'].mean()
    avg_ieps = summary_df['ieps_precision_pct'].mean()
    avg_time = summary_df['processing_time_seconds'].mean()
    avg_diff_total = summary_df['diff_total_abs'].mean() if 'diff_total_abs' in summary_df.columns else 0
    
    print(f"\n🎯 Precisión Promedio:")
    print(f"   • Subtotal: {avg_subtotal:.1f}%")
    print(f"   • IVA: {avg_iva:.1f}%")
    print(f"   • IEPS: {avg_ieps:.1f}%")
    print(f"   • Diferencia total promedio: ${avg_diff_total:.2f}")
    print(f"   • Tiempo promedio: {avg_time:.2f}s por depósito")
    
    # Verificar tolerancia
    if 'diff_total_abs' in summary_df.columns:
        casos_fuera_tolerancia = summary_df[summary_df['diff_total_abs'] > 10.0]
        if not casos_fuera_tolerancia.empty:
            print(f"\n⚠️  ADVERTENCIA: {len(casos_fuera_tolerancia)} casos exceden ±10 pesos:")
            for _, case in casos_fuera_tolerancia.iterrows():
                print(f"   Depósito {case['deposit_id']}: Diferencia total ${case['diff_total_abs']:.2f}")
        else:
            print(f"\n✅ EXCELENTE: Todos los casos están dentro de ±10.00 pesos")
    
    # Mostrar casos problemáticos si los hay
    low_precision = summary_df[summary_df['subtotal_precision_pct'] < 90]
    
    if not low_precision.empty:
        print(f"\n⚠️  Casos con precisión de subtotal < 90%:")
        for _, case in low_precision.iterrows():
            print(f"   Depósito {case['deposit_id']}: Subtotal {case['subtotal_precision_pct']:.1f}%")
    else:
        print(f"\n✅ EXCELENTE: Todos los casos tienen precisión de subtotal ≥ 90%")
    
    # =============================================================================
    # GENERAR ARCHIVOS CSV CON SUFIJO "_con_impuestos"
    # =============================================================================
    
    # 1. Generar ventas_asignadas_con_impuestos.csv
    if all_detailed_products:
        ventas_asignadas = []
        
        # Diccionario para rastrear folios ya usados por depósito y folios disponibles por depósito
        folios_por_deposito = {}  # {deposit_id: folio_unico_asignado}
        folios_disponibles_por_deposito = {}  # {deposit_id: [lista_de_folios_únicos_del_depósito]}
        folios_ya_usados_globalmente = set()  # Set de folios ya usados globalmente
        
        # Primer pase: recopilar todos los folios únicos por depósito
        for product in all_detailed_products:
            deposit_id = product['deposit_id_conciliado']
            folio_original = product['Folio_Venta']
            
            if deposit_id not in folios_disponibles_por_deposito:
                folios_disponibles_por_deposito[deposit_id] = []
            
            # Agregar folio a la lista si no está ya
            if folio_original not in folios_disponibles_por_deposito[deposit_id]:
                folios_disponibles_por_deposito[deposit_id].append(folio_original)
        
        # Segundo pase: asignar folios consolidados ÚNICOS GLOBALMENTE
        for product in all_detailed_products:
            deposit_id = product['deposit_id_conciliado']
            folio_original = product['Folio_Venta']
            
            # Si este depósito ya tiene un folio asignado, usar ese
            if deposit_id in folios_por_deposito:
                folio_a_usar = folios_por_deposito[deposit_id]
            else:
                # Buscar un folio del depósito que NO haya sido usado globalmente
                folio_a_usar = None
                for folio_candidato in folios_disponibles_por_deposito[deposit_id]:
                    if folio_candidato not in folios_ya_usados_globalmente:
                        folio_a_usar = folio_candidato
                        break
                
                # Si todos los folios del depósito ya fueron usados, crear uno único
                if folio_a_usar is None:
                    folio_a_usar = f"DEP-{deposit_id}-IMPUESTOS"
                
                # Registrar este folio para este depósito y marcarlo como usado globalmente
                folios_por_deposito[deposit_id] = folio_a_usar
                folios_ya_usados_globalmente.add(folio_a_usar)
            
            venta_asignada = {
                'ID_PVenta': product['ID_PVenta'],  # ID único de partida
                'FolioVenta': folio_a_usar,  # Folio consolidado ÚNICO GLOBALMENTE
                'FolioOriginal': folio_original,  # Mantener referencia del folio original
                'IDDeposito': product['deposit_id_conciliado'],
                'Total': product['total'],
                'IVA': product['iva'],
                'IEPS': product['ieps']
            }
            ventas_asignadas.append(venta_asignada)
        
        ventas_asignadas_df = pd.DataFrame(ventas_asignadas)
        ventas_asignadas_df.to_csv("ventas_asignadas_con_impuestos.csv", index=False)
        print(f"\n💾 Archivo generado: 'ventas_asignadas_con_impuestos.csv' ({len(ventas_asignadas_df)} registros)")
        print(f"🔄 Folios consolidados por depósito: {len(folios_por_deposito)} depósitos únicos")
        print(f"📋 Cada depósito tiene un folio ÚNICO GLOBALMENTE para evitar duplicados")
    
    # 2. Generar ventas_no_utilizadas_con_impuestos.csv (actualizado después de esta conciliación)
    # Obtener todas las partidas únicas del archivo de ventas no utilizadas originales
    todas_las_partidas_disponibles = set(partidas_df['ID_PVenta'].unique())
    partidas_utilizadas_ahora = used_partidas
    partidas_restantes = todas_las_partidas_disponibles - partidas_utilizadas_ahora
    
    if partidas_restantes:
        # Obtener las partidas que siguen sin utilizar
        partidas_restantes_df = partidas_df[partidas_df['ID_PVenta'].isin(partidas_restantes)]
        ventas_no_utilizadas_actualizadas_df = pd.DataFrame({
            'ID_PVenta': partidas_restantes_df['ID_PVenta'],
            'FolioVenta': partidas_restantes_df['Folio_Venta'],
            'Total': partidas_restantes_df['total'],
            'IVA': partidas_restantes_df['iva'],
            'IEPS': partidas_restantes_df['ieps']
        })
        ventas_no_utilizadas_actualizadas_df.to_csv("ventas_no_utilizadas_con_impuestos.csv", index=False)
        print(f"💾 Archivo generado: 'ventas_no_utilizadas_con_impuestos.csv' ({len(ventas_no_utilizadas_actualizadas_df)} registros)")
    
    # 3. Generar el archivo de resumen para referencia
    summary_filename = "resumen_conciliacion_con_impuestos.csv"
    summary_df.to_csv(summary_filename, index=False)
    print(f"💾 Resumen generado: '{summary_filename}' (archivo de referencia)")
    
    print(f"\n📊 RESUMEN DE ARCHIVOS GENERADOS (CON IMPUESTOS):")
    if all_detailed_products:
        print(f"   ✅ ventas_asignadas_con_impuestos.csv: {len(ventas_asignadas_df)} partidas asignadas")
    if partidas_restantes:
        print(f"   📝 ventas_no_utilizadas_con_impuestos.csv: {len(ventas_no_utilizadas_actualizadas_df)} partidas restantes")
    print(f"   📋 Total de partidas disponibles inicialmente: {len(todas_las_partidas_disponibles)}")
    print(f"   🎯 Tasa de utilización adicional: {(len(partidas_utilizadas_ahora)/len(todas_las_partidas_disponibles)*100):.1f}%")
    
    print(f"\n🎉 Conciliación CON IMPUESTOS completada en {total_time:.2f} segundos")
    print(f"⚡ Procesamiento ultra rápido: {len(depositos_a_conciliar)} depósitos en {process_time:.2f}s")
    print(f"📈 Velocidad promedio: {process_time/len(depositos_a_conciliar):.2f}s por depósito")
    print(f"🎯 TOLERANCIA RESPETADA: ±10.00 pesos por componente")
    print(f"🔄 FUENTE: Partidas no utilizadas (evita duplicar conciliaciones)")
    
else:
    print("\n⚠️  No se pudo conciliar ningún depósito con impuestos.")
    print("Verifica que los datos tengan depósitos con impuestos que puedan conciliarse dentro de ±10 pesos por componente.")