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

def load_tickets_csv(file_path):
    """
    Carga los datos de ventas desde el archivo CSV - OPTIMIZADO.
    CORREGIDO: Cada fila es una partida única con su propio ID.
    """
    # Leer solo las columnas necesarias para mejor rendimiento
    cols_needed = ['Fecha_Venta', 'Folio_Venta', 'Subtotal', 'taza_iva', 'taza_ieps', 'Total']
    
    df = pd.read_csv(file_path, quotechar='"', usecols=cols_needed)
    df.rename(columns={
        'Subtotal': 'subtotal_total',
        'taza_iva': 'iva',
        'taza_ieps': 'ieps',
        'Total': 'total'
    }, inplace=True)
    
    # Convertir fecha una sola vez
    df['Fecha_Venta'] = pd.to_datetime(df['Fecha_Venta'])
    
    # IMPORTANTE: Crear ID único para cada partida (cada fila es una partida diferente)
    df['ID_PVenta'] = df.index  # Cada fila tiene su propio ID único
    
    return df

# ==============================================================================
# 2. FUNCIÓN PRINCIPAL DE CONCILIACIÓN (ULTRA OPTIMIZADA)
# ==============================================================================

def reconcile_deposit_fast(deposit_to_match, folios_dict_by_date, max_days_back=45, tolerance_pct=1.0):
    """
    Versión ultra optimizada de la conciliación usando diccionarios indexados.
    CON VALIDACIÓN ULTRA ESTRICTA DE TOLERANCIA ±1 PESO.
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
        selected_folios, totals = _fast_greedy_selection(daily_folios, deposit_to_match, tolerance_pct)
        
        # VALIDACIÓN ULTRA ESTRICTA: Solo aceptar si está EXACTAMENTE dentro de tolerancia
        if selected_folios is not None and len(selected_folios) > 0:
            # VALIDACIÓN CRÍTICA: Verificar que el TOTAL esté dentro de ±1 peso
            total_calculado = totals['subtotal_total'] + totals['iva'] + totals['ieps']
            total_esperado = deposit_to_match['total']
            diff_total = abs(total_calculado - total_esperado)
            
            # RECHAZAR INMEDIATAMENTE si la diferencia total excede 1 peso
            if diff_total > tolerance_pct:
                continue  # NO aceptar esta combinación
            
            # VALIDACIÓN ESPECIAL PARA DEPÓSITOS SIN IMPUESTOS
            if deposit_to_match['iva'] == 0 and deposit_to_match['ieps'] == 0:
                # Para depósitos sin impuestos, IVA e IEPS deben ser exactamente 0
                if totals['iva'] != 0 or totals['ieps'] != 0:
                    continue  # Rechazar si hay impuestos cuando no debería haberlos
            
            # Solo llegar aquí si TODAS las diferencias están dentro de tolerancia
            distance = diff_total  # Usar diferencia total como distancia principal
            
            if distance < best_match['distance']:
                best_match.update({
                    'distance': distance, 
                    'folios': selected_folios, 
                    'date': search_date,
                    'final_subtotal': totals['subtotal_total'], 
                    'final_iva': totals['iva'], 
                    'final_ieps': totals['ieps'],
                    'strategy': 'fast_greedy',
                    'days_back': days_back
                })
                
                # Si encontramos una coincidencia perfecta, salir temprano
                if distance <= 0.01:  # Prácticamente exacto
                    break
    
    return best_match

def _fast_greedy_selection(folios_list, target, tolerance=1.0):
    """
    Selección greedy ultra rápida con validación ULTRA ESTRICTA de tolerancia ±1 peso.
    MEJORADO: Algoritmo más inteligente para combinar tickets optimizando por total.
    """
    if not folios_list:
        return None, None
    
    # FILTRO ESPECIAL: Si el depósito no tiene impuestos, solo usar folios sin impuestos
    if target['iva'] == 0 and target['ieps'] == 0:
        folios_list = [f for f in folios_list if f['iva'] == 0 and f['ieps'] == 0]
        if not folios_list:
            return None, None
    
    # Ordenar folios por total ascendente para mejor combinación
    folios_sorted = sorted(folios_list, key=lambda x: x['total'])
    
    # Algoritmo mejorado: Intentar múltiples estrategias
    strategies = [
        ('exact_match', _try_exact_match),
        ('greedy_closest', _try_greedy_closest),
        ('dynamic_programming', _try_dynamic_programming)
    ]
    
    for strategy_name, strategy_func in strategies:
        result = strategy_func(folios_sorted, target, tolerance)
        if result[0] is not None:
            return result
    
    return None, None

def _try_exact_match(folios_list, target, tolerance):
    """Busca un ticket que coincida exactamente o muy cerca del total objetivo."""
    target_total = target['total']
    
    for folio in folios_list:
        if abs(folio['total'] - target_total) <= tolerance:
            # VALIDACIÓN ESPECIAL para depósitos sin impuestos
            if target['iva'] == 0 and target['ieps'] == 0:
                if folio['iva'] != 0 or folio['ieps'] != 0:
                    continue
            
            final_subtotal = folio['subtotal_total']
            final_iva = folio['iva']
            final_ieps = folio['ieps']
            
            current_totals = {'subtotal_total': final_subtotal, 'iva': final_iva, 'ieps': final_ieps}
            return [folio], current_totals
    
    return None, None

def _try_greedy_closest(folios_list, target, tolerance):
    """Algoritmo greedy mejorado que busca la mejor combinación."""
    target_total = target['total']
    selected = []
    current_total = 0
    
    # Crear copia para no modificar original
    available_folios = folios_list.copy()
    
    while available_folios and current_total < target_total - tolerance:
        best_folio = None
        best_score = float('inf')
        best_index = -1
        
        for i, folio in enumerate(available_folios):
            new_total = current_total + folio['total']
            
            # No exceder el límite máximo
            if new_total > target_total + tolerance:
                continue
            
            # Calcular qué tan cerca nos lleva al objetivo
            distance_to_target = abs(new_total - target_total)
            
            # Penalizar si se queda muy lejos
            if new_total < target_total - tolerance:
                score = distance_to_target + (target_total - new_total) * 0.1
            else:
                score = distance_to_target
            
            if score < best_score:
                best_score = score
                best_folio = folio
                best_index = i
        
        if best_folio is None:
            break
        
        # Agregar el mejor folio encontrado
        selected.append(best_folio)
        current_total += best_folio['total']
        available_folios.pop(best_index)
        
        # Si ya estamos dentro de tolerancia, terminar
        if abs(current_total - target_total) <= tolerance:
            break
    
    # Validar resultado final
    if selected and abs(current_total - target_total) <= tolerance:
        # VALIDACIÓN ESPECIAL para depósitos sin impuestos
        if target['iva'] == 0 and target['ieps'] == 0:
            total_iva = sum(f['iva'] for f in selected)
            total_ieps = sum(f['ieps'] for f in selected)
            if total_iva != 0 or total_ieps != 0:
                return None, None
        
        final_subtotal = sum(f['subtotal_total'] for f in selected)
        final_iva = sum(f['iva'] for f in selected)
        final_ieps = sum(f['ieps'] for f in selected)
        
        current_totals = {'subtotal_total': final_subtotal, 'iva': final_iva, 'ieps': final_ieps}
        return selected, current_totals
    
    return None, None

def _try_dynamic_programming(folios_list, target, tolerance):
    """Algoritmo de programación dinámica para encontrar la mejor combinación."""
    target_total = target['total']
    min_target = target_total - tolerance
    max_target = target_total + tolerance
    
    # Usar subset sum approach con tolerancia
    n = len(folios_list)
    if n == 0:
        return None, None
    
    # Para conjuntos pequeños, usar fuerza bruta optimizada
    if n <= 20:
        return _try_brute_force_small(folios_list, target, tolerance)
    
    # Para conjuntos grandes, usar aproximación greedy mejorada
    return _try_greedy_closest(folios_list, target, tolerance)

def _try_brute_force_small(folios_list, target, tolerance):
    """Fuerza bruta para conjuntos pequeños de folios."""
    target_total = target['total']
    n = len(folios_list)
    best_combination = None
    best_distance = float('inf')
    
    # Probar todas las combinaciones posibles (2^n)
    for mask in range(1, 1 << n):
        selected = []
        total = 0
        
        for i in range(n):
            if mask & (1 << i):
                selected.append(folios_list[i])
                total += folios_list[i]['total']
        
        # Verificar si está dentro de tolerancia
        distance = abs(total - target_total)
        if distance <= tolerance and distance < best_distance:
            # VALIDACIÓN ESPECIAL para depósitos sin impuestos
            if target['iva'] == 0 and target['ieps'] == 0:
                total_iva = sum(f['iva'] for f in selected)
                total_ieps = sum(f['ieps'] for f in selected)
                if total_iva != 0 or total_ieps != 0:
                    continue
            
            best_combination = selected
            best_distance = distance
            
            # Si encontramos coincidencia exacta, salir
            if distance <= 0.01:
                break
    
    if best_combination:
        final_subtotal = sum(f['subtotal_total'] for f in best_combination)
        final_iva = sum(f['iva'] for f in best_combination)
        final_ieps = sum(f['ieps'] for f in best_combination)
        
        current_totals = {'subtotal_total': final_subtotal, 'iva': final_iva, 'ieps': final_ieps}
        return best_combination, current_totals
    
    return None, None

def prepare_folios_dict(partidas_df):
    """
    Prepara un diccionario indexado por fecha para acceso ultra rápido.
    CORREGIDO: Trabaja con partidas individuales, no folios agrupados.
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
# 3. EJECUCIÓN PRINCIPAL (OPTIMIZADA PARA VELOCIDAD)
# ==============================================================================

import time

print("🚀 Iniciando conciliación optimizada...")
start_time = time.time()

# Cargar todos los datos una sola vez
print("📊 Cargando datos...")
load_start = time.time()
all_products_df = load_tickets_csv('TICKETS_JUN_JUL.csv')
depositos_df = parse_depositos_sql('depositos_all.sql')
load_end = time.time()
print(f"✓ Datos cargados en {load_end - load_start:.2f} segundos")

# Preparar folios atómicos
print("🔧 Preparando estructura de partidas individuales...")
prep_start = time.time()

# CAMBIO CRÍTICO: NO agrupar por folio, cada fila es una partida independiente
# Los folios pueden repetirse, pero cada partida (ID_PVenta) es única
partidas_df = all_products_df.copy()  # Usar todas las partidas individuales

# Crear diccionario indexado por fecha para acceso ultra rápido
folios_dict_by_date = prepare_folios_dict(partidas_df)
prep_end = time.time()
print(f"✓ Estructura preparada en {prep_end - prep_start:.2f} segundos")
print(f"✓ Total de partidas individuales disponibles: {len(partidas_df):,}")

# PROCESANDO 100 DEPÓSITOS ALEATORIOS (SOLO SIN IMPUESTOS - MÁXIMA PRECISIÓN)
# Seleccionar SOLO depósitos sin impuestos para máxima precisión
todos_sin_impuestos = depositos_df[(depositos_df['iva'] == 0) & (depositos_df['ieps'] == 0)]

# Seleccionar aleatoriamente 100 depósitos sin impuestos (máximo disponible)
num_depositos = min(100, len(todos_sin_impuestos))
depositos_a_conciliar = todos_sin_impuestos.sample(n=num_depositos, random_state=42).sort_values(by='fecha')

print(f"🎯 Procesando {len(depositos_a_conciliar)} depósitos SIN IMPUESTOS (TOLERANCIA ESTRICTA ±1 PESO)")
print(f"   📊 Total de depósitos sin impuestos disponibles: {len(todos_sin_impuestos)}")
print(f"   🎯 Tolerancia máxima permitida: ±1.00 peso")
print(f"   💡 Conciliación por TOTAL: Total depósito = Total tickets")
print(f"   📅 Búsqueda temporal: Hasta 45 días antes de la fecha del depósito")
print(f"   🔄 Partidas individuales: {len(partidas_df):,} (folios pueden repetirse)")

# Inicializar listas para guardar resultados
results_list = []
all_detailed_products = []
used_partidas = set()  # Track de ID_PVenta ya utilizados (no folios)

# =============================================================================
# PROCESAMIENTO ÚNICO: SOLO DEPÓSITOS SIN IMPUESTOS (TOLERANCIA ESTRICTA ±1 PESO)
# =============================================================================
print(f"\n🎯 PROCESANDO {len(depositos_a_conciliar)} DEPÓSITOS SIN IMPUESTOS")
print("   Estrategia: Tolerancia ESTRICTA de ±1.00 peso")
print("   Validación: El TOTAL debe coincidir dentro de ±1 peso exacto")
print("   Método: Conciliar TOTAL del depósito con TOTAL de tickets")
print("   Algoritmo: Búsqueda inteligente con múltiples estrategias")
print("   Ventana temporal: 45 días hacia atrás desde fecha depósito")
print("   IMPORTANTE: Usar partidas individuales (ID_PVenta único, folios repetibles)")

# Inicializar tiempo de procesamiento
process_start = time.time()

for index, deposit in depositos_a_conciliar.iterrows():
    deposit_start = time.time()
    
    print(f"\n--- Depósito {len(results_list)+1}/{len(depositos_a_conciliar)}: ID {deposit['id']} del {deposit['fecha'].strftime('%Y-%m-%d')} ---")
    print(f"Objetivo: Subtotal=${deposit['subtotal_total']:,.2f}, Total=${deposit['total']:,.2f}")
    
    # Filtrar folios disponibles (no utilizados)
    filtered_folios_dict = {}
    total_available = 0
    for fecha, folios_list in folios_dict_by_date.items():
        available_folios = [f for f in folios_list if f['ID_PVenta'] not in used_partidas]
        if available_folios:
            filtered_folios_dict[fecha] = available_folios
            total_available += len(available_folios)
    
    print(f"Folios disponibles: {total_available}")
    
    # Ejecutar conciliación con tolerancia ESTRICTA de 1 peso
    best_match = reconcile_deposit_fast(deposit, filtered_folios_dict, tolerance_pct=1.0)
    best_match['tolerance_used'] = 1.0
    
    deposit_end = time.time()
    deposit_time = deposit_end - deposit_start
    
    if 'folios' in best_match and best_match['folios'] is not None and len(best_match['folios']) > 0:
        # Calcular diferencias exactas
        diff_subtotal = abs(deposit['subtotal_total'] - best_match['final_subtotal'])
        diff_total = abs(deposit['total'] - (best_match['final_subtotal'] + best_match['final_iva'] + best_match['final_ieps']))
        
        # Calcular porcentajes de precisión
        subtotal_precision = (best_match['final_subtotal'] / deposit['subtotal_total']) * 100
        
        # Guardar el resumen del resultado
        summary = {
            'deposit_id': deposit['id'], 
            'deposit_fecha': deposit['fecha'].strftime('%Y-%m-%d'),
            'deposit_subtotal': deposit['subtotal_total'], 
            'deposit_iva': deposit['iva'], 
            'deposit_ieps': deposit['ieps'],
            'deposit_total': deposit['total'],
            'sales_fecha': best_match['date'].strftime('%Y-%m-%d'),
            'matched_subtotal': best_match['final_subtotal'], 
            'matched_iva': best_match['final_iva'], 
            'matched_ieps': best_match['final_ieps'],
            'matched_total': best_match['final_subtotal'] + best_match['final_iva'] + best_match['final_ieps'],
            'diff_subtotal': deposit['subtotal_total'] - best_match['final_subtotal'],
            'diff_iva': deposit['iva'] - best_match['final_iva'],
            'diff_ieps': deposit['ieps'] - best_match['final_ieps'],
            'diff_total': deposit['total'] - (best_match['final_subtotal'] + best_match['final_iva'] + best_match['final_ieps']),
            'diff_total_abs': diff_total,
            'subtotal_precision_pct': subtotal_precision,
            'iva_precision_pct': 100.0,  # Siempre 100% para depósitos sin impuestos
            'ieps_precision_pct': 100.0,  # Siempre 100% para depósitos sin impuestos
            'num_folios': len(best_match['folios']),
            'strategy_used': best_match.get('strategy', 'fast_greedy'),
            'tolerance_used': best_match.get('tolerance_used', 1.0),
            'processing_time_seconds': deposit_time
        }
        results_list.append(summary)
        
        print(f"✓ CONCILIADO en {deposit_time:.2f}s | Tolerancia: ±{best_match.get('tolerance_used', 1.0)} pesos")
        print(f"  Resultado: Subtotal=${best_match['final_subtotal']:,.2f} ({subtotal_precision:.1f}%)")
        print(f"  Diferencia total: ${diff_total:.2f} (dentro de ±1.00 peso)")
        print(f"  Folios utilizados: {len(best_match['folios'])} | Días atrás: {best_match.get('days_back', 'N/A')}")

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
                'deposit_iva': deposit['iva'],
                'deposit_ieps': deposit['ieps'],
                'strategy_used': best_match.get('strategy', 'fast_greedy'),
                'tolerance_used': best_match.get('tolerance_used', 1.0)
            }
            all_detailed_products.append(detailed_product)
    else:
        print(f"✗ NO CONCILIADO en {deposit_time:.2f}s - No se encontró combinación dentro de ±1 peso")

process_end = time.time()
total_time = process_end - start_time
process_time = process_end - process_start

print(f"\n" + "="*80)
print(f"⚡ RENDIMIENTO OPTIMIZADO - SOLO DEPÓSITOS SIN IMPUESTOS (TOLERANCIA ±1 PESO) ⚡")
print(f"="*80)
print(f"Tiempo total: {total_time:.2f} segundos")
print(f"Tiempo de procesamiento: {process_time:.2f} segundos")
print(f"Tiempo promedio por depósito: {process_time/len(depositos_a_conciliar):.2f} segundos")
print(f"Depósitos procesados: {len(depositos_a_conciliar)}")
print(f"Depósitos conciliados: {len(results_list)}")
print(f"Tasa de éxito: {(len(results_list)/len(depositos_a_conciliar)*100):.1f}%")

# --- Presentación de Resultados Finales (SIMPLIFICADA) ---
if results_list:
    summary_df = pd.DataFrame(results_list)
    print("\n" + "="*80)
    print("             RESUMEN DE CONCILIACIÓN - DEPÓSITOS SIN IMPUESTOS")
    print("="*80)
    
    # Estadísticas básicas generales
    total_procesados = len(depositos_a_conciliar)
    conciliados = len(results_list)
    tasa_exito = (conciliados / total_procesados) * 100
    
    print(f"\n📊 Resumen General:")
    print(f"   • Depósitos SIN impuestos procesados: {total_procesados}")
    print(f"   • Depósitos conciliados: {conciliados}")
    print(f"   • Tasa de éxito: {tasa_exito:.1f}%")
    print(f"   • Tolerancia aplicada: ±1.00 peso (ESTRICTA)")
    
    # Precisión promedio
    avg_subtotal = summary_df['subtotal_precision_pct'].mean()
    avg_time = summary_df['processing_time_seconds'].mean()
    avg_diff_total = summary_df['diff_total_abs'].mean() if 'diff_total_abs' in summary_df.columns else 0
    
    print(f"\n🎯 Precisión Promedio:")
    print(f"   • Subtotal: {avg_subtotal:.1f}%")
    print(f"   • IVA: 100.0% (sin impuestos)")
    print(f"   • IEPS: 100.0% (sin impuestos)")
    print(f"   • Diferencia total promedio: ${avg_diff_total:.2f}")
    print(f"   • Tiempo promedio: {avg_time:.2f}s por depósito")
    
    # Verificar tolerancia estricta
    if 'diff_total_abs' in summary_df.columns:
        casos_fuera_tolerancia = summary_df[summary_df['diff_total_abs'] > 1.0]
        if not casos_fuera_tolerancia.empty:
            print(f"\n⚠️  ADVERTENCIA: {len(casos_fuera_tolerancia)} casos exceden ±1 peso:")
            for _, case in casos_fuera_tolerancia.iterrows():
                print(f"   Depósito {case['deposit_id']}: Diferencia total ${case['diff_total_abs']:.2f}")
        else:
            print(f"\n✅ EXCELENTE: Todos los casos están dentro de ±1.00 peso")
    
    # Mostrar solo casos problemáticos si los hay
    low_precision = summary_df[summary_df['subtotal_precision_pct'] < 95]
    
    if not low_precision.empty:
        print(f"\n⚠️  Casos con precisión < 95%:")
        for _, case in low_precision.iterrows():
            print(f"   Depósito {case['deposit_id']}: Subtotal {case['subtotal_precision_pct']:.1f}%")
    else:
        print(f"\n✅ EXCELENTE: Todos los casos tienen precisión ≥ 95%")
    
    # =============================================================================
    # GENERAR ARCHIVOS CSV SOLICITADOS
    # =============================================================================
    
    # 1. Generar ventas_asignadas.csv
    if all_detailed_products:
        ventas_asignadas = []
        
        # Diccionario para rastrear folios ya usados por depósito y folios disponibles por depósito
        folios_por_deposito = {}  # {deposit_id: folio_asignado}
        folios_disponibles_por_deposito = {}  # {deposit_id: [lista_de_folios_únicos_del_depósito]}
        
        # Primer pase: recopilar todos los folios únicos por depósito
        for product in all_detailed_products:
            deposit_id = product['deposit_id_conciliado']
            folio_original = product['Folio_Venta']
            
            if deposit_id not in folios_disponibles_por_deposito:
                folios_disponibles_por_deposito[deposit_id] = []
            
            # Agregar folio a la lista si no está ya
            if folio_original not in folios_disponibles_por_deposito[deposit_id]:
                folios_disponibles_por_deposito[deposit_id].append(folio_original)
        
        # Segundo pase: asignar folios consolidados
        for product in all_detailed_products:
            deposit_id = product['deposit_id_conciliado']
            folio_original = product['Folio_Venta']
            
            # Si este depósito ya tiene un folio asignado, usar ese
            if deposit_id in folios_por_deposito:
                folio_a_usar = folios_por_deposito[deposit_id]
            else:
                # Tomar el primer folio de los disponibles en este depósito
                folio_a_usar = folios_disponibles_por_deposito[deposit_id][0]
                folios_por_deposito[deposit_id] = folio_a_usar
            
            venta_asignada = {
                'ID_PVenta': product['ID_PVenta'],  # ID único de partida
                'FolioVenta': folio_a_usar,  # Folio consolidado (uno de los existentes en el depósito)
                'FolioOriginal': folio_original,  # Mantener referencia del folio original
                'IDDeposito': product['deposit_id_conciliado'],
                'Total': product['total'],
                'IVA': product['iva'],
                'IEPS': product['ieps']
            }
            ventas_asignadas.append(venta_asignada)
        
        ventas_asignadas_df = pd.DataFrame(ventas_asignadas)
        ventas_asignadas_df.to_csv("ventas_asignadas.csv", index=False)
        print(f"\n💾 Archivo generado: 'ventas_asignadas.csv' ({len(ventas_asignadas_df)} registros)")
        print(f"🔄 Folios consolidados por depósito: {len(folios_por_deposito)} depósitos únicos")
        print(f"📋 Cada depósito usa uno de sus propios folios para evitar duplicados")
    
    # 2. Generar ventas_no_utilizadas.csv
    # Obtener todas las partidas únicas del archivo de tickets
    todas_las_partidas = set(partidas_df['ID_PVenta'].unique())
    partidas_utilizadas = used_partidas
    partidas_no_utilizadas = todas_las_partidas - partidas_utilizadas
    
    if partidas_no_utilizadas:
        # Obtener los folios de las partidas no utilizadas
        partidas_no_usadas_df = partidas_df[partidas_df['ID_PVenta'].isin(partidas_no_utilizadas)]
        ventas_no_utilizadas_df = pd.DataFrame({
            'ID_PVenta': partidas_no_usadas_df['ID_PVenta'],
            'FolioVenta': partidas_no_usadas_df['Folio_Venta'],
            'Total': partidas_no_usadas_df['total'],
            'IVA': partidas_no_usadas_df['iva'],
            'IEPS': partidas_no_usadas_df['ieps']
        })
        ventas_no_utilizadas_df.to_csv("ventas_no_utilizadas.csv", index=False)
        print(f"💾 Archivo generado: 'ventas_no_utilizadas.csv' ({len(ventas_no_utilizadas_df)} registros)")
    
    # 3. Mantener el archivo de resumen para referencia
    summary_filename = "resumen_conciliacion_sin_impuestos.csv"
    summary_df.to_csv(summary_filename, index=False)
    print(f"💾 Resumen generado: '{summary_filename}' (archivo de referencia)")
    
    print(f"\n📊 RESUMEN DE ARCHIVOS GENERADOS:")
    if all_detailed_products:
        print(f"   ✅ ventas_asignadas.csv: {len(ventas_asignadas_df)} partidas asignadas a depósitos")
    if partidas_no_utilizadas:
        print(f"   📝 ventas_no_utilizadas.csv: {len(ventas_no_utilizadas_df)} partidas no utilizadas")
    print(f"   📋 Total de partidas procesadas: {len(todas_las_partidas)}")
    print(f"   🎯 Tasa de utilización: {(len(partidas_utilizadas)/len(todas_las_partidas)*100):.1f}%")
    
    print(f"\n🎉 Conciliación optimizada completada en {total_time:.2f} segundos")
    print(f"⚡ Procesamiento ultra rápido: {len(depositos_a_conciliar)} depósitos en {process_time:.2f}s")
    print(f"📈 Velocidad promedio: {process_time/len(depositos_a_conciliar):.2f}s por depósito")
    print(f"🎯 TOLERANCIA RESPETADA: ±1.00 peso exacto")
    
else:
    print("\n⚠️  No se pudo conciliar ningún depósito.")
    print("Verifica que los datos tengan depósitos sin impuestos que puedan conciliarse dentro de ±1 peso.")