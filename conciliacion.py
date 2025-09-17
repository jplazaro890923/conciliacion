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
    
    return df

# ==============================================================================
# 2. FUNCIÓN PRINCIPAL DE CONCILIACIÓN (ULTRA OPTIMIZADA)
# ==============================================================================

def reconcile_deposit_fast(deposit_to_match, folios_dict_by_date, max_days_back=45, tolerance_pct=1.0):
    """
    Versión ultra optimizada de la conciliación usando diccionarios indexados.
    CON VALIDACIÓN ESTRICTA DE TOLERANCIA.
    """
    best_match = {'distance': float('inf')}
    
    # Búsqueda optimizada por fechas
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
        
        # VALIDACIÓN ESTRICTA: Solo aceptar si está EXACTAMENTE dentro de tolerancia
        if selected_folios is not None and len(selected_folios) > 0:
            diff_subtotal = abs(deposit_to_match['subtotal_total'] - totals['subtotal_total'])
            diff_iva = abs(deposit_to_match['iva'] - totals['iva'])
            diff_ieps = abs(deposit_to_match['ieps'] - totals['ieps'])
            
            # VALIDACIÓN ESPECIAL PARA DEPÓSITOS SIN IMPUESTOS
            if deposit_to_match['iva'] == 0 and deposit_to_match['ieps'] == 0:
                # Para depósitos sin impuestos, IVA e IEPS deben ser exactamente 0
                if totals['iva'] != 0 or totals['ieps'] != 0:
                    continue  # Rechazar si hay impuestos cuando no debería haberlos
                
                # Verificar que el total coincida exactamente
                calculated_total = totals['subtotal_total'] + totals['iva'] + totals['ieps']
                expected_total = deposit_to_match['total']
                if abs(calculated_total - expected_total) > tolerance_pct:
                    continue  # Rechazar si el total no coincide
            
            # RECHAZAR si cualquier diferencia excede la tolerancia
            if (diff_subtotal > tolerance_pct or
                diff_iva > tolerance_pct or
                diff_ieps > tolerance_pct):
                continue  # No aceptar esta combinación
            
            # Solo llegar aquí si TODAS las diferencias están dentro de tolerancia
            distance = diff_subtotal + diff_iva + diff_ieps
            
            if distance < best_match['distance']:
                best_match.update({
                    'distance': distance, 
                    'folios': selected_folios, 
                    'date': search_date,
                    'final_subtotal': totals['subtotal_total'], 
                    'final_iva': totals['iva'], 
                    'final_ieps': totals['ieps'],
                    'strategy': 'fast_greedy'
                })
                
                # Si encontramos una coincidencia perfecta, salir temprano
                if distance <= 0.1:  # Prácticamente exacto
                    break
    
    return best_match

def _fast_greedy_selection(folios_list, target, tolerance=1.0):
    """
    Selección greedy ultra rápida con validación estricta de tolerancia.
    ESPECIAL: Para depósitos sin impuestos, solo usar folios sin impuestos.
    """
    if not folios_list:
        return None, None
    
    # FILTRO ESPECIAL: Si el depósito no tiene impuestos, solo usar folios sin impuestos
    if target['iva'] == 0 and target['ieps'] == 0:
        folios_list = [f for f in folios_list if f['iva'] == 0 and f['ieps'] == 0]
        if not folios_list:
            return None, None
    
    # Ordenar por score de eficiencia
    folios_with_score = []
    for folio in folios_list:
        # Score simple basado en proximidad al objetivo
        score = (min(folio['subtotal_total'] / target['subtotal_total'], 1.0) +
                min(folio['iva'] / max(target['iva'], 0.01), 1.0) +
                min(folio['ieps'] / max(target['ieps'], 0.01), 1.0))
        folios_with_score.append((score, folio))
    
    # Ordenar por score descendente
    folios_with_score.sort(key=lambda x: x[0], reverse=True)
    
    selected = []
    current_totals = {'subtotal_total': 0, 'iva': 0, 'ieps': 0}
    
    # Selección greedy con límites estrictos
    for score, folio in folios_with_score:
        new_subtotal = current_totals['subtotal_total'] + folio['subtotal_total']
        new_iva = current_totals['iva'] + folio['iva']
        new_ieps = current_totals['ieps'] + folio['ieps']
        
        # NO agregar folios que harían que excedamos el objetivo + tolerancia
        if (new_subtotal <= target['subtotal_total'] + tolerance and
            new_iva <= target['iva'] + tolerance and
            new_ieps <= target['ieps'] + tolerance):
            
            selected.append(folio)
            current_totals = {'subtotal_total': new_subtotal, 'iva': new_iva, 'ieps': new_ieps}
    
    # VALIDACIÓN FINAL ESTRICTA: Verificar que el resultado esté dentro de tolerancia
    if selected:
        diff_subtotal = abs(current_totals['subtotal_total'] - target['subtotal_total'])
        diff_iva = abs(current_totals['iva'] - target['iva'])
        diff_ieps = abs(current_totals['ieps'] - target['ieps'])
        
        # ESPECIAL: Para depósitos sin impuestos, IVA e IEPS deben ser exactamente 0
        if target['iva'] == 0 and target['ieps'] == 0:
            if current_totals['iva'] != 0 or current_totals['ieps'] != 0:
                return None, None
        
        # Si CUALQUIER diferencia excede tolerancia, rechazar TODO
        if (diff_subtotal > tolerance or
            diff_iva > tolerance or
            diff_ieps > tolerance):
            return None, None
    
    return selected, current_totals

def prepare_folios_dict(folio_summary_df):
    """
    Prepara un diccionario indexado por fecha para acceso ultra rápido.
    """
    folios_dict = defaultdict(list)
    
    # Convertir DataFrame a diccionario de listas para velocidad
    for _, row in folio_summary_df.iterrows():
        fecha = row['Fecha_Venta']
        folio_dict = {
            'Folio_Venta': row['Folio_Venta'],
            'Fecha_Venta': row['Fecha_Venta'],
            'subtotal_total': row['subtotal_total'],
            'iva': row['iva'],
            'ieps': row['ieps'],
            'total': row['total']
        }
        folios_dict[fecha].append(folio_dict)
    
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
print("🔧 Preparando estructura de folios...")
prep_start = time.time()
folio_summary_df = all_products_df.groupby(['Fecha_Venta', 'Folio_Venta']).agg({
    'subtotal_total': 'sum', 'iva': 'sum', 'ieps': 'sum', 'total': 'sum'
}).reset_index()

# Crear diccionario indexado por fecha para acceso ultra rápido
folios_dict_by_date = prepare_folios_dict(folio_summary_df)
prep_end = time.time()
print(f"✓ Estructura preparada en {prep_end - prep_start:.2f} segundos")

# PROCESANDO 10 DEPÓSITOS ALEATORIOS (8 SIN IMPUESTOS, 2 CON IMPUESTOS)
# Separar todos los depósitos por tipo de impuestos primero
todos_sin_impuestos = depositos_df[(depositos_df['iva'] == 0) & (depositos_df['ieps'] == 0)]
todos_con_impuestos = depositos_df[(depositos_df['iva'] > 0) | (depositos_df['ieps'] > 0)]

# Seleccionar aleatoriamente 8 sin impuestos y 2 con impuestos
depositos_sin_impuestos = todos_sin_impuestos.sample(n=8, random_state=42).sort_values(by='fecha')
depositos_con_impuestos = todos_con_impuestos.sample(n=2, random_state=42).sort_values(by='fecha')

# Combinar para tener los 10 depósitos a procesar
depositos_a_conciliar = pd.concat([depositos_sin_impuestos, depositos_con_impuestos]).sort_values(by='fecha')

print(f"🎯 Procesando {len(depositos_a_conciliar)} depósitos (ALEATORIOS - 8 SIN IMPUESTOS, 2 CON IMPUESTOS)")
print(f"   📊 Depósitos SIN impuestos: {len(depositos_sin_impuestos)}")
print(f"   📊 Depósitos CON impuestos: {len(depositos_con_impuestos)}")

# Inicializar listas para guardar resultados
results_list = []
all_detailed_products = []
used_folios = set()  # Track de folios ya utilizados

# =============================================================================
# FASE 1: PROCESAR DEPÓSITOS CON IMPUESTOS PRIMERO (PRIORIDAD ALTA)
# =============================================================================
print(f"\n🎯 FASE 1: PROCESANDO {len(depositos_con_impuestos)} DEPÓSITOS CON IMPUESTOS (PRIORIDAD)")
print("   Estrategia: Tolerancia fija de 1 peso (máxima precisión)")

# Inicializar tiempo de procesamiento
process_start = time.time()

for index, deposit in depositos_con_impuestos.iterrows():
    deposit_start = time.time()
    
    print(f"\n--- Depósito {len(results_list)+1}/{len(depositos_a_conciliar)}: ID {deposit['id']} del {deposit['fecha'].strftime('%Y-%m-%d')} (CON IMPUESTOS) ---")
    print(f"Objetivo: Subtotal=${deposit['subtotal_total']:,.2f}, IVA=${deposit['iva']:,.2f}, IEPS=${deposit['ieps']:,.2f}")
    
    # Filtrar folios disponibles (no utilizados)
    filtered_folios_dict = {}
    total_available = 0
    for fecha, folios_list in folios_dict_by_date.items():
        available_folios = [f for f in folios_list if f['Folio_Venta'] not in used_folios]
        if available_folios:
            filtered_folios_dict[fecha] = available_folios
            total_available += len(available_folios)
    
    print(f"Folios disponibles: {total_available}")
    
    # Ejecutar conciliación con tolerancia fija de 1 peso
    best_match = reconcile_deposit_fast(deposit, filtered_folios_dict, tolerance_pct=1.0)
    best_match['tolerance_used'] = 1
    
    deposit_end = time.time()
    deposit_time = deposit_end - deposit_start
    
    if 'folios' in best_match and best_match['folios'] is not None and len(best_match['folios']) > 0:
        # Calcular porcentajes de precisión
        subtotal_precision = (best_match['final_subtotal'] / deposit['subtotal_total']) * 100
        iva_precision = (best_match['final_iva'] / max(deposit['iva'], 0.01)) * 100 if deposit['iva'] > 0 else 100
        ieps_precision = (best_match['final_ieps'] / max(deposit['ieps'], 0.01)) * 100 if deposit['ieps'] > 0 else 100
        
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
            'subtotal_precision_pct': subtotal_precision,
            'iva_precision_pct': iva_precision,
            'ieps_precision_pct': ieps_precision,
            'num_folios': len(best_match['folios']),
            'strategy_used': best_match.get('strategy', 'fast_greedy'),
            'tolerance_used': best_match.get('tolerance_used', 1),
            'processing_time_seconds': deposit_time,
            'fase': 'con_impuestos'
        }
        results_list.append(summary)
        
        print(f"✓ Conciliado en {deposit_time:.2f}s | Estrategia: {best_match.get('strategy', 'fast_greedy')} | Tolerancia: {best_match.get('tolerance_used', 1)} pesos")
        print(f"  Resultado: Subtotal=${best_match['final_subtotal']:,.2f} ({subtotal_precision:.1f}%), " +
              f"IVA=${best_match['final_iva']:,.2f} ({iva_precision:.1f}%), " +
              f"IEPS=${best_match['final_ieps']:,.2f} ({ieps_precision:.1f}%)")
        print(f"  Folios utilizados: {len(best_match['folios'])}")

        # Marcar folios como utilizados
        for folio in best_match['folios']:
            used_folios.add(folio['Folio_Venta'])
        
        # Guardar detalles de productos
        for folio in best_match['folios']:
            detailed_product = {
                'Folio_Venta': folio['Folio_Venta'],
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
                'tolerance_used': best_match.get('tolerance_used', 1),
                'fase': 'con_impuestos'
            }
            all_detailed_products.append(detailed_product)
    else:
        print(f"✗ No conciliado en {deposit_time:.2f}s")

# =============================================================================
# FASE 2: PROCESAR DEPÓSITOS SIN IMPUESTOS (DESPUÉS DE LOS CON IMPUESTOS)
# =============================================================================
print(f"\n🔥 FASE 2: PROCESANDO {len(depositos_sin_impuestos)} DEPÓSITOS SIN IMPUESTOS")
print("   Estrategia: Tolerancia fija de 1 peso (máxima precisión)")

# Bucle principal optimizado
for index, deposit in depositos_sin_impuestos.iterrows():
    deposit_start = time.time()
    
    print(f"\n--- Depósito {len(results_list)+1}/{len(depositos_a_conciliar)}: ID {deposit['id']} del {deposit['fecha'].strftime('%Y-%m-%d')} (SIN IMPUESTOS) ---")
    print(f"Objetivo: Subtotal=${deposit['subtotal_total']:,.2f}, IVA=${deposit['iva']:,.2f}, IEPS=${deposit['ieps']:,.2f}")
    
    # Filtrar folios disponibles (no utilizados)
    filtered_folios_dict = {}
    total_available = 0
    for fecha, folios_list in folios_dict_by_date.items():
        available_folios = [f for f in folios_list if f['Folio_Venta'] not in used_folios]
        if available_folios:
            filtered_folios_dict[fecha] = available_folios
            total_available += len(available_folios)
    
    print(f"Folios disponibles: {total_available}")
    
    # Ejecutar conciliación con tolerancia fija de 1 peso (máxima precisión)
    best_match = reconcile_deposit_fast(deposit, filtered_folios_dict, tolerance_pct=1.0)
    best_match['tolerance_used'] = 1
    
    deposit_end = time.time()
    deposit_time = deposit_end - deposit_start
    
    if 'folios' in best_match and best_match['folios'] is not None and len(best_match['folios']) > 0:
        # Calcular porcentajes de precisión
        subtotal_precision = (best_match['final_subtotal'] / deposit['subtotal_total']) * 100
        iva_precision = (best_match['final_iva'] / max(deposit['iva'], 0.01)) * 100 if deposit['iva'] > 0 else 100
        ieps_precision = (best_match['final_ieps'] / max(deposit['ieps'], 0.01)) * 100 if deposit['ieps'] > 0 else 100
        
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
            'subtotal_precision_pct': subtotal_precision,
            'iva_precision_pct': iva_precision,
            'ieps_precision_pct': ieps_precision,
            'num_folios': len(best_match['folios']),
            'strategy_used': best_match.get('strategy', 'fast_greedy'),
            'tolerance_used': best_match.get('tolerance_used', 1),
            'processing_time_seconds': deposit_time,
            'fase': 'sin_impuestos'
        }
        results_list.append(summary)
        
        print(f"✓ Conciliado en {deposit_time:.2f}s | Estrategia: {best_match.get('strategy', 'fast_greedy')} | Tolerancia: {best_match.get('tolerance_used', 1)} pesos")
        print(f"  Resultado: Subtotal=${best_match['final_subtotal']:,.2f} ({subtotal_precision:.1f}%), " +
              f"IVA=${best_match['final_iva']:,.2f} ({iva_precision:.1f}%), " +
              f"IEPS=${best_match['final_ieps']:,.2f} ({ieps_precision:.1f}%)")
        print(f"  Folios utilizados: {len(best_match['folios'])}")

        # Marcar folios como utilizados
        for folio in best_match['folios']:
            used_folios.add(folio['Folio_Venta'])
        
        # Guardar detalles de productos
        for folio in best_match['folios']:
            detailed_product = {
                'Folio_Venta': folio['Folio_Venta'],
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
                'tolerance_used': best_match.get('tolerance_used', 1),
                'fase': 'sin_impuestos'
            }
            all_detailed_products.append(detailed_product)
    else:
        print(f"✗ No conciliado en {deposit_time:.2f}s")

process_end = time.time()
total_time = process_end - start_time
process_time = process_end - process_start

print(f"\n" + "="*80)
print(f"⚡ RENDIMIENTO DE LA VERSIÓN OPTIMIZADA (SEPARACIÓN POR FASES) ⚡")
print(f"="*80)
print(f"Tiempo total: {total_time:.2f} segundos")
print(f"Tiempo de procesamiento: {process_time:.2f} segundos")
print(f"Tiempo promedio por depósito: {process_time/len(depositos_a_conciliar):.2f} segundos")
print(f"Depósitos procesados: {len(depositos_a_conciliar)}")
print(f"Depósitos conciliados: {len(results_list)}")
print(f"Tasa de éxito: {(len(results_list)/len(depositos_a_conciliar)*100):.1f}%")

# --- Presentación de Resultados Finales (OPTIMIZADA) ---
if results_list:
    summary_df = pd.DataFrame(results_list)
    print("\n" + "="*80)
    print("                RESUMEN DE CONCILIACIÓN OPTIMIZADA POR FASES")
    print("="*80)
    
    # Estadísticas por fase
    fase_con_impuestos = summary_df[summary_df['fase'] == 'con_impuestos'] if 'fase' in summary_df.columns else pd.DataFrame()
    fase_sin_impuestos = summary_df[summary_df['fase'] == 'sin_impuestos'] if 'fase' in summary_df.columns else pd.DataFrame()
    
    print(f"\n📊 Estadísticas por Fase:")
    print(f"   🎯 FASE 1 - CON IMPUESTOS (PRIORIDAD):")
    print(f"      • Depósitos procesados: {len(depositos_con_impuestos)}")
    print(f"      • Depósitos conciliados: {len(fase_con_impuestos)}")
    print(f"      • Tasa de éxito: {(len(fase_con_impuestos)/max(len(depositos_con_impuestos),1)*100):.1f}%")
    print(f"      • Tolerancia usada: 1 peso (fija)")
    
    print(f"   🔥 FASE 2 - SIN IMPUESTOS:")
    print(f"      • Depósitos procesados: {len(depositos_sin_impuestos)}")
    print(f"      • Depósitos conciliados: {len(fase_sin_impuestos)}")
    print(f"      • Tasa de éxito: {(len(fase_sin_impuestos)/max(len(depositos_sin_impuestos),1)*100):.1f}%")
    print(f"      • Tolerancia usada: 1 peso (fija)")
    
    # Estadísticas básicas generales
    total_procesados = len(depositos_a_conciliar)
    conciliados = len(results_list)
    tasa_exito = (conciliados / total_procesados) * 100
    
    print(f"\n📊 Resumen General:")
    print(f"   • Depósitos procesados: {total_procesados}")
    print(f"   • Depósitos conciliados: {conciliados}")
    print(f"   • Tasa de éxito: {tasa_exito:.1f}%")
    
    # Precisión promedio
    avg_subtotal = summary_df['subtotal_precision_pct'].mean()
    avg_iva = summary_df['iva_precision_pct'].mean()
    avg_ieps = summary_df['ieps_precision_pct'].mean()
    avg_time = summary_df['processing_time_seconds'].mean()
    
    print(f"\n🎯 Precisión Promedio:")
    print(f"   • Subtotal: {avg_subtotal:.1f}%")
    print(f"   • IVA: {avg_iva:.1f}%")
    print(f"   • IEPS: {avg_ieps:.1f}%")
    print(f"   • Tiempo promedio: {avg_time:.2f}s por depósito")
    
    # Mostrar solo casos problemáticos si los hay
    low_precision = summary_df[
        (summary_df['subtotal_precision_pct'] < 95) |
        (summary_df['iva_precision_pct'] < 95) |
        (summary_df['ieps_precision_pct'] < 95)
    ]
    
    if not low_precision.empty:
        print(f"\n⚠️  Casos con precisión < 95%:")
        for _, case in low_precision.iterrows():
            print(f"   Depósito {case['deposit_id']}: Subtotal {case['subtotal_precision_pct']:.1f}%, " +
                  f"IVA {case['iva_precision_pct']:.1f}%, IEPS {case['ieps_precision_pct']:.1f}%")
    
    # =============================================================================
    # GENERAR ARCHIVOS CSV SOLICITADOS
    # =============================================================================
    
    # 1. Generar ventas_asignadas.csv
    if all_detailed_products:
        ventas_asignadas = []
        for product in all_detailed_products:
            venta_asignada = {
                'FolioVenta': product['Folio_Venta'],
                'IDDeposito': product['deposit_id_conciliado'],
                'Subtotal': product['subtotal_total'],  # subtotal_total es el subtotal (subtotal_0 + subtotal_16)
                'IVA': product['iva'],
                'IEPS': product['ieps']
            }
            ventas_asignadas.append(venta_asignada)
        
        ventas_asignadas_df = pd.DataFrame(ventas_asignadas)
        ventas_asignadas_df.to_csv("ventas_asignadas.csv", index=False)
        print(f"\n💾 Archivo generado: 'ventas_asignadas.csv' ({len(ventas_asignadas_df)} registros)")
    
    # 2. Generar ventas_no_utilizadas.csv
    # Obtener todos los folios únicos del archivo de tickets
    todos_los_folios = set(folio_summary_df['Folio_Venta'].unique())
    folios_utilizados = used_folios
    folios_no_utilizados = todos_los_folios - folios_utilizados
    
    if folios_no_utilizados:
        ventas_no_utilizadas_df = pd.DataFrame({
            'FolioVenta': list(folios_no_utilizados)
        })
        ventas_no_utilizadas_df.to_csv("ventas_no_utilizadas.csv", index=False)
        print(f"💾 Archivo generado: 'ventas_no_utilizadas.csv' ({len(ventas_no_utilizadas_df)} registros)")
    
    # 3. Mantener el archivo de resumen para referencia
    summary_filename = "resumen_conciliacion_completa.csv"
    summary_df.to_csv(summary_filename, index=False)
    print(f"💾 Resumen generado: '{summary_filename}' (archivo de referencia)")
    
    print(f"\n📊 RESUMEN DE ARCHIVOS GENERADOS:")
    if all_detailed_products:
        print(f"   ✅ ventas_asignadas.csv: {len(ventas_asignadas_df)} folios asignados a depósitos")
    if folios_no_utilizados:
        print(f"   📝 ventas_no_utilizadas.csv: {len(ventas_no_utilizadas_df)} folios no utilizados")
    print(f"   📋 Total de folios procesados: {len(todos_los_folios)}")
    print(f"   🎯 Tasa de utilización: {(len(folios_utilizados)/len(todos_los_folios)*100):.1f}%")
    
    print(f"\n🎉 Conciliación optimizada completada en {total_time:.2f} segundos")
    print(f"⚡ Procesamiento ultra rápido: {len(depositos_a_conciliar)} depósitos en {process_time:.2f}s")
    print(f"📈 Velocidad promedio: {process_time/len(depositos_a_conciliar):.2f}s por depósito")
    
else:
    print("\n⚠️  No se pudo conciliar ningún depósito.")
    print("Verifica los datos de entrada o ajusta los umbrales si es necesario.")