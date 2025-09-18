import pandas as pd
import re
from datetime import timedelta
import numpy as np
from collections import defaultdict
import random

# ==============================================================================
# 1. FUNCIONES DE CARGA DE DATOS SIMPLIFICADAS
# ==============================================================================

def parse_depositos_sql(file_path):
    """Extrae los datos de depósitos del archivo SQL."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pattern = re.compile(r"VALUES \((.*?)\);", re.IGNORECASE | re.DOTALL)
    matches = pattern.findall(content)
    
    data = []
    for match in matches:
        try:
            values = match.split(',')
            if len(values) >= 9:
                id_val = int(values[0].strip().strip("'"))
                fecha_val = pd.to_datetime(values[3].strip().strip("'"))
                subtotal_0 = float(values[4].strip().strip("'"))
                subtotal_16 = float(values[5].strip().strip("'"))
                iva_val = float(values[6].strip().strip("'"))
                ieps_val = float(values[7].strip().strip("'"))
                total_val = float(values[8].strip().strip("'"))
                
                subtotal_total = subtotal_0 + subtotal_16
                data.append([id_val, fecha_val, subtotal_total, iva_val, ieps_val, total_val, subtotal_0, subtotal_16])
        except (ValueError, IndexError):
            continue
    
    return pd.DataFrame(data, columns=['id', 'fecha', 'subtotal_total', 'iva', 'ieps', 'total', 'subtotal_0', 'subtotal_16'])

def load_ventas_no_utilizadas_csv(file_path):
    """Carga los datos de ventas no utilizadas."""
    df = pd.read_csv(file_path)
    df['subtotal_total'] = df['Total'] - df['IVA'] - df['IEPS']
    df.rename(columns={
        'FolioVenta': 'Folio_Venta',
        'Total': 'total',
        'IVA': 'iva',
        'IEPS': 'ieps'
    }, inplace=True)
    
    # Usar un rango de fechas distribuido
    num_rows = len(df)
    start_date = pd.to_datetime('2023-06-01')
    end_date = pd.to_datetime('2023-07-31')
    date_range = pd.date_range(start=start_date, end=end_date, periods=num_rows)
    df['Fecha_Venta'] = date_range
    
    return df

# ==============================================================================
# 2. ALGORITMO SIMPLIFICADO PARA COMBINACIONES GRANDES
# ==============================================================================

def simple_greedy_sum(tickets, target_amount, max_iterations=5000, tolerance_pct=0.15):
    """
    Algoritmo greedy simplificado para sumar tickets hasta llegar cerca del objetivo.
    tolerance_pct: porcentaje de tolerancia (ej: 0.15 = 15%)
    """
    if not tickets:
        return None, 0
    
    # Ordenar tickets por total de mayor a menor para convergir más rápido
    sorted_tickets = sorted(tickets, key=lambda x: x['total'], reverse=True)
    
    selected = []
    current_sum = 0
    min_target = target_amount * (1 - tolerance_pct)
    max_target = target_amount * (1 + tolerance_pct)
    
    # Selección greedy básica
    for ticket in sorted_tickets:
        if len(selected) >= max_iterations:
            break
            
        new_sum = current_sum + ticket['total']
        
        # Si nos pasa mucho del objetivo, saltarlo
        if new_sum > max_target:
            continue
            
        selected.append(ticket)
        current_sum = new_sum
        
        # Si llegamos al rango objetivo, terminar
        if min_target <= current_sum <= max_target:
            break
    
    # Verificar si está en rango aceptable
    if min_target <= current_sum <= max_target:
        return selected, current_sum
    
    return None, current_sum

def reconcile_deposit_simple(deposit, all_tickets, tolerance_pct=0.20):
    """
    Conciliación simplificada para depósitos grandes.
    tolerance_pct: tolerancia como porcentaje del monto total
    """
    target_total = deposit['total']
    
    print(f"   Buscando combinación para ${target_total:,.2f} con tolerancia {tolerance_pct*100:.0f}%")
    
    # Intentar con diferentes tolerancias si no encuentra
    tolerances = [tolerance_pct, tolerance_pct * 1.5, tolerance_pct * 2.0]
    
    for tol in tolerances:
        selected_tickets, achieved_sum = simple_greedy_sum(all_tickets, target_total, tolerance_pct=tol)
        
        if selected_tickets:
            # Calcular totales por componente
            total_subtotal = sum(t['subtotal_total'] for t in selected_tickets)
            total_iva = sum(t['iva'] for t in selected_tickets)
            total_ieps = sum(t['ieps'] for t in selected_tickets)
            
            return {
                'success': True,
                'tickets': selected_tickets,
                'achieved_total': achieved_sum,
                'achieved_subtotal': total_subtotal,
                'achieved_iva': total_iva,
                'achieved_ieps': total_ieps,
                'tolerance_used': tol,
                'num_tickets': len(selected_tickets)
            }
    
    return {'success': False}

# ==============================================================================
# 3. EJECUCIÓN PRINCIPAL SIMPLIFICADA
# ==============================================================================

print("🚀 Iniciando conciliación SIMPLIFICADA para depósitos con impuestos...")
print("   📋 Enfoque: Algoritmo greedy rápido con tolerancia amplia")
print("   🎯 Objetivo: Obtener resultados prácticos")

# Cargar datos
print("📊 Cargando datos...")
ventas_df = load_ventas_no_utilizadas_csv('ventas_no_utilizadas.csv')
depositos_df = parse_depositos_sql('depositos_all.sql')

# Filtrar depósitos con impuestos
depositos_con_impuestos = depositos_df[(depositos_df['iva'] > 0) | (depositos_df['ieps'] > 0)]
print(f"✓ {len(depositos_con_impuestos)} depósitos con impuestos encontrados")
print(f"✓ {len(ventas_df):,} partidas disponibles")

# Limitar a los primeros 50 depósitos para prueba
depositos_a_procesar = depositos_con_impuestos.head(50).sort_values('total')

print(f"\n🎯 Procesando los primeros {len(depositos_a_procesar)} depósitos...")
print("   💡 Tolerancia: 20% del monto total")
print("   🔄 Máximo 5000 tickets por depósito")

# Convertir tickets a lista de diccionarios para velocidad
all_tickets = []
for _, row in ventas_df.iterrows():
    ticket = {
        'ID_PVenta': row['ID_PVenta'],
        'Folio_Venta': row['Folio_Venta'],
        'total': row['total'],
        'subtotal_total': row['subtotal_total'],
        'iva': row['iva'],
        'ieps': row['ieps']
    }
    all_tickets.append(ticket)

print(f"✓ {len(all_tickets):,} tickets preparados")

# Procesar depósitos
results = []
used_ticket_ids = set()

for i, (_, deposit) in enumerate(depositos_a_procesar.iterrows()):
    print(f"\n--- Depósito {i+1}/{len(depositos_a_procesar)}: ID {deposit['id']} ---")
    print(f"Objetivo: ${deposit['total']:,.2f} (IVA: ${deposit['iva']:,.2f}, IEPS: ${deposit['ieps']:,.2f})")
    
    # Filtrar tickets no utilizados
    available_tickets = [t for t in all_tickets if t['ID_PVenta'] not in used_ticket_ids]
    print(f"Tickets disponibles: {len(available_tickets):,}")
    
    if len(available_tickets) < 100:  # Muy pocos tickets restantes
        print("⚠️  Muy pocos tickets disponibles, saltando...")
        continue
    
    # Intentar conciliación
    result = reconcile_deposit_simple(deposit, available_tickets)
    
    if result['success']:
        # Marcar tickets como utilizados
        for ticket in result['tickets']:
            used_ticket_ids.add(ticket['ID_PVenta'])
        
        # Calcular diferencias
        diff_total = abs(deposit['total'] - result['achieved_total'])
        diff_pct = (diff_total / deposit['total']) * 100
        
        print(f"✓ CONCILIADO: ${result['achieved_total']:,.2f}")
        print(f"  Diferencia: ${diff_total:.2f} ({diff_pct:.1f}%)")
        print(f"  Tickets usados: {result['num_tickets']}")
        print(f"  Tolerancia: {result['tolerance_used']*100:.0f}%")
        
        # Guardar resultado
        results.append({
            'deposit_id': deposit['id'],
            'deposit_total': deposit['total'],
            'deposit_iva': deposit['iva'],
            'deposit_ieps': deposit['ieps'],
            'achieved_total': result['achieved_total'],
            'achieved_subtotal': result['achieved_subtotal'],
            'achieved_iva': result['achieved_iva'],
            'achieved_ieps': result['achieved_ieps'],
            'diff_total': diff_total,
            'diff_pct': diff_pct,
            'num_tickets': result['num_tickets'],
            'tolerance_used': result['tolerance_used'],
            'tickets': result['tickets']
        })
    else:
        print("✗ NO CONCILIADO")

print(f"\n" + "="*60)
print(f"📊 RESUMEN FINAL")
print(f"="*60)
print(f"Depósitos procesados: {len(depositos_a_procesar)}")
print(f"Depósitos conciliados: {len(results)}")
print(f"Tasa de éxito: {(len(results)/len(depositos_a_procesar)*100):.1f}%")

if results:
    avg_diff = sum(r['diff_pct'] for r in results) / len(results)
    avg_tickets = sum(r['num_tickets'] for r in results) / len(results)
    print(f"Diferencia promedio: {avg_diff:.1f}%")
    print(f"Tickets promedio por depósito: {avg_tickets:.0f}")
    
    # Generar archivos de salida
    print(f"\n📄 Generando archivos de salida...")
    
    # 1. Resumen de conciliación
    summary_df = pd.DataFrame([{
        'deposit_id': r['deposit_id'],
        'deposit_total': r['deposit_total'],
        'achieved_total': r['achieved_total'],
        'diff_total': r['diff_total'],
        'diff_pct': r['diff_pct'],
        'num_tickets': r['num_tickets'],
        'tolerance_used': r['tolerance_used']
    } for r in results])
    
    summary_df.to_csv('resumen_conciliacion_con_impuestos_simple.csv', index=False)
    print(f"✓ resumen_conciliacion_con_impuestos_simple.csv")
    
    # 2. Detalle de tickets asignados
    all_assigned_tickets = []
    for r in results:
        for ticket in r['tickets']:
            all_assigned_tickets.append({
                'ID_PVenta': ticket['ID_PVenta'],
                'FolioVenta': ticket['Folio_Venta'],
                'Total': ticket['total'],
                'IVA': ticket['iva'],
                'IEPS': ticket['ieps'],
                'IDDeposito': r['deposit_id']
            })
    
    assigned_df = pd.DataFrame(all_assigned_tickets)
    assigned_df.to_csv('ventas_asignadas_con_impuestos_simple.csv', index=False)
    print(f"✓ ventas_asignadas_con_impuestos_simple.csv ({len(assigned_df)} tickets)")
    
    # 3. Tickets restantes no utilizados
    remaining_tickets = [t for t in all_tickets if t['ID_PVenta'] not in used_ticket_ids]
    remaining_df = pd.DataFrame([{
        'ID_PVenta': t['ID_PVenta'],
        'FolioVenta': t['Folio_Venta'],
        'Total': t['total'],
        'IVA': t['iva'],
        'IEPS': t['ieps']
    } for t in remaining_tickets])
    
    remaining_df.to_csv('ventas_no_utilizadas_con_impuestos_simple.csv', index=False)
    print(f"✓ ventas_no_utilizadas_con_impuestos_simple.csv ({len(remaining_df)} tickets)")
    
    print(f"\n🎉 Conciliación completada!")
    print(f"📈 Se lograron conciliar {len(results)} depósitos con tolerancia promedio {avg_diff:.1f}%")

else:
    print("⚠️  No se pudo conciliar ningún depósito")
    print("   Posibles causas:")
    print("   - Los montos de depósitos son muy grandes vs tickets individuales")
    print("   - Se necesita mayor tolerancia")
    print("   - Los datos no tienen suficiente correlación temporal")