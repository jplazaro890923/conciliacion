import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

def cargar_datos():
    """
    Carga y pre-procesa los datos necesarios para la conciliación.
    """
    # Cargar ventas no utilizadas con impuestos
    ventas = pd.read_csv('ventas_no_utilizadas_con_impuestos_simple.csv')
    
    # IEPS negativos se consideran como cero
    ventas['IEPS'] = np.where(ventas['IEPS'] < 0, 0, ventas['IEPS'])
    ventas['IVA'] = abs(ventas['IVA'])
    ventas['Total'] = abs(ventas['Total'])
    
    # Calcular el subtotal correctamente
    # Para ventas con IVA, usamos la tasa del 16%
    # Para ventas sin IVA, el subtotal es el total menos IEPS
    ventas['Subtotal'] = np.where(
        ventas['IVA'] > 0,
        (ventas['Total'] - ventas['IEPS']) / 1.16,  # Con IVA: Subtotal = (Total - IEPS) / (1 + 0.16)
        ventas['Total'] - ventas['IEPS']  # Sin IVA: Subtotal = Total - IEPS
    )
    
    # Recalcular IVA para asegurar consistencia
    ventas['IVA'] = np.where(
        ventas['IVA'] > 0,
        ventas['Subtotal'] * 0.16,  # 16% del subtotal
        0
    )
    
    # Agregar ID_PVenta si no existe
    if 'ID_PVenta' not in ventas.columns:
        ventas['ID_PVenta'] = range(len(ventas))
    
    # Cargar tickets con fechas
    tickets = pd.read_csv('TICKETS_JUN_JUL.csv')
    tickets['Fecha_Venta'] = pd.to_datetime(tickets['Fecha_Venta'])
    
    # Procesar depósitos
    depositos_data = []
    with open('depositos_all.sql', 'r', encoding='utf-8') as file:
        for line in file:
            if 'VALUES' in line:
                try:
                    values = line.split('VALUES')[1].strip()[1:-2].split(',')
                    depositos_data.append({
                        'IDDeposito': int(values[0].strip().strip("'")),
                        'Fecha': pd.to_datetime(values[3].strip().strip("'")),
                        'Subtotal0': float(values[4].strip().strip("'")),
                        'Subtotal16': float(values[5].strip().strip("'")),
                        'Subtotal': float(values[4].strip().strip("'")) + float(values[5].strip().strip("'")),
                        'IVA': float(values[6].strip().strip("'")),
                        'IEPS': float(values[7].strip().strip("'")),
                        'Total': float(values[8].strip().strip("'"))
                    })
                except (ValueError, IndexError):
                    continue
    
    depositos = pd.DataFrame(depositos_data)
    
    # Crear estructura optimizada para ventas
    ventas_con_fechas = ventas.merge(
        tickets[['Folio_Venta', 'Fecha_Venta']], 
        left_on='FolioVenta',
        right_on='Folio_Venta',
        how='inner'
    )
    
    # Pre-calcular índices para búsqueda rápida
    ventas_dict = {}  # Para búsqueda rápida por ID_PVenta
    folios_dict = defaultdict(list)  # Para búsqueda rápida por FolioVenta
    ventas_por_fecha = defaultdict(list)  # Mantener estructura por fecha
    
    for _, venta in ventas_con_fechas.iterrows():
        venta_dict = {
            'ID_PVenta': venta['ID_PVenta'],
            'FolioVenta': venta['FolioVenta'],
            'Subtotal': venta['Subtotal'],
            'IVA': venta['IVA'],
            'IEPS': venta['IEPS'],
            'Total': venta['Total'],
            'Fecha': venta['Fecha_Venta']
        }
        
        # Guardar en diccionarios de búsqueda
        ventas_dict[venta['ID_PVenta']] = venta_dict
        folios_dict[venta['FolioVenta']].append(venta_dict)
        ventas_por_fecha[venta['Fecha_Venta'].date()].append(venta_dict)
    
    return ventas_dict, folios_dict, ventas_por_fecha, depositos

def seleccionar_todos_los_depositos(depositos):
    """
    Selecciona TODOS los depósitos que tienen impuestos.
    """
    depositos_con_impuestos = depositos[
        (depositos['IVA'] > 0) | (depositos['IEPS'] > 0)
    ].copy()
    return depositos_con_impuestos

def encontrar_combinacion(ventas_candidatas, deposito, partidas_usadas, folios_usados_global):
    """
    Encuentra una combinación de ventas que maximiza IVA y IEPS dentro de los límites.
    Asegura que:
    1. ID_PVenta sea único globalmente
    2. FolioVenta solo se repita dentro del mismo depósito
    3. Si un ID_PVenta tiene un folio usado, busca una alternativa
    4. El subtotal total de tickets debe coincidir con subtotal0 + subtotal16
    """
    print(f"\nBuscando combinación para depósito:")
    print(f"IVA objetivo: ${deposito['IVA']:,.2f}")
    print(f"IEPS objetivo: ${deposito['IEPS']:,.2f}")
    print(f"Subtotal objetivo: ${deposito['Subtotal']:,.2f}")
    print(f"  - Subtotal 0%: ${deposito['Subtotal0']:,.2f}")
    print(f"  - Subtotal 16%: ${deposito['Subtotal16']:,.2f}")
    
    # Inicializar variables
    combinacion = []
    iva_acum = 0
    ieps_acum = 0
    subtotal_acum = 0
    subtotal_con_iva_acum = 0  # Para rastrear el subtotal de tickets con IVA
    partidas_en_uso = set()
    folios_en_uso = set()  # Folios usados en este depósito
    
    # Función auxiliar para verificar si una venta es válida
    def es_venta_valida(venta):
        # El ID_PVenta no debe estar usado globalmente
        if venta['ID_PVenta'] in partidas_usadas or venta['ID_PVenta'] in partidas_en_uso:
            return False
        
        # El FolioVenta puede estar en folios_en_uso (mismo depósito)
        # pero no en folios_usados_global (otros depósitos) a menos que sea de folios_en_uso
        if venta['FolioVenta'] in folios_usados_global and venta['FolioVenta'] not in folios_en_uso:
            return False
            
        return True
    
    # Pre-filtrar y ordenar ventas
    ventas_disponibles = [v for v in ventas_candidatas if es_venta_valida(v)]
    
    # Separar ventas por tipo
    ventas_con_iva = sorted([v for v in ventas_disponibles if v['IVA'] > 0.01], 
                           key=lambda x: x['IVA'], reverse=True)
    ventas_con_ieps = sorted([v for v in ventas_disponibles if v['IEPS'] > 0.01], 
                            key=lambda x: x['IEPS'], reverse=True)
    ventas_sin_impuestos = sorted([v for v in ventas_disponibles 
                                  if v['IVA'] <= 0.01 and v['IEPS'] <= 0.01],
                                  key=lambda x: x['Subtotal'], reverse=True)
    
    # Primera pasada: maximizar IVA (estas ventas contribuyen al subtotal16)
    for venta in ventas_con_iva:
        nuevo_iva = iva_acum + venta['IVA']
        nuevo_subtotal = subtotal_acum + venta['Subtotal']
        nuevo_ieps = ieps_acum + venta['IEPS']
        nuevo_subtotal_con_iva = subtotal_con_iva_acum + venta['Subtotal']
        
        if (nuevo_iva <= deposito['IVA'] and 
            nuevo_subtotal <= deposito['Subtotal'] and
            nuevo_subtotal_con_iva <= deposito['Subtotal16'] and
            (deposito['IEPS'] == 0 or nuevo_ieps <= deposito['IEPS']) and
            es_venta_valida(venta)):
            combinacion.append(venta)
            iva_acum = nuevo_iva
            ieps_acum = nuevo_ieps
            subtotal_acum = nuevo_subtotal
            subtotal_con_iva_acum = nuevo_subtotal_con_iva
            partidas_en_uso.add(venta['ID_PVenta'])
            folios_en_uso.add(venta['FolioVenta'])
            
            if iva_acum >= deposito['IVA'] * 0.99:
                break
    
    # Segunda pasada: maximizar IEPS
    if deposito['IEPS'] > 0 and ieps_acum < deposito['IEPS']:
        for venta in ventas_con_ieps:
            nuevo_ieps = ieps_acum + venta['IEPS']
            nuevo_iva = iva_acum + venta['IVA']
            nuevo_subtotal = subtotal_acum + venta['Subtotal']
            nuevo_subtotal_con_iva = subtotal_con_iva_acum + (venta['Subtotal'] if venta['IVA'] > 0.01 else 0)
            
            if (nuevo_ieps <= deposito['IEPS'] and
                nuevo_iva <= deposito['IVA'] and
                nuevo_subtotal <= deposito['Subtotal'] and
                nuevo_subtotal_con_iva <= deposito['Subtotal16'] and
                es_venta_valida(venta)):
                combinacion.append(venta)
                ieps_acum = nuevo_ieps
                iva_acum = nuevo_iva
                subtotal_acum = nuevo_subtotal
                subtotal_con_iva_acum = nuevo_subtotal_con_iva
                partidas_en_uso.add(venta['ID_PVenta'])
                folios_en_uso.add(venta['FolioVenta'])
    
    # Tercera pasada: maximizar subtotal con ventas sin impuestos (contribuyen al subtotal0)
    subtotal0_objetivo = deposito['Subtotal0']
    subtotal0_acum = subtotal_acum - subtotal_con_iva_acum
    
    if subtotal0_acum < subtotal0_objetivo * 0.99:
        for venta in ventas_sin_impuestos:
            nuevo_subtotal = subtotal_acum + venta['Subtotal']
            nuevo_subtotal0 = subtotal0_acum + venta['Subtotal']
            
            if (nuevo_subtotal <= deposito['Subtotal'] and
                nuevo_subtotal0 <= subtotal0_objetivo and
                es_venta_valida(venta)):
                combinacion.append(venta)
                subtotal_acum = nuevo_subtotal
                subtotal0_acum = nuevo_subtotal0
                iva_acum += venta['IVA']
                ieps_acum += venta['IEPS']
                partidas_en_uso.add(venta['ID_PVenta'])
                folios_en_uso.add(venta['FolioVenta'])
    
    # Mostrar resultados
    if combinacion:
        print("\nCombinación encontrada:")
        if deposito['IVA'] > 0:
            print(f"IVA: ${iva_acum:,.2f} ({iva_acum/deposito['IVA']*100:.1f}% del objetivo)")
        else:
            print(f"IVA: ${iva_acum:,.2f}")
            
        if deposito['IEPS'] > 0:
            print(f"IEPS: ${ieps_acum:,.2f} ({ieps_acum/deposito['IEPS']*100:.1f}% del objetivo)")
        else:
            print(f"IEPS: ${ieps_acum:,.2f}")
            
        print(f"Subtotal total: ${subtotal_acum:,.2f} ({subtotal_acum/deposito['Subtotal']*100:.1f}% del objetivo)")
        
        if deposito['Subtotal16'] > 0:
            print(f"  - Subtotal con IVA: ${subtotal_con_iva_acum:,.2f} ({subtotal_con_iva_acum/deposito['Subtotal16']*100:.1f}% del objetivo)")
        else:
            print(f"  - Subtotal con IVA: $0.00 (no hay subtotal con IVA)")
            
        if deposito['Subtotal0'] > 0:
            print(f"  - Subtotal sin IVA: ${subtotal0_acum:,.2f} ({subtotal0_acum/deposito['Subtotal0']*100:.1f}% del objetivo)")
        else:
            print(f"  - Subtotal sin IVA: $0.00 (no hay subtotal sin IVA)")
    
    # Si encontramos combinación, agregar los folios usados al conjunto global
    if combinacion:
        folios_usados_global.update(folios_en_uso)
    
    return combinacion if combinacion else None

def conciliar_depositos(ventas_dict, folios_dict, ventas_por_fecha, depositos):
    """
    Realiza la conciliación entre ventas y depósitos.
    """
    resultados = []
    partidas_usadas = set()  # ID_PVenta usados globalmente
    folios_usados_global = set()  # FolioVenta usados en otros depósitos
    total_depositos = len(depositos)
    
    print("\nIniciando proceso de conciliación...")
    
    for idx, deposito in depositos.iterrows():
        print(f"\nProcesando depósito {idx+1}/{total_depositos}:")
        print(f"ID: {deposito['IDDeposito']}")
        print(f"Total: ${deposito['Total']:,.2f}")
        
        fecha_deposito = deposito['Fecha'].date()
        combinacion = None
        ventana = 3  # Empezamos con ±3 días
        max_ventana = 60  # Máximo de días a buscar
        
        while not combinacion and ventana <= max_ventana:
            print(f"Intentando con ventana de ±{ventana} días...")
            ventas_candidatas = []
            
            # Recolectar ventas en la ventana actual
            for dias in range(-ventana, ventana + 1):
                fecha_busqueda = fecha_deposito + timedelta(days=dias)
                ventas_candidatas.extend(ventas_por_fecha.get(fecha_busqueda, []))
            
            # Intentar encontrar combinación
            combinacion = encontrar_combinacion(ventas_candidatas, deposito, 
                                              partidas_usadas, folios_usados_global)
            
            if not combinacion:
                ventana += 3  # Incrementar la ventana en 3 días si no encontramos nada
        
        # Si se encontró una combinación, registrar los resultados
        if combinacion:
            for venta in combinacion:
                resultados.append({
                    'IDDeposito': deposito['IDDeposito'],
                    'FolioVenta': venta['FolioVenta'],
                    'ID_PVenta': venta['ID_PVenta'],
                    'Subtotal': venta['Subtotal'],
                    'IVA': venta['IVA'],
                    'IEPS': venta['IEPS'],
                    'Total': venta['Total'],
                    'Fecha_Venta': venta['Fecha'],
                    'Fecha_Deposito': deposito['Fecha'],
                    'Subtotal_Deposito': deposito['Subtotal'],
                    'IVA_Deposito': deposito['IVA'],
                    'IEPS_Deposito': deposito['IEPS'],
                    'Total_Deposito': deposito['Total']
                })
                partidas_usadas.add(venta['ID_PVenta'])
    
    return pd.DataFrame(resultados)

def main():
    print("Iniciando proceso de conciliación...")
    
    # Cargar datos
    ventas_dict, folios_dict, ventas_por_fecha, depositos = cargar_datos()
    print("Datos cargados")
    
    # Seleccionar TODOS los depósitos con impuestos
    depositos_prueba = seleccionar_todos_los_depositos(depositos)
    print(f"Depósitos seleccionados: {len(depositos_prueba)} (TODOS los depósitos con impuestos)\n")
    
    # Realizar conciliación
    resultados = conciliar_depositos(ventas_dict, folios_dict, ventas_por_fecha, depositos_prueba)
    
    # Guardar resultados
    resultados.to_csv('resumen_conciliacion_todos_depositos_simple.csv', index=False)
    print(f"\nProceso completado. Se encontraron {len(resultados)} coincidencias.")

if __name__ == '__main__':
    main()