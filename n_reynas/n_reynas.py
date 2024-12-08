import random
import math

def calcular_conflictos(tablero):
    N = len(tablero)
    conflictos = 0
    for i in range(N):
        for j in range(i + 1, N):
            if tablero[i] == tablero[j] or abs(tablero[i] - tablero[j]) == abs(i - j):
                conflictos += 1
    return conflictos

def simulado_recocido(N, temperatura_inicial=1000, tasa_enfriamiento=0.95, iteraciones_por_temperatura=1000):
    solucion_actual = [random.randint(0, N - 1) for _ in range(N)]
    energia_actual = calcular_conflictos(solucion_actual)
    temperatura = temperatura_inicial
    
    mejor_solucion = solucion_actual[:]
    mejor_energia = energia_actual
    
    while temperatura > 1e-3:
        for _ in range(iteraciones_por_temperatura):
            nueva_solucion = solucion_actual[:]
            i = random.randint(0, N - 1)
            nueva_solucion[i] = random.randint(0, N - 1)
            energia_nueva = calcular_conflictos(nueva_solucion)
            
            if (energia_nueva < energia_actual or 
                random.random() < math.exp((energia_actual - energia_nueva) / temperatura)):
                solucion_actual = nueva_solucion
                energia_actual = energia_nueva
                
                if energia_actual < mejor_energia:
                    mejor_solucion = solucion_actual[:]
                    mejor_energia = energia_actual
        
        temperatura *= tasa_enfriamiento
    
    return mejor_solucion, mejor_energia

def visualizar_tablero(tablero):
    N = len(tablero)
    matriz = [[0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        matriz[tablero[i]][i] = "Reyna"  
    return matriz

N = 8
solucion, energia = simulado_recocido(N)
print(f"Para {N} reinas, la mejor solución encontrada: {solucion} con {energia} conflictos.")


tablero_visual = visualizar_tablero(solucion)
for fila in tablero_visual:
    print(fila)
