# ✅ CHECKLIST DE VALIDACIÓN MANUAL - PENNY STOCKS

## Guía de decisión paso a paso (sin código)

**Propósito:** Validar manualmente la estrategia antes de automatizar  
**Uso:** Aplicar a 20-30 casos históricos para calibrar umbrales  
**Tiempo:** 5-10 minutos por símbolo

---

## 📋 HOJA DE EVALUACIÓN RÁPIDA

```
SÍMBOLO: __________     FECHA: __________     ANALISTA: __________

PRECIO ACTUAL: $______     VOLUMEN HOY: ________

DECISIÓN FINAL: ⬜ COMPRAR  ⬜ WATCHLIST  ⬜ RECHAZAR
```

---

## FASE 1: FILTROS DE ELIMINACIÓN INMEDIATA ❌

**Si CUALQUIERA de estos es TRUE → RECHAZAR INMEDIATAMENTE**

```
⬜ Precio fuera de rango $0.50 - $10.00
⬜ Precio subió > 15% en últimos 3 días
⬜ Volumen explotó > 4x promedio AYER (no hoy)
⬜ RSI > 70 en cierre anterior
⬜ Gap up > 10% en apertura hoy
⬜ Cierre de ayer bajo VWAP del día
⬜ Mercado general: SPY -1.5% o más hoy
⬜ VIX > 28 (pánico)
⬜ Delisting warning o bankruptcy talk
⬜ Float > 100M shares (demasiado grande)

Si marcaste ALGUNO → STOP. Este símbolo NO califica.
Score = 0. No continuar.
```

---

## FASE 2: ANÁLISIS DE SETUP ESTRUCTURAL (40 puntos máximo)

### 2.1 Compresión de Precio (0-15 puntos)

**Análisis visual en gráfico:**
```
¿Cuántos días consecutivos en rango estrecho?
⬜ 8+ días en rango < 8%        → +15 puntos ⭐⭐⭐
⬜ 5-7 días en rango < 10%      → +12 puntos ⭐⭐
⬜ 3-4 días en rango < 12%      → +8 puntos ⭐
⬜ < 3 días o rango > 12%       → +0 puntos

Bollinger Bands:
⬜ Bandas muy estrechas (squeeze visible)   → +3 puntos extra
⬜ Bandas normales                          → +0 puntos

SUBTOTAL COMPRESIÓN: _____ / 15 puntos
```

### 2.2 Volumen Seco (0-10 puntos)

**Comportamiento de volumen pre-explosión:**
```
Últimos 3-5 días:
⬜ Volumen consistentemente < 80% del promedio 20d    → +10 puntos ⭐⭐⭐
⬜ Volumen variable pero trending down               → +6 puntos ⭐⭐
⬜ Volumen estable cerca del promedio                → +3 puntos ⭐
⬜ Volumen ya aumentando                             → +0 puntos

Gráfico de volumen:
⬜ Sin picos > 2x en últimos 5 días    → +0 puntos (bueno, no suma)
⬜ Hay picos recientes > 2x             → -5 puntos (mala señal)

SUBTOTAL VOLUMEN SECO: _____ / 10 puntos
```

### 2.3 Estructura Fundamental (0-10 puntos)

**Datos clave para squeeze potential:**
```
Short Interest:
⬜ > 20%        → +4 puntos ⭐⭐⭐
⬜ 15-20%       → +3 puntos ⭐⭐
⬜ 10-15%       → +2 puntos ⭐
⬜ < 10%        → +0 puntos

Days to Cover:
⬜ > 3.0 días    → +3 puntos ⭐⭐
⬜ 2.0-3.0 días  → +2 puntos ⭐
⬜ < 2.0 días    → +0 puntos

Borrow Rate:
⬜ > 40%        → +3 puntos ⭐⭐⭐
⬜ 25-40%       → +2 puntos ⭐⭐
⬜ < 25%        → +1 punto ⭐
⬜ Desconocido  → +1 punto (estimado)

SUBTOTAL FUNDAMENTAL: _____ / 10 puntos
```

### 2.4 Float y Liquidez (0-5 puntos)

```
Float size:
⬜ < 20M shares     → +3 puntos ⭐⭐⭐
⬜ 20-50M shares    → +2 puntos ⭐⭐
⬜ 50-100M shares   → +1 punto ⭐
⬜ > 100M shares    → +0 puntos

Volumen promedio diario:
⬜ > 500K shares    → +2 puntos ⭐⭐
⬜ 200-500K shares  → +1 punto ⭐
⬜ < 200K shares    → +0 puntos

SUBTOTAL FLOAT/LIQUIDEZ: _____ / 5 puntos
```

### TOTAL FASE 2 (SETUP): _____ / 40 puntos

```
Evaluación:
⬜ 30-40 puntos: Setup EXCELENTE - Alta prioridad watchlist ⭐⭐⭐
⬜ 20-29 puntos: Setup BUENO - Monitorear de cerca ⭐⭐
⬜ 10-19 puntos: Setup DÉBIL - Watchlist baja prioridad ⭐
⬜ < 10 puntos: Setup INSUFICIENTE - Rechazar ❌
```

---

## FASE 3: TRIGGER DE ENTRADA (40 puntos máximo)

**IMPORTANTE:** Solo evalúa esta fase si FASE 2 ≥ 20 puntos

### 3.1 Volumen Explosivo (0-15 puntos)

```
Ratio de volumen hoy:
⬜ > 4x promedio 20d        → +15 puntos ⭐⭐⭐
⬜ 3-4x promedio 20d        → +12 puntos ⭐⭐
⬜ 2.5-3x promedio 20d      → +9 puntos ⭐⭐
⬜ 2-2.5x promedio 20d      → +6 puntos ⭐
⬜ < 2x promedio 20d        → +0 puntos

Aceleración intradía (compara primera hora vs segunda hora):
⬜ Volumen acelera > 1.5x    → +3 puntos extra
⬜ Volumen estable           → +0 puntos

Día de explosión:
⬜ PRIMER día con vol > 2.5x     → +0 puntos (perfecto, pero ya contado)
⬜ SEGUNDO día con vol > 2.5x    → -10 puntos ❌ (TARDE)
⬜ TERCER+ día con vol > 2.5x    → -20 puntos ❌❌ (MUY TARDE)

SUBTOTAL VOLUMEN EXPLOSIVO: _____ / 15 puntos
```

### 3.2 Breakout Técnico (0-10 puntos)

```
Precio vs máximos previos:
⬜ > Máximo de 10 días    → +5 puntos ⭐⭐
⬜ > Máximo de 5 días     → +3 puntos ⭐
⬜ No hay breakout        → +0 puntos

Medias móviles:
⬜ Precio > SMA(20) > SMA(50)      → +3 puntos ⭐⭐
⬜ Precio > SMA(20), pero SMA(20) < SMA(50)  → +2 puntos ⭐
⬜ Precio < SMA(20)                 → +0 puntos

Posición en el rango del día:
⬜ Cierre en top 30% del rango     → +2 puntos ⭐
⬜ Cierre en middle 40% del rango  → +1 punto
⬜ Cierre en bottom 30% del rango  → +0 puntos

SUBTOTAL BREAKOUT: _____ / 10 puntos
```

### 3.3 Momentum (0-10 puntos)

```
RSI actual:
⬜ 55-65 (zona dorada)     → +5 puntos ⭐⭐⭐
⬜ 50-55 (iniciando)       → +4 puntos ⭐⭐
⬜ 65-70 (fuerte)          → +2 puntos ⭐
⬜ > 70 (sobrecomprado)    → -10 puntos ❌
⬜ < 50 (débil)            → +0 puntos

Dirección RSI:
⬜ Cruzó 55 desde abajo HOY      → +3 puntos ⭐⭐
⬜ Ya estaba > 55 ayer            → +1 punto
⬜ Bajando                        → +0 puntos

MACD:
⬜ Histograma verde y creciendo    → +2 puntos ⭐
⬜ Histograma verde pero plano     → +1 punto
⬜ Histograma rojo                 → +0 puntos

SUBTOTAL MOMENTUM: _____ / 10 puntos
```

### 3.4 Confirmación Intradía (0-5 puntos)

```
VWAP:
⬜ Precio cierra > VWAP           → +2 puntos ⭐⭐
⬜ Precio cerca de VWAP           → +1 punto ⭐
⬜ Precio cierra < VWAP           → -5 puntos ❌

Bid/Ask spread:
⬜ Spread < 5%      → +2 puntos ⭐
⬜ Spread 5-10%     → +1 punto
⬜ Spread > 10%     → +0 puntos

Horario de evaluación:
⬜ 10:30 AM - 3:00 PM    → +1 punto (timing ideal)
⬜ Otra hora              → +0 puntos

SUBTOTAL CONFIRMACIÓN: _____ / 5 puntos
```

### TOTAL FASE 3 (TRIGGER): _____ / 40 puntos

```
Evaluación:
⬜ 30-40 puntos: Trigger EXCELENTE - Señal fuerte ⭐⭐⭐
⬜ 20-29 puntos: Trigger BUENO - Señal válida ⭐⭐
⬜ 10-19 puntos: Trigger DÉBIL - Esperar mejor momento ⭐
⬜ < 10 puntos: Trigger INSUFICIENTE - No entrar ❌
```

---

## FASE 4: CONTEXTO DE MERCADO (20 puntos máximo)

### 4.1 Mercado General (0-10 puntos)

```
S&P 500 (SPY) hoy:
⬜ Verde > +0.5%        → +5 puntos ⭐⭐⭐
⬜ Verde 0% a +0.5%     → +3 puntos ⭐⭐
⬜ Rojo 0% a -0.5%      → +1 punto ⭐
⬜ Rojo -0.5% a -1.5%   → -3 puntos
⬜ Rojo < -1.5%         → -10 puntos ❌ (no operar)

Nasdaq (QQQ) hoy:
⬜ Confirma dirección de SPY    → +2 puntos ⭐
⬜ Contradice SPY               → -2 puntos

Trend de 5 días SPY:
⬜ Alcista clara       → +3 puntos ⭐⭐
⬜ Lateral             → +1 punto ⭐
⬜ Bajista             → -2 puntos

SUBTOTAL MERCADO: _____ / 10 puntos
```

### 4.2 Volatilidad (VIX) (0-5 puntos)

```
VIX actual:
⬜ < 15 (calma)         → +5 puntos ⭐⭐⭐
⬜ 15-20 (normal)       → +3 puntos ⭐⭐
⬜ 20-25 (elevado)      → +1 punto ⭐
⬜ > 25 (pánico)        → -10 puntos ❌ (no operar)

SUBTOTAL VIX: _____ / 5 puntos
```

### 4.3 Sector (0-5 puntos)

```
Sector del símbolo:
⬜ Sector en rally (ej: biotech hot)       → +3 puntos ⭐⭐
⬜ Sector neutral                           → +1 punto ⭐
⬜ Sector en pánico                         → -5 puntos ❌

Correlación con sector:
⬜ Se mueve independiente del sector    → +2 puntos ⭐⭐
⬜ Se mueve con el sector               → +0 puntos

SUBTOTAL SECTOR: _____ / 5 puntos
```

### TOTAL FASE 4 (CONTEXTO): _____ / 20 puntos

```
Evaluación:
⬜ 15-20 puntos: Contexto EXCELENTE - Viento a favor ⭐⭐⭐
⬜ 10-14 puntos: Contexto BUENO - Neutral/positivo ⭐⭐
⬜ 5-9 puntos: Contexto DÉBIL - Precaución ⭐
⬜ < 5 puntos: Contexto MALO - Evitar ❌
```

---

## CÁLCULO FINAL Y DECISIÓN

### Score Total

```
FASE 2 (Setup):     _____ / 40 puntos
FASE 3 (Trigger):   _____ / 40 puntos  
FASE 4 (Contexto):  _____ / 20 puntos

SUBTOTAL:           _____ / 100 puntos

PENALIZACIONES aplicadas en Fase 1 y 3: _____ puntos

SCORE FINAL:        _____ / 100 puntos
```

### Matriz de Decisión

```
SCORE 70-100: COMPRA FUERTE ⭐⭐⭐
   ✅ Entrar con posición 3-5% del capital
   ✅ Setup + Trigger + Contexto alineados
   ✅ Alta convicción
   
   Acción: EJECUTAR COMPRA
   Stop loss: 8% bajo entrada o bajo breakout (lo que sea mayor)
   TP1: +12-15% (vender 30%)
   TP2: +25-30% (vender 30%)
   TP3: Trailing stop desde +15% (40% restante)

SCORE 55-69: COMPRA MODERADA ⭐⭐
   ⚠️  Entrar con posición 2-3% del capital
   ⚠️  Setup bueno pero trigger o contexto débil
   ⚠️  Convicción media
   
   Acción: EJECUTAR COMPRA (con cautela)
   Stop loss: 7% bajo entrada
   TP1: +10% (vender 40%)
   TP2: +20% (vender 30%)
   TP3: Trailing stop desde +12% (30% restante)

SCORE 40-54: WATCHLIST ACTIVA ⚪
   👁️  NO comprar aún
   👁️  Monitorear diariamente
   👁️  Esperar mejora de score
   
   Acción: AGREGAR A WATCHLIST
   Condiciones para upgrade: Score sube a 55+ en próximos 1-3 días
   Condiciones para eliminar: Score baja < 35 o pasan 5 días

SCORE < 40: RECHAZAR ❌
   ⛔ No califica
   ⛔ Alto riesgo / baja recompensa
   ⛔ Sin convicción
   
   Acción: ELIMINAR DE LISTA
   Re-evaluar: Solo si cambia fundamentalmente (nuevo catalizador)
```

---

## REGISTRO DE EVALUACIÓN

**Para documentar y aprender de cada evaluación:**

```
SÍMBOLO: __________
FECHA: __________
SCORE FINAL: _____ / 100

DECISIÓN: ⬜ COMPRA FUERTE  ⬜ COMPRA MODERADA  ⬜ WATCHLIST  ⬜ RECHAZAR

RAZONES PRINCIPALES (top 3):
1. _________________________________________________
2. _________________________________________________
3. _________________________________________________

BANDERAS ROJAS (si hay):
- _________________________________________________
- _________________________________________________

PRECIO ENTRADA (si aplica): $______
STOP LOSS: $______
TAKE PROFITS: TP1 $______ | TP2 $______ | TP3 Trailing

SEGUIMIENTO (actualizar diariamente):
Día 1: Precio $______ | Status: ____________
Día 2: Precio $______ | Status: ____________
Día 3: Precio $______ | Status: ____________
Día 4: Precio $______ | Status: ____________
Día 5: Precio $______ | Status: ____________

RESULTADO FINAL:
Salida: Día ___ a $______ 
P&L: _____%
Lección aprendida: _________________________________
```

---

## CASOS DE PRÁCTICA RECOMENDADOS

### Para calibrar tu juicio, evalúa estos 10 símbolos históricos:

```
1. BYND - Noviembre 2023 (el fracaso documentado)
2. GME - Enero 2021 (el squeeze legendario)
3. AMC - Mayo-Junio 2021 (squeeze múltiple)
4. DWAC - Octubre 2021 (explosión política)
5. BBBY - Agosto 2022 (rally y colapso)
6. MULN - Marzo 2022 (penny stock típico)
7. APRN - Agosto 2022 (squeeze fallido)
8. CVNA - Noviembre 2022 (recuperación post-colapso)
9. ENVX - Enero 2023 (momentum tech)
10. IONQ - Diciembre 2023 (quantum computing hype)

Para cada uno:
   a) Evalúa el día ANTES del gran movimiento
   b) Evalúa el día DEL gran movimiento (cuando otros compraron)
   c) Compara scores
   d) Documenta qué habría pasado con cada decisión
```

---

## VALIDACIÓN DE LA CHECKLIST

**Criterios de éxito después de 20 evaluaciones:**

```
✅ Win rate de decisiones "COMPRA FUERTE" > 60%
✅ Win rate de decisiones "COMPRA MODERADA" > 45%
✅ % de rechazos correctos (no perdiste nada) > 80%
✅ Ninguna compra en activos que colapsaron > 20% en 5 días
✅ Al menos 3 trades exitosos con > +20% ganancia

Si no cumples estos criterios:
   → Ajustar umbrales de puntuación
   → Revisar pesos de cada categoría
   → Agregar filtros adicionales
```

---

## 💡 TIPS FINALES PARA USO EFECTIVO

1. **Sé Honesto en la Evaluación**
   - No busques justificar una entrada
   - Si el score es bajo, es bajo
   - FOMO mata cuentas

2. **Documenta TODAS las Evaluaciones**
   - Incluso (especialmente) los rechazos
   - Aprenderás más de lo que NO hiciste

3. **Revisa Semanalmente tus Evaluaciones Pasadas**
   - ¿Qué funcionó?
   - ¿Qué falló?
   - ¿Qué patrones ves?

4. **No Modifiques Scores Después de Decidir**
   - La tentación será grande
   - Pero destruye el aprendizaje
   - El score es lo que es

5. **Usa un Timer de 10 Minutos**
   - No más de 10min por evaluación
   - Si necesitas más, el símbolo es muy complejo
   - Simplicidad = claridad

6. **Practica con Dinero de Papel Primero**
   - Al menos 20-30 evaluaciones
   - Hasta que tengas confianza
   - La confianza viene de los datos, no de la suerte

---

## 🎯 PRÓXIMO PASO DESPUÉS DE LA CHECKLIST

Una vez que hayas evaluado manualmente 20-30 símbolos:

1. ✅ Calcula tu win rate real
2. ✅ Identifica qué categorías fueron más predictivas
3. ✅ Ajusta pesos si es necesario
4. ✅ Codifica el sistema con confianza
5. ✅ El código implementará TU criterio validado

**La checklist es el blueprint del algoritmo.**

Si la checklist funciona manualmente → El código funcionará automatizado  
Si la checklist no funciona manualmente → Arreglar la estrategia, no el código

---

**Documento v1.0 - Octubre 23, 2025**  
**Autor: Framework de validación manual**  
**Uso: Calibración pre-implementación**

¿Listo para empezar a evaluar símbolos manualmente?
