🤖 RESUMEN DEL ALGORITMO DE TRADING - PENNY STOCK ROBOT ADVISOR
🎯 OBJETIVO PRINCIPAL
Detectar penny stocks con alto potencial de short squeeze antes de que exploten (como HTOO +60% o LCFY +20%).

📊 CÓMO FUNCIONA EL ALGORITMO
1. ANÁLISIS DE 5 SEÑALES CLAVE
El algoritmo analiza cada stock usando 5 señales específicas, cada una con un peso diferente:
🔍 SEÑALES ANALIZADAS:
├── 25% - Short Interest Cualificado
├── 25% - Confirmación de Momentum  
├── 20% - Riesgo de Delisting
├── 20% - Calidad del Volumen
└── 10% - Filtro de Liquidez
2. SISTEMA DE PUNTUACIÓN

Cada señal recibe un score de 0 a 1.0
Score final = suma ponderada de las 5 señales
Umbrales de decisión:

Score ≥ 0.75 → COMPRAR FUERTE
Score ≥ 0.60 → COMPRAR MODERADO
Score ≥ 0.45 → COMPRAR LIGERO
Score < 0.45 → ESPERAR




🔍 LAS 5 SEÑALES EN DETALLE
1️⃣ SHORT INTEREST CUALIFICADO (25%)
¿Qué mide? Si hay muchos "shorts" atrapados y es difícil/caro mantenerlos.
Analogía: Como contar soldados enemigos sin munición ni ruta de escape.
Datos que necesita:

short_interest_pct - % de acciones en short
days_to_cover - Días para cerrar todas las posiciones short
borrow_rate - Costo de mantener el short

Lógica:
pythonif short_interest > 30% AND days_to_cover > 3 AND borrow_rate > 50%:
    score = 1.0  # ¡Squeeze muy probable!
elif short_interest > 15% AND days_to_cover > 2:
    score = 0.6  # Posible squeeze
else:
    score = 0.2  # Poco probable

2️⃣ CONFIRMACIÓN DE MOMENTUM (25%)
¿Qué mide? Si el movimiento de precio tiene fuerza técnica real.
Analogía: No solo ver gente corriendo, sino verificar que corren en nuestra dirección.
Datos que necesita:

price vs vwap - Precio actual vs promedio ponderado
rsi - Fuerza relativa (30-80 = bueno)
volume_ratio - Volumen actual vs promedio
price_change_pct - Cambio de precio del día

Lógica:
pythonmomentum_signals = 0
if price > vwap: momentum_signals += 1           # Presión compradora
if 35 <= rsi <= 75: momentum_signals += 1       # RSI en zona ideal
if 2 <= volume_ratio <= 15: momentum_signals += 1  # Volumen óptimo
if price_change > 1%: momentum_signals += 1     # Subiendo
if price_change <= 25%: momentum_signals += 1   # Subida sustentable

score = momentum_signals / 5  # Score de 0 a 1

3️⃣ RIESGO DE DELISTING (20%)
¿Qué mide? Si hay catalizadores de "reverse split" que atraen especuladores.
Analogía: Apostar por diamantes en bruto antes de pulir.
Datos que necesita:

price - Precio actual
has_delisting_warning - Si tiene aviso de delisting

Lógica:
pythonif price < $1 AND has_delisting_warning:
    score = 0.8  # Catalizador fuerte
elif price < $1:
    score = 0.4  # Posible catalizador
else:
    score = 0.2  # Sin catalizador

4️⃣ CALIDAD DEL VOLUMEN (20%)
¿Qué mide? Si el alto volumen es por compras (bueno) o ventas (malo).
Analogía: Distinguir entre multitud comprando vs multitud huyendo.
Datos que necesita:

volume vs avg_volume_20d - Volumen actual vs promedio
price_change_pct - Dirección del precio
bid_ask_spread_pct - Liquidez real

Lógica:
pythonvolume_ratio = current_volume / avg_volume

if price_change > 5%:
    quality = 1.3  # Alto volumen subiendo = buying pressure
elif price_change > 0%:
    quality = 1.0  # Neutral positivo
else:
    quality = 0.4  # Alto volumen bajando = dumping

if bid_ask_spread > 8%:
    penalty = 0.7  # Penalizar baja liquidez

score = (volume_ratio / 8) * quality * penalty

5️⃣ FILTRO DE LIQUIDEZ (10%)
¿Qué mide? Si puedes entrar Y salir sin quedar atrapado.
Analogía: Verificar que existe puerta de salida antes de entrar.
Datos que necesita:

bid_ask_spread_pct - Diferencia entre compra/venta
market_depth_dollars - Profundidad del mercado
daily_dollar_volume - Volumen en dólares diario

Lógica:
python# FILTRO CRÍTICO - Si falla cualquiera, score = 0
if bid_ask_spread > 15%: return 0  # Spread prohibitivo
if market_depth < $5000: return 0  # Sin liquidez
if daily_volume < $50000: return 0  # Muy poco volumen

# Si pasa todos los filtros
return 0.8  # Liquidez aceptable

⚙️ CONFIGURACIONES DEL ALGORITMO
🔧 4 NIVELES DE AGRESIVIDAD:
ConfiguraciónUmbralesRSISpread MáxUsoConservative0.55, 0.70, 0.8545-6510%Alta precisiónBalanced0.50, 0.65, 0.8040-7012%EquilibrioAggressive0.45, 0.60, 0.7535-7515%Para HTOO/LCFYVery Aggressive0.40, 0.55, 0.7030-8020%Máximo trades

🎯 EJEMPLO PRÁCTICO: ¿POR QUÉ HTOO SUBIÓ +60%?
Datos hipotéticos de HTOO el día anterior:
pythonHTOO_data = {
    'price': 7.25,                    # Precio base
    'short_interest_pct': 28.5,       # Alto short interest ✅
    'days_to_cover': 4.2,             # Difícil cubrir ✅  
    'borrow_rate': 73.6,              # Caro mantener ✅
    'rsi': 74,                        # RSI alto (momentum) ✅
    'price_change_pct': 8.5,          # Subiendo fuerte ✅
    'volume_ratio': 6.3,              # Volumen 6x normal ✅
    'vwap': 6.9,                      # Precio > VWAP ✅
    'bid_ask_spread_pct': 8.2         # Liquidez aceptable ✅
}
Análisis del algoritmo:
🔍 SEÑALES:
├── Short Interest: 0.95 (28.5% + DTC 4.2d + Rate 73%)
├── Momentum: 1.00 (5/5 señales positivas)
├── Delisting: 0.20 (precio > $1)
├── Volume Quality: 0.88 (6x volumen subiendo)
└── Liquidez: 0.80 (spread aceptable)

📊 SCORE FINAL: 0.846
🎯 DECISIÓN: COMPRAR FUERTE (> 0.75)

💡 CONCEPTOS CLAVE PARA ENTENDER
🔥 Short Squeeze

Qué es: Cuando shorts se ven forzados a comprar para cerrar posiciones
Por qué ocurre: Precio sube → shorts pierden dinero → compran para limitar pérdidas → precio sube más
Indicadores: Alto SI% + difícil cubrir + caro mantener

📈 Momentum Técnico

RSI (Relative Strength Index): Mide si está "sobreprecio" (>70) o "barato" (<30)
VWAP (Volume Weighted Average Price): Precio promedio ponderado por volumen
Precio > VWAP = presión compradora dominante

💧 Liquidez

Bid-ask spread: Diferencia entre precio de compra y venta
Spread alto = difícil entrar/salir sin perder dinero
Market depth: Cuánto dinero hay esperando en el "order book"

⚖️ Risk Management

Position sizing: Nunca más del 2-3% del capital por trade
Stop loss: Salida automática si baja X% (basado en volatilidad)
Take profit: Vender por tramos en subidas (25%, 50%, 100%)


🚨 LIMITACIONES DEL ALGORITMO
❌ Lo que NO puede hacer:

Predecir noticias (FDA approvals, earnings surprises)
Detectar manipulación (pump & dump schemes)
Garantizar timing perfecto (puede ser días early/late)
Funcionar en bear markets extremos

⚠️ Riesgos principales:

Falsos positivos: Alto SI justificado (empresa realmente mal)
Timing imperfecto: Entrar demasiado temprano
Liquidity traps: Quedar atrapado en bid-ask spreads
Gap downs: Stops no funcionan en gaps extremos


🎯 RESUMEN EJECUTIVO
El algoritmo busca stocks donde:

Hay muchos shorts atrapados (difícil/caro salir)
El momentum técnico confirma presión compradora
Hay suficiente liquidez para entrar/salir
El volumen es de calidad (compradores, no vendedores)
Opcional: Catalizador de delisting/reverse split

Configuración recomendada para tu caso:

"Aggressive" para capturar HTOO/LCFY
Umbrales: 0.45, 0.60, 0.75
RSI expandido: 35-75
Máximo 2-3% por posición