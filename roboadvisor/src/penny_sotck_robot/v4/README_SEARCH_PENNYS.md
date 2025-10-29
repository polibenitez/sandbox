# 📊 PENNY STOCKS SCREENER - MEJORAS IMPLEMENTADAS

**Fecha:** 23 de Octubre, 2025
**Archivo:** `search_pennys.py`
**Estado:** ✅ Completado

---

## 🎯 MEJORAS IMPLEMENTADAS

### 1. ✅ Análisis de Sentimiento de Reddit

**Función:** `get_reddit_sentiment(symbol, max_results=50)`

**Qué hace:**
- Usa `snscrape reddit-search` para buscar menciones del símbolo
- Analiza hasta 50 posts recientes de Reddit
- Extrae título y contenido de cada post
- Calcula sentimiento (positivo/negativo/neutral)

**Comando ejecutado:**
```bash
snscrape --jsonl --max-results 50 reddit-search "CLOV"
```

**Output:**
```python
{
    'mentions': 25,              # Número de menciones
    'sentiment_score': 0.68,     # Score de -1 a 1
    'positive': 17,              # Posts positivos
    'neutral': 5,                # Posts neutrales
    'negative': 3,               # Posts negativos
    'hype_level': 'ALTO'         # MUY ALTO | ALTO | MEDIO | BAJO | NINGUNO
}
```

---

### 2. ✅ Análisis de Sentimiento Simple

**Función:** `analyze_sentiment_simple(text)`

**Palabras clave positivas:**
- `moon`, `bullish`, `buy`, `rocket`, `squeeze`, `breakout`
- `pump`, `gain`, `profit`, `long`, `call`, `up`, `rally`
- `bounce`, `support`, `strong`, `winning`, `great`, `amazing`

**Palabras clave negativas:**
- `bear`, `bearish`, `sell`, `dump`, `crash`, `tank`, `drop`
- `fall`, `short`, `put`, `down`, `bag`, `holder`, `loss`
- `scam`, `fraud`, `avoid`, `warning`, `risky`

**Lógica:**
```python
if palabras_positivas > palabras_negativas:
    return 1  # Positivo
elif palabras_negativas > palabras_positivas:
    return -1  # Negativo
else:
    return 0  # Neutral
```

---

### 3. ✅ Nivel de Hype

**Niveles definidos:**

| Menciones | Sentimiento | Nivel de Hype |
|-----------|-------------|---------------|
| ≥ 30 | > 0.3 | **MUY ALTO** 🔥🔥🔥 |
| ≥ 20 | > 0.2 | **ALTO** 🔥🔥 |
| ≥ 10 | - | **MEDIO** 🔥 |
| > 0 | - | **BAJO** 💡 |
| 0 | - | **NINGUNO** - |

---

### 4. ✅ Scoring Mejorado

**Antes (solo Twitter):**
```python
score = change_pct * 0.6 + log(twitter) * 10 * 0.4
```

**Ahora (Twitter + Reddit):**
```python
score = (
    change_pct * 0.5 +                          # Técnica: 50%
    log(twitter) * 5 * 0.2 +                   # Twitter: 20%
    (log(reddit) * 8 + sentiment * 10) * 0.3   # Reddit: 30%
)
```

**Ponderación:**
- **50%** - Análisis técnico (cambio de precio)
- **20%** - Menciones en Twitter
- **30%** - Reddit (menciones + sentimiento)

**Ejemplo:**

```
Símbolo: CLOV
Cambio: +8.5%
Twitter: 25 menciones
Reddit: 30 menciones, sentimiento +0.68

Score técnico: 8.5 * 0.5 = 4.25
Score Twitter: log(26) * 5 * 0.2 = 3.23
Score Reddit: (log(31) * 8 + 0.68 * 10) * 0.3 = 10.53
TOTAL: 18.01
```

---

### 5. ✅ Formato de Archivo Mejorado

**Antes:**
```
penny_stocks_pro_2025-10-23.csv
```

**Ahora:**
```
pennys_2025-10-23_14-30-45.csv
```

**Ventajas:**
- Incluye hora exacta del análisis
- Permite múltiples análisis por día
- Más fácil de ordenar cronológicamente
- Formato más compacto (`pennys_` en lugar de `penny_stocks_pro_`)

---

### 6. ✅ Columnas del CSV

**Columnas incluidas:**

| Columna | Descripción | Ejemplo |
|---------|-------------|---------|
| `symbol` | Ticker del símbolo | CLOV |
| `price` | Precio actual | 2.45 |
| `change_pct` | Cambio porcentual | 8.5 |
| `volume_ratio` | Ratio de volumen | 3.2 |
| `avg_volume` | Volumen promedio | 15000000 |
| `twitter_mentions` | Menciones en Twitter | 25 |
| `reddit_mentions` | Menciones en Reddit | 30 |
| `reddit_sentiment` | Sentimiento Reddit | 0.68 |
| `reddit_positive` | Posts positivos | 20 |
| `reddit_neutral` | Posts neutrales | 7 |
| `reddit_negative` | Posts negativos | 3 |
| `reddit_hype` | Nivel de hype | ALTO |
| `total_social_mentions` | Total menciones | 55 |
| `hype_score` | Score de hype | 13.76 |
| `score` | **Score total** | **18.01** |

---

## 🚀 CÓMO USAR

### Instalación de dependencias

```bash
# Instalar snscrape (si no está instalado)
pip install snscrape

# Verificar instalación
snscrape --version
```

### Ejecución

```bash
cd /path/to/v4/
python search_pennys.py
```

### Output esperado

```
🔍 Analizando 250 tickers...
📊 Incluye: Análisis técnico + Twitter + Reddit

  ✓ AAPL: Precio 175.23, Reddit 0 menciones (NINGUNO)
  ✓ CLOV: Precio 2.45, Reddit 30 menciones (ALTO)
  ✓ BYND: Precio 2.15, Reddit 18 menciones (MEDIO)
  ...

📈 Progreso: 50/250 tickers analizados...
📈 Progreso: 100/250 tickers analizados...
...

================================================================================
🔥 PENNY STOCKS CON MAYOR POTENCIAL
================================================================================
 symbol  price  change_pct  volume_ratio  twitter_mentions  reddit_mentions  reddit_sentiment reddit_hype   score
   CLOV   2.45        8.50          3.20                25               30              0.68        ALTO   18.01
   BYND   2.15        7.20          2.80                15               18              0.45       MEDIO   14.23
   ...
================================================================================

📋 REPORTE DETALLADO:

1. 💰 $CLOV - $2.45
   📈 Cambio: +8.50%
   📊 Volumen ratio: 3.2x
   🐦 Twitter: 25 menciones
   🤖 Reddit: 30 menciones | Sentimiento: +0.68 | Hype: ALTO
   💯 Score total: 18.01

2. 💰 $BYND - $2.15
   📈 Cambio: +7.20%
   📊 Volumen ratio: 2.8x
   🐦 Twitter: 15 menciones
   🤖 Reddit: 18 menciones | Sentimiento: +0.45 | Hype: MEDIO
   💯 Score total: 14.23

...

💾 Resultados guardados en: pennys_2025-10-23_14-30-45.csv
📱 Reporte enviado por Telegram
```

---

## 📊 EJEMPLO DE DATOS

### Ejemplo de CSV generado

```csv
symbol,price,change_pct,volume_ratio,avg_volume,twitter_mentions,reddit_mentions,reddit_sentiment,reddit_positive,reddit_neutral,reddit_negative,reddit_hype,total_social_mentions,hype_score,score
CLOV,2.45,8.50,3.20,15000000,25,30,0.68,20,7,3,ALTO,55,13.76,18.01
BYND,2.15,7.20,2.80,12000000,15,18,0.45,12,4,2,MEDIO,33,9.87,14.23
OPEN,3.80,6.50,2.50,8000000,10,12,0.33,8,3,1,MEDIO,22,7.45,11.68
```

---

## 🔍 ANÁLISIS DE CÓDIGO

### Flujo de ejecución

```
1. get_all_tickers(250)
   └─> Obtiene lista de tickers del mercado

2. Para cada ticker:
   a) analyze_ticker(symbol)
      └─> Análisis técnico (precio, volumen, cambio %)

   b) get_twitter_mentions(symbol)
      └─> Busca menciones en Twitter

   c) get_reddit_sentiment(symbol)
      └─> Busca menciones en Reddit
      └─> Analiza sentimiento de cada post
      └─> Calcula nivel de hype

   d) Calcular score total
      └─> Técnica (50%) + Twitter (20%) + Reddit (30%)

3. Ordenar por score descendente

4. Seleccionar Top 10

5. Guardar en CSV con timestamp

6. Mostrar reporte detallado

7. Enviar a Telegram (opcional)
```

---

## ⚙️ CONFIGURACIÓN

### Parámetros ajustables

```python
# En el archivo search_pennys.py

PRICE_LIMIT = 5              # Precio máximo (penny stocks)
CHANGE_PCT_MIN = 5           # Cambio mínimo en % para calificar
VOLUME_RATIO_MIN = 2         # Ratio de volumen mínimo
TOP_N = 10                   # Número de resultados a mostrar

# Telegram (opcional)
TELEGRAM_TOKEN = "tu_token"
TELEGRAM_CHAT_ID = "tu_chat_id"
```

---

## 🎯 CASOS DE USO

### Caso 1: Encontrar acciones con hype en Reddit

**Buscar:** Penny stocks con alta actividad en Reddit

**Filtro:**
```python
df[df['reddit_hype'].isin(['ALTO', 'MUY ALTO'])]
```

**Resultado:**
```
symbol  price  reddit_mentions  reddit_sentiment  reddit_hype
CLOV    2.45              30              0.68         ALTO
GME     4.20              45              0.82    MUY ALTO
```

---

### Caso 2: Sentimiento positivo pero pocas menciones

**Buscar:** Acciones con sentimiento muy positivo pero aún poco conocidas

**Filtro:**
```python
df[(df['reddit_sentiment'] > 0.5) & (df['reddit_mentions'] < 15)]
```

**Uso:** Entrar temprano antes del hype masivo

---

### Caso 3: Comparar Twitter vs Reddit

**Buscar:** Dónde está el verdadero hype

```python
df['twitter_dominant'] = df['twitter_mentions'] > df['reddit_mentions']
```

**Interpretación:**
- Twitter dominante: Hype más retail/general
- Reddit dominante: Hype más especializado/dedicado

---

## 🔧 TROUBLESHOOTING

### Error: `snscrape: command not found`

**Solución:**
```bash
pip install snscrape
```

### Error: Timeout al buscar en Reddit

**Causa:** Reddit puede tener muchos resultados

**Solución:** Ya implementado con timeout de 30 segundos
```python
result = subprocess.run(cmd, timeout=30)
```

### No aparecen menciones de Reddit

**Posibles causas:**
1. El símbolo es muy genérico (ej: "GO", "IT")
2. No hay actividad reciente en Reddit
3. snscrape tiene límites de rate

**Solución:**
- Verificar manualmente: `snscrape reddit-search "CLOV"`
- Ajustar `max_results` si es necesario

---

## 📈 MEJORAS FUTURAS

### Posibles expansiones

1. **Subreddits específicos**
   ```python
   # Buscar solo en r/wallstreetbets, r/pennystocks
   cmd = ["snscrape", "reddit-search", f"{symbol} subreddit:wallstreetbets"]
   ```

2. **Análisis de tiempo**
   - Tendencia de menciones (subiendo/bajando)
   - Hype en últimas 24h vs última semana

3. **Análisis de comentarios**
   - No solo posts, también comentarios
   - Detección de spam/bots

4. **Integración con V4**
   - Usar Reddit hype como señal en el scoring V4
   - Penalizar si hype es excesivo (pump & dump)

5. **Sentiment analysis avanzado**
   - Usar VADER o TextBlob para mejor precisión
   - Detección de sarcasmo

---

## 📊 COMPARACIÓN CON VERSIÓN ANTERIOR

| Aspecto | Versión Anterior | Versión Nueva |
|---------|------------------|---------------|
| **Redes sociales** | Solo Twitter | Twitter + Reddit |
| **Sentimiento** | No | Sí (positivo/neutral/negativo) |
| **Nivel de hype** | No | Sí (5 niveles) |
| **Scoring** | 60/40 | 50/20/30 |
| **Nombre archivo** | `penny_stocks_pro_2025-10-23.csv` | `pennys_2025-10-23_14-30-45.csv` |
| **Columnas CSV** | 8 | 15 |
| **Reporte** | Básico | Detallado con emojis |

---

## ✅ CHECKLIST DE IMPLEMENTACIÓN

- [x] Función `get_reddit_sentiment()` implementada
- [x] Función `analyze_sentiment_simple()` implementada
- [x] Integración con `snscrape reddit-search`
- [x] Cálculo de nivel de hype (5 niveles)
- [x] Nuevo sistema de scoring (50/20/30)
- [x] Formato de archivo con timestamp completo
- [x] Columnas adicionales en CSV (15 total)
- [x] Reporte detallado mejorado
- [x] Documentación completa
- [x] Manejo de errores y timeouts

---

## 🎓 CONCLUSIÓN

El screener ahora combina:
- ✅ Análisis técnico (precio, volumen)
- ✅ Sentimiento de Twitter
- ✅ Sentimiento de Reddit (NUEVO)
- ✅ Nivel de hype (NUEVO)

**Resultado:** Mejor detección de penny stocks con potencial explosivo basado en análisis social + técnico.

---

**Archivo:** `search_pennys.py`
**Versión:** 2.0 - Reddit Enhanced
**Fecha:** 23 de Octubre, 2025

🚀 **Happy Screening!**
