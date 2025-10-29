# 📊 Guía de Análisis de Sentimientos - Sistema V5

## ¿Cómo Funciona Actualmente?

### Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│  PENNY STOCK ADVISOR V5                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Alternative Data Provider                           │   │
│  │  ├── Reddit Sentiment (mentions, sentiment, trending)│   │
│  │  └── Short Borrow Rates (borrow %, availability)    │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  FASE 3: Context + Alternative Data                 │   │
│  │  Score: 0-100 basado en:                            │   │
│  │  • Reddit sentiment (-15 a +15 pts)                 │   │
│  │  • Trending en Reddit (+10 pts)                     │   │
│  │  • Mentions alto (+5-10 pts)                        │   │
│  │  • Short borrow rate (+5-15 pts)                    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Flujo de Datos

```python
# 1. Leer datos locales
reddit = alt_data_provider.get_reddit_sentiment('ASST')
# Returns: {mentions: 350, sentiment: 'bullish', score: 0.85, trending: True}

short = alt_data_provider.get_short_borrow_rate('ASST')
# Returns: {borrow_rate: 80%, availability: 'hard_to_borrow', squeeze_risk: 'high'}

# 2. Calcular score combinado
combined_score = calculate_combined_score(reddit, short)
# Returns: 97.8/100

# 3. Incorporar en Fase 3 del análisis
phase3_score = context_score + alternative_data_score
```

---

## 🚨 Limitaciones del Sistema Actual

### CSV Estático

**Problema:**
```
data/reddit_sentiment.csv (creado: 27 Oct 2025, 12:55 PM)
    ↓
Pasan 24 horas...
    ↓
Datos desactualizados ❌
    ↓
Análisis usa sentiment viejo
```

**Impacto:**
- ✅ Funciona para **testing/desarrollo**
- ❌ **No recomendado para trading real**
- ❌ Sentiment puede cambiar en **horas** (especialmente en penny stocks volátiles)
- ❌ Trending flags incorrectos

### Ejemplo Real

```
27 Oct 12:00 PM: ASST tiene 350 mentions, bullish
                 → CSV guardado

28 Oct 09:00 AM: ASST explota en Reddit (1000+ mentions nuevos)
                 → Sistema lee CSV viejo (350 mentions) ❌
                 → Score incorrecto
```

---

## ✅ Soluciones para Mantener Actualizado

### Opción 1: Manual (Antes de cada análisis)

**Ventajas:**
- ✅ Control total
- ✅ No requiere configuración compleja
- ✅ Gratis

**Desventajas:**
- ❌ Requiere intervención manual
- ❌ Fácil olvidarlo

```bash
# Ejecutar antes del análisis
python update_reddit_sentiment.py
python integration_v5_trading_manager.py
```

---

### Opción 2: Cron Job Diario (Automático)

**Configuración:**

```bash
# Editar crontab
crontab -e

# Agregar línea (ejecutar a las 8:00 AM diariamente)
0 8 * * * cd /path/to/v5 && python update_reddit_sentiment.py

# Ejecutar análisis a las 8:30 AM
30 8 * * * cd /path/to/v5 && python integration_v5_trading_manager.py
```

**Ventajas:**
- ✅ Totalmente automático
- ✅ No requiere intervención
- ✅ Datos frescos todos los días

**Desventajas:**
- ❌ Solo se actualiza 1x al día
- ❌ Si hay volatilidad intraday, puede quedar desactualizado

---

### Opción 3: Script Integrado (Recomendado)

Modificar `integration_v5_trading_manager.py` para actualizar antes de analizar:

```python
def run_full_analysis(self):
    """Ejecuta análisis completo V5"""

    # 1. Actualizar datos alternativos (opcional)
    if self.update_alt_data:
        logger.info("Actualizando datos alternativos...")
        update_reddit_csv(self.watchlist)

    # 2. Analizar contexto de mercado
    context = self.market_context.get_market_context()

    # 3. Analizar símbolos...
```

**Ventajas:**
- ✅ Siempre actualizado antes de analizar
- ✅ Un solo comando
- ✅ Datos en tiempo real

**Desventajas:**
- ❌ Análisis tarda más (1-2 min por scraping)
- ❌ Rate limits de Reddit API

---

## 🔧 Configuración de Reddit API

### Paso 1: Crear App en Reddit

1. Ir a: https://www.reddit.com/prefs/apps
2. Scroll abajo → "create another app..."
3. Completar:
   ```
   Name: PennyStockAdvisor
   Type: script
   Description: Sentiment analysis
   Redirect URI: http://localhost:8080
   ```
4. Click "create app"
5. Obtener:
   - **client_id**: debajo de "personal use script"
   - **client_secret**: al lado de "secret"

### Paso 2: Configurar Credenciales

Editar `update_reddit_sentiment.py`:

```python
REDDIT_CONFIG = {
    'client_id': 'abc123xyz',              # Tu client_id
    'client_secret': 'def456uvw',          # Tu client_secret
    'user_agent': 'PennyStockAdvisor/1.0'  # Dejar así
}
```

### Paso 3: Instalar Dependencias

```bash
pip install praw textblob
python -m textblob.download_corpora
```

### Paso 4: Probar

```bash
python update_reddit_sentiment.py
```

Salida esperada:
```
======================================================================
ACTUALIZADOR DE REDDIT SENTIMENT
======================================================================
Conectando a Reddit API...
Buscando menciones de ASST...
   ASST: 350 menciones, sentiment: bullish (0.85)
Buscando menciones de VIVK...
   VIVK: 280 menciones, sentiment: bullish (0.90)
...

✅ Archivo actualizado: ./data/reddit_sentiment.csv
   • 42 símbolos actualizados
   • Timestamp: 2025-10-27 13:45:23

📊 Estadísticas:
   • Bullish: 18
   • Bearish: 3
   • Trending: 5
   • Total menciones: 2,450

🔥 Top 5 más mencionados:
   🔥 ASST: 350 menciones (bullish)
   🔥 VIVK: 280 menciones (bullish)
      BYND: 180 menciones (bullish)
      CLOV: 150 menciones (bullish)
      CHPT: 120 menciones (bullish)
```

---

## 📊 Alternativas al Scraping

### Opción A: APIs Comerciales

Servicios pagos con sentiment pre-calculado:

1. **Quiver Quantitative** (https://www.quiverquant.com/)
   - Sentiment de WallStreetBets
   - API disponible
   - $20-50/mes

2. **Social Sentiment** (https://socialsentiment.io/)
   - Agregado de Reddit + Twitter
   - API REST
   - $30-100/mes

3. **Swaggystocks** (https://swaggystocks.com/)
   - Específico para WSB
   - Gratis (con límites)

### Opción B: WebSocket Real-Time

Para traders muy activos:

```python
# Pseudo-código
import asyncio
import websocket

async def stream_reddit():
    subreddit = reddit.subreddit('wallstreetbets')
    for submission in subreddit.stream.submissions():
        # Analizar en tiempo real
        sentiment = analyze_sentiment(submission.title)
        update_cache(submission.ticker, sentiment)
```

---

## 🎯 Recomendación Final

### Para Trading Real:

**MEJOR OPCIÓN: Opción 3 (Script Integrado) + Cron Diario**

```bash
# 1. Configurar cron para actualizar datos a las 7:00 AM
0 7 * * * cd /path/to/v5 && python update_reddit_sentiment.py

# 2. Ejecutar análisis cuando quieras durante el día
python integration_v5_trading_manager.py
```

**Beneficios:**
- ✅ Datos actualizados todas las mañanas antes del mercado
- ✅ Puedes ejecutar análisis múltiples veces al día con datos frescos
- ✅ Balance entre actualización y rate limits
- ✅ No depende de intervención manual

### Para Testing/Desarrollo:

**CSV estático está bien** - Los datos de ejemplo sirven para:
- ✅ Probar el sistema
- ✅ Desarrollo de features
- ✅ Backtesting con datos históricos

---

## 📝 Checklist de Implementación

- [ ] Crear cuenta Reddit API
- [ ] Obtener client_id y client_secret
- [ ] Configurar `update_reddit_sentiment.py`
- [ ] Instalar dependencias: `pip install praw textblob`
- [ ] Descargar corpora: `python -m textblob.download_corpora`
- [ ] Probar manualmente: `python update_reddit_sentiment.py`
- [ ] Configurar cron job (opcional)
- [ ] Modificar `integration_v5_trading_manager.py` (opcional)

---

## ⚠️ Limitaciones de Reddit API

**Rate Limits:**
- 60 requests por minuto
- 1000 requests por día (sin OAuth)

**Cómo evitarlos:**
- ✅ Actualizar 1x al día (no intraday)
- ✅ Usar cache
- ✅ Sleep entre requests
- ✅ Considerar APIs comerciales para high-frequency

---

## 🔍 Validación de Datos

Para verificar que los datos están actualizados:

```python
import pandas as pd

df = pd.read_csv('data/reddit_sentiment.csv')
print(f"Último update: {os.path.getmtime('data/reddit_sentiment.csv')}")
print(f"Top trending: \n{df[df['trending'] == True]}")
```

---

## 📚 Referencias

- Reddit API Docs: https://www.reddit.com/dev/api/
- PRAW Docs: https://praw.readthedocs.io/
- TextBlob Docs: https://textblob.readthedocs.io/
