# 📋 CHANGELOG - V4 + SEARCH PENNYS

**Fecha:** 23 de Octubre, 2025

---

## 🚀 V4.0.0 - PARADIGM SHIFT (Primera parte del día)

### Nuevos archivos

1. **`penny_stock_advisor_v4.py`** (650 líneas)
   - Sistema de scoring de 3 capas
   - Detección de día de explosión
   - Penalizaciones severas

2. **`integration_v4_trading_manager.py`** (550 líneas)
   - Análisis de contexto de mercado (SPY/QQQ/VIX)
   - Gestión de salidas mejorada
   - Integración completa

3. **Documentación V4:**
   - `README_V4.md` - Guía completa (900+ líneas)
   - `QUICK_START.md` - Inicio rápido (300+ líneas)
   - `IMPLEMENTATION_SUMMARY.md` - Resumen de implementación

### Funciones implementadas (16 nuevas)

**FASE A - Scoring:**
- ✅ `detect_setup_compression()`
- ✅ `get_explosion_day_number()`
- ✅ `calculate_phase1_setup_score()`
- ✅ `calculate_phase2_trigger_score()`
- ✅ `calculate_phase3_context_score()`
- ✅ `apply_penalties()`

**FASE B - Contexto:**
- ✅ `get_market_context()`
- ✅ `get_vix_level()`
- ✅ `get_sector_sentiment()`

**FASE C - Salidas:**
- ✅ `detect_bearish_divergence()`
- ✅ `calculate_dynamic_trailing_stop()`
- ✅ `calculate_partial_exit_levels()`
- ✅ `detect_distribution_pattern()`

### Cambio fundamental

**V3:** Compraba la explosión (tardío) ❌
**V4:** Anticipa la compresión (temprano) ✅

---

## 📊 V2.0.0 - SEARCH PENNYS ENHANCED (Segunda parte del día)

### Archivo modificado

**`search_pennys.py`** - Screener de penny stocks

### Mejoras implementadas

#### 1. ✅ Análisis de Reddit

**Nueva función:** `get_reddit_sentiment(symbol, max_results=50)`

**Qué hace:**
- Usa `snscrape reddit-search` para buscar menciones
- Analiza hasta 50 posts de Reddit
- Calcula sentimiento (positivo/negativo/neutral)
- Determina nivel de hype (5 niveles)

**Tecnología:**
```bash
snscrape --jsonl --max-results 50 reddit-search "CLOV"
```

**Output:**
```python
{
    'mentions': 30,
    'sentiment_score': 0.68,  # -1 a +1
    'positive': 20,
    'neutral': 7,
    'negative': 3,
    'hype_level': 'ALTO'      # MUY ALTO | ALTO | MEDIO | BAJO | NINGUNO
}
```

#### 2. ✅ Análisis de Sentimiento

**Nueva función:** `analyze_sentiment_simple(text)`

**Palabras clave:**
- Positivas: moon, bullish, buy, rocket, squeeze, breakout, etc.
- Negativas: bear, bearish, sell, dump, crash, tank, etc.

**Retorna:** 1 (positivo), 0 (neutral), -1 (negativo)

#### 3. ✅ Scoring Mejorado

**Antes:**
```python
score = change_pct * 0.6 + log(twitter) * 10 * 0.4
```

**Ahora:**
```python
score = (
    change_pct * 0.5 +                         # Técnica: 50%
    log(twitter) * 5 * 0.2 +                  # Twitter: 20%
    (log(reddit) * 8 + sentiment * 10) * 0.3  # Reddit: 30%
)
```

#### 4. ✅ Formato de Archivo

**Antes:**
```
penny_stocks_pro_2025-10-23.csv
```

**Ahora:**
```
pennys_2025-10-23_14-30-45.csv
```

**Incluye:** Fecha + Hora completa

#### 5. ✅ Columnas Adicionales en CSV

**Nuevas columnas:**
- `reddit_mentions` - Menciones en Reddit
- `reddit_sentiment` - Score de sentimiento (-1 a 1)
- `reddit_positive` - Posts positivos
- `reddit_neutral` - Posts neutrales
- `reddit_negative` - Posts negativos
- `reddit_hype` - Nivel de hype
- `total_social_mentions` - Total menciones (Twitter + Reddit)
- `hype_score` - Score solo de redes sociales

**Total columnas:** 8 → 15

#### 6. ✅ Reporte Mejorado

**Output en consola:**
```
================================================================================
🔥 PENNY STOCKS CON MAYOR POTENCIAL
================================================================================
symbol  price  change_pct  twitter  reddit  sentiment  hype    score
CLOV    2.45   8.50        25       30      0.68       ALTO    18.01
================================================================================

📋 REPORTE DETALLADO:

1. 💰 $CLOV - $2.45
   📈 Cambio: +8.50%
   📊 Volumen ratio: 3.2x
   🐦 Twitter: 25 menciones
   🤖 Reddit: 30 menciones | Sentimiento: +0.68 | Hype: ALTO
   💯 Score total: 18.01
```

### Archivos adicionales

1. **`README_SEARCH_PENNYS.md`**
   - Documentación completa del screener
   - Ejemplos de uso
   - Troubleshooting

2. **`test_reddit_search.py`**
   - Script de prueba para verificar snscrape
   - Prueba búsqueda de Reddit
   - Validación de instalación

3. **`CHANGELOG.md`** (este archivo)
   - Historial de cambios

---

## 📊 RESUMEN DE CAMBIOS

### Estadísticas

| Métrica | Antes | Después | Cambio |
|---------|-------|---------|--------|
| **Archivos Python** | 2 | 5 | +3 |
| **Líneas de código** | ~300 | ~2,000 | +1,700 |
| **Funciones nuevas** | - | 19 | +19 |
| **Documentación (líneas)** | 0 | ~3,500 | +3,500 |
| **Redes sociales analizadas** | 1 (Twitter) | 2 (Twitter + Reddit) | +1 |
| **Sistema de scoring** | Simple | 3 capas + penalizaciones | Mejorado |

### Archivos del proyecto V4

```
v4/
├── penny_stock_advisor_v4.py            # Motor V4 (650 líneas)
├── integration_v4_trading_manager.py    # Gestor V4 (550 líneas)
├── search_pennys.py                     # Screener mejorado (350 líneas)
├── test_reddit_search.py                # Test de Reddit (150 líneas)
│
├── README_V4.md                         # Doc V4 (900 líneas)
├── QUICK_START.md                       # Inicio rápido V4 (300 líneas)
├── IMPLEMENTATION_SUMMARY.md            # Resumen V4 (400 líneas)
├── README_SEARCH_PENNYS.md              # Doc screener (800 líneas)
├── CHANGELOG.md                         # Este archivo
│
├── analisis_teorico_estrategia_trading.md  # Teoría (600 líneas)
└── fase1.txt                            # Plan original
```

---

## 🎯 FUNCIONALIDADES CLAVE

### Sistema V4

1. **Sistema de 3 capas:**
   - Fase 1: Setup (compresión)
   - Fase 2: Trigger (explosión día 1-2)
   - Fase 3: Context (mercado)

2. **Detección de timing:**
   - Identifica día 1, 2 o 3+ del movimiento
   - Penaliza entradas tardías (-30 pts)

3. **Contexto de mercado:**
   - Analiza SPY/QQQ
   - Monitorea VIX
   - Filtra por condiciones adversas

4. **Gestión de salidas:**
   - Divergencias RSI
   - Trailing stops dinámicos
   - Salidas parciales escalonadas
   - Patrones de distribución

### Screener Mejorado

1. **Análisis de Reddit:**
   - Menciones
   - Sentimiento positivo/negativo
   - Nivel de hype

2. **Scoring combinado:**
   - 50% técnico
   - 20% Twitter
   - 30% Reddit

3. **Output mejorado:**
   - CSV con timestamp
   - 15 columnas de datos
   - Reporte detallado
   - Telegram (opcional)

---

## 🔧 DEPENDENCIAS

### Nuevas dependencias

```python
# Ya existentes
numpy
pandas
yfinance
requests

# Nueva (para Reddit)
snscrape  # pip install snscrape
```

---

## 🚀 CÓMO USAR

### V4 - Trading System

```bash
cd /path/to/v4/
python integration_v4_trading_manager.py
```

### Screener con Reddit

```bash
cd /path/to/v4/

# 1. Probar instalación
python test_reddit_search.py

# 2. Ejecutar screener completo
python search_pennys.py
```

---

## 📈 MEJORAS ESPERADAS

### Sistema V4

| Métrica | V3 | V4 Objetivo |
|---------|----|----|
| Win Rate | ~45% | >55% |
| Profit Factor | ~1.5 | >2.0 |
| Max Drawdown | ~25% | <20% |
| Entradas tardías | 40% | <10% |
| Señales/semana | 5-10 | 0-3 |

### Screener

| Aspecto | Antes | Después |
|---------|-------|---------|
| Precisión | Media | Alta |
| Fuentes | 1 (Twitter) | 2 (Twitter + Reddit) |
| Sentimiento | No | Sí |
| Detección de hype | No | Sí (5 niveles) |

---

## ⚠️ NOTAS IMPORTANTES

### V4

1. **Paper trading recomendado** antes de usar dinero real
2. **Muy selectivo** - puede dar 0 señales en un día (esto es bueno)
3. **Requiere monitoreo activo** - no es "set and forget"

### Screener

1. **snscrape requerido** - instalar con `pip install snscrape`
2. **Reddit puede tardar** - cada búsqueda toma 10-30 segundos
3. **Rate limits** - no ejecutar más de 1 vez por hora

---

## 🔮 PRÓXIMOS PASOS

### Validación (1-2 semanas)

1. Paper trading con V4
2. Backtesting con datos históricos
3. Comparar V3 vs V4 en paralelo

### Mejoras futuras

**V4:**
- Machine Learning para día de explosión
- Backtesting automatizado
- Alertas en tiempo real

**Screener:**
- Subreddits específicos (r/wallstreetbets, r/pennystocks)
- Análisis de tendencia temporal
- Integración con V4 (usar hype como señal)

---

## ✅ TESTING

### V4

- [x] Código compila sin errores
- [x] Funciones principales testeadas manualmente
- [ ] Backtesting con datos históricos (pendiente)
- [ ] Paper trading en vivo (pendiente)

### Screener

- [x] Código compila sin errores
- [x] snscrape funciona correctamente
- [x] Reddit search retorna datos
- [x] Sentimiento se calcula correctamente
- [x] CSV se genera con timestamp
- [ ] Testing con 250 tickers completo (pendiente)

---

## 📞 SOPORTE

### Archivos de documentación

- **V4:** `README_V4.md`, `QUICK_START.md`
- **Screener:** `README_SEARCH_PENNYS.md`
- **Teoría:** `analisis_teorico_estrategia_trading.md`

### Problemas comunes

**V4:**
- Score muy bajo → Ver penalizaciones aplicadas
- No hay señales → Normal, el sistema es selectivo

**Screener:**
- snscrape no instalado → `pip install snscrape`
- Timeout en Reddit → Normal con símbolos populares
- No hay menciones → Símbolo sin actividad reciente

---

## 🎓 CONCLUSIÓN

**Hoy se implementaron dos mejoras mayores:**

1. **V4 - Paradigm Shift:**
   - Ya no compramos cohetes en vuelo
   - Ahora los encontramos en la plataforma de lanzamiento

2. **Screener Enhanced:**
   - Análisis de sentimiento de Reddit
   - Detección de nivel de hype
   - Scoring mejorado (técnico + social)

**Resultado:** Sistema completo de análisis y trading de penny stocks con:
- Entrada temprana (V4)
- Detección de hype (Screener)
- Gestión de riesgo (V4)
- Análisis social (Screener)

---

**Implementado por:** Claude Code
**Fecha:** 23 de Octubre, 2025
**Versión:** V4.0.0 + Screener 2.0.0
**Estado:** ✅ COMPLETADO

🚀 **Happy Trading!**
