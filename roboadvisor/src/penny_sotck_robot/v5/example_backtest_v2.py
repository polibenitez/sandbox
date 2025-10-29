#!/usr/bin/env python3
"""
EXAMPLE: Backtest Engine V2 Usage
==================================

Ejemplo de cómo usar el nuevo Backtest Engine V2 con tu estrategia
de Penny Stock Advisor.

Este script demuestra:
1. Configuración del backtest
2. Obtención de universo de stocks
3. Ejecución de backtest con datos reales
4. Generación de métricas y reportes
5. Validación estadística
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Importar el nuevo backtest engine
from backtest_engine_v2 import (
    BacktestEngineV2,
    BacktestConfig,
    Trade,
    Position,
    get_universe,
    temporal_split,
    walk_forward_split,
    calculate_features_realtime,
    calculate_all_metrics,
    monte_carlo_simulation,
    statistical_validation,
    fetch_benchmark_data,
    generate_tearsheet,
    print_critical_validations
)

# Importar módulos existentes
sys.path.append(os.path.dirname(__file__))
from penny_stock_advisor_v5 import PennyStockAdvisorV5
from utils.logging_config_v5 import setup_logging

import logging
logger = logging.getLogger('example_backtest')


def simple_backtest_demo():
    """
    Demo simple: Simula un backtest con datos sintéticos
    para demostrar todas las funcionalidades del engine
    """
    logger.info("="*80)
    logger.info("DEMO: Backtest Engine V2 con datos sintéticos")
    logger.info("="*80)

    # 1. Configurar backtest
    config = BacktestConfig(
        initial_capital=100000,
        position_size_pct=0.05,  # 5% por trade
        max_positions=5,
        commission=0.001,
        slippage=0.002,
        stop_loss_pct=0.15,
        take_profit_pct=0.20,
        max_holding_days=30
    )

    # 2. Crear engine
    engine = BacktestEngineV2(config)

    # 3. Simular algunos trades
    start_date = datetime(2023, 1, 1)

    # Trade 1: Ganador
    pos1 = engine.open_position('SNDL', 1.00, start_date)
    if pos1:
        engine.update_positions(
            start_date + timedelta(days=5),
            {'SNDL': 1.25}  # +25% ganancia
        )

    # Trade 2: Perdedor
    pos2 = engine.open_position('GNUS', 2.00, start_date + timedelta(days=1))
    if pos2:
        engine.update_positions(
            start_date + timedelta(days=3),
            {'GNUS': 1.70}  # -15% stop loss
        )

    # Trade 3: Ganador moderado
    pos3 = engine.open_position('ATOS', 1.50, start_date + timedelta(days=2))
    if pos3:
        engine.update_positions(
            start_date + timedelta(days=7),
            {'ATOS': 1.70}  # +13% ganancia
        )

    # Trade 4: Perdedor pequeño
    pos4 = engine.open_position('TELL', 3.00, start_date + timedelta(days=5))
    if pos4:
        engine.update_positions(
            start_date + timedelta(days=8),
            {'TELL': 2.85}  # -5% pérdida
        )

    # Trade 5: Ganador grande
    pos5 = engine.open_position('ZOM', 0.50, start_date + timedelta(days=10))
    if pos5:
        engine.update_positions(
            start_date + timedelta(days=15),
            {'ZOM': 0.75}  # +50% ganancia
        )

    # 4. Registrar equity curve (simulada)
    for i in range(20):
        date = start_date + timedelta(days=i)
        # Equity simulado basado en trades
        equity = config.initial_capital + sum(t.pnl_net for t in engine.closed_trades)
        engine.equity_curve.append((date, equity))

    # 5. Calcular métricas
    logger.info("\n" + "="*80)
    logger.info("CALCULANDO MÉTRICAS")
    logger.info("="*80)

    metrics = calculate_all_metrics(
        engine.closed_trades,
        engine.equity_curve,
        config,
        period_days=20
    )

    # Imprimir métricas
    print("\n📊 PERFORMANCE METRICS")
    print("="*80)
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']*100:.1f}%")
    print(f"Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"CAGR: {metrics['cagr']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Expectancy: ${metrics['expectancy']:.2f}")
    print("="*80)

    # 6. Monte Carlo Simulation
    logger.info("\n" + "="*80)
    logger.info("MONTE CARLO SIMULATION")
    logger.info("="*80)

    mc_results = monte_carlo_simulation(
        engine.closed_trades,
        n_simulations=1000,
        initial_capital=config.initial_capital
    )

    print("\n🎲 MONTE CARLO RESULTS (1000 simulaciones)")
    print("="*80)
    print(f"Return medio: {mc_results['return_mean']:.2f}%")
    print(f"Return std: {mc_results['return_std']:.2f}%")
    print("\nPercentiles de retorno:")
    for p, val in mc_results['return_percentiles'].items():
        print(f"  {p}: {val:.2f}%")
    print(f"\nProbabilidad de ruina (>50% pérdida): {mc_results['prob_ruin']*100:.2f}%")
    print(f"Peor caso: {mc_results['worst_case_return']:.2f}%")
    print(f"Mejor caso: {mc_results['best_case_return']:.2f}%")
    print("="*80)

    # 7. Statistical Validation (simulada)
    logger.info("\n" + "="*80)
    logger.info("STATISTICAL VALIDATION")
    logger.info("="*80)

    # Simular returns del benchmark
    strategy_returns = np.random.normal(0.001, 0.02, 100)  # Returns diarios
    benchmark_returns = np.random.normal(0.0005, 0.015, 100)  # SPY returns

    stat_val = statistical_validation(strategy_returns, benchmark_returns)

    print("\n📈 STATISTICAL VALIDATION vs SPY")
    print("="*80)
    print(f"T-statistic: {stat_val['t_statistic']:.4f}")
    print(f"P-value: {stat_val['p_value']:.4f}")
    print(f"Resultado: {stat_val['conclusion']}")
    print(f"Information Ratio: {stat_val['information_ratio']:.4f}")
    print(f"Beta: {stat_val['beta']:.4f}")
    print(f"Alpha (anualizado): {stat_val['alpha_annualized_pct']:.2f}%")
    print("="*80)

    # 8. Validaciones críticas
    print_critical_validations(metrics, stat_val)

    # 9. Generar tearsheet
    logger.info("\n" + "="*80)
    logger.info("GENERANDO TEARSHEET")
    logger.info("="*80)

    try:
        tearsheet_path = generate_tearsheet(
            metrics,
            engine.equity_curve,
            engine.closed_trades,
            mc_results,
            save_path='reports/'
        )
        print(f"\n✓ Tearsheet generado: {tearsheet_path}")
    except Exception as e:
        logger.error(f"Error generando tearsheet: {e}")
        print(f"\n⚠️  No se pudo generar el tearsheet: {e}")

    # 10. Guardar resultados
    logger.info("\n" + "="*80)
    logger.info("GUARDANDO RESULTADOS")
    logger.info("="*80)

    # Guardar trades
    trades_df = pd.DataFrame([t.to_dict() for t in engine.closed_trades])
    trades_df.to_csv('data/backtest_trades_v2.csv', index=False)
    print("✓ Trades guardados: data/backtest_trades_v2.csv")

    # Guardar equity curve
    equity_df = pd.DataFrame(engine.equity_curve, columns=['date', 'equity'])
    equity_df.to_csv('data/equity_curve_v2.csv', index=False)
    print("✓ Equity curve guardado: data/equity_curve_v2.csv")

    # Guardar métricas
    import json
    with open('data/metrics_v2.json', 'w') as f:
        # Convertir valores numpy a nativos de Python
        metrics_serializable = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                               for k, v in metrics.items()}
        json.dump(metrics_serializable, f, indent=2)
    print("✓ Métricas guardadas: data/metrics_v2.json")

    logger.info("\n✓ Demo completado exitosamente!")


def real_backtest_with_strategy():
    """
    Backtest real usando PennyStockAdvisorV5

    NOTA: Este es un ejemplo de cómo integrar el engine con tu estrategia.
    Necesita adaptarse a tu lógica específica de señales.
    """
    logger.info("="*80)
    logger.info("BACKTEST REAL: Penny Stock Advisor V5")
    logger.info("="*80)

    # Configuración
    config = BacktestConfig(
        initial_capital=100000,
        position_size_pct=0.02,
        max_positions=10,
        commission=0.001,
        slippage=0.002,
        stop_loss_pct=0.15,
        take_profit_pct=0.20
    )

    # Obtener universo
    universe = get_universe(
        min_price=1.0,
        max_price=10.0,
        start_date='2023-01-01',
        end_date='2023-12-31'
    )

    logger.info(f"Universo: {len(universe)} símbolos")

    # TODO: Implementar lógica de backtesting real
    # 1. Para cada símbolo en el universo
    # 2. Obtener datos históricos
    # 3. Hacer temporal split
    # 4. Walk-forward analysis
    # 5. En cada ventana:
    #    - Calcular features (sin look-ahead)
    #    - Generar señales con PennyStockAdvisorV5
    #    - Abrir/cerrar posiciones con BacktestEngineV2
    # 6. Calcular métricas finales
    # 7. Generar reportes

    logger.warning("⚠️  Implementación pendiente - ver backtest_engine_v2.py")
    logger.info("Este script es un template para integrar con tu estrategia específica")


def test_data_functions():
    """
    Test de funciones de data management
    """
    logger.info("="*80)
    logger.info("TEST: Data Management Functions")
    logger.info("="*80)

    # 1. Test temporal_split
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    data = pd.DataFrame({
        'date': dates,
        'value': np.random.randn(len(dates))
    }).set_index('date')

    train, val, test = temporal_split(data, train_pct=0.6, val_pct=0.2, test_pct=0.2)

    print("\n✓ Temporal Split:")
    print(f"  Train: {len(train)} rows ({train.index[0]} to {train.index[-1]})")
    print(f"  Val: {len(val)} rows ({val.index[0]} to {val.index[-1]})")
    print(f"  Test: {len(test)} rows ({test.index[0]} to {test.index[-1]})")

    # 2. Test walk_forward_split
    windows = walk_forward_split(data, train_days=180, test_days=30)

    print(f"\n✓ Walk-Forward Split: {len(windows)} ventanas")
    for i, (train_w, test_w) in enumerate(windows[:3]):  # Mostrar primeras 3
        print(f"  Window {i+1}:")
        print(f"    Train: {train_w.index[0]} to {train_w.index[-1]}")
        print(f"    Test: {test_w.index[0]} to {test_w.index[-1]}")

    # 3. Test calculate_features_realtime
    # Crear datos OHLCV simulados
    ohlcv = pd.DataFrame({
        'Open': np.random.uniform(1, 2, 100),
        'High': np.random.uniform(2, 3, 100),
        'Low': np.random.uniform(0.5, 1, 100),
        'Close': np.random.uniform(1, 2, 100),
        'Volume': np.random.randint(100000, 1000000, 100)
    })

    features = calculate_features_realtime(ohlcv, current_idx=50, lookback=20)

    print("\n✓ Features calculados (sin look-ahead):")
    for key, val in features.items():
        print(f"  {key}: {val:.4f}")

    logger.info("\n✓ Tests completados!")


if __name__ == "__main__":
    # Setup logging
    setup_logging(level="INFO")

    # Crear directorios necesarios
    os.makedirs('data', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    print("\n" + "="*80)
    print("BACKTEST ENGINE V2 - EJEMPLOS DE USO")
    print("="*80)
    print("\nSelecciona un ejemplo:")
    print("1. Demo simple (datos sintéticos)")
    print("2. Test de funciones de data management")
    print("3. Backtest real con estrategia (template)")
    print("="*80)

    choice = input("\nOpción (1-3): ").strip()

    if choice == '1':
        simple_backtest_demo()
    elif choice == '2':
        test_data_functions()
    elif choice == '3':
        real_backtest_with_strategy()
    else:
        print("Opción inválida. Ejecutando demo simple...")
        simple_backtest_demo()
