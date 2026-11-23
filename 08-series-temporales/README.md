# Modelos predictivos: series temporales (forecasting)

**Objetivo:** pronosticar las ventas diarias de los próximos **28 días**, como haría un equipo de
demanda/inventario. Es de los problemas más pedidos en empresas.

## Datos
Serie sintética de 4 años con los ingredientes de una serie real: tendencia, estacionalidad semanal
(pico el sábado) y anual, festivos (Navidad, Año Nuevo), promociones planificadas y ruido.

## Qué implementa
- **Baselines**: naive y naive estacional. *Si tu modelo no le gana al naive estacional, no sirve.*
- **Features sin fuga de información**: solo lags ≥ 28 días (el horizonte), medias móviles desplazadas,
  calendario (día de semana, mes, seno/coseno del día del año) y variables exógenas conocidas a futuro
  (festivos, promociones).
- **Ridge** y **Gradient Boosting** sobre esas features (estrategia *directa*, como en la competición M5).
- **LSTM** en PyTorch que predice los 28 días de una vez, con normalización por ventana.
- **Backtesting walk-forward**: 3 orígenes de pronóstico consecutivos, nunca se mezcla futuro con pasado.
- Métricas **MAE, RMSE, MAPE y MASE** (MASE < 1 = mejor que el naive estacional).
- **Intervalos de predicción** al 80% con regresión cuantílica y su cobertura real.
- **Importancia por permutación** para ver qué variables usa el modelo.

## Lecciones que salen del experimento
- **Un modelo simple bien diseñado puede ganar.** La serie es casi multiplicativa y las features
  capturan bien la estructura, así que Ridge supera al Gradient Boosting y a la LSTM.
- **Los defaults importan.** Con `min_samples_leaf=20` (por defecto) el GBM no aprendía el efecto de
  los festivos (solo hay ~12 en el histórico) y perdía contra el naive estacional. Con 5 lo aprende.
- **Deep learning no es automáticamente mejor** con una sola serie y ~1.000 puntos.
- **Revisa la cobertura de los intervalos**: aquí el intervalo "80%" solo cubre ~40% de los días reales.
  Está mal calibrado (los cuantiles sobreajustan con pocos datos). Arreglarlo es el reto 6.

## Retos
1. Entrena con `log(y)`: ¿mejora el Ridge? (pista: la serie es multiplicativa).
2. Añade **Prophet** o **statsmodels** (ETS / SARIMA) como modelos estadísticos clásicos.
3. Estrategia **recursiva** (predecir t+1 y realimentar) vs directa: compara el error a 28 días.
4. Hazlo con **muchas series** (p. ej., 50 tiendas) y entrena un solo modelo global con un ID de tienda.
5. Usa un dataset real: *M5 Forecasting* (Kaggle) o datos de demanda eléctrica.
6. Calibra los intervalos con **conformal prediction** (residuos del backtesting) hasta lograr ~80% de cobertura.
