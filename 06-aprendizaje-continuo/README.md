# Aprendizaje continuo (continual / online learning)

**Objetivo:** en producción los datos cambian. Un modelo debe aprender lo nuevo **sin olvidar lo viejo**.

## Parte 1 — Olvido catastrófico
Se entrena una red en dígitos 0-4 (tarea A) y luego en 5-9 (tarea B):
- **Fine-tuning ingenuo:** la accuracy en A se desploma (olvido catastrófico).
- **Experience Replay:** guardar solo 50 ejemplos de A y mezclarlos al entrenar B recupera gran parte.
- **EWC:** penaliza cambiar los pesos que la *información de Fisher* marca como importantes para A.

## Parte 2 — Aprendizaje online con *concept drift*
Un flujo de lotes cuya regla cambia a mitad de camino. Un modelo estático colapsa al 50%;
uno que se actualiza con `partial_fit` se adapta en pocos lotes. Se evalúa de forma **prequential**
(predecir primero, aprender después), que es como se mide en streaming real.

## Conceptos clave
- Escenarios: task-incremental, domain-incremental, class-incremental (el más difícil, el de este ejemplo).
- Dilema estabilidad–plasticidad.
- Detección de drift (ADWIN, Page-Hinkley) y re-entrenamiento en MLOps.

## Retos
1. Varía el tamaño del buffer de replay (10, 50, 200). ¿Dónde está el punto óptimo?
2. Ajusta `lam` de EWC. ¿Por qué EWC sufre más en class-incremental que en task-incremental?
3. Implementa un detector de drift simple que dispare un re-entrenamiento solo cuando haga falta.
4. Prueba la librería `river` para aprendizaje online.
