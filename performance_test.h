#ifndef PERFORMANCE_TEST_H
#define PERFORMANCE_TEST_H

#include <string>
#include "filter_kernel.h"

/**
 * @file performance_test.h
 * @brief Header file per i test di performance dell'elaborazione immagini
 *
 * Questo file definisce l'interfaccia pubblica per l'esecuzione
 * di test di performance su diverse implementazioni di elaborazione
 * immagini (sequenziale, CUDA, OpenMP).
 */

 /**
  * @brief Esegue una batteria completa di test di performance
  *
  * Questa funzione esegue test di performance comparativi tra:
  * - Elaborazione sequenziale (CPU)
  * - Elaborazione parallela (CUDA) con diversi tipi di memoria:
  *   - Memoria globale
  *   - Memoria costante
  *   - Memoria condivisa
  * - Elaborazione multi-thread (OpenMP) con diversi numeri di thread
  *
  * I test vengono eseguiti su diverse dimensioni di immagini e kernel,
  * e i risultati vengono salvati in un file CSV per l'analisi.
  *
  * @param filterType Tipo di filtro da utilizzare per i test
  * @param configuredFilter Filtro già configurato con i parametri desiderati
  * @return 0 se i test sono completati con successo, codice di errore altrimenti
  */
int runPerformanceTests(const std::string& filterType, const FilterKernel& configuredFilter);

#endif // PERFORMANCE_TEST_H