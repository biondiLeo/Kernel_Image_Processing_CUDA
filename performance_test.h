#ifndef PERFORMANCE_TEST_H
#define PERFORMANCE_TEST_H

#include <string>
#include "filter_kernel.h"

// Funzione principale per l'esecuzione dei test di performance
// Restituisce 0 in caso di successo, un codice di errore altrimenti
int runPerformanceTests(const std::string& filterType, const FilterKernel& configuredFilter);

#endif // PERFORMANCE_TEST_H