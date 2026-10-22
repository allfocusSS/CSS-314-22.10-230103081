
#if defined(__APPLE__)
#  define _DARWIN_C_SOURCE 1
#elif defined(__linux__)
#  define _GNU_SOURCE 1
#else
#  define _POSIX_C_SOURCE 200809L
#endif

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#if defined(__APPLE__)
#  include <sys/types.h>
#  include <sys/sysctl.h>
#elif defined(__linux__)
#  include <sched.h>
#  include <unistd.h>
#else
#  include <unistd.h>
#endif

#define STUDENT_ID UINT64_C(230103081)
#define TRIALS 9
#define SIZE_COUNT 6
#define SERIAL_TARGET_ELEMENTS 20000000UL

static const size_t SIZES[SIZE_COUNT] = {
    100UL, 1000UL, 10000UL, 100000UL, 1000000UL, 10000000UL
};

typedef struct {
    const double *x;
    double *y;
    size_t begin;
    size_t end;
    double a;
    double b;
} WorkerJob;

typedef struct {
    size_t n;
    double serial_ms;
    double parallel_ms;
} Result;

static int available_logical_cores(void) {
#if defined(__APPLE__)
    int count = 0;
    size_t bytes = sizeof(count);
    if (sysctlbyname("hw.logicalcpu", &count, &bytes, NULL, 0) == 0 && count > 0)
        return count;
    return 1;
#elif defined(__linux__)
    cpu_set_t mask;
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
        int count = CPU_COUNT(&mask);
        if (count > 0) return count;
    }
    long count = sysconf(_SC_NPROCESSORS_ONLN);
    return count > 0 ? (int)count : 1;
#else
    long count = sysconf(_SC_NPROCESSORS_ONLN);
    return count > 0 ? (int)count : 1;
#endif
}

static double wall_clock_ms(void) {
    struct timespec stamp;
    if (clock_gettime(CLOCK_MONOTONIC, &stamp) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
    return (double)stamp.tv_sec * 1000.0 + (double)stamp.tv_nsec / 1000000.0;
}

/* Prevent compiler from removing repeated benchmark calls. */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
static void compute_range(const double *x, double *y, size_t begin,
                          size_t end, double a, double b) {
    for (size_t i = begin; i < end; ++i)
        y[i] = a * x[i] + b;
}

static void *worker_entry(void *argument) {
    const WorkerJob *job = (const WorkerJob *)argument;
    compute_range(job->x, job->y, job->begin, job->end, job->a, job->b);
    return NULL;
}

static int run_parallel(const double *x, double *y, size_t n,
                        int workers, pthread_t *ids, WorkerJob *jobs,
                        double a, double b) {
    int created = 0;
    for (int i = 0; i < workers; ++i) {
        jobs[i] = (WorkerJob){
            .x = x, .y = y,
            .begin = n * (size_t)i / (size_t)workers,
            .end = n * (size_t)(i + 1) / (size_t)workers,
            .a = a, .b = b
        };
        int err = pthread_create(&ids[i], NULL, worker_entry, &jobs[i]);
        if (err != 0) {
            fprintf(stderr, "pthread_create worker %d failed: %d\n", i, err);
            for (int j = 0; j < created; ++j) pthread_join(ids[j], NULL);
            return 0;
        }
        ++created;
    }
    for (int i = 0; i < workers; ++i) {
        if (pthread_join(ids[i], NULL) != 0) {
            fprintf(stderr, "pthread_join failed\n");
            return 0;
        }
    }
    return 1;
}

static int compare_doubles(const void *left, const void *right) {
    const double a = *(const double *)left;
    const double b = *(const double *)right;
    return (a > b) - (a < b);
}

static double median(double samples[TRIALS]) {
    qsort(samples, TRIALS, sizeof(double), compare_doubles);
    return samples[TRIALS / 2];
}

static int verify(const double *x, const double *y, size_t n,
                  double a, double b) {
    for (size_t i = 0; i < n; ++i) {
        const double expected = a * x[i] + b;
        if (fabs(y[i] - expected) > 1e-12) {
            fprintf(stderr, "Verification failed at index %zu\n", i);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    const int workers = available_logical_cores();
    const double a = 1.125, b = 0.375;
    Result results[SIZE_COUNT];

    printf("CSS 314 - TASK 5: OVERHEAD CROSSOVER POINT\n");
    printf("Student ID: %" PRIu64 "\n", STUDENT_ID);
#if defined(__APPLE__)
    printf("Platform: macOS\n");
#elif defined(__linux__)
    printf("Platform: Linux\n");
#else
    printf("Platform: Other POSIX\n");
#endif
    printf("Available logical cores / pthread workers: %d\n", workers);
    printf("Operation: y[i] = 1.125 * x[i] + 0.375 (double)\n");
    printf("Trials per size: %d (median reported)\n", TRIALS);
    printf("Parallel time includes thread creation/join overhead.\n");
    printf("Speedup = serial time / parallel time; >1 means parallel wins.\n\n");

    pthread_t *ids = malloc((size_t)workers * sizeof(*ids));
    WorkerJob *jobs = malloc((size_t)workers * sizeof(*jobs));
    if (!ids || !jobs) {
        fprintf(stderr, "Could not allocate thread bookkeeping\n");
        free(ids); free(jobs);
        return EXIT_FAILURE;
    }

    printf("%12s | %16s | %16s | %12s | %s\n",
           "N", "Serial (ms)", "Parallel (ms)", "Speedup", "Winner");
    printf("-------------+------------------+------------------+--------------+---------\n");

    for (int k = 0; k < SIZE_COUNT; ++k) {
        const size_t n = SIZES[k];
        double *x = malloc(n * sizeof(*x));
        double *y = malloc(n * sizeof(*y));
        if (!x || !y) {
            fprintf(stderr, "Allocation failed for N=%zu\n", n);
            free(x); free(y); free(ids); free(jobs);
            return EXIT_FAILURE;
        }
        for (size_t i = 0; i < n; ++i) {
            const uint64_t v = (STUDENT_ID + (uint64_t)i * UINT64_C(17)) % UINT64_C(65536);
            x[i] = (double)v / 65536.0;
        }

        /* Both implementations write the same output and are warmed once. */
        compute_range(x, y, 0, n, a, b);
        if (!run_parallel(x, y, n, workers, ids, jobs, a, b)) {
            free(x); free(y); free(ids); free(jobs);
            return EXIT_FAILURE;
        }

        /* Serial tiny operations are batched to exceed timer resolution. */
        size_t serial_repeats = SERIAL_TARGET_ELEMENTS / n;
        if (serial_repeats == 0) serial_repeats = 1;

        double serial_samples[TRIALS], parallel_samples[TRIALS];
        for (int trial = 0; trial < TRIALS; ++trial) {
            double start, serial_elapsed, parallel_elapsed;
            if ((trial & 1) == 0) {
                start = wall_clock_ms();
                for (size_t j = 0; j < serial_repeats; ++j)
                    compute_range(x, y, 0, n, a, b);
                serial_elapsed = wall_clock_ms() - start;

                start = wall_clock_ms();
                if (!run_parallel(x, y, n, workers, ids, jobs, a, b)) {
                    free(x); free(y); free(ids); free(jobs);
                    return EXIT_FAILURE;
                }
                parallel_elapsed = wall_clock_ms() - start;
            } else {
                start = wall_clock_ms();
                if (!run_parallel(x, y, n, workers, ids, jobs, a, b)) {
                    free(x); free(y); free(ids); free(jobs);
                    return EXIT_FAILURE;
                }
                parallel_elapsed = wall_clock_ms() - start;

                start = wall_clock_ms();
                for (size_t j = 0; j < serial_repeats; ++j)
                    compute_range(x, y, 0, n, a, b);
                serial_elapsed = wall_clock_ms() - start;
            }
            serial_samples[trial] = serial_elapsed / (double)serial_repeats;
            parallel_samples[trial] = parallel_elapsed;
        }
        if (!verify(x, y, n, a, b)) {
            free(x); free(y); free(ids); free(jobs);
            return EXIT_FAILURE;
        }
        results[k] = (Result){n, median(serial_samples), median(parallel_samples)};
        const double ratio = results[k].serial_ms / results[k].parallel_ms;
        const char *winner = (results[k].serial_ms < results[k].parallel_ms)
                            ? "SERIAL" : (results[k].serial_ms > results[k].parallel_ms)
                            ? "PARALLEL" : "TIE";
        printf("%12zu | %16.6f | %16.6f | %12.6f | %s\n",
               n, results[k].serial_ms, results[k].parallel_ms, ratio, winner);
        fflush(stdout);
        free(x); free(y);
    }

    printf("\nCROSSOVER ANALYSIS (only tested N values):\n");
    int found = 0;
    for (int k = 1; k < SIZE_COUNT; ++k) {
        const int previous_slower = results[k - 1].parallel_ms > results[k - 1].serial_ms;
        const int current_faster = results[k].parallel_ms < results[k].serial_ms;
        if (previous_slower && current_faster) {
            printf("SLOWER -> FASTER crossover bracket: %zu < N <= %zu\n",
                   results[k - 1].n, results[k].n);
            ++found;
        }
    }
    if (!found) {
        printf("No SLOWER -> FASTER crossover among the six tested sizes.\n");
        if (results[0].parallel_ms < results[0].serial_ms)
            printf("Parallel is already faster at the smallest tested N=%zu.\n", results[0].n);
    }
    printf("A precise per-element threshold cannot be established from only six noisy benchmarks.\n");
    printf("Verification: PASSED for all N\n");
    free(ids); free(jobs);
    return EXIT_SUCCESS;
}
