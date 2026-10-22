/*
 * CSS 314 - TASK 7: Arithmetic Intensity vs. Memory Bottlenecks
 * Student ID: 230103081
 *
 * macOS (Apple Silicon):
 *   clang -O3 -std=c11 -Wall -Wextra -pthread task7.c -o task7 -lm
 *   ./task7 | tee task7_output.txt
 *
 * Both kernels process the SAME 30,000,000-element float64 input.
 * Threads own nonoverlapping slices of the output buffer; no data races.
 * Timings use CLOCK_MONOTONIC and exclude allocation/input initialization.
 * Thread creation is excluded with a common start gate. Reported times
 * are the median of 3 runs to reduce timing noise.
 *
 * This is a scaling experiment, not a direct memory-controller measurement.
 */

#ifdef __APPLE__
#define _DARWIN_C_SOURCE 1
#else
#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/utsname.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif
#ifdef __linux__
#include <sched.h>
#endif

#define STUDENT_ID 230103081u
#define ARRAY_SIZE 30000000u
#define TRIALS 3

typedef enum { COPY_KERNEL = 0, MATH_KERNEL = 1 } Kernel;

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t ready_cond;
    pthread_cond_t go_cond;
    int ready;
    int go;
} StartGate;

typedef struct {
    const double *a;
    double *b;
    size_t begin;
    size_t end;
    Kernel kernel;
    StartGate *gate;
} Worker;

static double now_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static int available_cores(void) {
#ifdef __APPLE__
    int n = 0;
    size_t len = sizeof(n);
    if (sysctlbyname("hw.logicalcpu", &n, &len, NULL, 0) == 0 && n > 0)
        return n;
#elif defined(__linux__)
    cpu_set_t mask;
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
        int n = CPU_COUNT(&mask);
        if (n > 0) return n;
    }
#endif
#ifdef _SC_NPROCESSORS_ONLN
    long fallback_cores = sysconf(_SC_NPROCESSORS_ONLN);
    if (fallback_cores > 0) return (int)fallback_cores;
#endif
    return 1;
}

static void *worker_main(void *opaque) {
    Worker *w = (Worker *)opaque;
    StartGate *gate = w->gate;
    pthread_mutex_lock(&gate->mutex);
    gate->ready++;
    pthread_cond_signal(&gate->ready_cond);
    while (!gate->go) pthread_cond_wait(&gate->go_cond, &gate->mutex);
    pthread_mutex_unlock(&gate->mutex);

    if (w->kernel == COPY_KERNEL) {
        for (size_t i = w->begin; i < w->end; ++i) {
            w->b[i] = w->a[i];
        }
    } else {
        for (size_t i = w->begin; i < w->end; ++i) {
            const double x = w->a[i];
            w->b[i] = sin(x) * cos(x) + tan(x);
        }
    }
    return NULL;
}

static void check_pthread(int code, const char *what) {
    if (code != 0) {
        fprintf(stderr, "%s: %s\n", what, strerror(code));
        exit(EXIT_FAILURE);
    }
}

static double run_once(const double *a, double *b, size_t n, int workers,
                       Kernel kernel) {
    StartGate gate = {0};
    check_pthread(pthread_mutex_init(&gate.mutex, NULL), "mutex init");
    check_pthread(pthread_cond_init(&gate.ready_cond, NULL), "ready cond init");
    check_pthread(pthread_cond_init(&gate.go_cond, NULL), "go cond init");

    pthread_t *threads = calloc((size_t)workers, sizeof(*threads));
    Worker *args = calloc((size_t)workers, sizeof(*args));
    if (!threads || !args) {
        perror("calloc");
        exit(EXIT_FAILURE);
    }

    for (int j = 0; j < workers; ++j) {
        args[j].a = a;
        args[j].b = b;
        args[j].begin = n * (size_t)j / (size_t)workers;
        args[j].end = n * (size_t)(j + 1) / (size_t)workers;
        args[j].kernel = kernel;
        args[j].gate = &gate;
        check_pthread(pthread_create(&threads[j], NULL, worker_main, &args[j]),
                      "pthread_create");
    }

    check_pthread(pthread_mutex_lock(&gate.mutex), "mutex lock");
    while (gate.ready != workers)
        check_pthread(pthread_cond_wait(&gate.ready_cond, &gate.mutex),
                      "cond wait");
    const double start = now_seconds();
    gate.go = 1;
    check_pthread(pthread_cond_broadcast(&gate.go_cond), "cond broadcast");
    check_pthread(pthread_mutex_unlock(&gate.mutex), "mutex unlock");

    for (int j = 0; j < workers; ++j)
        check_pthread(pthread_join(threads[j], NULL), "pthread_join");
    const double elapsed_ms = (now_seconds() - start) * 1000.0;

    free(args);
    free(threads);
    check_pthread(pthread_cond_destroy(&gate.go_cond), "cond destroy");
    check_pthread(pthread_cond_destroy(&gate.ready_cond), "cond destroy");
    check_pthread(pthread_mutex_destroy(&gate.mutex), "mutex destroy");
    return elapsed_ms;
}

static double median3(double values[TRIALS]) {
    for (int i = 0; i < TRIALS; ++i) {
        for (int j = i + 1; j < TRIALS; ++j) {
            if (values[i] > values[j]) {
                double t = values[i];
                values[i] = values[j];
                values[j] = t;
            }
        }
    }
    return values[TRIALS / 2];
}

static void verify_output(const double *a, const double *b, size_t n,
                          Kernel kernel) {
    /* Read output after each benchmark to verify work was actually done. */
    const size_t positions[] = {0, n / 4, n / 2, 3 * n / 4, n - 1};
    for (size_t j = 0; j < sizeof(positions) / sizeof(positions[0]); ++j) {
        size_t i = positions[j];
        double expected = (kernel == COPY_KERNEL)
            ? a[i] : sin(a[i]) * cos(a[i]) + tan(a[i]);
        if (!isfinite(b[i]) || fabs(b[i] - expected) > 1e-12) {
            fprintf(stderr, "Verification FAILED: kernel=%d index=%zu\n",
                    (int)kernel, i);
            exit(EXIT_FAILURE);
        }
    }
}

int main(void) {
    const size_t n = ARRAY_SIZE;
    int max_cores = available_cores();
    if (max_cores < 1 || max_cores > 4096) {
        fprintf(stderr, "Invalid detected core count: %d\n", max_cores);
        return EXIT_FAILURE;
    }

    double *a = NULL;
    double *b = NULL;
    if (posix_memalign((void **)&a, 64, n * sizeof(double)) != 0 ||
        posix_memalign((void **)&b, 64, n * sizeof(double)) != 0) {
        fprintf(stderr, "Unable to allocate two 30M-element double arrays.\n");
        free(a);
        free(b);
        return EXIT_FAILURE;
    }

    uint64_t seed = STUDENT_ID;
    for (size_t i = 0; i < n; ++i) {
        seed = seed * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
        /* Input in [0.1, 0.9], safely away from tan() singularities. */
        a[i] = 0.1 + 0.8 * (double)(seed >> 11) / 9007199254740992.0;
    }
    memset(b, 0, n * sizeof(double));

    int counts[] = {1, 2, max_cores};
    /* Deduplicate if available cores is 1 or 2; don't request more than available. */
    if (max_cores == 1) { counts[1] = 1; counts[2] = 1; }
    else if (max_cores == 2) counts[2] = 2;
    int number_of_counts = 0;
    int unique_counts[3];
    for (int i = 0; i < 3; ++i) {
        int exists = 0;
        for (int j = 0; j < number_of_counts; ++j)
            if (unique_counts[j] == counts[i]) exists = 1;
        if (!exists) unique_counts[number_of_counts++] = counts[i];
    }

    struct utsname platform;
    if (uname(&platform) != 0) {
        perror("uname");
        return EXIT_FAILURE;
    }
    printf("TASK 7 - Arithmetic Intensity vs. Memory Bottlenecks\n");
    printf("Student ID: %u\n", STUDENT_ID);
    printf("OS / CPU architecture: %s %s\n", platform.sysname, platform.machine);
    printf("Available logical cores: %d\n", max_cores);
    printf("Array elements: %zu (float64; 30,000,000 per array)\n", n);
    printf("Trials per test: %d (median time, milliseconds)\n", TRIALS);
    printf("Timing: kernel wall-clock, synchronized start, excludes setup/allocation\n");

    double timing[2][3] = {{0}};
    double speedup[2][3] = {{0}};
    double efficiency[2][3] = {{0}};
    for (int k = 0; k < 2; ++k) {
        printf("\nKernel %c - %s\n", k == 0 ? 'A' : 'B',
               k == 0 ? "Memory-bound: B[i] = A[i]" :
                        "Compute-bound: B[i] = sin(A[i])*cos(A[i])+tan(A[i])");
        printf("%8s | %14s | %9s | %13s\n",
               "Workers", "Wall Time (ms)", "Speedup", "Efficiency (%)");
        printf("---------+----------------+-----------+---------------\n");
        for (int j = 0; j < number_of_counts; ++j) {
            double samples[TRIALS];
            for (int trial = 0; trial < TRIALS; ++trial) {
                samples[trial] = run_once(a, b, n, unique_counts[j], (Kernel)k);
            }
            verify_output(a, b, n, (Kernel)k);
            timing[k][j] = median3(samples);
            speedup[k][j] = timing[k][0] / timing[k][j];
            efficiency[k][j] = speedup[k][j] / (double)unique_counts[j];
            printf("%8d | %14.3f | %9.3f | %13.2f\n",
                   unique_counts[j], timing[k][j], speedup[k][j],
                   efficiency[k][j] * 100.0);
        }
    }

    const int last = number_of_counts - 1;
    const double a_eff = efficiency[0][last];
    const double b_eff = efficiency[1][last];
    printf("\nScaling interpretation at %d workers:\n", unique_counts[last]);
    printf("  Kernel A efficiency: %.2f%%\n", a_eff * 100.0);
    printf("  Kernel B efficiency: %.2f%%\n", b_eff * 100.0);
    if (unique_counts[last] <= 1) {
        puts("  Memory-channel saturation: cannot assess with one core.");
    } else if (a_eff < 0.60 && speedup[0][last] < speedup[1][last] * 0.80) {
        puts("  Kernel A (vector copy) STALLED in scaling, consistent with RAM bandwidth saturation.");
        puts("  Kernel B scaled better; its higher arithmetic intensity reduces bandwidth pressure.");
    } else {
        puts("  Memory-channel saturation NOT conclusively demonstrated in this run.");
        puts("  Kernel A is the memory-bandwidth-bound candidate; inspect its speedup vs Kernel B.");
    }
    puts("  Note: timing alone cannot prove hardware memory-channel saturation.");
    puts("Verification: PASSED (sample outputs from both kernels)");
    free(a);
    free(b);
    return EXIT_SUCCESS;
}
