#if defined(__APPLE__)
/* macOS needs Darwin extensions for BSD types used by <sys/sysctl.h>. */
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

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif
#ifdef __linux__
#include <sched.h>
#endif

#define STUDENT_ID 230103081ULL
#define MATRIX_SIZE 8000U
#define REPEATS 3

typedef enum {
    ROW_MAJOR = 0,
    COLUMN_MAJOR = 1
} Traversal;

typedef struct {
    const double *matrix;
    size_t first;
    size_t last;
    Traversal traversal;
    double partial_sum;
} WorkerArgs;

static uint64_t next_random(uint64_t *state) {
    /* Reproducible 64-bit xorshift* sequence; STUDENT_ID is the seed. */
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * UINT64_C(2685821657736338717);
}

static void fill_matrix(double *matrix, size_t elements) {
    uint64_t state = STUDENT_ID;
    for (size_t k = 0; k < elements; ++k) {
        matrix[k] = (double)(next_random(&state) >> 11) * (1.0 / 9007199254740992.0);
    }
}

static double monotonic_ms(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
    return (double)now.tv_sec * 1000.0 + (double)now.tv_nsec / 1000000.0;
}

static int logical_core_count(void) {
#ifdef __APPLE__
    int count = 0;
    size_t size = sizeof(count);
    if (sysctlbyname("hw.logicalcpu", &count, &size, NULL, 0) == 0 && count > 0)
        return count;
    count = 0;
    size = sizeof(count);
    if (sysctlbyname("hw.ncpu", &count, &size, NULL, 0) == 0 && count > 0)
        return count;
    return 1;
#else
    long online_count = sysconf(_SC_NPROCESSORS_ONLN);
    return online_count > 0 ? (int)online_count : 1;
#endif
}

static int physical_core_count(void) {
#ifdef __APPLE__
    int count = 0;
    size_t size = sizeof(count);
    if (sysctlbyname("hw.physicalcpu", &count, &size, NULL, 0) == 0 && count > 0)
        return count;
#elif defined(__linux__)
    /* Count distinct (package, core) identities within our CPU affinity mask. */
    cpu_set_t mask;
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
        int seen_packages[CPU_SETSIZE];
        int seen_cores[CPU_SETSIZE];
        int n = 0;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (!CPU_ISSET(cpu, &mask))
                continue;
            char path[256];
            int core_id, package_id;
            snprintf(path, sizeof(path),
                     "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
            FILE *file = fopen(path, "r");
            if (file == NULL)
                continue;
            int valid_core = fscanf(file, "%d", &core_id);
            fclose(file);
            snprintf(path, sizeof(path),
                     "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
            file = fopen(path, "r");
            if (file == NULL)
                continue;
            int valid_package = fscanf(file, "%d", &package_id);
            fclose(file);
            if (valid_core != 1 || valid_package != 1)
                continue;
            int duplicate = 0;
            for (int k = 0; k < n; ++k) {
                if (seen_packages[k] == package_id && seen_cores[k] == core_id) {
                    duplicate = 1;
                    break;
                }
            }
            if (!duplicate) {
                seen_packages[n] = package_id;
                seen_cores[n] = core_id;
                ++n;
            }
        }
        if (n > 0)
            return n;
        int available = CPU_COUNT(&mask);
        if (available > 0)
            return available;
    }
#endif
    return logical_core_count();
}

static void *run_worker(void *opaque) {
    WorkerArgs *args = (WorkerArgs *)opaque;
    const double *matrix = args->matrix;
    double local = 0.0;

    if (args->traversal == ROW_MAJOR) {
        /* Consecutive columns are adjacent in row-major C memory. */
        for (size_t row = args->first; row < args->last; ++row) {
            const double *row_ptr = matrix + row * MATRIX_SIZE;
            for (size_t col = 0; col < MATRIX_SIZE; ++col)
                local += row_ptr[col];
        }
    } else {
        /* Fixed column, increasing row: jumps 8000 doubles (64000 B). */
        for (size_t col = args->first; col < args->last; ++col) {
            for (size_t row = 0; row < MATRIX_SIZE; ++row)
                local += matrix[row * MATRIX_SIZE + col];
        }
    }
    args->partial_sum = local;
    return NULL;
}

static int benchmark(const double *matrix, int thread_count, Traversal traversal,
                     double *elapsed_ms, double *total_sum) {
    pthread_t *threads = calloc((size_t)thread_count, sizeof(*threads));
    WorkerArgs *args = calloc((size_t)thread_count, sizeof(*args));
    if (!threads || !args) {
        fprintf(stderr, "Could not allocate thread metadata.\n");
        free(threads);
        free(args);
        return -1;
    }

    /* Wall-clock timing includes thread launch and joins. */
    double started = monotonic_ms();
    int created = 0;
    for (int t = 0; t < thread_count; ++t) {
        args[t].matrix = matrix;
        args[t].first = MATRIX_SIZE * (size_t)t / (size_t)thread_count;
        args[t].last = MATRIX_SIZE * (size_t)(t + 1) / (size_t)thread_count;
        args[t].traversal = traversal;
        int error = pthread_create(&threads[t], NULL, run_worker, &args[t]);
        if (error != 0) {
            fprintf(stderr, "pthread_create: %s\n", strerror(error));
            break;
        }
        ++created;
    }
    for (int t = 0; t < created; ++t) {
        int error = pthread_join(threads[t], NULL);
        if (error != 0) {
            fprintf(stderr, "pthread_join: %s\n", strerror(error));
            free(threads);
            free(args);
            return -1;
        }
    }
    double stopped = monotonic_ms();
    if (created != thread_count) {
        free(threads);
        free(args);
        return -1;
    }

    double sum = 0.0;
    for (int t = 0; t < thread_count; ++t)
        sum += args[t].partial_sum;

    *elapsed_ms = stopped - started;
    *total_sum = sum;
    free(threads);
    free(args);
    return 0;
}

static double median_three(double values[REPEATS]) {
    double a = values[0], b = values[1], c = values[2];
    if (a > b) { double tmp = a; a = b; b = tmp; }
    if (b > c) { double tmp = b; b = c; c = tmp; }
    if (a > b) { double tmp = a; a = b; b = tmp; }
    return b;
}

static void print_report(FILE *out, int logical, int physical, int threads,
                         double a_ms, double b_ms, double sum_a, double sum_b) {
    fprintf(out, "TASK 1 - Row-Major vs. Column-Major Parallel Cache Traversal\n");
    fprintf(out, "Student ID: %llu\n", (unsigned long long)STUDENT_ID);
    fprintf(out, "Logical core count: %d\n", logical);
    fprintf(out, "Physical core count (available): %d\n", physical);
    fprintf(out, "Parallel workers: %d\n", threads);
    fprintf(out, "Matrix: %u x %u, float64 (%.2f MiB)\n", MATRIX_SIZE,
            MATRIX_SIZE, (double)MATRIX_SIZE * MATRIX_SIZE * sizeof(double) / 1048576.0);
    fprintf(out, "Trials per mode: %d (median wall-clock time, includes thread overhead)\n", REPEATS);
    fprintf(out, "Mode A (Row-Major) runtime: %.3f ms\n", a_ms);
    fprintf(out, "Mode B (Column-Major / Strided) runtime: %.3f ms\n", b_ms);
    fprintf(out, "Cache penalty factor (B / A): %.3fx\n", b_ms / a_ms);
    fprintf(out, "Mode A matrix sum: %.8f\n", sum_a);
    fprintf(out, "Mode B matrix sum: %.8f\n", sum_b);
    fprintf(out, "Sum verification: PASS\n");
}

int main(void) {
    size_t elements = (size_t)MATRIX_SIZE * MATRIX_SIZE;
    size_t bytes = elements * sizeof(double);
    double *matrix = NULL;
    int mem_error = posix_memalign((void **)&matrix, 64, bytes);
    if (mem_error != 0) {
        fprintf(stderr, "Matrix allocation failed (%zu bytes): %s\n",
                bytes, strerror(mem_error));
        return EXIT_FAILURE;
    }
    fill_matrix(matrix, elements);

    int logical = logical_core_count();
    int physical = physical_core_count();
    int threads = physical;
    if (threads > (int)MATRIX_SIZE)
        threads = (int)MATRIX_SIZE;

    double a_times[REPEATS], b_times[REPEATS];
    double sum_a = 0.0, sum_b = 0.0;

    /* Alternate mode order to reduce measurement-order bias. */
    for (int trial = 0; trial < REPEATS; ++trial) {
        if (trial % 2 == 0) {
            if (benchmark(matrix, threads, ROW_MAJOR, &a_times[trial], &sum_a) != 0 ||
                benchmark(matrix, threads, COLUMN_MAJOR, &b_times[trial], &sum_b) != 0)
                goto fail;
        } else {
            if (benchmark(matrix, threads, COLUMN_MAJOR, &b_times[trial], &sum_b) != 0 ||
                benchmark(matrix, threads, ROW_MAJOR, &a_times[trial], &sum_a) != 0)
                goto fail;
        }
    }

    if (fabs(sum_a - sum_b) > 1e-10 * fmax(1.0, fabs(sum_a))) {
        fprintf(stderr, "Sum mismatch: %.12f vs %.12f\n", sum_a, sum_b);
        goto fail;
    }

    double a_ms = median_three(a_times);
    double b_ms = median_three(b_times);
    print_report(stdout, logical, physical, threads, a_ms, b_ms, sum_a, sum_b);

    FILE *out = fopen("task1_output.txt", "w");
    if (!out) {
        fprintf(stderr, "Cannot write task1_output.txt: %s\n", strerror(errno));
        goto fail;
    }
    print_report(out, logical, physical, threads, a_ms, b_ms, sum_a, sum_b);
    if (fclose(out) != 0) {
        perror("fclose(task1_output.txt)");
        goto fail;
    }
    free(matrix);
    return EXIT_SUCCESS;

fail:
    free(matrix);
    return EXIT_FAILURE;
}
