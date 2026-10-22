/*
 * TASK 3 — Prefix Scan (Step Efficiency vs. Work Efficiency)
 * Student ID: 230103081
 *
 * macOS / Linux:
 *   clang -O3 -std=c11 -Wall -Wextra -pthread task3.c -o task3 -lm
 *   ./task3
 *
 * The program writes task3_output.txt using measurements from the host
 * computer. Benchmarks use the median of 9 wall-clock runs, including
 * thread creation and joining (input generation is excluded).
 *
 * Parallel algorithm:
 *   Pass 1: Each of four workers scans its own contiguous chunk, recording
 *           that chunk's sum.
 *   Between passes: Main thread computes prefix offsets from chunk sums.
 *   Pass 2: Workers add their offsets to the local prefix values.
 */

#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/utsname.h>
#include <time.h>

#define STUDENT_ID UINT64_C(230103081)
#define INPUT_SIZE 2000000U
#define THREAD_COUNT 4U
#define BENCHMARK_RUNS 9U

typedef struct {
    const int32_t *input;
    int64_t *output;
    size_t begin;
    size_t end;
    int64_t local_sum;
    int64_t offset;
} Chunk;

/* Reproducible integer values from 1 to 10, seeded by Student ID. */
static uint64_t splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static double milliseconds_now(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime");
        exit(1);
    }
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static void serial_scan(const int32_t *input, int64_t *output) {
    int64_t running = 0;
    for (size_t i = 0; i < INPUT_SIZE; ++i) {
        running += input[i];
        output[i] = running; /* inclusive: output[i] = output[i-1] + input[i] */
    }
}

static void *scan_local_chunk(void *arg) {
    Chunk *c = (Chunk *)arg;
    int64_t running = 0;
    for (size_t i = c->begin; i < c->end; ++i) {
        running += c->input[i];
        c->output[i] = running;
    }
    c->local_sum = running;
    return NULL;
}

static void *apply_chunk_offset(void *arg) {
    Chunk *c = (Chunk *)arg;
    for (size_t i = c->begin; i < c->end; ++i) {
        c->output[i] += c->offset;
    }
    return NULL;
}

static void spawn_and_join(pthread_t threads[THREAD_COUNT],
                           Chunk chunks[THREAD_COUNT],
                           void *(*worker)(void *)) {
    unsigned int created = 0;
    for (unsigned int t = 0; t < THREAD_COUNT; ++t) {
        int err = pthread_create(&threads[t], NULL, worker, &chunks[t]);
        if (err != 0) {
            fprintf(stderr, "pthread_create failed: %s\n", strerror(err));
            for (unsigned int j = 0; j < created; ++j) {
                (void)pthread_join(threads[j], NULL);
            }
            exit(1);
        }
        ++created;
    }
    for (unsigned int t = 0; t < THREAD_COUNT; ++t) {
        int err = pthread_join(threads[t], NULL);
        if (err != 0) {
            fprintf(stderr, "pthread_join failed: %s\n", strerror(err));
            exit(1);
        }
    }
}

static void parallel_scan(const int32_t *input, int64_t *output) {
    pthread_t threads[THREAD_COUNT];
    Chunk chunks[THREAD_COUNT];

    for (unsigned int t = 0; t < THREAD_COUNT; ++t) {
        chunks[t].input = input;
        chunks[t].output = output;
        chunks[t].begin = (size_t)INPUT_SIZE * t / THREAD_COUNT;
        chunks[t].end = (size_t)INPUT_SIZE * (t + 1U) / THREAD_COUNT;
        chunks[t].local_sum = 0;
        chunks[t].offset = 0;
    }

    /* Pass 1: independent local inclusive scans and chunk sums. */
    spawn_and_join(threads, chunks, scan_local_chunk);

    /* Prefix reduction over chunk totals: exclusive offset for each chunk. */
    int64_t running_offset = 0;
    for (unsigned int t = 0; t < THREAD_COUNT; ++t) {
        chunks[t].offset = running_offset;
        running_offset += chunks[t].local_sum;
    }

    /* Pass 2: convert local prefixes to global inclusive prefixes. */
    spawn_and_join(threads, chunks, apply_chunk_offset);
}

static int double_compare(const void *a, const void *b) {
    const double x = *(const double *)a;
    const double y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median_ms(double timings[BENCHMARK_RUNS]) {
    double values[BENCHMARK_RUNS];
    memcpy(values, timings, sizeof(values));
    qsort(values, BENCHMARK_RUNS, sizeof(values[0]), double_compare);
    return values[BENCHMARK_RUNS / 2U];
}

static void write_results(FILE *stream, const struct utsname *host,
                          double serial_ms, double parallel_ms,
                          int64_t serial_final, int64_t parallel_final,
                          int verification_passed) {
    fprintf(stream, "TASK 3: Prefix Scan (Step Efficiency vs. Work Efficiency)\n");
    fprintf(stream, "Host: %s (%s)\n", host->sysname, host->machine);
    fprintf(stream, "Student ID: %" PRIu64 "\n", STUDENT_ID);
    fprintf(stream, "Array size: %u integers\n", INPUT_SIZE);
    fprintf(stream, "Input range: 1..10 (SplitMix64, Student ID seed)\n");
    fprintf(stream, "Parallel threads: %u\n", THREAD_COUNT);
    fprintf(stream, "Benchmark: median of %u runs (wall clock, ms)\n", BENCHMARK_RUNS);
    fprintf(stream, "Serial Time (ms): %.3f\n", serial_ms);
    fprintf(stream, "Parallel Time (ms): %.3f\n", parallel_ms);
    fprintf(stream, "Speedup (serial / parallel): %.3fx\n",
            serial_ms / parallel_ms);
    fprintf(stream, "Serial output[-1]: %" PRId64 "\n", serial_final);
    fprintf(stream, "Parallel output[-1]: %" PRId64 "\n", parallel_final);
    fprintf(stream, "Verification Status: %s\n",
            verification_passed ? "PASS" : "FAIL");
    fprintf(stream, "Full array equality: %s\n",
            verification_passed ? "PASS" : "FAIL");
}

int main(void) {
    struct utsname host;
    if (uname(&host) != 0) {
        perror("uname");
        return 1;
    }

    int32_t *input = malloc((size_t)INPUT_SIZE * sizeof(*input));
    int64_t *serial_output = malloc((size_t)INPUT_SIZE * sizeof(*serial_output));
    int64_t *parallel_output = malloc((size_t)INPUT_SIZE * sizeof(*parallel_output));
    if (input == NULL || serial_output == NULL || parallel_output == NULL) {
        fprintf(stderr, "Allocation failed: expected about 40 MB of memory.\n");
        free(input);
        free(serial_output);
        free(parallel_output);
        return 1;
    }

    uint64_t rng_state = STUDENT_ID;
    for (size_t i = 0; i < INPUT_SIZE; ++i) {
        input[i] = (int32_t)(splitmix64_next(&rng_state) % 10U) + 1;
    }

    /* Warm up both implementations before collecting measured runs. */
    serial_scan(input, serial_output);
    parallel_scan(input, parallel_output);

    double serial_timings[BENCHMARK_RUNS];
    double parallel_timings[BENCHMARK_RUNS];
    for (unsigned int run = 0; run < BENCHMARK_RUNS; ++run) {
        double start = milliseconds_now();
        serial_scan(input, serial_output);
        serial_timings[run] = milliseconds_now() - start;

        start = milliseconds_now();
        parallel_scan(input, parallel_output);
        parallel_timings[run] = milliseconds_now() - start;
    }

    /* An assertion with explicit exit(1) instead of assert()'s SIGABRT exit. */
    const int last_element_matches =
        (parallel_output[INPUT_SIZE - 1U] == serial_output[INPUT_SIZE - 1U]);
    const int full_array_matches =
        (memcmp(parallel_output, serial_output,
                (size_t)INPUT_SIZE * sizeof(*parallel_output)) == 0);
    const int verification_passed = last_element_matches && full_array_matches;
    const double serial_ms = median_ms(serial_timings);
    const double parallel_ms = median_ms(parallel_timings);

    write_results(stdout, &host, serial_ms, parallel_ms,
                  serial_output[INPUT_SIZE - 1U],
                  parallel_output[INPUT_SIZE - 1U], verification_passed);

    FILE *output_file = fopen("task3_output.txt", "w");
    if (output_file == NULL) {
        fprintf(stderr, "Cannot write task3_output.txt: %s\n", strerror(errno));
        free(input);
        free(serial_output);
        free(parallel_output);
        return 1;
    }
    write_results(output_file, &host, serial_ms, parallel_ms,
                  serial_output[INPUT_SIZE - 1U],
                  parallel_output[INPUT_SIZE - 1U], verification_passed);
    if (fclose(output_file) != 0) {
        perror("fclose(task3_output.txt)");
        free(input);
        free(serial_output);
        free(parallel_output);
        return 1;
    }
    puts("Results saved to task3_output.txt");

    free(input);
    free(serial_output);
    free(parallel_output);

    /* Explicit assertion: either mismatch returns exit status 1. */
    if (!last_element_matches || !full_array_matches) {
        fprintf(stderr, "ASSERTION FAILED: parallel_output[-1] must equal "
                        "serial_output[-1] (and full arrays must match).\n");
        return 1;
    }
    return 0;
}
