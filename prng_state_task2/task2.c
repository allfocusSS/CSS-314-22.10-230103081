/*
 * TASK 2 — Pseudo-Random Generator State Bottleneck
 * Student ID: 230103081
 *
 * Compile on macOS or Linux:
 *   clang -O3 -std=c11 -Wall -Wextra -pthread task2.c -o task2 -lm
 * Run:
 *   ./task2
 *
 * Generates 40 million Monte Carlo sample points with four pthreads.
 * Both modes use the same SplitMix64 PRNG and the same point-in-circle test.
 * In shared mode a mutex protects the two draws that make one (x, y) point.
 * In independent mode each thread owns its generator state, seeded as
 *   (Student_ID * 10) + thread_index  (indices 0, 1, 2, 3).
 *
 * Timings include thread creation and joining but exclude file I/O.
 * The result file is created afresh by this program on the machine running it.
 */

#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/utsname.h>
#include <time.h>

#define STUDENT_ID UINT64_C(230103081)
#define THREAD_COUNT 4
#define TOTAL_SAMPLES UINT64_C(40000000)

/* Small, deterministic generator: exactly the same algorithm in both modes. */
static uint64_t splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

static double random_unit(uint64_t *state) {
    /* Use the 53 most significant random bits to produce [0, 1). */
    return (double)(splitmix64_next(state) >> 11) * (1.0 / 9007199254740992.0);
}

typedef struct {
    uint64_t prng_state;
    pthread_mutex_t mutex;
} SharedGenerator;

typedef struct {
    int index;
    int shared_mode;
    uint64_t samples;
    uint64_t hits;
    uint64_t local_state;
    SharedGenerator *shared;
} Worker;

typedef struct {
    uint64_t hits;
    double pi_estimate;
    double absolute_error;
    double elapsed_ms;
} BenchmarkResult;

static double now_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static void *run_worker(void *data) {
    Worker *w = (Worker *)data;
    uint64_t inside = 0;
    uint64_t local_state = w->local_state;

    for (uint64_t sample = 0; sample < w->samples; ++sample) {
        double x;
        double y;
        if (w->shared_mode) {
            /* Exactly one shared generator; both draws are mutex-protected. */
            int rc = pthread_mutex_lock(&w->shared->mutex);
            if (rc != 0) {
                fprintf(stderr, "pthread_mutex_lock: %s\n", strerror(rc));
                exit(EXIT_FAILURE);
            }
            x = random_unit(&w->shared->prng_state);
            y = random_unit(&w->shared->prng_state);
            rc = pthread_mutex_unlock(&w->shared->mutex);
            if (rc != 0) {
                fprintf(stderr, "pthread_mutex_unlock: %s\n", strerror(rc));
                exit(EXIT_FAILURE);
            }
        } else {
            /* No locks: this thread exclusively owns local_state. */
            x = random_unit(&local_state);
            y = random_unit(&local_state);
        }
        if (x * x + y * y <= 1.0) {
            ++inside;
        }
    }

    w->hits = inside;
    w->local_state = local_state;
    return NULL;
}

static BenchmarkResult measure_mode(int shared_mode, double true_pi) {
    pthread_t threads[THREAD_COUNT];
    Worker workers[THREAD_COUNT] = {0};
    SharedGenerator shared;
    shared.prng_state = STUDENT_ID * UINT64_C(10);

    int rc = pthread_mutex_init(&shared.mutex, NULL);
    if (rc != 0) {
        fprintf(stderr, "pthread_mutex_init: %s\n", strerror(rc));
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < THREAD_COUNT; ++i) {
        workers[i].index = i;
        workers[i].shared_mode = shared_mode;
        workers[i].samples = TOTAL_SAMPLES / THREAD_COUNT;
        workers[i].local_state = (STUDENT_ID * UINT64_C(10)) + (uint64_t)i;
        workers[i].shared = &shared;
    }

    double start = now_ms();
    for (int i = 0; i < THREAD_COUNT; ++i) {
        rc = pthread_create(&threads[i], NULL, run_worker, &workers[i]);
        if (rc != 0) {
            fprintf(stderr, "pthread_create: %s\n", strerror(rc));
            exit(EXIT_FAILURE);
        }
    }
    for (int i = 0; i < THREAD_COUNT; ++i) {
        rc = pthread_join(threads[i], NULL);
        if (rc != 0) {
            fprintf(stderr, "pthread_join: %s\n", strerror(rc));
            exit(EXIT_FAILURE);
        }
    }
    double elapsed = now_ms() - start;

    uint64_t hits = 0;
    for (int i = 0; i < THREAD_COUNT; ++i) {
        hits += workers[i].hits;
    }

    rc = pthread_mutex_destroy(&shared.mutex);
    if (rc != 0) {
        fprintf(stderr, "pthread_mutex_destroy: %s\n", strerror(rc));
        exit(EXIT_FAILURE);
    }

    BenchmarkResult result;
    result.hits = hits;
    result.pi_estimate = 4.0 * (double)hits / (double)TOTAL_SAMPLES;
    result.absolute_error = fabs(result.pi_estimate - true_pi);
    result.elapsed_ms = elapsed;
    return result;
}

static void report(FILE *file, const char *label, BenchmarkResult r) {
    printf("%s: pi = %.9f | absolute error = %.9f | wall time = %.3f ms\n",
           label, r.pi_estimate, r.absolute_error, r.elapsed_ms);
    fprintf(file, "%s: pi = %.9f | absolute error = %.9f | wall time = %.3f ms\n",
            label, r.pi_estimate, r.absolute_error, r.elapsed_ms);
}

int main(void) {
    const double true_pi = acos(-1.0);
    struct utsname host;
    if (uname(&host) != 0) {
        perror("uname");
        return EXIT_FAILURE;
    }

    printf("TASK 2: Pseudo-Random Generator State Bottleneck\n");
    printf("Host: %s (%s)\n", host.sysname, host.machine);
    printf("Student ID: %llu\n", (unsigned long long)STUDENT_ID);
    printf("Threads: %d\n", THREAD_COUNT);
    printf("Sample points: %llu\n", (unsigned long long)TOTAL_SAMPLES);
    printf("Independent seeds: %llu, %llu, %llu, %llu\n",
           (unsigned long long)(STUDENT_ID * 10 + 0),
           (unsigned long long)(STUDENT_ID * 10 + 1),
           (unsigned long long)(STUDENT_ID * 10 + 2),
           (unsigned long long)(STUDENT_ID * 10 + 3));
    printf("Reference Math.PI: %.15f\n", true_pi);
    printf("Running benchmarks...\n");
    fflush(stdout);

    BenchmarkResult shared = measure_mode(1, true_pi);
    BenchmarkResult independent = measure_mode(0, true_pi);
    if (independent.elapsed_ms <= 0.0) {
        fprintf(stderr, "Invalid elapsed time for independent PRNG\n");
        return EXIT_FAILURE;
    }
    double speedup = shared.elapsed_ms / independent.elapsed_ms;

    FILE *file = fopen("task2_output.txt", "w");
    if (file == NULL) {
        fprintf(stderr, "Could not write task2_output.txt: %s\n", strerror(errno));
        return EXIT_FAILURE;
    }
    fprintf(file, "TASK 2: Pseudo-Random Generator State Bottleneck\n");
    fprintf(file, "Host: %s (%s)\n", host.sysname, host.machine);
    fprintf(file, "Student ID: %llu\n", (unsigned long long)STUDENT_ID);
    fprintf(file, "Threads: %d\n", THREAD_COUNT);
    fprintf(file, "Sample points: %llu\n", (unsigned long long)TOTAL_SAMPLES);
    fprintf(file, "Independent seeds: %llu, %llu, %llu, %llu\n",
            (unsigned long long)(STUDENT_ID * 10 + 0),
            (unsigned long long)(STUDENT_ID * 10 + 1),
            (unsigned long long)(STUDENT_ID * 10 + 2),
            (unsigned long long)(STUDENT_ID * 10 + 3));
    fprintf(file, "Reference Math.PI: %.15f\n", true_pi);

    report(file, "Shared PRNG (mutex)", shared);
    report(file, "Independent PRNG (per thread)", independent);
    printf("Speedup (shared / independent): %.3fx\n", speedup);
    fprintf(file, "Speedup (shared / independent): %.3fx\n", speedup);
    if (fclose(file) != 0) {
        perror("fclose");
        return EXIT_FAILURE;
    }
    printf("Results saved to task2_output.txt\n");
    return EXIT_SUCCESS;
}
