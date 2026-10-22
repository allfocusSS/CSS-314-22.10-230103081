/*
 * CSS 314 - Task 6: Synchronization Cost & Data Loss
 * Student ID: 230103081
 *
 * Four pthreads each attempt 2,000,000 increments in three modes:
 *  1) UNPROTECTED: separate relaxed atomic load and store. The individual
 *     accesses are well-defined, but the combined increment is NOT atomic;
 *     overlapping threads lose updates. No lock or synchronizing RMW is used.
 *  2) ATOMIC RMW: atomic_fetch_add, which cannot lose updates.
 *  3) MUTEX: pthread_mutex_lock/unlock around each increment; this separately
 *     demonstrates the cost of an actual synchronization lock.
 *
 * The first approach deliberately shows a logical race without invoking the
 * undefined behavior of ordinary non-atomic shared C variable accesses.
 *
 * macOS: clang -O2 -std=c11 -Wall -Wextra -pthread task6.c -o task6
 * Run:   ./task6 | tee task6_output.txt
 */

#define _POSIX_C_SOURCE 200809L

#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/utsname.h>
#include <time.h>

#define STUDENT_ID 230103081
#define THREAD_COUNT 4
#define INCREMENTS_PER_THREAD UINT64_C(2000000)
#define EXPECTED_COUNT (THREAD_COUNT * INCREMENTS_PER_THREAD)

typedef enum {
    MODE_UNPROTECTED,
    MODE_ATOMIC_RMW,
    MODE_MUTEX
} Mode;

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t ready_cv;
    pthread_cond_t start_cv;
    int ready;
    int start;
} StartGate;

typedef struct {
    _Atomic uint64_t counter;
    pthread_mutex_t counter_mutex;
    StartGate gate;
    Mode mode;
} SharedState;

typedef struct {
    SharedState *state;
} WorkerArg;

typedef struct {
    uint64_t actual;
    double ms;
} Result;

static void check_pthread(int status, const char *action) {
    if (status != 0) {
        fprintf(stderr, "pthread %s failed: %s\n", action, strerror(status));
        exit(EXIT_FAILURE);
    }
}

static double monotonic_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static void *worker(void *opaque) {
    WorkerArg *arg = (WorkerArg *)opaque;
    SharedState *s = arg->state;

    /* Start all workers together for more realistic contention. */
    check_pthread(pthread_mutex_lock(&s->gate.mu), "lock gate");
    s->gate.ready++;
    if (s->gate.ready == THREAD_COUNT) {
        check_pthread(pthread_cond_signal(&s->gate.ready_cv), "signal ready");
    }
    while (!s->gate.start) {
        check_pthread(pthread_cond_wait(&s->gate.start_cv, &s->gate.mu),
                      "wait for start");
    }
    check_pthread(pthread_mutex_unlock(&s->gate.mu), "unlock gate");

    if (s->mode == MODE_UNPROTECTED) {
        for (uint64_t k = 0; k < INCREMENTS_PER_THREAD; k++) {
            /* Two individually atomic accesses != one atomic increment. */
            uint64_t old = atomic_load_explicit(&s->counter, memory_order_relaxed);
            atomic_store_explicit(&s->counter, old + 1, memory_order_relaxed);
        }
    } else if (s->mode == MODE_ATOMIC_RMW) {
        for (uint64_t k = 0; k < INCREMENTS_PER_THREAD; k++) {
            atomic_fetch_add_explicit(&s->counter, UINT64_C(1), memory_order_relaxed);
        }
    } else {
        for (uint64_t k = 0; k < INCREMENTS_PER_THREAD; k++) {
            check_pthread(pthread_mutex_lock(&s->counter_mutex), "lock counter");
            uint64_t old = atomic_load_explicit(&s->counter, memory_order_relaxed);
            atomic_store_explicit(&s->counter, old + 1, memory_order_relaxed);
            check_pthread(pthread_mutex_unlock(&s->counter_mutex), "unlock counter");
        }
    }
    return NULL;
}

static Result run(Mode mode) {
    SharedState s;
    atomic_init(&s.counter, 0);
    s.mode = mode;
    s.gate.ready = 0;
    s.gate.start = 0;

    check_pthread(pthread_mutex_init(&s.counter_mutex, NULL), "init counter mutex");
    check_pthread(pthread_mutex_init(&s.gate.mu, NULL), "init gate mutex");
    check_pthread(pthread_cond_init(&s.gate.ready_cv, NULL), "init ready cv");
    check_pthread(pthread_cond_init(&s.gate.start_cv, NULL), "init start cv");

    pthread_t threads[THREAD_COUNT];
    WorkerArg args[THREAD_COUNT];
    for (int i = 0; i < THREAD_COUNT; i++) {
        args[i].state = &s;
        check_pthread(pthread_create(&threads[i], NULL, worker, &args[i]),
                      "create thread");
    }

    check_pthread(pthread_mutex_lock(&s.gate.mu), "lock main gate");
    while (s.gate.ready < THREAD_COUNT) {
        check_pthread(pthread_cond_wait(&s.gate.ready_cv, &s.gate.mu),
                      "wait until ready");
    }
    double start_ms = monotonic_ms();
    s.gate.start = 1;
    check_pthread(pthread_cond_broadcast(&s.gate.start_cv), "broadcast start");
    check_pthread(pthread_mutex_unlock(&s.gate.mu), "unlock main gate");

    for (int i = 0; i < THREAD_COUNT; i++) {
        check_pthread(pthread_join(threads[i], NULL), "join thread");
    }
    double elapsed_ms = monotonic_ms() - start_ms;
    Result r = {atomic_load_explicit(&s.counter, memory_order_relaxed), elapsed_ms};

    check_pthread(pthread_cond_destroy(&s.gate.start_cv), "destroy start cv");
    check_pthread(pthread_cond_destroy(&s.gate.ready_cv), "destroy ready cv");
    check_pthread(pthread_mutex_destroy(&s.gate.mu), "destroy gate mutex");
    check_pthread(pthread_mutex_destroy(&s.counter_mutex), "destroy counter mutex");
    return r;
}

int main(void) {
    struct utsname hardware;
    if (uname(&hardware) != 0) {
        perror("uname");
        return EXIT_FAILURE;
    }

    printf("TASK 6 - Synchronization Cost & Data Loss\n");
    printf("Student ID: %d\n", STUDENT_ID);
    printf("Platform: %s %s (%s)\n", hardware.sysname, hardware.release, hardware.machine);
    printf("Worker threads: %d\n", THREAD_COUNT);
    printf("Increments per thread: %" PRIu64 "\n", INCREMENTS_PER_THREAD);
    printf("Expected final value: %" PRIu64 "\n\n", (uint64_t)EXPECTED_COUNT);

    Result unprotected = run(MODE_UNPROTECTED);
    Result atomic = run(MODE_ATOMIC_RMW);
    Result mutex = run(MODE_MUTEX);

    uint64_t lost = EXPECTED_COUNT - unprotected.actual;
    double lost_pct = 100.0 * (double)lost / (double)EXPECTED_COUNT;

    printf("UNPROTECTED (relaxed load + store; no atomic increment)\n");
    printf("  Actual final value: %" PRIu64 "\n", unprotected.actual);
    printf("  Lost updates: %" PRIu64 "\n", lost);
    printf("  Lost updates: %.4f%%\n", lost_pct);
    printf("  Wall-clock time: %.3f ms\n\n", unprotected.ms);

    printf("ATOMIC (atomic_fetch_add, correct result)\n");
    printf("  Actual final value: %" PRIu64 "\n", atomic.actual);
    printf("  Wall-clock time: %.3f ms\n\n", atomic.ms);

    printf("MUTEX (pthread_mutex_lock/unlock, correct result)\n");
    printf("  Actual final value: %" PRIu64 "\n", mutex.actual);
    printf("  Wall-clock time: %.3f ms\n\n", mutex.ms);

    printf("Synchronization overhead versus unprotected run:\n");
    printf("  Atomic RMW penalty: %+.3f ms (%.2fx time)\n",
           atomic.ms - unprotected.ms, atomic.ms / unprotected.ms);
    printf("  Mutex lock penalty: %+.3f ms (%.2fx time)\n",
           mutex.ms - unprotected.ms, mutex.ms / unprotected.ms);

    if (atomic.actual != EXPECTED_COUNT || mutex.actual != EXPECTED_COUNT ||
        unprotected.actual > EXPECTED_COUNT) {
        fprintf(stderr, "Verification: FAILED\n");
        return EXIT_FAILURE;
    }
    printf("Verification: PASSED (atomic and mutex both reached 8,000,000)\n");
    if (lost == 0) {
        printf("Note: no updates were lost in this run; rerun to observe the race.\n");
    }
    return EXIT_SUCCESS;
}
