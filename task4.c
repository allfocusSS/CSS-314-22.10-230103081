#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_ITEMS 1000
#define NUM_WORKERS 4
#define DYNAMIC_CHUNK 5

typedef enum {
    STATIC_SCHEDULING,
    DYNAMIC_SCHEDULING
} Strategy;

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t ready_cond;
    pthread_cond_t start_cond;
    int ready_count;
    int started;
    int next_item;
    double start_time;
    Strategy strategy;
} Scheduler;

typedef struct {
    int id;
    int completed_items;
    double completion_time;
    Scheduler *scheduler;
} Worker;

static double monotonic_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void simulate_work(int i) {
    long nanoseconds = (long)(i % 20) * 100000L;
    if (nanoseconds == 0) return;

    struct timespec remaining = {0, nanoseconds};
    while (nanosleep(&remaining, &remaining) != 0) {
        if (errno != EINTR) {
            perror("nanosleep");
            exit(EXIT_FAILURE);
        }
    }
}

static void *worker_main(void *argument) {
    Worker *worker = (Worker *)argument;
    Scheduler *scheduler = worker->scheduler;

    /* Release all four workers together for comparable measurements. */
    pthread_mutex_lock(&scheduler->mutex);
    scheduler->ready_count++;
    pthread_cond_signal(&scheduler->ready_cond);
    while (!scheduler->started) {
        pthread_cond_wait(&scheduler->start_cond, &scheduler->mutex);
    }
    pthread_mutex_unlock(&scheduler->mutex);

    if (scheduler->strategy == STATIC_SCHEDULING) {
        int begin = worker->id * NUM_ITEMS / NUM_WORKERS;
        int end = (worker->id + 1) * NUM_ITEMS / NUM_WORKERS;
        for (int i = begin; i < end; i++) {
            simulate_work(i);
            worker->completed_items++;
        }
    } else {
        /* Shared task queue: idle workers claim the next 5 items. */
        for (;;) {
            pthread_mutex_lock(&scheduler->mutex);
            int begin = scheduler->next_item;
            scheduler->next_item += DYNAMIC_CHUNK;
            pthread_mutex_unlock(&scheduler->mutex);

            if (begin >= NUM_ITEMS) break;
            int end = begin + DYNAMIC_CHUNK;
            if (end > NUM_ITEMS) end = NUM_ITEMS;
            for (int i = begin; i < end; i++) {
                simulate_work(i);
                worker->completed_items++;
            }
        }
    }

    worker->completion_time = monotonic_seconds() - scheduler->start_time;
    return NULL;
}

static int run_strategy(Strategy strategy) {
    Scheduler scheduler = {0};
    scheduler.strategy = strategy;
    if (pthread_mutex_init(&scheduler.mutex, NULL) != 0 ||
        pthread_cond_init(&scheduler.ready_cond, NULL) != 0 ||
        pthread_cond_init(&scheduler.start_cond, NULL) != 0) {
        fputs("Failed to initialize thread synchronization.\n", stderr);
        return 1;
    }

    pthread_t threads[NUM_WORKERS];
    Worker workers[NUM_WORKERS] = {0};
    for (int w = 0; w < NUM_WORKERS; w++) {
        workers[w].id = w;
        workers[w].scheduler = &scheduler;
        if (pthread_create(&threads[w], NULL, worker_main, &workers[w]) != 0) {
            fputs("Failed to create a worker thread.\n", stderr);
            return 1;
        }
    }

    pthread_mutex_lock(&scheduler.mutex);
    while (scheduler.ready_count != NUM_WORKERS) {
        pthread_cond_wait(&scheduler.ready_cond, &scheduler.mutex);
    }
    scheduler.start_time = monotonic_seconds();
    scheduler.started = 1;
    pthread_cond_broadcast(&scheduler.start_cond);
    pthread_mutex_unlock(&scheduler.mutex);

    for (int w = 0; w < NUM_WORKERS; w++) {
        pthread_join(threads[w], NULL);
    }
    double total_time = monotonic_seconds() - scheduler.start_time;

    printf("\nStrategy %s: %s\n",
           strategy == STATIC_SCHEDULING ? "A" : "B",
           strategy == STATIC_SCHEDULING ? "Static equal blocks" : "Dynamic shared queue (chunk=5)");

    double earliest = workers[0].completion_time;
    double latest = workers[0].completion_time;
    int total_items = 0;
    for (int w = 0; w < NUM_WORKERS; w++) {
        double t = workers[w].completion_time;
        if (t < earliest) earliest = t;
        if (t > latest) latest = t;
        total_items += workers[w].completed_items;
        printf("  Worker %d: %3d items, finished at %.6f s\n",
               w, workers[w].completed_items, t);
    }
    printf("  Total items processed:    %d / %d\n", total_items, NUM_ITEMS);
    printf("  Total execution time:     %.6f s\n", total_time);
    printf("  Min completion time:      %.6f s\n", earliest);
    printf("  Max completion time:      %.6f s\n", latest);
    printf("  Completion delta max-min: %.6f s\n", latest - earliest);

    pthread_cond_destroy(&scheduler.start_cond);
    pthread_cond_destroy(&scheduler.ready_cond);
    pthread_mutex_destroy(&scheduler.mutex);
    return total_items == NUM_ITEMS ? 0 : 1;
}

int main(void) {
    puts("TASK 4 - Work Stealing & Scheduling Under Skew");
    printf("Items: %d (indices 0..999); workers: %d\n", NUM_ITEMS, NUM_WORKERS);
    puts("Workload per item: sleep(0.0001 * (i % 20)) seconds");
    puts("Times use CLOCK_MONOTONIC; completion times measured from common start.");

    if (run_strategy(STATIC_SCHEDULING) != 0) return EXIT_FAILURE;
    if (run_strategy(DYNAMIC_SCHEDULING) != 0) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
