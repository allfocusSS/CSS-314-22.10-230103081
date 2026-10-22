#ifdef __APPLE__
/* Expose Darwin's BSD sysctl types such as u_int and u_char. */
#define _DARWIN_C_SOURCE 1
#else
#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include <inttypes.h>
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

#define STUDENT_ID 230103081
#define INCREMENTS_PER_THREAD UINT64_C(50000000)
/* Apple Silicon reports 128-byte cache lines; Linux commonly uses 64. */
#ifdef __APPLE__
#define CACHE_LINE_BYTES 128
#else
#define CACHE_LINE_BYTES 64
#endif

typedef struct {
    volatile uint64_t value;
    unsigned char pad[CACHE_LINE_BYTES - sizeof(uint64_t)];
} padded_counter_t;
_Static_assert(sizeof(padded_counter_t) == CACHE_LINE_BYTES,
               "Padded counter stride must be exactly 64 bytes");

typedef struct {
    volatile uint64_t *counter;
} worker_args_t;

static void *increment_worker(void *argument) {
    worker_args_t *args = (worker_args_t *)argument;
    volatile uint64_t *counter = args->counter;
    for (uint64_t j = 0; j < INCREMENTS_PER_THREAD; ++j) {
        ++(*counter);
    }
    return NULL;
}

#ifdef __linux__
/* Count distinct (socket, core) pairs among CPUs available to this process. */
static int linux_physical_cores(void) {
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return 0;

    int socket_ids[CPU_SETSIZE];
    int core_ids[CPU_SETSIZE];
    int unique = 0;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) continue;
        char path[160];
        int socket_id = 0, core_id = 0;
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
        FILE *file = fopen(path, "r");
        if (!file) return 0;
        int ok = fscanf(file, "%d", &socket_id);
        fclose(file);
        if (ok != 1) return 0;
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
        file = fopen(path, "r");
        if (!file) return 0;
        ok = fscanf(file, "%d", &core_id);
        fclose(file);
        if (ok != 1) return 0;
        int seen = 0;
        for (int j = 0; j < unique; ++j) {
            if (socket_ids[j] == socket_id && core_ids[j] == core_id) {
                seen = 1;
                break;
            }
        }
        if (!seen) {
            socket_ids[unique] = socket_id;
            core_ids[unique] = core_id;
            ++unique;
        }
    }
    return unique;
}
#endif

static int logical_core_count(void) {
#ifdef __APPLE__
    int count = 0;
    size_t size = sizeof(count);
    if (sysctlbyname("hw.logicalcpu", &count, &size, NULL, 0) == 0 && count > 0)
        return count;
    return 1;
#else
    long count = sysconf(_SC_NPROCESSORS_ONLN);
    return count > 0 ? (int)count : 1;
#endif
}

static int physical_core_count(void) {
#ifdef __APPLE__
    int count = 0;
    size_t size = sizeof(count);
    if (sysctlbyname("hw.physicalcpu", &count, &size, NULL, 0) == 0 && count > 0)
        return count;
#elif defined(__linux__)
    int count = linux_physical_cores();
    if (count > 0) return count;
#endif
    return logical_core_count();
}

static double current_time_ms(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
    return (double)now.tv_sec * 1000.0 + (double)now.tv_nsec / 1e6;
}

/* The same worker and timing function are used for both memory layouts. */
static double benchmark(worker_args_t *args, int count) {
    pthread_t *threads = calloc((size_t)count, sizeof(*threads));
    if (!threads) {
        perror("calloc threads");
        exit(EXIT_FAILURE);
    }

    double start = current_time_ms();
    int started = 0;
    for (int i = 0; i < count; ++i) {
        int err = pthread_create(&threads[i], NULL, increment_worker, &args[i]);
        if (err != 0) {
            fprintf(stderr, "pthread_create error: %s\n", strerror(err));
            for (int j = 0; j < started; ++j) pthread_join(threads[j], NULL);
            free(threads);
            exit(EXIT_FAILURE);
        }
        ++started;
    }
    for (int i = 0; i < count; ++i) {
        int err = pthread_join(threads[i], NULL);
        if (err != 0) {
            fprintf(stderr, "pthread_join error: %s\n", strerror(err));
            free(threads);
            exit(EXIT_FAILURE);
        }
    }
    double elapsed = current_time_ms() - start;
    free(threads);
    return elapsed;
}

int main(void) {
    const int workers = physical_core_count();
    const int logical_cores = logical_core_count();
    if (workers < 1 || workers > 4096) {
        fprintf(stderr, "Invalid physical core count: %d\n", workers);
        return EXIT_FAILURE;
    }

    volatile uint64_t *shared_counts = calloc((size_t)workers, sizeof(uint64_t));
    padded_counter_t *padded_counts = NULL;
    worker_args_t *args = calloc((size_t)workers, sizeof(*args));
    if (!shared_counts || !args ||
        posix_memalign((void **)&padded_counts, CACHE_LINE_BYTES,
                       (size_t)workers * sizeof(*padded_counts)) != 0) {
        fprintf(stderr, "Could not allocate counters or worker arguments\n");
        free((void *)shared_counts);
        free(args);
        return EXIT_FAILURE;
    }
    memset(padded_counts, 0, (size_t)workers * sizeof(*padded_counts));

    struct utsname info;
    if (uname(&info) != 0) {
        perror("uname");
        return EXIT_FAILURE;
    }
    printf("TASK 3: Cache Line Contention (False Sharing)\n");
    printf("Student ID: %d\n", STUDENT_ID);
    printf("OS: %s %s (%s)\n", info.sysname, info.release, info.machine);
    printf("Physical cores / threads spawned: %d\n", workers);
    printf("Logical processors reported: %d\n", logical_cores);
    printf("Increments per thread: %" PRIu64 "\n", INCREMENTS_PER_THREAD);
    printf("Unpadded shared_counts size: %d elements (8-byte stride)\n", workers);
    printf("Padded counter stride: %zu bytes (%d-byte aligned)\n", sizeof(*padded_counts), CACHE_LINE_BYTES);
    puts("Timing: wall clock, includes thread creation/join, excludes allocation");
    puts("");

    for (int i = 0; i < workers; ++i) args[i].counter = &shared_counts[i];
    double unpadded_ms = benchmark(args, workers);

    for (int i = 0; i < workers; ++i) args[i].counter = &padded_counts[i].value;
    double padded_ms = benchmark(args, workers);

    const uint64_t expected = INCREMENTS_PER_THREAD * (uint64_t)workers;
    uint64_t unpadded_total = 0, padded_total = 0;
    for (int i = 0; i < workers; ++i) {
        unpadded_total += shared_counts[i];
        padded_total += padded_counts[i].value;
        if (shared_counts[i] != INCREMENTS_PER_THREAD ||
            padded_counts[i].value != INCREMENTS_PER_THREAD) {
            fprintf(stderr, "Incorrect value at worker %d\n", i);
            free((void *)shared_counts);
            free(padded_counts);
            free(args);
            return EXIT_FAILURE;
        }
    }

    printf("%-22s | %15s\n", "Memory Layout", "Wall Time (ms)");
    puts("-----------------------------------------");
    printf("%-22s | %15.3f\n", "Unpadded Shared", unpadded_ms);
    char padded_label[48];
    snprintf(padded_label, sizeof(padded_label), "Padded (%d-byte)", CACHE_LINE_BYTES);
    printf("%-22s | %15.3f\n", padded_label, padded_ms);
    printf("\nExecution ratio (Unpadded / Padded): %.3fx\n", unpadded_ms / padded_ms);
    printf("Expected total per layout: %" PRIu64 "\n", expected);
    printf("Unpadded total: %" PRIu64 "\n", unpadded_total);
    printf("Padded total: %" PRIu64 "\n", padded_total);
    puts("Verification: PASSED (each worker performed 50,000,000 increments)");

    free((void *)shared_counts);
    free(padded_counts);
    free(args);
    return EXIT_SUCCESS;
}
