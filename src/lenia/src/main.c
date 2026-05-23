#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lenia.h"

#define DEFAULT_ROWS 128
#define DEFAULT_COLS 128
#define DEFAULT_STEPS 100
#define DEFAULT_DT 0.1
#define DEFAULT_KERNEL_SIZE 26
#define NUM_ORBIUMS 2

enum parse_status {
    PARSE_OK = 0,
    PARSE_HELP = 1,
    PARSE_ERROR = 2
};

static void print_help(const char *exe)
{
    printf("Usage: %s [options]\n", exe);
    printf("\n");
    printf("Options:\n");
    printf("  --rows N          Number of rows in the world (default: %d)\n", DEFAULT_ROWS);
    printf("  --cols N          Number of columns in the world (default: %d)\n", DEFAULT_COLS);
    printf("  --steps N         Number of simulation steps (default: %d)\n", DEFAULT_STEPS);
    printf("  --dt X            Simulation time step (default: %.3f)\n", DEFAULT_DT);
    printf("  --kernel-size N   Convolution kernel size (default: %d)\n", DEFAULT_KERNEL_SIZE);
    printf("  --gif FILE        Gather each iteration to rank 0 and generate a GIF\n");
    printf("  --output FILE     Write the final world state to a text file\n");
    printf("  --help, -h        Show this help message\n");
}

static int parse_unsigned_int(const char *text, unsigned int *value)
{
    char *endptr = NULL;
    errno = 0;
    unsigned long parsed = strtoul(text, &endptr, 10);

    if (text[0] == '\0' || endptr == NULL || *endptr != '\0' || errno != 0 || parsed > UINT_MAX)
    {
        return 0;
    }

    *value = (unsigned int)parsed;
    return 1;
}

static int parse_double(const char *text, double *value)
{
    char *endptr = NULL;
    errno = 0;
    double parsed = strtod(text, &endptr);

    if (text[0] == '\0' || endptr == NULL || *endptr != '\0' || errno != 0)
    {
        return 0;
    }

    *value = parsed;
    return 1;
}

static enum parse_status parse_arguments(
    int argc,
    char **argv,
    struct lenia_config *config,
    const char **output_path,
    int myid
)
{
    *output_path = NULL;
    config->rows = DEFAULT_ROWS;
    config->cols = DEFAULT_COLS;
    config->steps = DEFAULT_STEPS;
    config->kernel_size = DEFAULT_KERNEL_SIZE;
    config->dt = DEFAULT_DT;
    config->gif_path = NULL;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
        {
            if (myid == 0)
            {
                print_help(argv[0]);
            }
            return PARSE_HELP;
        }
        else if (strcmp(argv[i], "--rows") == 0)
        {
            if (i + 1 >= argc || !parse_unsigned_int(argv[++i], &config->rows))
            {
                if (myid == 0)
                {
                    fprintf(stderr, "Invalid value for --rows.\n");
                }
                return PARSE_ERROR;
            }
        }
        else if (strcmp(argv[i], "--cols") == 0)
        {
            if (i + 1 >= argc || !parse_unsigned_int(argv[++i], &config->cols))
            {
                if (myid == 0)
                {
                    fprintf(stderr, "Invalid value for --cols.\n");
                }
                return PARSE_ERROR;
            }
        }
        else if (strcmp(argv[i], "--steps") == 0)
        {
            if (i + 1 >= argc || !parse_unsigned_int(argv[++i], &config->steps))
            {
                if (myid == 0)
                {
                    fprintf(stderr, "Invalid value for --steps.\n");
                }
                return PARSE_ERROR;
            }
        }
        else if (strcmp(argv[i], "--dt") == 0)
        {
            if (i + 1 >= argc || !parse_double(argv[++i], &config->dt))
            {
                if (myid == 0)
                {
                    fprintf(stderr, "Invalid value for --dt.\n");
                }
                return PARSE_ERROR;
            }
        }
        else if (strcmp(argv[i], "--kernel-size") == 0)
        {
            if (i + 1 >= argc || !parse_unsigned_int(argv[++i], &config->kernel_size))
            {
                if (myid == 0)
                {
                    fprintf(stderr, "Invalid value for --kernel-size.\n");
                }
                return PARSE_ERROR;
            }
        }
        else if (strcmp(argv[i], "--gif") == 0)
        {
            if (i + 1 >= argc)
            {
                if (myid == 0)
                {
                    fprintf(stderr, "Missing value for --gif.\n");
                }
                return PARSE_ERROR;
            }
            config->gif_path = argv[++i];
        }
        else if (strcmp(argv[i], "--output") == 0)
        {
            if (i + 1 >= argc)
            {
                if (myid == 0)
                {
                    fprintf(stderr, "Missing value for --output.\n");
                }
                return PARSE_ERROR;
            }
            *output_path = argv[++i];
        }
        else
        {
            if (myid == 0)
            {
                fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            }
            return PARSE_ERROR;
        }
    }

    if (config->rows == 0 || config->cols == 0)
    {
        if (myid == 0)
        {
            fprintf(stderr, "The world dimensions must be positive.\n");
        }
        return PARSE_ERROR;
    }

    if (config->kernel_size == 0 || config->kernel_size % 2 != 0)
    {
        if (myid == 0)
        {
            fprintf(stderr, "The kernel size must be a positive even number.\n");
        }
        return PARSE_ERROR;
    }

    if (config->rows < config->kernel_size || config->cols < config->kernel_size)
    {
        if (myid == 0)
        {
            fprintf(stderr, "The world dimensions must be at least as large as the kernel size.\n");
        }
        return PARSE_ERROR;
    }

    if (config->dt <= 0.0)
    {
        if (myid == 0)
        {
            fprintf(stderr, "The time step must be positive.\n");
        }
        return PARSE_ERROR;
    }

    return PARSE_OK;
}

int main(int argc, char **argv)
{
    int myid, procs;
    struct lenia_config config;
    const char *output_path = NULL;
    enum parse_status parse_status;
    double elapsed_time = 0.0;
    double *world = NULL;
    int exit_code = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    parse_status = parse_arguments(argc, argv, &config, &output_path, myid);
    if (parse_status == PARSE_HELP)
    {
        MPI_Finalize();
        return 0;
    }
    if (parse_status == PARSE_ERROR)
    {
        if (myid == 0)
        {
            fprintf(stderr, "Use --help to see the supported options.\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (procs > (int)config.rows)
    {
        if (myid == 0)
        {
            fprintf(stderr, "The number of MPI processes must not exceed the number of rows.\n");
        }
        MPI_Finalize();
        return 1;
    }

    struct orbium_coo orbiums[NUM_ORBIUMS] = {
        {0, (int)(config.cols / 3), 0},
        {(int)(config.rows / 3), 0, 180}
    };

    if (myid == 0)
    {
        printf("Lenia configuration: rows=%u cols=%u steps=%u dt=%.3f kernel=%u procs=%d\n",
               config.rows,
               config.cols,
               config.steps,
               config.dt,
               config.kernel_size,
               procs);
        if (config.gif_path != NULL)
        {
            printf("GIF output: %s\n", config.gif_path);
        }
        if (output_path != NULL)
        {
            printf("Final world output: %s\n", output_path);
        }
    }

    world = evolve_lenia(&config, orbiums, NUM_ORBIUMS, MPI_COMM_WORLD, &elapsed_time);

    if (myid == 0)
    {
        printf("Simulation time %u steps: %.6f seconds\n", config.steps, elapsed_time);
        printf("RESULT rows=%u cols=%u steps=%u procs=%d time=%.6f\n",
               config.rows,
               config.cols,
               config.steps,
               procs,
               elapsed_time);

        if (output_path != NULL && !write_world_to_file(output_path, world, config.rows, config.cols))
        {
            fprintf(stderr, "Failed to write the final world state to '%s'.\n", output_path);
            exit_code = 1;
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

    free(world);
    MPI_Finalize();
    return exit_code;
}
