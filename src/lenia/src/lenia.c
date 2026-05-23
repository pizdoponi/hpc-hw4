#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gifenc.h"
#include "lenia.h"
#include "orbium.h"

struct halo_buffers {
    double *top_halo;
    double *bottom_halo;
    double *top_send;
    double *top_recv;
    double *bottom_send;
    double *bottom_recv;
};

static void fail_and_abort(MPI_Comm comm, int myid, const char *message)
{
    fprintf(stderr, "Rank %d: %s\n", myid, message);
    MPI_Abort(comm, EXIT_FAILURE);
}

static void *checked_calloc(MPI_Comm comm, int myid, size_t count, size_t size, const char *message)
{
    void *ptr = calloc(count, size);
    if (ptr == NULL)
    {
        fail_and_abort(comm, myid, message);
    }
    return ptr;
}

static inline double gauss(double x, double mu, double sigma)
{
    double t = (x - mu) / sigma;
    return exp(-0.5 * t * t);
}

static inline double growth_lenia(double u)
{
    const double mu = 0.15;
    const double sigma = 0.015;
    return -1.0 + 2.0 * gauss(u, mu, sigma);
}

static inline double clamp01(double value)
{
    if (value < 0.0)
    {
        return 0.0;
    }
    if (value > 1.0)
    {
        return 1.0;
    }
    return value;
}

static void build_row_distribution(unsigned int rows, int procs, int *counts_rows, int *displs_rows)
{
    unsigned int base_rows = rows / (unsigned int)procs;
    unsigned int remainder = rows % (unsigned int)procs;
    int displacement = 0;

    for (int rank = 0; rank < procs; rank++)
    {
        counts_rows[rank] = (int)base_rows + (rank < (int)remainder ? 1 : 0);
        displs_rows[rank] = displacement;
        displacement += counts_rows[rank];
    }
}

static int compute_halo_rounds(const int *counts_rows, int procs, unsigned int halo_rows)
{
    int max_rounds = 0;

    for (int rank = 0; rank < procs; rank++)
    {
        unsigned int covered = 0;
        int rounds = 0;
        int other = rank;

        while (covered < halo_rows)
        {
            other = (other + procs - 1) % procs;
            covered += (unsigned int)counts_rows[other];
            rounds++;
        }
        if (rounds > max_rounds)
        {
            max_rounds = rounds;
        }

        covered = 0;
        rounds = 0;
        other = rank;

        while (covered < halo_rows)
        {
            other = (other + 1) % procs;
            covered += (unsigned int)counts_rows[other];
            rounds++;
        }
        if (rounds > max_rounds)
        {
            max_rounds = rounds;
        }
    }

    return max_rounds;
}

static void generate_kernel(double *kernel, unsigned int size)
{
    const double mu = 0.5;
    const double sigma = 0.15;
    int radius = (int)size / 2;
    double sum = 0.0;

    for (int y = -radius; y < radius; y++)
    {
        for (int x = -radius; x < radius; x++)
        {
            double distance = sqrt((double)((1 + x) * (1 + x) + (1 + y) * (1 + y))) / radius;
            double value = gauss(distance, mu, sigma);

            if (distance > 1.0)
            {
                value = 0.0;
            }

            kernel[(size_t)(y + radius) * size + (x + radius)] = value;
            sum += value;
        }
    }

    for (unsigned int y = 0; y < size; y++)
    {
        for (unsigned int x = 0; x < size; x++)
        {
            kernel[(size_t)y * size + x] /= sum;
        }
    }
}

static double *initialize_world(
    unsigned int rows,
    unsigned int cols,
    const struct orbium_coo *orbiums,
    unsigned int num_orbiums
)
{
    double *world = (double *)calloc((size_t)rows * cols, sizeof(double));
    if (world == NULL)
    {
        return NULL;
    }

    for (unsigned int i = 0; i < num_orbiums; i++)
    {
        place_orbium(world, rows, cols, orbiums[i].row, orbiums[i].col, orbiums[i].angle);
    }

    return world;
}

static void copy_last_rows(
    double *destination,
    const double *source,
    unsigned int source_rows,
    unsigned int cols,
    unsigned int rows_to_copy
)
{
    size_t offset = (size_t)(source_rows - rows_to_copy) * cols;
    memcpy(destination, source + offset, (size_t)rows_to_copy * cols * sizeof(double));
}

static void copy_first_rows(
    double *destination,
    const double *source,
    unsigned int cols,
    unsigned int rows_to_copy
)
{
    memcpy(destination, source, (size_t)rows_to_copy * cols * sizeof(double));
}

static unsigned int compose_last_rows(
    double *destination,
    const double *prefix,
    unsigned int prefix_rows,
    const double *local_world,
    unsigned int local_rows,
    unsigned int cols,
    unsigned int max_rows
)
{
    unsigned int total_rows = prefix_rows + local_rows;
    unsigned int rows_to_copy = total_rows < max_rows ? total_rows : max_rows;
    unsigned int rows_from_local = local_rows < rows_to_copy ? local_rows : rows_to_copy;
    unsigned int rows_from_prefix = rows_to_copy - rows_from_local;

    if (rows_from_prefix > 0)
    {
        memcpy(
            destination,
            prefix + (size_t)(prefix_rows - rows_from_prefix) * cols,
            (size_t)rows_from_prefix * cols * sizeof(double)
        );
    }

    if (rows_from_local > 0)
    {
        memcpy(
            destination + (size_t)rows_from_prefix * cols,
            local_world + (size_t)(local_rows - rows_from_local) * cols,
            (size_t)rows_from_local * cols * sizeof(double)
        );
    }

    return rows_to_copy;
}

static unsigned int compose_first_rows(
    double *destination,
    const double *local_world,
    unsigned int local_rows,
    const double *suffix,
    unsigned int suffix_rows,
    unsigned int cols,
    unsigned int max_rows
)
{
    unsigned int total_rows = local_rows + suffix_rows;
    unsigned int rows_to_copy = total_rows < max_rows ? total_rows : max_rows;
    unsigned int rows_from_local = local_rows < rows_to_copy ? local_rows : rows_to_copy;
    unsigned int rows_from_suffix = rows_to_copy - rows_from_local;

    if (rows_from_local > 0)
    {
        memcpy(destination, local_world, (size_t)rows_from_local * cols * sizeof(double));
    }

    if (rows_from_suffix > 0)
    {
        memcpy(
            destination + (size_t)rows_from_local * cols,
            suffix,
            (size_t)rows_from_suffix * cols * sizeof(double)
        );
    }

    return rows_to_copy;
}

static void exchange_halos(
    const double *local_world,
    unsigned int local_rows,
    unsigned int cols,
    unsigned int halo_rows,
    int myid,
    int procs,
    int halo_rounds,
    MPI_Comm comm,
    struct halo_buffers *buffers
)
{
    if (procs == 1)
    {
        copy_last_rows(buffers->top_halo, local_world, local_rows, cols, halo_rows);
        copy_first_rows(buffers->bottom_halo, local_world, cols, halo_rows);
        return;
    }

    unsigned int top_send_rows = local_rows < halo_rows ? local_rows : halo_rows;
    unsigned int bottom_send_rows = local_rows < halo_rows ? local_rows : halo_rows;

    copy_last_rows(buffers->top_send, local_world, local_rows, cols, top_send_rows);
    copy_first_rows(buffers->bottom_send, local_world, cols, bottom_send_rows);

    int rank_up = (myid + procs - 1) % procs;
    int rank_down = (myid + 1) % procs;

    for (int round = 0; round < halo_rounds; round++)
    {
        unsigned int top_recv_rows = 0;
        unsigned int bottom_recv_rows = 0;
        unsigned int next_top_send_rows;
        unsigned int next_bottom_send_rows;

        MPI_Sendrecv(
            &top_send_rows, 1, MPI_UNSIGNED, rank_down, 100,
            &top_recv_rows, 1, MPI_UNSIGNED, rank_up, 100,
            comm, MPI_STATUS_IGNORE
        );
        MPI_Sendrecv(
            buffers->top_send, (int)((size_t)top_send_rows * cols), MPI_DOUBLE, rank_down, 101,
            buffers->top_recv, (int)((size_t)top_recv_rows * cols), MPI_DOUBLE, rank_up, 101,
            comm, MPI_STATUS_IGNORE
        );

        MPI_Sendrecv(
            &bottom_send_rows, 1, MPI_UNSIGNED, rank_up, 200,
            &bottom_recv_rows, 1, MPI_UNSIGNED, rank_down, 200,
            comm, MPI_STATUS_IGNORE
        );
        MPI_Sendrecv(
            buffers->bottom_send, (int)((size_t)bottom_send_rows * cols), MPI_DOUBLE, rank_up, 201,
            buffers->bottom_recv, (int)((size_t)bottom_recv_rows * cols), MPI_DOUBLE, rank_down, 201,
            comm, MPI_STATUS_IGNORE
        );

        memcpy(buffers->top_halo, buffers->top_recv, (size_t)top_recv_rows * cols * sizeof(double));
        memcpy(buffers->bottom_halo, buffers->bottom_recv, (size_t)bottom_recv_rows * cols * sizeof(double));

        next_top_send_rows = compose_last_rows(
            buffers->top_send,
            buffers->top_recv,
            top_recv_rows,
            local_world,
            local_rows,
            cols,
            halo_rows
        );
        next_bottom_send_rows = compose_first_rows(
            buffers->bottom_send,
            local_world,
            local_rows,
            buffers->bottom_recv,
            bottom_recv_rows,
            cols,
            halo_rows
        );

        top_send_rows = next_top_send_rows;
        bottom_send_rows = next_bottom_send_rows;
    }
}

static void build_world_with_halo(
    double *world_with_halo,
    const struct halo_buffers *buffers,
    const double *local_world,
    unsigned int local_rows,
    unsigned int cols,
    unsigned int halo_rows
)
{
    memcpy(world_with_halo, buffers->top_halo, (size_t)halo_rows * cols * sizeof(double));
    memcpy(world_with_halo + (size_t)halo_rows * cols, local_world, (size_t)local_rows * cols * sizeof(double));
    memcpy(
        world_with_halo + (size_t)(halo_rows + local_rows) * cols,
        buffers->bottom_halo,
        (size_t)halo_rows * cols * sizeof(double)
    );
}

static void evolve_local_rows(
    const double *world_with_halo,
    const double *local_world,
    double *next_world,
    const double *kernel,
    unsigned int local_rows,
    unsigned int cols,
    unsigned int kernel_size,
    double dt
)
{
    int halo = (int)kernel_size / 2;
    int ncols = (int)cols;

    for (unsigned int i = 0; i < local_rows; i++)
    {
        for (unsigned int j = 0; j < cols; j++)
        {
            double convolution = 0.0;

            for (int ki = (int)kernel_size - 1, kri = 0; ki >= 0; ki--, kri++)
            {
                const double *kernel_row = kernel + (size_t)ki * kernel_size;
                const double *input_row = world_with_halo + (size_t)(i + kri) * cols;

                for (int kj = (int)kernel_size - 1, kcj = 0; kj >= 0; kj--, kcj++)
                {
                    int col = (int)j - halo + kcj;

                    if (col < 0)
                    {
                        col += ncols;
                    }
                    else if (col >= ncols)
                    {
                        col -= ncols;
                    }

                    convolution += kernel_row[kj] * input_row[col];
                }
            }

            next_world[(size_t)i * cols + j] = clamp01(local_world[(size_t)i * cols + j] + dt * growth_lenia(convolution));
        }
    }
}

static void store_gif_frame(ge_GIF *gif, const double *world, unsigned int rows, unsigned int cols)
{
    size_t total_cells = (size_t)rows * cols;

    for (size_t i = 0; i < total_cells; i++)
    {
        gif->frame[i] = (uint8_t)(world[i] * 255.0);
    }

    ge_add_frame(gif, 5);
}

double *evolve_lenia(
    const struct lenia_config *config,
    const struct orbium_coo *orbiums,
    unsigned int num_orbiums,
    MPI_Comm comm,
    double *elapsed_time
)
{
    int myid, procs;
    int *counts_rows;
    int *displs_rows;
    int *counts_elems;
    int *displs_elems;
    unsigned int local_rows;
    unsigned int halo_rows;
    int halo_rounds;
    double *kernel;
    double *global_world = NULL;
    double *local_world;
    double *next_world;
    double *world_with_halo;
    struct halo_buffers halo;
    ge_GIF *gif = NULL;

    MPI_Comm_rank(comm, &myid);
    MPI_Comm_size(comm, &procs);

    halo_rows = config->kernel_size / 2;

    counts_rows = (int *)checked_calloc(comm, myid, (size_t)procs, sizeof(int), "Failed to allocate row counts.");
    displs_rows = (int *)checked_calloc(comm, myid, (size_t)procs, sizeof(int), "Failed to allocate row displacements.");
    counts_elems = (int *)checked_calloc(comm, myid, (size_t)procs, sizeof(int), "Failed to allocate element counts.");
    displs_elems = (int *)checked_calloc(comm, myid, (size_t)procs, sizeof(int), "Failed to allocate element displacements.");

    build_row_distribution(config->rows, procs, counts_rows, displs_rows);
    for (int rank = 0; rank < procs; rank++)
    {
        counts_elems[rank] = counts_rows[rank] * (int)config->cols;
        displs_elems[rank] = displs_rows[rank] * (int)config->cols;
    }

    local_rows = (unsigned int)counts_rows[myid];
    halo_rounds = compute_halo_rounds(counts_rows, procs, halo_rows);

    kernel = (double *)checked_calloc(
        comm,
        myid,
        (size_t)config->kernel_size * config->kernel_size,
        sizeof(double),
        "Failed to allocate the convolution kernel."
    );
    local_world = (double *)checked_calloc(
        comm,
        myid,
        (size_t)local_rows * config->cols,
        sizeof(double),
        "Failed to allocate the local world."
    );
    next_world = (double *)checked_calloc(
        comm,
        myid,
        (size_t)local_rows * config->cols,
        sizeof(double),
        "Failed to allocate the next local world."
    );
    world_with_halo = (double *)checked_calloc(
        comm,
        myid,
        (size_t)(local_rows + 2 * halo_rows) * config->cols,
        sizeof(double),
        "Failed to allocate the local world with halos."
    );

    halo.top_halo = (double *)checked_calloc(
        comm,
        myid,
        (size_t)halo_rows * config->cols,
        sizeof(double),
        "Failed to allocate the top halo buffer."
    );
    halo.bottom_halo = (double *)checked_calloc(
        comm,
        myid,
        (size_t)halo_rows * config->cols,
        sizeof(double),
        "Failed to allocate the bottom halo buffer."
    );
    halo.top_send = (double *)checked_calloc(
        comm,
        myid,
        (size_t)halo_rows * config->cols,
        sizeof(double),
        "Failed to allocate the top send buffer."
    );
    halo.top_recv = (double *)checked_calloc(
        comm,
        myid,
        (size_t)halo_rows * config->cols,
        sizeof(double),
        "Failed to allocate the top receive buffer."
    );
    halo.bottom_send = (double *)checked_calloc(
        comm,
        myid,
        (size_t)halo_rows * config->cols,
        sizeof(double),
        "Failed to allocate the bottom send buffer."
    );
    halo.bottom_recv = (double *)checked_calloc(
        comm,
        myid,
        (size_t)halo_rows * config->cols,
        sizeof(double),
        "Failed to allocate the bottom receive buffer."
    );

    generate_kernel(kernel, config->kernel_size);

    if (myid == 0)
    {
        global_world = initialize_world(config->rows, config->cols, orbiums, num_orbiums);
        if (global_world == NULL)
        {
            fail_and_abort(comm, myid, "Failed to allocate the global world.");
        }

        if (config->gif_path != NULL)
        {
            gif = ge_new_gif(
                config->gif_path,
                (uint16_t)config->cols,
                (uint16_t)config->rows,
                inferno_pallete,
                8,
                -1,
                0
            );

            if (gif == NULL)
            {
                fail_and_abort(comm, myid, "Failed to create the GIF output file.");
            }
        }
    }

    MPI_Scatterv(
        myid == 0 ? global_world : local_world,
        counts_elems,
        displs_elems,
        MPI_DOUBLE,
        local_world,
        counts_elems[myid],
        MPI_DOUBLE,
        0,
        comm
    );

    MPI_Barrier(comm);
    double start = MPI_Wtime();

    for (unsigned int step = 0; step < config->steps; step++)
    {
        exchange_halos(
            local_world,
            local_rows,
            config->cols,
            halo_rows,
            myid,
            procs,
            halo_rounds,
            comm,
            &halo
        );

        build_world_with_halo(world_with_halo, &halo, local_world, local_rows, config->cols, halo_rows);
        evolve_local_rows(
            world_with_halo,
            local_world,
            next_world,
            kernel,
            local_rows,
            config->cols,
            config->kernel_size,
            config->dt
        );

        double *tmp = local_world;
        local_world = next_world;
        next_world = tmp;

        if (config->gif_path != NULL)
        {
            MPI_Gatherv(
                local_world,
                counts_elems[myid],
                MPI_DOUBLE,
                myid == 0 ? global_world : local_world,
                counts_elems,
                displs_elems,
                MPI_DOUBLE,
                0,
                comm
            );

            if (myid == 0)
            {
                store_gif_frame(gif, global_world, config->rows, config->cols);
            }
        }
    }

    double local_elapsed = MPI_Wtime() - start;
    MPI_Reduce(&local_elapsed, elapsed_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (config->gif_path == NULL)
    {
        MPI_Gatherv(
            local_world,
            counts_elems[myid],
            MPI_DOUBLE,
            myid == 0 ? global_world : local_world,
            counts_elems,
            displs_elems,
            MPI_DOUBLE,
            0,
            comm
        );
    }

    if (myid == 0 && gif != NULL)
    {
        ge_close_gif(gif);
    }

    free(counts_rows);
    free(displs_rows);
    free(counts_elems);
    free(displs_elems);
    free(kernel);
    free(local_world);
    free(next_world);
    free(world_with_halo);
    free(halo.top_halo);
    free(halo.bottom_halo);
    free(halo.top_send);
    free(halo.top_recv);
    free(halo.bottom_send);
    free(halo.bottom_recv);

    return global_world;
}

int write_world_to_file(const char *path, const double *world, unsigned int rows, unsigned int cols)
{
    FILE *file = fopen(path, "w");
    if (file == NULL)
    {
        return 0;
    }

    for (unsigned int i = 0; i < rows; i++)
    {
        for (unsigned int j = 0; j < cols; j++)
        {
            if (fprintf(file, j + 1 == cols ? "%.12f" : "%.12f ", world[(size_t)i * cols + j]) < 0)
            {
                fclose(file);
                return 0;
            }
        }

        if (fputc('\n', file) == EOF)
        {
            fclose(file);
            return 0;
        }
    }

    if (fclose(file) != 0)
    {
        return 0;
    }

    return 1;
}
