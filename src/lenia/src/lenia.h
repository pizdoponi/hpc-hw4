#ifndef LENIA_H
#define LENIA_H

#include <mpi.h>

struct orbium_coo {
    int row;
    int col;
    int angle;
};

struct lenia_config {
    unsigned int rows;
    unsigned int cols;
    unsigned int steps;
    unsigned int kernel_size;
    double dt;
    const char *gif_path;
};

double *evolve_lenia(
    const struct lenia_config *config,
    const struct orbium_coo *orbiums,
    unsigned int num_orbiums,
    MPI_Comm comm,
    double *elapsed_time
);

int write_world_to_file(const char *path, const double *world, unsigned int rows, unsigned int cols);

#endif
