#include "utils_1d.h"
#include <limits.h>

int heat_eqn_solver_1d(double *heat_array, double *err, size_t num_pts) {
  int rank, ranks;
  if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS ||
      MPI_Comm_size(MPI_COMM_WORLD, &ranks) != MPI_SUCCESS ||
      num_pts > INT_MAX) {
    return -1;
  }

  const int base = (int)num_pts / ranks;
  const int extra = (int)num_pts % ranks;
  const int count = base + (rank < extra);
  const int offset = rank * base + (rank < extra ? rank : extra);
  const int left = rank > 0 ? rank - 1 : MPI_PROC_NULL;
  const int right = offset + count < (int)num_pts ? rank + 1 : MPI_PROC_NULL;

  double *block = heat_array + offset;
  double *scratch = malloc(((size_t)count + 2) * sizeof(*scratch));
  int *counts = NULL, *offsets = NULL;

  if (rank == 0) {
    counts = malloc((size_t)ranks * sizeof(*counts));
    offsets = malloc((size_t)ranks * sizeof(*offsets));
  }

  int result = -1;
  int ready = scratch != NULL && (rank != 0 || (counts && offsets));
  int all_ready;

  if (MPI_Allreduce(&ready, &all_ready, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD) !=
          MPI_SUCCESS ||
      !all_ready) {
    goto done;
  }

  const double dx = 1.0 / (num_pts + 1);
  const double dt = 0.5 * dx * dx;
  const double coefficient = dt / (dx * dx);
  double local_error = 0.0;
  double *current = block;
  double *next = scratch;

  if (count > 0) {
    // MPI_PROC_NULL leaves the physical endpoints fixed in both buffers
    scratch[0] = block[0];
    scratch[(size_t)count + 1] = block[(size_t)count + 1];

    for (size_t step = 0; step < ITER; ++step) {
      if (MPI_Sendrecv(current + 1, 1, MPI_DOUBLE, left, 0, current + count + 1,
                       1, MPI_DOUBLE, right, 0, MPI_COMM_WORLD,
                       MPI_STATUS_IGNORE) != MPI_SUCCESS ||
          MPI_Sendrecv(current + count, 1, MPI_DOUBLE, right, 1, current, 1,
                       MPI_DOUBLE, left, 1, MPI_COMM_WORLD,
                       MPI_STATUS_IGNORE) != MPI_SUCCESS) {
        goto done;
      }

      for (size_t j = 1; j <= (size_t)count; ++j) {
        next[j] =
            current[j] +
            coefficient * (current[j - 1] - 2.0 * current[j] + current[j + 1]);

        local_error += dx * fabs(next[j] - current[j]);
      }

      double *swap = current;
      current = next;
      next = swap;
    }
  }

  if (rank == 0) {
    for (int r = 0; r < ranks; ++r) {
      counts[r] = base + (r < extra);
      offsets[r] = r * base + (r < extra ? r : extra);
    }

    if (current != block) {
      memcpy(block + 1, current + 1, (size_t)count * sizeof(*block));
    }
  }

  double total_error;

  if (MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : current + 1, count, MPI_DOUBLE,
                  heat_array + 1, counts, offsets, MPI_DOUBLE, 0,
                  MPI_COMM_WORLD) != MPI_SUCCESS ||
      MPI_Reduce(&local_error, &total_error, 1, MPI_DOUBLE, MPI_SUM, 0,
                 MPI_COMM_WORLD) != MPI_SUCCESS) {
    goto done;
  }

  if (rank == 0) {
    *err += total_error;
  }

  result = 0;

done:
  free(scratch);
  free(counts);
  free(offsets);
  return result;
}
