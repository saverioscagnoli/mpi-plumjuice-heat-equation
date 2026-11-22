#include "utils_1d.h"
#include <limits.h>

static inline double temperature(const double *current, ptrdiff_t j,
                                 double coefficient) {
  return current[j] +
         coefficient * (current[j - 1] - 2.0 * current[j] + current[j + 1]);
}

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

  // Each step consumes one halo point. Keep the exchange within one neighbor
  const int batch = base == 0 ? 1 : (base < 32 ? base : 32);
  const int left_halo = left == MPI_PROC_NULL ? 1 : batch;
  const int right_halo = right == MPI_PROC_NULL ? 1 : batch;
  double *block = heat_array + offset;
  double *scratch = malloc(((size_t)left_halo + count + right_halo) *
                           sizeof(*scratch));
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
  if (count > 0) {
    // Index 1 is the first owned point in both buffers.
    double *next = scratch + left_halo - 1;
    next[0] = block[0];
    next[(size_t)count + 1] = block[(size_t)count + 1];

    for (int step = 0; step < ITER; step += batch) {
      const int steps = ITER - step < batch ? ITER - step : batch;
      // The input already supplies both halos for the first batch
      if (step > 0 && (left != MPI_PROC_NULL || right != MPI_PROC_NULL)) {
        // Use zero counts and valid buffer addresses at the physical ends
        const int to_left = left == MPI_PROC_NULL ? 0 : steps;
        const int to_right = right == MPI_PROC_NULL ? 0 : steps;
        if (MPI_Sendrecv(current + 1, to_left, MPI_DOUBLE, left, 0,
                         current + count + 1, to_right, MPI_DOUBLE, right, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE) != MPI_SUCCESS ||
            MPI_Sendrecv(current + count + 1 - to_right, to_right, MPI_DOUBLE,
                         right, 1, current + 1 - to_left, to_left, MPI_DOUBLE,
                         left, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
          goto done;
        }
      }

      for (int remaining = steps - 1; remaining >= 0; --remaining) {
        // Shrink the halo each step. Only owned points contribute to the error
        if (left != MPI_PROC_NULL) {
          for (int j = 1 - remaining; j <= 0; ++j) {
            next[j] = temperature(current, j, coefficient);
          }
        }
        double step_error = 0.0;
#pragma omp simd reduction(+:step_error)
        for (size_t j = 1; j <= (size_t)count; ++j) {
          next[j] = temperature(current, j, coefficient);
          step_error += dx * fabs(next[j] - current[j]);
        }
        local_error += step_error;
        if (right != MPI_PROC_NULL) {
          for (ptrdiff_t j = (ptrdiff_t)count + 1; j <= (ptrdiff_t)count + remaining; ++j) {
            next[j] = temperature(current, j, coefficient);
          }
        }
        double *swap = current;
        current = next;
        next = swap;
      }
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
