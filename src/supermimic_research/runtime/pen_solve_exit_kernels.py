"""Observer-owned launch counter; never writes simulator Model/Data/context."""
import warp as wp


@wp.kernel
def count_solve_exit(counter: wp.array[int]):
    counter[0] = counter[0] + 1
