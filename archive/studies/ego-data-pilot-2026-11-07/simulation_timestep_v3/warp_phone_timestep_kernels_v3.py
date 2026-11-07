"""GPU trace kernels, imported only by an explicit timestep-v3 Warp bench --run."""
import warp as wp


@wp.kernel
def record_post_state(
    qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
    time_s: wp.array(dtype=float), xfrc: wp.array2d(dtype=wp.spatial_vector),
    overflow: wp.array(dtype=int), nefc: wp.array(dtype=int),
    cursor: wp.array(dtype=int), phone_body: int,
    out_qpos: wp.array3d(dtype=float), out_qvel: wp.array3d(dtype=float),
    out_time: wp.array2d(dtype=float), out_force: wp.array2d(dtype=float),
    out_overflow: wp.array2d(dtype=int), out_nefc: wp.array2d(dtype=int),
):
    world = wp.tid()
    row = cursor[0] + 1
    for j in range(7):
        out_qpos[row, world, j] = qpos[world, j]
    for j in range(6):
        out_qvel[row, world, j] = qvel[world, j]
    out_time[row, world] = time_s[world]
    out_force[row, world] = xfrc[world, phone_body][0]
    out_overflow[row, world] = overflow[world]
    out_nefc[row, world] = nefc[world]


@wp.kernel
def record_pre_contacts(
    nacon: wp.array(dtype=int), dist: wp.array(dtype=float),
    pos: wp.array(dtype=wp.vec3), frame: wp.array(dtype=wp.mat33),
    geom: wp.array(dtype=wp.vec2i), worldid: wp.array(dtype=int),
    contact_type: wp.array(dtype=int), dim: wp.array(dtype=int),
    cursor: wp.array(dtype=int), capacity: int,
    out_count: wp.array(dtype=int), out_dist: wp.array2d(dtype=float),
    out_pos: wp.array2d(dtype=wp.vec3), out_frame: wp.array2d(dtype=wp.mat33),
    out_geom: wp.array2d(dtype=wp.vec2i), out_worldid: wp.array2d(dtype=int),
    out_type: wp.array2d(dtype=int), out_dim: wp.array2d(dtype=int),
):
    slot = wp.tid()
    row = cursor[0]
    if slot == 0:
        out_count[row] = nacon[0]
    if slot < nacon[0] and slot < capacity:
        out_dist[row, slot] = dist[slot]
        out_pos[row, slot] = pos[slot]
        out_frame[row, slot] = frame[slot]
        out_geom[row, slot] = geom[slot]
        out_worldid[row, slot] = worldid[slot]
        out_type[row, slot] = contact_type[slot]
        out_dim[row, slot] = dim[slot]


@wp.kernel
def advance_cursor(cursor: wp.array(dtype=int)):
    cursor[0] = cursor[0] + 1


@wp.kernel
def record_pre_diagnostics(
    qacc: wp.array2d(dtype=float), nefc: wp.array(dtype=int),
    solver_niter: wp.array(dtype=int), ncollision: wp.array(dtype=int),
    energy: wp.array(dtype=wp.vec2),
    cursor: wp.array(dtype=int),
    out_qacc: wp.array3d(dtype=float), out_nefc: wp.array2d(dtype=int),
    out_solver_niter: wp.array2d(dtype=int), out_ncollision: wp.array(dtype=int),
    out_energy: wp.array2d(dtype=wp.vec2),
):
    world = wp.tid()
    row = cursor[0]
    for j in range(6):
        out_qacc[row, world, j] = qacc[world, j]
    out_nefc[row, world] = nefc[world]
    out_solver_niter[row, world] = solver_niter[world]
    out_energy[row, world] = energy[world]
    if world == 0:
        out_ncollision[row] = ncollision[0]
