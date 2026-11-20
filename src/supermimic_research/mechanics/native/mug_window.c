/* Native stepping and measurement for the 20-servo rigid-mug experiment. */
#include <math.h>
#include <stdint.h>
#include <mujoco/mujoco.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

typedef void (*Step)(const mjModel*, mjData*);
typedef void (*Force)(const mjModel*, const mjData*, int, mjtNum*);

EXPORT int mug_header_version(void) { return mjVERSION_HEADER; }
EXPORT int mug_scalar_bytes(void) { return (int)sizeof(mjtNum); }

EXPORT int mug_window(const mjModel* m, mjData* d, Step step, Force force,
                      const uint8_t* hand, const uint8_t* mug, int table,
                      int ticks, double* peaks) {
  if (m->nu != 20 || table < 0 || table >= m->ngeom || ticks <= 0) return 1;
  for (int j = 0; j < 4; ++j) peaks[j] = 0.0;
  for (int tick = 0; tick < ticks; ++tick) {
    step(m, d);
    for (int k = 0; k < d->ncon; ++k) {
      const mjContact* contact = d->contact + k;
      int a = contact->geom[0], b = contact->geom[1];
      if (a < 0 || b < 0 || a >= m->ngeom || b >= m->ngeom) return 2;
      double depth = -1000.0 * contact->dist;
      if (!isfinite(depth)) return 3;
      if ((hand[a] && mug[b]) || (hand[b] && mug[a])) {
        if (depth > peaks[1]) peaks[1] = depth;
      }
      if (hand[a] && hand[b] && depth > peaks[2]) peaks[2] = depth;
      if ((a == table && mug[b]) || (b == table && mug[a])) {
        mjtNum local[6] = {0};
        force(m, d, k, local);
        if (!isfinite(local[0])) return 3;
        if (local[0] > peaks[0]) peaks[0] = local[0];
      }
    }
    for (int k = 0; k < m->nu; ++k) {
      int joint = m->actuator_trnid[2*k];
      if (joint < 0 || joint >= m->njnt || !m->jnt_limited[joint] ||
          m->jnt_type[joint] != mjJNT_HINGE) return 4;
      double q = d->qpos[m->jnt_qposadr[joint]];
      if (!isfinite(q)) return 3;
      double lower = m->jnt_range[2*joint] - q;
      double upper = q - m->jnt_range[2*joint+1];
      if (lower > peaks[3]) peaks[3] = lower;
      if (upper > peaks[3]) peaks[3] = upper;
    }
  }
  return 0;
}
