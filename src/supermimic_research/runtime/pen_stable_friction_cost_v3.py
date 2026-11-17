"""FP64 same-regime costs; return the original vector directly on fallback."""
import warp as wp


@wp.func
def with_stable_cost(point:wp.vec3,jaref:float,jv:float,d:float,loss:float,alpha:float)->wp.vec3:
    j=wp.float64(jaref); direction=wp.float64(jv); diagonal=wp.float64(d); f=wp.float64(loss)
    advance=wp.float64(alpha)*direction; x=j+advance
    if d>0.0 and loss>=0.0:
        radius=f/diagonal
        if -radius<j and j<radius and -radius<x and x<radius:
            delta=advance*(diagonal*j+wp.float64(0.5)*diagonal*advance)
            return wp.vec3(float(delta),point[1],point[2])
        elif j<=-radius and x<=-radius:
            return wp.vec3(float(-f*advance),point[1],point[2])
        elif j>=radius and x>=radius:
            return wp.vec3(float(f*advance),point[1],point[2])
    return point
