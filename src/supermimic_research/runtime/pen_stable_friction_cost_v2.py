"""Independent FP64 cost arithmetic; retain the supplied original FP32 slope/H."""
import warp as wp


@wp.func
def with_stable_cost(point:wp.vec3,jaref:float,jv:float,d:float,loss:float,alpha:float)->wp.vec3:
    # Do not share an FP32 alpha*jv temporary with the original derivative path.
    j=wp.float64(jaref); direction=wp.float64(jv); diagonal=wp.float64(d); f=wp.float64(loss)
    advance=wp.float64(alpha)*direction; x=j+advance; delta=wp.float64(point[0])
    if d>0.0 and loss>=0.0:
        radius=f/diagonal
        if -radius<j and j<radius and -radius<x and x<radius:
            delta=advance*(diagonal*j+wp.float64(0.5)*diagonal*advance)
        elif j<=-radius and x<=-radius:
            delta=-f*advance
        elif j>=radius and x>=radius:
            delta=f*advance
    return wp.vec3(float(delta),point[1],point[2])
