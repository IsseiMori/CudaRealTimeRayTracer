#ifndef AABBH
#define AABBH

#include <math.h>
#include <stdlib.h>
#include <iostream>

#include "vec3.h"
#include "ray.h"

class aabb {
public:
	__device__ aabb() {}
	__device__ aabb(const vec3& a, const vec3& b) { _min = a, _max = b; }

	__device__  vec3 min() const { return _min; }
	__device__  vec3 max() const { return _max; }
	

	// Return true if ray hits this bb within tmax and tmin distance
	__device__  inline bool hit(const ray& r, float tmin, float tmax) const {
		for (int a = 0; a < 3; a++) {
			auto t0 = fmin(_min[a] - r.origin()[a] / r.direction()[a],
						  (_max[a] - r.origin()[a] / r.direction()[a]));
			auto t1 = fmax(_min[a] - r.origin()[a] / r.direction()[a],
						  (_max[a] - r.origin()[a] / r.direction()[a]));
			tmin = fmax(t0, tmin);
			tmax = fmin(t1, tmax);
			if (tmax <= tmin)
				return false;
		}
		return true;
	}


	__device__ int max_extent() {
		float x = _max.x() - _min.x();
		float y = _max.y() - _min.y();
		float z = _max.z() - _min.z();
	
		if (x >= y && x >= z) return 0;
		if (y >= x && y >= z) return 1;
		else return 2;
	}

	__device__ vec3 center() {
		return 0.5f * _min + 0.5f * _max;
	}

	__device__ void copy(const aabb box) {
		_min = box._min;
		_max = box._max;
	}

	__device__ vec3 offset(const vec3& p) const {
		vec3 o = p - _min;
		if (_max[0] > _min[0]) o[0] /= _max[0] - _min[0];
		if (_max[1] > _min[1]) o[1] /= _max[1] - _min[1];
		if (_max[2] > _min[2]) o[2] /= _max[2] - _min[2];
		return o;
	}


	vec3 _min;
	vec3 _max;
};

__device__ aabb surrounding_box(aabb box0, aabb box1) {
	vec3 small(fmin(box0.min().x(), box1.min().x()),
			   fmin(box0.min().y(), box1.min().y()),
			   fmin(box0.min().z(), box1.min().z()));

	vec3 big(fmax(box0.max().x(), box1.max().x()),
			 fmax(box0.max().y(), box1.max().y()),
			 fmax(box0.max().z(), box1.max().z()));

	return aabb(small, big);
}


#endif