#include <stdio.h>
#include <iostream>
#include <float.h>
#include <time.h>
#include <chrono>
#include <ctime>

// OpenGL
#include <gl/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// CUDA
#include "cuda_runtime.h"
#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"
#include <curand_kernel.h>

// CUDA Helper
#include "libs/helper_cuda.h"
#include "libs/helper_cuda_gl.h"

#include "shader_tools/GLSLProgram.h"
#include "shader_tools/GLSLShader.h"

// My Objects
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "camera.h"
#include "hitable_list.h"
#include "material.h"
#include "aabb.h"
#include "bvh.h"
#include "opengl_utils.h"
#include "global.h"


__device__ vec3 color(const ray& r, hitable **world, curandState *local_rand_state) {
	
	ray cur_ray = r;
	vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);

	
	for (int i = 0; i < 50; i++) {
		hit_record rec;
		if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
			ray scattered;
			vec3 attenuation;
			if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
				cur_attenuation *= attenuation;
				cur_ray = scattered;
			}
			else {
				return vec3(0.0, 0.0, 0.0);
			}
		}
		else {
			vec3 unit_direction = unit_vector(cur_ray.direction());
			float t = 0.5f*(unit_direction.y() + 1.0f);
			vec3 c = (1.0f - t)*vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
			return cur_attenuation * c;
		}
	}
	// reached max recursion
	return vec3(0.0, 0.0, 0.0);

}

__global__ void rand_init(curandState *rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curand_init(1004, 0, 0, rand_state);
	}
}

/* Initialize rand function so that each thread will have guaranteed distinct random numbrs*/
__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam, hitable **world, curandState *rand_state, inputPointers pointers, int frame) {
	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;

	int pixel_index = j * max_x + i;
	curandState local_rand_state = rand_state[pixel_index];
	vec3 col(0, 0, 0);

	float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
	float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
	ray r = (*cam)->get_ray(u, v, &local_rand_state);
	col += color(r, world, &local_rand_state);

	rand_state[pixel_index] = local_rand_state;

	int firstPos = (max_x * j + i) * 4;
	col /= float(ns);
	// pointers.image1[firstPos] = sqrt(col[0])*255;
	// pointers.image1[firstPos+1] = sqrt(col[1])*255;
	// pointers.image1[firstPos+2] = sqrt(col[2])*255;
	if (frame == 0) {
		pointers.acc_img[firstPos] = sqrt(col[0]) * 255;
		pointers.acc_img[firstPos + 1] = sqrt(col[1]) * 255;
		pointers.acc_img[firstPos + 2] = sqrt(col[2]) * 255;
	}
	else {
		pointers.acc_img[firstPos + 0] = pointers.acc_img[firstPos + 0] * (float)(frame - 1) / (float)(frame) + sqrt(col[0]) * 255 / (float(frame));
		pointers.acc_img[firstPos + 1] = pointers.acc_img[firstPos + 1] * (float)(frame - 1) / (float)(frame) + sqrt(col[1]) * 255 / (float(frame));
		pointers.acc_img[firstPos + 2] = pointers.acc_img[firstPos + 2] * (float)(frame - 1) / (float)(frame) + sqrt(col[2]) * 255 / (float(frame));
	}

	pointers.image1[firstPos + 0] = (int)(pointers.acc_img[firstPos + 0]);
	pointers.image1[firstPos + 1] = (int)(pointers.acc_img[firstPos + 1]);
	pointers.image1[firstPos + 2] = (int)(pointers.acc_img[firstPos + 2]);
		
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera, int nx, int ny, curandState *rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curandState local_rand_state = *rand_state;

		d_list[0] = new sphere(vec3(0, -1000.0, -1), 1000,
			new lambertian(vec3(0.5, 0.5, 0.5)));
		
		int i = 1;
		/*for (int a = -2; a < 2; a++) {
			for (int b = -2; b < 2; b++) {
				float choose_mat = RND;
				vec3 center(a + RND, 0.2, b + RND);
				if (choose_mat < 0.8f) {
					d_list[i++] = new sphere(center, 0.2,
						new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
				}
				else if (choose_mat < 0.95f) {
					d_list[i++] = new sphere(center, 0.2,
						new metal(vec3(0.5f*(1.0f + RND), 0.5f*(1.0f + RND), 0.5f*(1.0f + RND)), 0.5f*RND));
				}
				else {
					d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
				}
			}
		}*/
		d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
		d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
		d_list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
		*rand_state = local_rand_state;
		*d_world = new hitable_list(d_list, 1 + 3);

		vec3 lookfrom(13, 2, 3);
		vec3 lookat(0, 0, 0);
		float dist_to_focus = 10.0; (lookfrom - lookat).length();
		float aperture = 0.1;
		*d_camera = new camera(lookfrom,
			lookat,
			vec3(0, 1, 0),
			30.0,
			float(nx) / float(ny),
			aperture,
			dist_to_focus);
	}
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {
	for (int i = 0; i < 1+3; i++) {
		delete ((sphere *)d_list[i])->mat_ptr;
		delete d_list[i];
	}
	delete *d_world;
	delete *d_camera;
}

__global__ void create_bvh(hitable** list, int list_size, bvh_info** bvh_info_list, morton_info** morton_info_list) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		
		for (int i = 0; i < list_size; ++i) {
			bvh_info_list[i] = new bvh_info{ i, aabb(), vec3(0.0f, 0.0f, 0.0f) };
			list[i]->bounding_box(0, 0, bvh_info_list[i]->box);
			bvh_info_list[i]->centroid = bvh_info_list[i]->box.center();
		}
	
	}
}

__global__ void free_bvh(hitable** list, int list_size, bvh_info** bvh_info_list, morton_info** morton_info_list) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {

		for (int i = 0; i < list_size; ++i) {
			delete bvh_info_list[i];
		}
	}
}



void generateCUDAImage(vec3 *fb, int max_x, int max_y, int ns, 
					   camera **d_camera, hitable **d_world, curandState *d_rand_state, inputPointers pointers, 
					   std::chrono::duration<double> totalTime, std::chrono::duration<double> deltaTime, int frame) {

	dim3 blocks(WIDTH, HEIGHT);
	dim3 threads(ns);

	render << <blocks, threads >> > (fb, WIDTH, HEIGHT, ns, d_camera, d_world, d_rand_state, pointers, frame);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	cudaArray* texture_ptr;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_resource, 0));
	checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_resource, 0, 0));

	checkCudaErrors(cudaMemcpyToArray(texture_ptr, 0, 0, cuda_dev_render_buffer, size_tex_data, cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_resource, 0));

	cudaDeviceSynchronize();
}


int main(int argc, char* argv[]) {
	
	// OpenGL Setting

	initGLFW();
	initGL();
	
	// Find best cuda device
	findCudaGLDevice(argc, (const char**)argv);
	initGLBuffers();
	initCUDABuffers();

	// Generate buffers
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);

	// Buffer setup
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// Position attribute (3 floats)
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);

	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)(3*sizeof(GLfloat)));
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);


	glBindVertexArray(VAO);
	glClearColor(0.0f, 0.0f, 0.3f, 0.1f);
	glClear(GL_COLOR_BUFFER_BIT);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, opengl_tex_cuda);

	shdrawtex.use();
	glUniform1i(glGetUniformLocation(shdrawtex.program, "tex"), 0);


	// Rendering Settings

	// samples per pixel
	int ns = 1;

	// thread block dimension
	int tx = 32;
	int ty = 32;

	int num_pixels = WIDTH * HEIGHT;
	size_t fb_size = num_pixels * sizeof(vec3);

	inputPointers pointers{ (unsigned int*)cuda_dev_render_buffer, (float*)cuda_dev_render_buffer_2 };

	vec3 *fb;
	checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

	curandState *d_rand_state;
	checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));
	curandState *d_rand_state2;
	checkCudaErrors(cudaMalloc((void**)&d_rand_state2, 1 * sizeof(curandState)));

	rand_init << <1, 1 >> > (d_rand_state2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	hitable **d_list;
	int num_hitables = 1 + 3;
	checkCudaErrors(cudaMalloc((void**)&d_list, num_hitables * sizeof(hitable*)));
	hitable **d_world;
	checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable_list*)));
	camera **d_camera;
	checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
	create_world << <1, 1 >> > (d_list, d_world, d_camera, WIDTH, HEIGHT, d_rand_state2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	dim3 blocks(WIDTH, HEIGHT);
	dim3 threads(ns);
	render_init << <blocks, threads >> > (WIDTH, HEIGHT, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	auto firstTime = std::chrono::system_clock::now();
	auto lastTime = firstTime;
	auto lastMeasureTime = firstTime;
	int frameNum = 0;	// For FPS count
	int frame = 1;	// For total frame accumulation

	// Debugging for BVH

	// Malloc temporary arrays because they will be parallelized later on.
	bvh_info **d_bvh_info_list;
	checkCudaErrors(cudaMalloc((void**)&d_bvh_info_list, num_hitables * sizeof(bvh_info*)));
	morton_info** d_morton_info_list;
	checkCudaErrors(cudaMalloc((void**)&d_morton_info_list, num_hitables * sizeof(morton_info*)));
	create_bvh << <1, 1 >> > (d_list, num_hitables, d_bvh_info_list, d_morton_info_list);



	free_bvh << <1, 1 >> > (d_list, num_hitables, d_bvh_info_list, d_morton_info_list);
	checkCudaErrors(cudaFree(d_bvh_info_list));
	checkCudaErrors(cudaFree(d_morton_info_list));


	/*
	// Loop frame
	while (glfwWindowShouldClose(window) == GL_FALSE) {		
	//for (int i = 0; i < 1; i++) {

		// Calculate duration
		auto currTime = std::chrono::system_clock::now();
		auto totalTime = currTime - firstTime;
		auto deltaTime = currTime - lastTime;

		// Reset display and call render function
		glClear(GL_COLOR_BUFFER_BIT);
		generateCUDAImage(fb, WIDTH, HEIGHT, ns, d_camera, d_world, d_rand_state, pointers, totalTime, deltaTime, frame);
		glfwPollEvents();
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

		// Swap the screen buffer
		glfwSwapBuffers(window);

		std::chrono::duration<double> elapsed_seconds = currTime - lastMeasureTime;
		frameNum++;
		frame++;

		// show fps every  second
		if (elapsed_seconds.count() >= 1.0) {

			std::cout << "fps: " << (frameNum / elapsed_seconds.count()) << ", total: " << frame << "\n";
			frameNum = 0;
			lastMeasureTime = currTime;
		}
		lastTime = currTime;
	}
	glBindVertexArray(0);
	*/

	
	do
	{
		std::cout << '\n' << "Press a key to continue...";
	} while (std::cin.get() != '\n');

	// Free rendering memory
	checkCudaErrors(cudaDeviceSynchronize());
	free_world << <1, 1 >> > (d_list, d_world, d_camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_camera));
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_rand_state2));
	checkCudaErrors(cudaFree(fb));


	// End GLFW
	glfwDestroyWindow(window);
	glfwTerminate();

	cudaDeviceReset();

	return 0;
}
