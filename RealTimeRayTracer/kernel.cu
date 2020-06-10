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

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__)

GLFWwindow* window;
int WIDTH = 480;
int HEIGHT = 360;
int num_texels = WIDTH * HEIGHT;
int num_values = num_texels * 4;
int size_tex_data = sizeof(GLuint) * num_values;
int size_tex_data_f = sizeof(GLfloat) * num_values;

// OpenGL
GLuint VBO, VAO, EBO;
GLSLShader drawtex_f;
GLSLShader drawtex_v;
GLSLProgram shdrawtex;

// CUDA buffers
void* cuda_dev_render_buffer;	// Stores initial
void* cuda_dev_render_buffer_2;	// Stores final output
void* cuda_ping_buffer;	// Stores intermediate effects


struct inputPointers {
	unsigned int* image1; // texture position
	float* acc_img;	// accumulated image
};


struct cudaGraphicsResource* cuda_tex_resource;
GLuint opengl_tex_cuda;	// OpenGL Texture for cuda result

static const char* glsl_drawtex_vertshader_src =
"#version 330 core\n"
"layout (location = 0) in vec3 position;\n"
"layout (location = 1) in vec2 texCoord;\n"
"\n"
"out vec2 ourTexCoord;\n"
"\n"
"void main()\n"
"{\n"
"	gl_Position = vec4(position, 1.0f);\n"
"	ourTexCoord = texCoord;\n"
"}\n";

static const char* glsl_drawtex_fragshader_src =
"#version 330 core\n"
"uniform usampler2D tex;\n"
"in vec2 ourTexCoord;\n"
"out vec4 color;\n"
"void main()\n"
"{\n"
"   	vec4 c = texture(tex, ourTexCoord);\n"
"   	color = c / 255.0;\n"
"}\n";

// QUAD GEOMETRY
GLfloat vertices[] = {
	// Positions             // Texture Coords
	1.0f, 1.0f, 0.5f,1.0f, 1.0f,  // Top Right
	1.0f, -1.0f, 0.5f, 1.0f, 0.0f,  // Bottom Right
	-1.0f, -1.0f, 0.5f, 0.0f, 0.0f,  // Bottom Left
	-1.0f, 1.0f, 0.5f,  0.0f, 1.0f // Top Left 
};
// you can also put positions, colors and coordinates in seperate VBO's
GLuint indices[] = {
	0, 1, 3,
	1, 2, 3
};


void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << "at " <<
			file << ":" << line << "'" << func << "/n";
		cudaDeviceReset();
		exit(99);
	}
}


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

void createGLTextureForCUDA(GLuint* gl_tex, cudaGraphicsResource** cuda_tex, unsigned int size_x, unsigned int size_y) {

	// Create an OpenGL texture
	glGenTextures(1, gl_tex);	//generate 1 texture
	glBindTexture(GL_TEXTURE_2D, *gl_tex);	// set it as current target

	// Set basic texture parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// Specify 2D texture
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32UI_EXT, size_x, size_y, 0, GL_RGB_INTEGER_EXT, GL_UNSIGNED_BYTE, NULL);
	
	// Register this texture with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterImage(cuda_tex, *gl_tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
}

bool initGLFW() {

	// Initialize GLFW
	if (glfwInit() == GL_FALSE) {
		exit(EXIT_FAILURE);
	}

	// Version Setting
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);



	// Create Window
	window = glfwCreateWindow(WIDTH, HEIGHT, "Raytracer", NULL, NULL);
	if (!window) {
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	// Create Context
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	// Do keyboard stuff here

	return true;
}

void initGLBuffers() {
	// Create texture that will receive the result of cuda rendering
	createGLTextureForCUDA(&opengl_tex_cuda, &cuda_tex_resource, WIDTH, HEIGHT);
	
	// Create shader program
	drawtex_v = GLSLShader("Textured draw vertex shader", glsl_drawtex_vertshader_src, GL_VERTEX_SHADER);
	drawtex_f = GLSLShader("Textured draw fragment shader", glsl_drawtex_fragshader_src, GL_FRAGMENT_SHADER);
	shdrawtex = GLSLProgram(&drawtex_v, &drawtex_f);
	shdrawtex.compile();
}

bool initGL() {
	glewExperimental = GL_TRUE;	// need this for core profile
	GLenum err = glewInit();
	glGetError();
	if (err != GLEW_OK) {
		printf("glewInit failed : %s /n", glewGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	glViewport(0, 0, WIDTH, HEIGHT);

	return true;
}

void initCUDABuffers() {
	cudaError_t stat;

	checkCudaErrors(cudaMalloc(&cuda_dev_render_buffer, size_tex_data));
	checkCudaErrors(cudaMalloc(&cuda_dev_render_buffer_2, size_tex_data_f));
	checkCudaErrors(cudaMalloc(&cuda_ping_buffer, size_tex_data));

	cudaDeviceSynchronize();
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

	printf("sphere size %d/n", sizeof(sphere));
	printf("sphere* size %d/n", sizeof(sphere*));

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

	/*
	do
	{
		std::cout << '\n' << "Press a key to continue...";
	} while (std::cin.get() != '\n');*/

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
