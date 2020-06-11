#ifndef OPENGL_UTILS
#define OPENGL_UTILS

#include "global.h"

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

#endif