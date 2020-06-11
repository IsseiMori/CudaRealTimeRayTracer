#ifndef GLOBALH
#define GLOBALH


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


void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << "at " <<
			file << ":" << line << "'" << func << "/n";
		cudaDeviceReset();
		exit(99);
	}
}


#endif