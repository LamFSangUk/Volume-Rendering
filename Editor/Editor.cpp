#include "Editor.h"

#include "volume.h"

#include <iostream>
#include <regex>

#include <cmath>
#include <algorithm>
#include "test.cuh"

using namespace std;

Editor::Editor(uint32_t width, uint32_t height)
	:m_window(nullptr),
	m_context(nullptr),
	m_width(width),
	m_height(height),
	m_isRunning(true),
	m_hasTexture(false) {
}

Editor::~Editor() {
	ImGui_ImplSdlGL3_Shutdown();
	SDL_GL_DeleteContext(m_context);
	SDL_DestroyWindow(m_window);
	SDL_Quit();
}

bool Editor::Initialize() {
	// Setup SDL
	if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
		std::cout << ("Error: %s\n", SDL_GetError()) << std::endl;
		return false;
	}

	// Setup window
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
	SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
	SDL_DisplayMode current;
	SDL_GetCurrentDisplayMode(0, &current);
	m_window = SDL_CreateWindow("Volume Renderer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, m_width, m_height, SDL_WINDOW_OPENGL);
	SDL_GLContext glcontext = SDL_GL_CreateContext(m_window);
	glewInit();

	// Setup ImGui binding
	ImGui_ImplSdlGL3_Init(m_window);
	Process();
	return true; // Return initialization result
}

void Editor::Run() {
	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
	while (m_isRunning) {
		// Handle SDL events
		SDL_Event event;
		if (SDL_PollEvent(&event)) {
			ImGui_ImplSdlGL3_ProcessEvent(&event);
			HandleSDLEvent(&event);
		}
		ImGui_ImplSdlGL3_NewFrame(m_window);
		// Editor
		{
			ControlPanel(m_width - 720, 720);
			Scene(720, 720);
			// Code sample of ImGui (Remove comment when you want to see it)
			ImGui::ShowTestWindow();
		}
		// Rendering
		glViewport(0, 0, (int)ImGui::GetIO().DisplaySize.x, (int)ImGui::GetIO().DisplaySize.y);
		glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
		glClear(GL_COLOR_BUFFER_BIT);

		ImGui::Render();
		SDL_GL_SwapWindow(m_window);
	}
}

void Editor::UpdateTexture(const void * buffer, int width, int height) {
	if (!m_hasTexture) {
		auto err = glGetError();
		glGenTextures(1, &m_textureID);
		if (err != GL_NO_ERROR) {
			throw std::runtime_error("Not able to create texture from buffer" + std::to_string(glGetError()));
		}
		else {
			m_hasTexture = true;
		}
	}
	glBindTexture(GL_TEXTURE_2D, m_textureID);
	// set texture sampling methods
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, buffer);
	glBindTexture(GL_TEXTURE_2D, 0);
}

//bool intersect(Eigen::Vector4f origin, Eigen::Vector4f dir, 
//		Eigen::Vector4f* bounds, float &t) {
//	Eigen::Vector4f inv_dir(1. / dir[0], 1. / dir[1], 1. / dir[2], 0.);
//
//	double t1 = (bounds[0][0] - origin[0])*inv_dir[0];
//	double t2 = (bounds[1][0] - origin[0])*inv_dir[0];
//
//	double tmin = min(t1, t2);
//	double tmax = max(t1, t2);
//
//	for (int i = 1; i < 3; ++i) {
//		t1 = (bounds[0][i] - origin[i])*inv_dir[i];
//		t2 = (bounds[1][i] - origin[i])*inv_dir[i];
//
//		tmin = max(tmin, min(t1, t2));
//		tmax = min(tmax, max(t1, t2));
//	}
//
//	t = tmin;
//
//	return tmax > max(tmin, 0.0);
//}
//
//// Raycast into the volume
//Eigen::Vector4f RayCast(Eigen::Vector4f origin, Eigen::Vector4f dir, vdcm::Volume* vol, int width, int height, int depth) {
//	// Find start and end points.
//	Eigen::Vector4f start, end, min_bound, max_bound;
//	min_bound = Eigen::Vector4f(-width / 2.f, -height / 2.f, -depth / 2.f, 1);
//	max_bound = Eigen::Vector4f(width / 2.f, height / 2.f, depth / 2.f, 1);
//
//	Eigen::Vector4f bounds[2];
//	bounds[0] = min_bound; bounds[1] = max_bound;
//
//	//printf("orogin: %f %f %f\n", origin.x(), origin.y(), origin.z());
//
//
//	float t;
//	bool is_intersect = intersect(origin, dir, bounds, t);
//
//	if (!is_intersect) {
//		// RGBA color
//		Eigen::Vector4f color(0, 0, 0, 255);
//		return color;
//	}
//	//printf("Intersect!\n");
//	float max_val = 0.0f;
//	// TODO: Sampling from start to end
//	Eigen::Vector4f cur = origin + dir * t;
//
//	/*printf("origin: %f %f %f\n", origin.x(), origin.y(), origin.z());
//	printf("dir: %f %f %f %f\n", dir.x(), dir.y(), dir.z(), t);
//	printf("start: %f %f %f\n", cur.x(), cur.y(), cur.z());*/
//	while (true) {
//		// Terminate condition
//		if (cur.x() > max_bound.x() || cur.x() < min_bound.x()
//			|| cur.y() > max_bound.y() || cur.y() < min_bound.y()
//			|| cur.z() > max_bound.z() || cur.z() < min_bound.z())
//			break;
//
//		// Get origin coordinates
//		float x = cur.x() + width / 2.f;
//		float y = cur.y() + height / 2.f;
//		float z = cur.z() + depth / 2.f;
//		//printf("real: %f %f %f\n", x, y, z);
//
//		if (x < 0 || y < 0 || z < 0 || x >= width || y >= height || z >= depth) {
//			// Ray go
//			cur.x() += dir.x();
//			cur.y() += dir.y();
//			cur.z() += dir.z();
//			continue;
//		}
//		/*printf("moved: %f %f %f\n", cur.x(), cur.y(), cur.z());
//		printf("real: %f %f %f\n", x, y, z);*/
//
//		// Interpolate with x, y, z
//		//float val = vol_data[(int)floor(z) * width * height + (int)floor(y) * width + (int)floor(x)];
//		float val = vol->getIntensity(x,y,z);
//		if (max_val < val) max_val = val;
//		// printf("%f\n", max_val);
//		// Ray go
//		cur.x() += dir.x();
//		cur.y() += dir.y();
//		cur.z() += dir.z();
//	}
//
//	// RGBA color
//	Eigen::Vector4f color(max_val*255,0,0,255);
//	return color;
//}

void Editor::Process() {

	/* TODO : Process volume data & pass raw buffer to UpdateTexture method*/
	//vdcm::Volume* vol = vdcm::read("D:/AnnotationProject/MedView/MedView/dicom_ct_sample");


	// Screen
	unsigned char* screen = new unsigned char[m_width * m_height * 4];
	//Eigen::Vector4f screen_center(0, 0, 500, 1);
	//Eigen::Vector4f volume_center(0, 0, 0, 1);

	///* Temporal Camera pos and Screen pos */
	//Eigen::Vector4f direction;
	//float screen_x, screen_y;	// Start point left bottom
	//float delta_x, delta_y; 
	//delta_x = 1.f;
	//delta_y = 1.f;
	//screen_x = screen_center.x() - m_width * delta_x / 2.f;
	//screen_y = screen_center.y() - m_height * delta_y / 2.f;

	rayCastCuda(m_width, m_height, screen);

	// Assume that only consider parallel ray
	//direction = volume_center - screen_center;
	//direction.normalize();

	float cur_x, cur_y;
	int idx = 0;

	// CPU Raycasting
	//for (int y = 0; y < m_height; y++) {

	//	for (int x = 0; x < m_width; x++) {
	//		//printf("cur: %f %f\n", cur_x, cur_y);

	//		// TODO: replace with octree data
	//		cur_x = screen_x + delta_x * x;
	//		cur_y = screen_y + delta_y * y;
	//		Eigen::Vector4f origin(cur_x, cur_y, screen_center.z(), 1);
	//		Eigen::Vector4f color = RayCast(origin, direction, vol, width, height, depth);

	//		screen[idx + 0] = color.x();
	//		screen[idx + 1] = color.y();
	//		screen[idx + 2] = color.z();
	//		screen[idx + 3] = color.w();

	//		idx += 4;
	//	}
	//}

	UpdateTexture(screen, m_width, m_height);

}

void Editor::ControlPanel(uint32_t width, uint32_t height) {
	// Control Panel Window
	ImGui::SetNextWindowSize(ImVec2((float)width, (float)height));
	ImGui::SetNextWindowPos(ImVec2(0, 0));
	ImGui::Begin("Control Panel", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);

	/* TODO : Write UI Functions */

	ImGui::End();
}

void Editor::Scene(uint32_t width, uint32_t height) {
	// Scene Window
	ImGui::SetNextWindowSize(ImVec2((float)width, (float)height));
	ImGui::SetNextWindowPos(ImVec2((float)(m_width - width), 0.f));
	ImGui::Begin("Scene", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
	// Draw texture if there is one
	if (m_hasTexture) {
		ImGui::Image(ImTextureID(m_textureID), ImGui::GetContentRegionAvail());
	}
	ImGui::End();
}

void Editor::OnResize(uint32_t width, uint32_t height) {
	m_width = width;
	m_height = height;
}

void Editor::HandleSDLEvent(SDL_Event * event) {
	// SDL_Event wiki : https://wiki.libsdl.org/SDL_Event
	static bool mouseIsDown = false;
	static bool isDragging = false;
	int degreeStep = 5;
	switch (event->type) {
	case SDL_QUIT:
		m_isRunning = false;
		break;
	case SDL_KEYDOWN:
		break;
	case SDL_MOUSEWHEEL:
		break;
	case SDL_MOUSEMOTION:
		break;
	case SDL_MOUSEBUTTONDOWN:
		break;
	case SDL_MOUSEBUTTONUP:
		mouseIsDown = false;
		break;
	case SDL_WINDOWEVENT:
		switch (event->window.event) {
		case SDL_WINDOWEVENT_RESIZED:
			OnResize(event->window.data1, event->window.data2);
			break;
		default:
			break;
		}
	default:
		break;
	}
}
