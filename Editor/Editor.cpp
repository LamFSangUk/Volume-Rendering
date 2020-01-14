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

	// Load Volume
	m_vol = vdcm::read("D:/AnnotationProject/MedView/MedView/dicom_ct_sample");

	// Set default Screen
	m_screen_center = glm::vec3(0, 0, 500);
	m_screen_delta_x = glm::vec3(1, 0, 0);
	m_screen_delta_y = glm::vec3(0, 1, 0);

	// Create Arcball
	m_arcball = new ArcBall(720, m_height);

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

void Editor::Process() {

	/* TODO : Process volume data & pass raw buffer to UpdateTexture method*/

	// Screen
	unsigned char* screen = new unsigned char[m_width * m_height * 4];

	glm::mat4 rot_m = m_arcball->getRotationMatrix();
	m_screen_center = glm::vec4(m_screen_center, 1) * rot_m;

	m_screen_delta_x = glm::vec4(m_screen_delta_x, 1) * rot_m;
	m_screen_delta_y = glm::vec4(m_screen_delta_y, 1) * rot_m;

	rayCastCuda(m_vol, 
		m_screen_center, m_screen_delta_x, m_screen_delta_y, 
		m_width, m_height, screen);

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
		if (m_is_right_pressed && event->motion.x >= (1000-720)) {
			printf("%d %d\n", event->motion.x - (1000 - 720), event->motion.y);

			m_arcball->rotate(event->motion.x - (1000-720), event->motion.y);
			Process();
		}
		break;
	case SDL_MOUSEBUTTONDOWN:
		if (event->button.button == SDL_BUTTON_RIGHT) {
			m_is_right_pressed = true;

			printf("%d %d\n", event->button.x - (1000 - 720), event->button.y);
			m_arcball->setStart(event->button.x - (1000 - 720), event->button.y);
		}
		break;
	case SDL_MOUSEBUTTONUP:
		if (event->button.button == SDL_BUTTON_RIGHT) {
			m_is_right_pressed = false;
		}
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
