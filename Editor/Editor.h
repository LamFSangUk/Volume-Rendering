#include <cstdint>
#include <imgui\imgui.h>
#include "imgui_impl_sdl_gl3.h"
#include <GL\glew.h>  
#include <SDL\SDL.h>
#include <chrono>

#include "volume.h"
#include "ArcBall.h"
#include "Octree.h"

#undef main // undef to remove sdl_main

#pragma once
class VolumeData;
class Editor {
public:
	Editor(uint32_t width, uint32_t height);
	~Editor();
	bool Initialize();
	void Run();
private:
	// Volume renderer functionality
	void UpdateTexture(const void* buffer, const int width, const int height);
	void Process();
	// UI
	void ControlPanel(uint32_t width, uint32_t height);
	void Scene(uint32_t width, uint32_t height);
	// SDL event related functions
	void OnResize(uint32_t width, uint32_t height);
	void HandleSDLEvent(SDL_Event* event);

	// SDL & window
	SDL_Window* m_window;
	SDL_GLContext m_context;
	uint32_t m_width, m_height;
	// Status
	bool m_isRunning;
	// Output texture
	bool m_hasTexture;
	GLuint m_textureID;

	// Volume
	vdcm::Volume* m_vol;

	// Octree
	Octree* tree;

	// Screen
	glm::vec3 m_screen_center;
	glm::vec3 m_screen_delta_x;
	glm::vec3 m_screen_delta_y;

	// Arcball
	ArcBall* m_arcball;

	bool m_is_right_pressed = false;
};

