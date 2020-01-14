#include "ArcBall.h"
#include <algorithm>

ArcBall::ArcBall(int width, int height) {
	this->setBounds(width, height);

	this->m_quat_anchor = glm::quat(1, 0, 0, 0);
	this->m_quat_now = glm::quat(1, 0, 0, 0);
}

ArcBall::~ArcBall() {

}

void ArcBall::reset() {
	this->m_quat_anchor = glm::quat(1, 0, 0, 0);
	this->m_quat_now = glm::quat(1, 0, 0, 0);
}

/**
 * Resize screen and arcball
 */
void ArcBall::setBounds(int width, int height) {
	this->m_window_width = width;
	this->m_window_height = height;

	this->m_radius = std::min(width, height) / 2.0f;
	this->m_origin = glm::vec3(width / 2.0, height / 2.0, 0);
}

void ArcBall::setStart(int x, int y) {
	ScreenCoord screen_coord;
	screen_coord.x = x; screen_coord.y = y;

	m_anchor = _toSphereCoord(screen_coord);

	m_quat_anchor = m_quat_now;
}

void ArcBall::rotate(int cur_x, int cur_y) {
	ScreenCoord cur_screen;
	cur_screen.x = cur_x; cur_screen.y = cur_y;

	SpehreCoord prev, cur;
	prev = m_anchor;
	cur = _toSphereCoord(cur_screen);

	glm::vec3 prev_vector = glm::normalize(glm::vec3(prev.x, prev.y, prev.z));
	glm::vec3 cur_vector = glm::normalize(glm::vec3(cur.x, cur.y, cur.z));

	float dot = glm::dot(prev_vector, cur_vector);
	float angle = 2 * acos(std::min(1.0f, dot));

	glm::vec3 cam_axis = glm::normalize(glm::cross(prev_vector, cur_vector));

	glm::quat rot;
	rot = glm::angleAxis(angle, cam_axis);
	m_quat_now = rot * m_quat_anchor;
}

glm::mat4 ArcBall::getRotationMatrix() {
	return glm::mat4_cast(m_quat_now);
}


/**
 * Translate coordinate
 */
SphereCoord ArcBall::_toSphereCoord(ScreenCoord screen) {
	SphereCoord sphere;
	sphere.x = (screen.x - m_origin.x) / m_radius;
	sphere.y = -(screen.y - m_origin.y) / m_radius;

	float length_squared = (sphere.x * sphere.x) + (sphere.y * sphere.y);

	if (length_squared > 1.0f) {
		float norm = (float)(1.0f / sqrt(length_squared));

		sphere.x = sphere.x * norm;
		sphere.y = sphere.y * norm;
		sphere.z = 0.0f;
	}
	else {
		sphere.z = sqrt(1.0f - length_squared);
	}

	return sphere;
}