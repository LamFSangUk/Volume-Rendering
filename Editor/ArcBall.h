#ifndef __ARCBALL_H__
#define __ARCBALL_H__

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/common.hpp>

typedef struct ScreenCoord {
	float x, y;
}ScreenCoord;

typedef struct SpehreCoord {
	float x, y, z;
}SphereCoord;

class ArcBall {
public:
	ArcBall(int, int);
	~ArcBall();

	void setBounds(int, int);
	void setStart(int, int);
	glm::mat4 getRotationMatrix();
	void rotate(int, int);

	void reset();

private:
	int m_window_width;
	int m_window_height;

	glm::vec3 m_origin;
	float m_radius;

	glm::quat m_quat_anchor;
	glm::quat m_quat_now;

	SpehreCoord m_anchor;

	SphereCoord _toSphereCoord(ScreenCoord);

};

#endif // __ARCBALL_H__