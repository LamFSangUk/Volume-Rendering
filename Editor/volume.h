#ifndef __VOLUME_H__
#define __VOLUME_H__

#define _CRT_SECURE_NO_WARNINGS
#include <vector>

namespace vdcm {
	class Volume
	{
	public:
		Volume();

		friend Volume* read(const char*);

		//Slice* getSlice(Mode, Axes*, int, int, Eigen::Vector3f, float);

		int getWidth();
		int getHeight();
		int getDepth();

		float* getBuffer();

		void setMaxIntensity(int);
		void setMinIntensity(int);

		std::pair<int, int> getDefaultWindowing();

		float getIntensity(float x, float y, float z);
		std::vector<std::vector<int16_t> > m_volume_data;

	private:
		int m_width;
		int m_height;
		int m_depth;

		int m_rescale_slope;
		int m_rescale_intercept;

		int m_window_level;
		int m_window_width;

		float m_pixel_spacing_x;
		float m_pixel_spacing_y;
		float m_slice_thickness;
		float m_slice_spacing;

		int m_max_intensity;
		int m_min_intensity;
	};

	inline void Volume::setMaxIntensity(int intensity) { m_max_intensity = intensity; }
	inline void Volume::setMinIntensity(int intensity) { m_min_intensity = intensity; }
}

#endif // __VOLUME_H__