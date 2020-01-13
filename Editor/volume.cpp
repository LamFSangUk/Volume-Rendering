#include "volume.h"
#include "gdcmScanner.h"
#include "gdcmImageReader.h"
#include "gdcmPixmap.h"
#include "gdcmAttribute.h"

#include <cmath>

namespace vdcm {
	Volume* read(const char* dirname) {

		// Directory 
		gdcm::Directory dir;
		dir.Load(dirname, false);

		// File List of Directory
		gdcm::Directory::FilenamesType files = dir.GetFilenames();

		if (files.size() == 0) {
			throw std::runtime_error("ERROR! Empty dir\n");
		}

		// For debug
		printf("files %d\n", files.size());
		
		// Read multi-frame DICOMs
		Volume *volume = new Volume();

		volume->m_depth = files.size();

		for (auto filename : files) {
			
			// Read DICOM file
			gdcm::ImageReader reader;
			reader.SetFileName(filename.c_str());
			if (!reader.Read()) {
				throw std::runtime_error("ERROR! None file\n");
			}

			// Save some attributes of DICOM slice
			if (volume->m_width == 0 || volume->m_height == 0) {
				const gdcm::File &f = reader.GetFile();
				const gdcm::DataSet &ds = f.GetDataSet();

				// row, col
				gdcm::Attribute<0x0028, 0x0010> at_rows;
				gdcm::Attribute<0x0028, 0x0011> at_cols;
				at_rows.Set(ds);
				at_cols.Set(ds);
				volume->m_height = at_rows.GetValue();
				volume->m_width = at_cols.GetValue();

				// Rescale parameter
				gdcm::Attribute<0x0028, 0x1052> at_rescale_intercept;
				gdcm::Attribute<0x0028, 0x1053> at_rescale_slope;
				at_rescale_intercept.Set(ds);
				at_rescale_slope.Set(ds);
				volume->m_rescale_intercept = at_rescale_intercept.GetValue();
				volume->m_rescale_slope = at_rescale_slope.GetValue();

				// Windowing parameter
				gdcm::Attribute<0x0028, 0x1050> at_window_level;
				gdcm::Attribute<0x0028, 0x1051> at_window_width;
				at_window_level.Set(ds);
				at_window_width.Set(ds);
				volume->m_window_level = at_window_level.GetValue();
				volume->m_window_width = at_window_width.GetValue();

			//	// Slice spacing parameter
			//	gdcm::Attribute<0x0028, 0x0030> at_pixel_spacing;
			//	gdcm::Attribute<0x0018, 0x0050> at_slice_thickness;
			//	gdcm::Attribute<0x0018, 0x1088> at_slice_spacing;
			//	at_pixel_spacing.Set(ds);
			//	at_slice_thickness.Set(ds);
			//	at_slice_spacing.Set(ds);
			//	// TODO:: save window parameter
			//	//volume->m_rescale_intercept = at_rescale_intercept.GetValue();
			//	//volume->m_rescale_slope = at_rescale_slope.GetValue();
			}

			const gdcm::Pixmap &image = reader.GetPixmap();
			int length = image.GetBufferLength();
			
			int16_t* buffer = new int16_t[length/2];
			char* temp_buffer = new char[length];
			image.GetBuffer(temp_buffer);

			int max_val = INT_MIN;
			int min_val = INT_MAX;
			for (int i = 0; i < length; i+=2) {
				int16_t val = (unsigned char)temp_buffer[i] | ((unsigned char)temp_buffer[i+1] << 8);
				
				buffer[i/2] = (val * volume->m_rescale_slope + volume->m_rescale_intercept); // Rescale
				if (buffer[i / 2] > max_val) max_val = buffer[i / 2];
				if (buffer[i / 2] < min_val) min_val = buffer[i / 2];
			}
			
			std::vector<int16_t> slice(buffer, buffer + length/2);

			volume->m_volume_data.push_back(slice);
			volume->setMaxIntensity(max_val);
			volume->setMinIntensity(min_val);

			delete temp_buffer;
			delete buffer;
		}
		printf("Read done\n", files.size());


		return volume;
	}

	Volume::Volume() : m_width(0),m_height(0),m_depth(0) {

	}

	//Slice* Volume::getSlice(Mode mode, Axes* axes, int width, int height, Eigen::Vector3f center, float zoom) {
	//	Slice* s = new Slice(width, height, (width -1)/2.0f, (height -1)/2.0f);

	//	// Transformation
	//	s->refTransform(mode, axes->getCenter(), center, axes->getYaw(), axes->getRoll(), axes->getPitch(), zoom);

	//	// Interpolate
	//	for (int i = 0; i < s->getHeight(); i++) {
	//		for (int j = 0; j < s->getWidth(); j++) {
	//			Eigen::Vector4f pos = s->getVoxelCoord(j, i);
	//			int intensity = 0;

	//			int x_0 = (int)pos(0);
	//			int y_0 = (int)pos(1);
	//			int z_0 = (int)pos(2);
	//			int x_1 = (int)(pos(0) + 1);
	//			int y_1 = (int)(pos(1) + 1);
	//			int z_1 = (int)(pos(2) + 1);

	//			// Check max and min boundary
	//			if (x_0 < 0 || y_0 < 0 || z_0 < 0 || x_1 >= m_width || y_1 >= m_height || z_1 >= m_depth) {
	//				s->setVoxelIntensity(j, i, 0);
	//				continue;
	//			}

	//			double x_d = pos(0) - x_0;
	//			double y_d = pos(1) - y_0;
	//			double z_d = pos(2) - z_0;

	//			double c_00 = m_volume_data[z_0][y_0*m_width + x_0] * (1 - x_d) + m_volume_data[z_0][y_0*m_width + x_1] * x_d;
	//			double c_01 = m_volume_data[z_1][y_0*m_width + x_0] * (1 - x_d) + m_volume_data[z_1][y_0*m_width + x_1] * x_d;
	//			double c_10 = m_volume_data[z_0][y_1*m_width + x_0] * (1 - x_d) + m_volume_data[z_0][y_1*m_width + x_1] * x_d;
	//			double c_11 = m_volume_data[z_1][y_1*m_width + x_0] * (1 - x_d) + m_volume_data[z_1][y_1*m_width + x_1] * x_d;

	//			double c_0 = c_00 * (1 - y_d) + c_10 * y_d;
	//			double c_1 = c_01 * (1 - y_d) + c_11 * y_d;

	//			double c = c_0 * (1 - z_d) + c_1 * z_d;

	//			intensity = (int16_t)(c + 0.5);

	//			s->setVoxelIntensity(j, i, intensity);
	//		}
	//	}

	//	return s;
	//}

	float* Volume::getBuffer() {
		int size = m_width * m_height * m_depth;

		float* buffer = new float[size];

		for (int i = 0; i < m_depth; i++) {
			for (int j = 0; j < m_height; j++) {
				for (int k = 0; k < m_width; k++) {
					buffer[i * m_width * m_height + j * m_width + k]
						= (m_volume_data[i][j * m_width + k] - INT16_MIN)
						/(float)(INT16_MAX - INT16_MIN);
				}
			}
		}

		return buffer;
	}

	int Volume::getWidth() {
		return m_width;
	}
	int Volume::getHeight() {
		return m_height;
	}
	int Volume::getDepth() {
		return m_depth;
	}

	std::pair<int, int> Volume::getDefaultWindowing() {
		return std::make_pair(m_window_level, m_window_width);
	}

	float Volume::getIntensity(float x, float y, float z) {
		float val = m_volume_data[floor(z)][floor(y)*m_width + floor(x)];

		float window_min = m_window_level - m_window_width / 2;
		float window_max = m_window_level + m_window_width / 2;

		if (val < window_min) val = window_min;
		if (val > window_max) val = window_max;

		return (val - window_min) / (float)(window_max-window_min);
	}
}