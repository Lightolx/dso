/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once

#include "util/NumType.h"
#include "algorithm"

namespace dso
{


class FrameShell
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	int id; 			// INTERNAL ID, starting at zero.
	int incoming_id;	// ID passed into DSO
	double timestamp;		// timestamp passed into DSO.

	// set once after tracking
	SE3 camToTrackingRef;
	FrameShell* trackingRef;

	// constantly adapted.
	SE3 camToWorld;				// Write: TRACKING, while frame is still fresh; MAPPING: only when locked [shellPoseMutex].
	AffLight aff_g2l;       // 光度映射函数，矫正图像灰度的affine matrix
	bool poseValid;         // 构造函数中直接初始化为true，有点强悍啊，意味着来了一帧我就有信心必然能算出它的pose

	// statisitcs
	int statistics_outlierResOnThis;
	int statistics_goodResOnThis;
	int marginalizedAt;
	double movedByOpt;


	inline FrameShell()
	{
		id=0;
		poseValid=true;
		camToWorld = SE3();
		timestamp=0;
		marginalizedAt=-1;
		movedByOpt=0;
		statistics_outlierResOnThis=statistics_goodResOnThis=0;
		trackingRef=nullptr;    // 指向其他FrameShell的指针
		camToTrackingRef = SE3();
	}
};


}

