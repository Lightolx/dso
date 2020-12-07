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

namespace dso
{

enum PixelSelectorStatus {PIXSEL_VOID=0, PIXSEL_1, PIXSEL_2, PIXSEL_3};


class FrameHessian;

class PixelSelector
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	int makeMaps(
			const FrameHessian* const fh,
			float* map_out, float density, int recursionsLeft=1, bool plot=false, float thFactor=1);

	PixelSelector(int w, int h);
	~PixelSelector();
	int currentPotential;


	bool allowFast;
	void makeHists(const FrameHessian* const fh);
private:

    // map_out作为出参输出的是与fh的size相同的4值图，每个像素值可能是0,1,2,4．
    // 0表示该像素处没提取出keypoint
    // 1表示该像素处在第0层提取出了keypoint
    // 2表示该像素处在第1层提取出了keypoint
    // 4表示该像素处在第2层提取出了keypoint
	Eigen::Vector3i select(const FrameHessian* const fh,
			float* map_out, int pot, float thFactor=1);


	unsigned char* randomPattern;   // w*h个，范围为0~255的随机数


	int* gradHist;
	float* ths;         // 每个格子内选定为keypoint的梯度阈值，也就是3.2节,step1中的 g+gth
	float* thsSmoothed; // smoothed过后的阈值ths，其实就是ths的3*3的高斯平滑
	int thsStep;        // 表示原图上每一行有几个32*32的网格
	const FrameHessian* gradHistFrame;
};




}

