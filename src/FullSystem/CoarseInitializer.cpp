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


/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/CoarseInitializer.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "util/nanoflann.h"
#include <glog/logging.h>

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{

CoarseInitializer::CoarseInitializer(int ww, int hh) : thisToNext_aff(0,0), thisToNext(SE3())
{
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		points[lvl] = 0;
		numPoints[lvl] = 0;
	}

	JbBuffer = new Vec10f[ww*hh];
	JbBuffer_new = new Vec10f[ww*hh];


	frameID=-1;
	fixAffine=true;
	printDebug=false;

	wM.diagonal()[0] = wM.diagonal()[1] = wM.diagonal()[2] = SCALE_XI_ROT;
	wM.diagonal()[3] = wM.diagonal()[4] = wM.diagonal()[5] = SCALE_XI_TRANS;
	wM.diagonal()[6] = SCALE_A;
	wM.diagonal()[7] = SCALE_B;
}
CoarseInitializer::~CoarseInitializer()
{
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		if(points[lvl] != 0) delete[] points[lvl];
	}

	delete[] JbBuffer;
	delete[] JbBuffer_new;
}


bool CoarseInitializer::trackFrame(FrameHessian* newFrameHessian, std::vector<IOWrap::Output3DWrapper*> &wraps)
{
	newFrame = newFrameHessian;

    for(IOWrap::Output3DWrapper* ow : wraps)
        ow->pushLiveFrame(newFrameHessian);

	int maxIterations[] = {5,5,10,30,50};



	alphaK = 2.5*2.5;//*freeDebugParam1*freeDebugParam1;
	alphaW = 150*150;//*freeDebugParam2*freeDebugParam2;
	regWeight = 0.8;//*freeDebugParam4;
	couplingWeight = 1;//*freeDebugParam5;

	if(!snapped)    // setFirst()中被设置为了false
	{
		thisToNext.translation().setZero();     // thisToNext在setFirst()中置为了单位阵
		for(int lvl=0;lvl<pyrLevelsUsed;lvl++)  // 重置每一层每个keypoint的iR, iDepth_new, lastHessian
		{
			int npts = numPoints[lvl];
			Pnt* ptsl = points[lvl];
			for(int i=0;i<npts;i++)
			{
				ptsl[i].iR = 1;
				ptsl[i].idepth_new = 1;
				ptsl[i].lastHessian = 0;
			}
		}
	}


	SE3 refToNew_current = thisToNext;  // 第0帧相对于第1帧的pose
    // CoarseInitializer的构造函数中，thisToNext_aff的两个光度矫正参数(a,b)都被初始化为了0，也就是这个Affine Matrix是单位阵
	AffLight refToNew_aff_current = thisToNext_aff;
//    std::cout << "a1 = " << refToNew_aff_current.a << ", b1 = " << refToNew_aff_current.b << std::endl;
    // 如果两帧的曝光时间均已知的话，那么a就可以初始化为log[(t1*exp(0))/(t0*exp(0))]，也就是
    // a = log(t1/t0), b = 0，这样就有方程
    // exp(a)*I0 + b， 就可以把光度从第0帧变换到第1帧
	if(firstFrame->ab_exposure>0 && newFrame->ab_exposure>0)
		refToNew_aff_current = AffLight(logf(newFrame->ab_exposure /  firstFrame->ab_exposure),0); // coarse approximation.
		// 因为firstFrame->ab_exposure和newFrame->ab_exposure都设置为了1s，所以其实a,b也都等于0，Affine Matrix仍然是单位阵
//    std::cout << "a2 = " << refToNew_aff_current.a << ", b2 = " << refToNew_aff_current.b << std::endl;

	Vec3f latestRes = Vec3f::Zero();
	for(int lvl=pyrLevelsUsed-1; lvl>=0; lvl--)     // 两帧图像之间找2d patch匹配，当然是从高层往底层一步步refine
	{



		if(lvl<pyrLevelsUsed-1)     // 如果不是金字塔最顶层（因为最顶层没有上一层，老老实实地从逆深度等于0开始迭代）
			propagateDown(lvl+1);   // 将当前层所有keypoints的逆深度初始化为上一层的parent的逆深度，加速本层keypoints逆深度的收敛

		Mat88f H,Hsc; Vec8f b,bsc;  // normal equation, 即 H*Δx=b 中的H,b. 8维是因为周围有8个点？
		resetPoints(lvl);   // 重置每个keypoint的idepth_new = idepth
		Vec3f resOld = calcResAndGS(lvl, H, b, Hsc, bsc, refToNew_current, refToNew_aff_current, false);
        // 更新当前层每一个kpt的energy, isGood, idepth, lastHessian，
        // 之所以写在while()循环外面的意思是，无论残差是否增大了，都强制执行一次迭代？
		applyStep(lvl);

		float lambda = 0.1;
		float eps = 1e-4;
		int fails=0;

		if(printDebug)
		{
			printf("lvl %d, it %d (l=%f) %s: %.3f+%.5f -> %.3f+%.5f (%.3f->%.3f) (|inc| = %f)! \t",
					lvl, 0, lambda,
					"INITIA",
					sqrtf((float)(resOld[0] / resOld[2])),
					sqrtf((float)(resOld[1] / resOld[2])),
					sqrtf((float)(resOld[0] / resOld[2])),
					sqrtf((float)(resOld[1] / resOld[2])),
					(resOld[0]+resOld[1]) / resOld[2],
					(resOld[0]+resOld[1]) / resOld[2],
					0.0f);
			std::cout << refToNew_current.log().transpose() << " AFF " << refToNew_aff_current.vec().transpose() <<"\n";
		}

		int iteration=0;
		while(true)
		{
			Mat88f Hl = H;
			for(int i=0;i<8;i++) Hl(i,i) *= (1+lambda);     // 这里看得出来是LM方法
			Hl -= Hsc*(1/(1+lambda));
			Vec8f bl = b - bsc*(1/(1+lambda));

            // H和b都乘以(0.01f/(w[lvl]*h[lvl]))应该是为了减小数值, 使得整个方程更稳定
			Hl = wM * Hl * wM * (0.01f/(w[lvl]*h[lvl]));
			bl = wM * bl * (0.01f/(w[lvl]*h[lvl]));


			Vec8f inc;
			if(fixAffine)   // 不优化光度校正系数a,b，只优化相对pose
			{
			    // 可以看到只有左上角6*6的子矩阵块生效
				inc.head<6>() = - (wM.toDenseMatrix().topLeftCorner<6,6>() * (Hl.topLeftCorner<6,6>().ldlt().solve(bl.head<6>())));
				inc.tail<2>().setZero();    // a,b的增量为0
			}
			else
				inc = - (wM * (Hl.ldlt().solve(bl)));	//=-H^-1 * b.   正常的正规方程求解增量

            // 更新pose与光度矫正系数，不过注意的是现在更新的结果都还保留在局部变量中，也就是还未确定是否要接受这一步更新
            // 所以下面会再调用一次calcResAndGS()函数，计算更新后的残差．如果更新后残差减小了，就接受这一次更新
			SE3 refToNew_new = SE3::exp(inc.head<6>().cast<double>()) * refToNew_current;
			AffLight refToNew_aff_new = refToNew_aff_current;
			refToNew_aff_new.a += inc[6];
			refToNew_aff_new.b += inc[7];
			doStep(lvl, lambda, inc);   // 更新每个keypoint的逆深度，这是真的更新，不是存储在局部变量中


			Mat88f H_new, Hsc_new; Vec8f b_new, bsc_new;
			// 在新的相对pose，光度affine matrix，以及kpt逆深度下，再计算一次残差
			Vec3f resNew = calcResAndGS(lvl, H_new, b_new, Hsc_new, bsc_new, refToNew_new, refToNew_aff_new, false);
			Vec3f regEnergy = calcEC(lvl);  // 这个refEnergy表示了idepth regulate相关的残差，表示整张图上depth的光滑度

			float eTotalNew = (resNew[0]+resNew[1]+regEnergy[1]);
			float eTotalOld = (resOld[0]+resOld[1]+regEnergy[0]);


			bool accept = eTotalOld > eTotalNew;    // 如果新的残差小于旧的残差，那么就接受这一次迭代更新

			if(printDebug)
			{
				printf("lvl %d, it %d (l=%f) %s: %.5f + %.5f + %.5f -> %.5f + %.5f + %.5f (%.2f->%.2f) (|inc| = %f)! \t",
						lvl, iteration, lambda,
						(accept ? "ACCEPT" : "REJECT"),
						sqrtf((float)(resOld[0] / resOld[2])),
						sqrtf((float)(regEnergy[0] / regEnergy[2])),
						sqrtf((float)(resOld[1] / resOld[2])),
						sqrtf((float)(resNew[0] / resNew[2])),
						sqrtf((float)(regEnergy[1] / regEnergy[2])),
						sqrtf((float)(resNew[1] / resNew[2])),
						eTotalOld / resNew[2],
						eTotalNew / resNew[2],
						inc.norm());
				std::cout << refToNew_new.log().transpose() << " AFF " << refToNew_aff_new.vec().transpose() <<"\n";
			}

			if(accept)
			{

				if(resNew[1] == alphaK*numPoints[lvl])  // 为啥是等于号，而不是大于号，应该没这么巧吧
					snapped = true; // 当前帧相对于第0帧的位移足够大，那么就可以认定初始化成功了
				H = H_new;
				b = b_new;
				Hsc = Hsc_new;
				bsc = bsc_new;
				resOld = resNew;
				refToNew_aff_current = refToNew_aff_new;
				refToNew_current = refToNew_new;
				applyStep(lvl); // 更新每一个kpt的energy, isGood, idepth, lastHessian
				optReg(lvl);    // 更新iR，也就是fake的逆深度的ground truth
				lambda *= 0.5;
				fails=0;
				if(lambda < 0.0001) lambda = 0.0001;
			}
			else    // 如果迭代后的残差比老的残差还要大，说明优化函数陷入了困境，需要一个大的冲量跳出局部最优
			{
				fails++;
				lambda *= 4;
				if(lambda > 10000) lambda = 10000;
			}

			bool quitOpt = false;

			// 迭代退出条件，无非就是增量Δx过小，达到迭代次数，或者两次迭代残差都变大了，说明迭代陷入了死胡同
			if(!(inc.norm() > eps) || iteration >= maxIterations[lvl] || fails >= 2)
			{
				Mat88f H,Hsc; Vec8f b,bsc;

				quitOpt = true;
			}


			if(quitOpt) break;
			iteration++;
		}
		latestRes = resOld;

	}


    // 初始化最终的优化结果，两帧的相对pose(Twc)和光度矫正系数(a,b)
	thisToNext = refToNew_current;
	thisToNext_aff = refToNew_aff_current;

	for(int i=0;i<pyrLevelsUsed-1;i++)
		propagateUp(i); // 使用归一化积，从底层往上层更新每个kpt的parent的idepth




	frameID++;
	if(!snapped) snappedAt=0;

	if(snapped && snappedAt==0)
		snappedAt = frameID;    // 与第0帧相比，拥有足够位移的帧的id



    debugPlot(0,wraps);


    // 从拥有足够位移的帧往后再连续track到了5帧，这样才算初始化成功了
	return snapped && frameID > snappedAt+5;
}

void CoarseInitializer::debugPlot(int lvl, std::vector<IOWrap::Output3DWrapper*> &wraps)
{
    bool needCall = false;
    for(IOWrap::Output3DWrapper* ow : wraps)
        needCall = needCall || ow->needPushDepthImage();
    if(!needCall) return;


	int wl = w[lvl], hl = h[lvl];
	Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];

	MinimalImageB3 iRImg(wl,hl);

	for(int i=0;i<wl*hl;i++)
		iRImg.at(i) = Vec3b(colorRef[i][0],colorRef[i][0],colorRef[i][0]);


	int npts = numPoints[lvl];

	float nid = 0, sid=0;
	for(int i=0;i<npts;i++)
	{
		Pnt* point = points[lvl]+i;
		if(point->isGood)
		{
			nid++;
			sid += point->iR;
		}
	}
	float fac = nid / sid;



	for(int i=0;i<npts;i++)
	{
		Pnt* point = points[lvl]+i;

		if(!point->isGood)
			iRImg.setPixel9(point->u+0.5f,point->v+0.5f,Vec3b(0,0,0));

		else
			iRImg.setPixel9(point->u+0.5f,point->v+0.5f,makeRainbow3B(point->iR*fac));
	}


	//IOWrap::displayImage("idepth-R", &iRImg, false);
    for(IOWrap::Output3DWrapper* ow : wraps)
        ow->pushDepthImage(&iRImg);
}

// calculates residual, Hessian and Hessian-block needed for re-substituting depth.
Vec3f CoarseInitializer::calcResAndGS(
		int lvl, Mat88f &H_out, Vec8f &b_out,
		Mat88f &H_out_sc, Vec8f &b_out_sc,      // sc是schur，即舒尔补的意思
		const SE3 &refToNew, AffLight refToNew_aff,     // 两帧之间的相对pose，两帧之间的光度affine matrix
		bool plot)
{
	int wl = w[lvl], hl = h[lvl];
	Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];   // 第0帧的I,du,dv
	Eigen::Vector3f* colorNew = newFrame->dIp[lvl];     // 第1帧的I,du,dv

	Mat33f RKi = (refToNew.rotationMatrix() * Ki[lvl]).cast<float>();   // Rwc*K.inverse()
	Vec3f t = refToNew.translation().cast<float>(); // twc
	// 从第0帧到第1帧的光度映射矩阵，但其实是个单位阵
	Eigen::Vector2f r2new_aff = Eigen::Vector2f(exp(refToNew_aff.a), refToNew_aff.b);
	float fxl = fx[lvl];
	float fyl = fy[lvl];
	float cxl = cx[lvl];
	float cyl = cy[lvl];


	Accumulator11 E;    // 1*1的累加器，也就是只累加一个数
	acc9.initialize();
	E.initialize();     // initialize()函数是在为累加器里面的数组分配空间


	int npts = numPoints[lvl];  // first_frame的本层的keypoints数目
	Pnt* ptsl = points[lvl];
	for(int i=0;i<npts;i++)     // 遍历本层的所有keypoint
	{

		Pnt* point = ptsl+i;

		point->maxstep = 1e10;
		if(!point->isGood)      // setFirst()的时候在第0帧上提取出的kpt肯定都是good，但随着迭代的进行，可能有的kpt会被置为false
		{
			E.updateSingle((float)(point->energy[0]));
			point->energy_new = point->energy;
			point->isGood_new = false;
			continue;
		}

        VecNRf dp0;     // 8行1列的Eigen::Vector
        VecNRf dp1;
        VecNRf dp2;
        VecNRf dp3;
        VecNRf dp4;
        VecNRf dp5;
        VecNRf dp6;
        VecNRf dp7;
        VecNRf dd;
        VecNRf r;
		JbBuffer_new[i].setZero();

		// sum over all residuals.
		bool isGood = true;
		float energy=0;
		for(int idx=0;idx<patternNum;idx++) // patternNum = 8, 表示遍历该kpt的8个菱形角点
		{
			int dx = patternP[idx][0];  // (1,-1)等等，表示菱形角点相对于中心kpt的相对坐标
			int dy = patternP[idx][1];

            // Vec3f(point->u+dx, point->v+dy, 1)是该菱形角点的uv坐标的齐次表达式
            // 从第0帧的camera系转到第1帧的camera系
			Vec3f pt = RKi * Vec3f(point->u+dx, point->v+dy, 1) + t*point->idepth_new;
			float v = pt[1] / pt[2];
            float u = pt[0] / pt[2];    // 转到camera1的归一化平面
            float Ku = fxl * u + cxl;   // 转到camera1的像素平面
			float Kv = fyl * v + cyl;
			float new_idepth = point->idepth_new/pt[2];     // 该keypoint在camera1的相机系下的逆深度

			// 重投影位置落在了图像范围内，并且在camera1的相机系下的逆深度（深度）为正，表示在成像平面前面
			// 满足上面两个条件的话，表示在当前的相对pose下，第0帧上的这个kpt的逆深度估计是有效的
			if(!(Ku > 1 && Kv > 1 && Ku < wl-2 && Kv < hl-2 && new_idepth > 0))
			{
				isGood = false;
				break;
			}

			// 双线性插值出该kpt重投影到第1帧上后，重投影位置的I,du,dv
			Vec3f hitColor = getInterpolatedElement33(colorNew, Ku, Kv, wl);
			//Vec3f hitColor = getInterpolatedElement33BiCub(colorNew, Ku, Kv, wl);

			//float rlR = colorRef[point->u+dx + (point->v+dy) * wl][0];
			// 双线性插值出在原图（第0帧）上，该kpt的该菱形角点的I，与上个函数getInterpolatedElement33()的区别在于只计算灰度I
			float rlR = getInterpolatedElement31(colorRef, point->u+dx, point->v+dy, wl);

			// 如果在原图（第0帧）或重投影图（第1帧）双线性插值计算得到的灰度为无穷大？一般不太会发生吧
			if(!std::isfinite(rlR) || !std::isfinite((float)hitColor[0]))
			{
				isGood = false;
				break;
			}


			// 计算原图的keypoint和重投影点的灰度差，也就是直接法最终的优化目标：光度残差
			float residual = hitColor[0] - r2new_aff[0] * rlR - r2new_aff[1];   // r2new_aff是光度映射矩阵，把第0帧的灰度映射到第1帧
			// hw,huber权重，鲁棒核函数，在光度残差小于9的时候，不使用核函数免得收敛震荡；大于9的时候才启用
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
			energy += hw *residual*residual*(2-hw);     // 累加残差



            // 对逆深度求导
			float dxdd = (t[0]-t[2]*u)/pt[2];
			float dydd = (t[1]-t[2]*v)/pt[2];

			if(hw < 1) hw = sqrtf(hw);
			float dxInterp = hw*hitColor[1]*fxl;    // du*fx，梯度乘焦距
			float dyInterp = hw*hitColor[2]*fyl;
			// 对位姿(新状态)求导
			dp0[idx] = new_idepth*dxInterp;
			dp1[idx] = new_idepth*dyInterp;
			dp2[idx] = -new_idepth*(u*dxInterp + v*dyInterp);
			dp3[idx] = -u*v*dxInterp - (1+v*v)*dyInterp;
			dp4[idx] = (1+u*u)*dxInterp + u*v*dyInterp;
			dp5[idx] = -v*dxInterp + u*dyInterp;
			// 对光度参数求导
			dp6[idx] = - hw*r2new_aff[0] * rlR;
			dp7[idx] = - hw*1;
            // 对逆深度(旧状态)求导
			dd[idx] = dxInterp * dxdd  + dyInterp * dydd;
			r[idx] = hw*residual;

			float maxstep = 1.0f / Vec2f(dxdd*fxl, dydd*fyl).norm();
			if(maxstep < point->maxstep) point->maxstep = maxstep;

			// immediately compute dp*dd' and dd*dd' in JbBuffer1.
			JbBuffer_new[i][0] += dp0[idx]*dd[idx];
			JbBuffer_new[i][1] += dp1[idx]*dd[idx];
			JbBuffer_new[i][2] += dp2[idx]*dd[idx];
			JbBuffer_new[i][3] += dp3[idx]*dd[idx];
			JbBuffer_new[i][4] += dp4[idx]*dd[idx];
			JbBuffer_new[i][5] += dp5[idx]*dd[idx];
			JbBuffer_new[i][6] += dp6[idx]*dd[idx];
			JbBuffer_new[i][7] += dp7[idx]*dd[idx];
			JbBuffer_new[i][8] += r[idx]*dd[idx];
			JbBuffer_new[i][9] += dd[idx]*dd[idx];
		}

		if(!isGood || energy > point->outlierTH*20)
		{
			E.updateSingle((float)(point->energy[0]));
			point->isGood_new = false;
			point->energy_new = point->energy;
			continue;
		}


		// add into energy.
		E.updateSingle(energy);
		point->isGood_new = true;
		point->energy_new[0] = energy;

		// update Hessian matrix.
        //　因为使用128位相当于每次加4个数, 因此i+=4
		for(int i=0;i+3<patternNum;i+=4)
			acc9.updateSSE(
					_mm_load_ps(((float*)(&dp0))+i),
					_mm_load_ps(((float*)(&dp1))+i),
					_mm_load_ps(((float*)(&dp2))+i),
					_mm_load_ps(((float*)(&dp3))+i),
					_mm_load_ps(((float*)(&dp4))+i),
					_mm_load_ps(((float*)(&dp5))+i),
					_mm_load_ps(((float*)(&dp6))+i),
					_mm_load_ps(((float*)(&dp7))+i),
					_mm_load_ps(((float*)(&r))+i));


		for(int i=((patternNum>>2)<<2); i < patternNum; i++)
			acc9.updateSingle(
					(float)dp0[i],(float)dp1[i],(float)dp2[i],(float)dp3[i],
					(float)dp4[i],(float)dp5[i],(float)dp6[i],(float)dp7[i],
					(float)r[i]);


	}

	E.finish();
	acc9.finish();






	// calculate alpha energy, and decide if we cap it.
	Accumulator11 EAlpha;
	EAlpha.initialize();
	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;
		if(!point->isGood_new)
		{
			E.updateSingle((float)(point->energy[1]));
		}
		else
		{
			point->energy_new[1] = (point->idepth_new-1)*(point->idepth_new-1);
			E.updateSingle((float)(point->energy_new[1]));
		}
	}
	EAlpha.finish();
	float alphaEnergy = alphaW*(EAlpha.A + refToNew.translation().squaredNorm() * npts);

	//printf("AE = %f * %f + %f\n", alphaW, EAlpha.A, refToNew.translation().squaredNorm() * npts);


	// compute alpha opt.
	float alphaOpt;
	if(alphaEnergy > alphaK*npts)
	{
		alphaOpt = 0;
		alphaEnergy = alphaK*npts;
	}
	else
	{
		alphaOpt = alphaW;
	}


	acc9SC.initialize();
	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;
		if(!point->isGood_new)
			continue;

		point->lastHessian_new = JbBuffer_new[i][9];

		JbBuffer_new[i][8] += alphaOpt*(point->idepth_new - 1);
		JbBuffer_new[i][9] += alphaOpt;

		if(alphaOpt==0)
		{
			JbBuffer_new[i][8] += couplingWeight*(point->idepth_new - point->iR);
			JbBuffer_new[i][9] += couplingWeight;
		}

		JbBuffer_new[i][9] = 1/(1+JbBuffer_new[i][9]);
		acc9SC.updateSingleWeighted(
				(float)JbBuffer_new[i][0],(float)JbBuffer_new[i][1],(float)JbBuffer_new[i][2],(float)JbBuffer_new[i][3],
				(float)JbBuffer_new[i][4],(float)JbBuffer_new[i][5],(float)JbBuffer_new[i][6],(float)JbBuffer_new[i][7],
				(float)JbBuffer_new[i][8],(float)JbBuffer_new[i][9]);
	}
	acc9SC.finish();


	//printf("nelements in H: %d, in E: %d, in Hsc: %d / 9!\n", (int)acc9.num, (int)E.num, (int)acc9SC.num*9);
	H_out = acc9.H.topLeftCorner<8,8>();// / acc9.num;
	b_out = acc9.H.topRightCorner<8,1>();// / acc9.num;
	H_out_sc = acc9SC.H.topLeftCorner<8,8>();// / acc9.num;
	b_out_sc = acc9SC.H.topRightCorner<8,1>();// / acc9.num;



	H_out(0,0) += alphaOpt*npts;
	H_out(1,1) += alphaOpt*npts;
	H_out(2,2) += alphaOpt*npts;

	Vec3f tlog = refToNew.log().head<3>().cast<float>();
	b_out[0] += tlog[0]*alphaOpt*npts;
	b_out[1] += tlog[1]*alphaOpt*npts;
	b_out[2] += tlog[2]*alphaOpt*npts;





	return Vec3f(E.A, alphaEnergy ,E.num);
}

float CoarseInitializer::rescale()
{
	float factor = 20*thisToNext.translation().norm();
//	float factori = 1.0f/factor;
//	float factori2 = factori*factori;
//
//	for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
//	{
//		int npts = numPoints[lvl];
//		Pnt* ptsl = points[lvl];
//		for(int i=0;i<npts;i++)
//		{
//			ptsl[i].iR *= factor;
//			ptsl[i].idepth_new *= factor;
//			ptsl[i].lastHessian *= factori2;
//		}
//	}
//	thisToNext.translation() *= factori;

	return factor;
}


Vec3f CoarseInitializer::calcEC(int lvl)
{
	if(!snapped) return Vec3f(0,0,numPoints[lvl]);
	AccumulatorX<2> E;
	E.initialize();
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)     // 遍历当前层的每一个kpt
	{
		Pnt* point = points[lvl]+i;
		if(!point->isGood_new) continue;
		float rOld = (point->idepth-point->iR);     // 预测值-真值，虽然这个真值iR是fake的
		float rNew = (point->idepth_new-point->iR);
		E.updateNoWeight(Vec2f(rOld*rOld,rNew*rNew));   // 本质上就是累加

		//printf("%f %f %f!\n", point->idepth, point->idepth_new, point->iR);
	}
	E.finish();

	//printf("ER: %f %f %f!\n", couplingWeight*E.A1m[0], couplingWeight*E.A1m[1], (float)E.num.numIn1m);
	// 返回新旧两个sum和累计的个数
	return Vec3f(couplingWeight*E.A1m[0], couplingWeight*E.A1m[1], E.num);  // couplingWeight=1.0,常数
}
void CoarseInitializer::optReg(int lvl)
{
	int npts = numPoints[lvl];
	Pnt* ptsl = points[lvl];
	if(!snapped)
	{
		for(int i=0;i<npts;i++)
			ptsl[i].iR = 1;
		return;
	}


	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;
		if(!point->isGood) continue;

		float idnn[10];
		int nnn=0;
		for(int j=0;j<10;j++)
		{
			if(point->neighbours[j] == -1) continue;
			Pnt* other = ptsl+point->neighbours[j];
			if(!other->isGood) continue;
			idnn[nnn] = other->iR;
			nnn++;
		}

		if(nnn > 2)
		{
			std::nth_element(idnn,idnn+nnn/2,idnn+nnn);
			point->iR = (1-regWeight)*point->idepth + regWeight*idnn[nnn/2];
		}
	}

}



void CoarseInitializer::propagateUp(int srcLvl)
{
	assert(srcLvl+1<pyrLevelsUsed);
	// set idepth of target

	int nptss= numPoints[srcLvl];
	int nptst= numPoints[srcLvl+1];
	Pnt* ptss = points[srcLvl];
	Pnt* ptst = points[srcLvl+1];

	// set to zero.
	for(int i=0;i<nptst;i++)
	{
		Pnt* parent = ptst+i;
		parent->iR=0;
		parent->iRSumNum=0;
	}

	for(int i=0;i<nptss;i++)
	{
		Pnt* point = ptss+i;
		if(!point->isGood) continue;

		Pnt* parent = ptst + point->parent;
		parent->iR += point->iR * point->lastHessian;   //! 均值*信息矩阵 ∑ (sigma*u)
		parent->iRSumNum += point->lastHessian;         //! 新的信息矩阵 ∑ sigma
	}

	for(int i=0;i<nptst;i++)
	{
		Pnt* parent = ptst+i;
		if(parent->iRSumNum > 0)
		{
            //! 高斯归一化积后的均值
			parent->idepth = parent->iR = (parent->iR / parent->iRSumNum);
			parent->isGood = true;
		}
	}

	optReg(srcLvl+1);   // 使用附近的点来更新IR
}

void CoarseInitializer::propagateDown(int srcLvl)
{
    // 因为为这个函数的作用是用本层keypoints的逆深度初始化下一层keypoints的逆深度，所以不能是最底层．（否则它哪来的下一层？）
	assert(srcLvl>0);
	// set idepth of target

	int nptst= numPoints[srcLvl-1]; // 下一层的keypoints的数量
	Pnt* ptss = points[srcLvl];     // 本层的所有keypoints
	Pnt* ptst = points[srcLvl-1];   // 下一层的所有keypoints

	for(int i=0;i<nptst;i++)    // 遍历下一层的每个keypoint，赋给其逆深度初值
	{
		Pnt* point = ptst+i;                // 待处理的keypoint
		Pnt* parent = ptss+point->parent;   // 该keypoint的parent

		if(!parent->isGood || parent->lastHessian < 0.1) continue;
		if(!point->isGood)
		{
            // 该keypoint非good,则它不能提供任何信息，就只有把parent的idepth直接给它, 并且置为good
			point->iR = point->idepth = point->idepth_new = parent->iR;
			point->isGood=true;
			point->lastHessian=0;
		}
		else
		{
		    // 高斯归一化积融合该keypoint与parent的iR
			float newiR = (point->iR*point->lastHessian*2 + parent->iR*parent->lastHessian) / (point->lastHessian*2+parent->lastHessian);
			point->iR = point->idepth = point->idepth_new = newiR;
		}
	}
	optReg(srcLvl-1);   // 更新iR,平滑下一层的idepth
}


void CoarseInitializer::makeGradients(Eigen::Vector3f** data)
{
	for(int lvl=1; lvl<pyrLevelsUsed; lvl++)
	{
		int lvlm1 = lvl-1;
		int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

		Eigen::Vector3f* dINew_l = data[lvl];
		Eigen::Vector3f* dINew_lm = data[lvlm1];

		for(int y=0;y<hl;y++)
			for(int x=0;x<wl;x++)
				dINew_l[x + y*wl][0] = 0.25f * (dINew_lm[2*x   + 2*y*wlm1][0] +
													dINew_lm[2*x+1 + 2*y*wlm1][0] +
													dINew_lm[2*x   + 2*y*wlm1+wlm1][0] +
													dINew_lm[2*x+1 + 2*y*wlm1+wlm1][0]);

		for(int idx=wl;idx < wl*(hl-1);idx++)
		{
			dINew_l[idx][1] = 0.5f*(dINew_l[idx+1][0] - dINew_l[idx-1][0]);
			dINew_l[idx][2] = 0.5f*(dINew_l[idx+wl][0] - dINew_l[idx-wl][0]);
		}
	}
}
void CoarseInitializer::setFirst(	CalibHessian* HCalib, FrameHessian* newFrameHessian)
{

	makeK(HCalib);
	firstFrame = newFrameHessian;

	PixelSelector sel(w[0],h[0]);

	float* statusMap = new float[w[0]*h[0]];
	bool* statusMapB = new bool[w[0]*h[0]];

	float densities[] = {0.03,0.05,0.15,0.5,1};     // 金字塔越高层keypoints采样密度越高
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		sel.currentPotential = 3;
		int npts;
		if(lvl == 0)
		    // 在第0层提取keypoints，也包括2d,4d的cell内提取到的keypoints．返回的是keypoints的总数
			npts = sel.makeMaps(firstFrame, statusMap,densities[lvl]*w[0]*h[0],1,false,2);
		else
		    // 在高层提取keypoints，这里的提取就不会扩展到2d,4d的大cell了，而且每个5*5的cell的keypoint梯度阈值都是一样的
		    // 而在上面的第0层提取时，每个32*32的cell的梯度阈值都是动态变化的，等于g+gth
			npts = makePixelStatus(firstFrame->dIp[lvl], statusMapB, w[lvl], h[lvl], densities[lvl]*w[0]*h[0]);



		if(points[lvl] != nullptr) delete[] points[lvl];    // 不是空指针就delete指针指向的对象，免得内存泄漏
		points[lvl] = new Pnt[npts];    // 这里为数组分配大小为npts的空间

		// set idepth map to initially 1 everywhere.
		int wl = w[lvl], hl = h[lvl];
		Pnt* pl = points[lvl];      // 当前层的keypoints的数组的头指针
		int nl = 0;
		for(int y=patternPadding+1;y<hl-patternPadding-2;y++)   // patternPadding指原图的border
		for(int x=patternPadding+1;x<wl-patternPadding-2;x++)
		{
			//if(x==2) printf("y=%d!\n",y);
			if((lvl!=0 && statusMapB[x+y*wl]) || (lvl==0 && statusMap[x+y*wl] != 0)) // 这个像素提取出了keypoint
			{
				//assert(patternNum==9);
				pl[nl].u = x+0.1;       // 这个keypoint的uv
				pl[nl].v = y+0.1;
				pl[nl].idepth = 1;      // keypoint的深度初始化为1
				pl[nl].iR = 1;
				pl[nl].isGood=true;
				pl[nl].energy.setZero();
				pl[nl].lastHessian=0;
				pl[nl].lastHessian_new=0;
                // 在多大的cell上提取出的keypoint．如果当前不是第0层，那么就统一是5*5的cell上的,如果是第0层，那就可能是2d*2d或4d*4d的cell上提出来的
				pl[nl].my_type= (lvl!=0) ? 1 : statusMap[x+y*wl];

				Eigen::Vector3f* cpt = firstFrame->dIp[lvl] + x + y*w[lvl]; // 指向该像素的指针
				float sumGrad2=0;
				for(int idx=0;idx<patternNum;idx++) // 遍历菱形的8个点
				{
					int dx = patternP[idx][0];  // 这个菱形的第idx点的局部uv坐标
					int dy = patternP[idx][1];
					// 菱形角点的梯度
					float absgrad = cpt[dx + dy*w[lvl]].tail<2>().squaredNorm();
					sumGrad2 += absgrad;    // 统计菱形角点的梯度和，虽然这个值后面没被用到
				}

//				float gth = setting_outlierTH * (sqrtf(sumGrad2)+setting_outlierTHSumComponent);
//				pl[nl].outlierTH = patternNum*gth*gth;
//

				pl[nl].outlierTH = patternNum*setting_outlierTH;



				nl++;
                // 这一层构造出的keypoints的总数当然会少于提取出的总数,因为可能有些图像边界上的点不会被选中
				assert(nl <= npts);
			}
		}


		numPoints[lvl]=nl;  // 这一层构造出的keypoints的总数
	}
	delete[] statusMap;     // 指向数组的指针用完就赶紧delete掉，免得内存泄漏
	delete[] statusMapB;

	// 计算每个keypoint在本层最邻近的10个neighbors以及在上一层最邻近的1个neighbor
	makeNN();

	thisToNext=SE3();
	snapped = false;
	frameID = snappedAt = 0;    // 此时Initializer已经加入了first frame，因此frameID更新为0

	for(int i=0;i<pyrLevelsUsed;i++)
		dGrads[i].setZero();

}

void CoarseInitializer::resetPoints(int lvl)
{
	Pnt* pts = points[lvl];
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		pts[i].energy.setZero();
		pts[i].idepth_new = pts[i].idepth;


		if(lvl==pyrLevelsUsed-1 && !pts[i].isGood)  // 如果是最高层，而且这个点不是good．（之所以会非good是因为在gaussian-newton迭代中被判定为了outlier）
		{
			float snd=0, sn=0;
			for(int n = 0;n<10;n++)
			{
				if(pts[i].neighbours[n] == -1 || !pts[pts[i].neighbours[n]].isGood) continue;
				snd += pts[pts[i].neighbours[n]].iR;
				sn += 1;
			}

			if(sn > 0)
			{
                // 如果这个keypoint不是good，但是它有neighbors，那么就将neighbors的depth的均值赋给它，并且重新设置为good
                // 当然还要注意下这招仅仅对最高层有效哈，可能是因为最高层keypoints本身数量就很少，所以适当补充点
				pts[i].isGood=true;
				pts[i].iR = pts[i].idepth = pts[i].idepth_new = snd/sn;
			}
		}
	}
}
void CoarseInitializer::doStep(int lvl, float lambda, Vec8f inc)
{

	const float maxPixelStep = 0.25;
	const float idMaxStep = 1e10;
	Pnt* pts = points[lvl];
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		if(!pts[i].isGood) continue;


		float b = JbBuffer[i][8] + JbBuffer[i].head<8>().dot(inc);
		float step = - b * JbBuffer[i][9] / (1+lambda);


		float maxstep = maxPixelStep*pts[i].maxstep;
		if(maxstep > idMaxStep) maxstep=idMaxStep;

		if(step >  maxstep) step = maxstep;
		if(step < -maxstep) step = -maxstep;

		float newIdepth = pts[i].idepth + step;
		if(newIdepth < 1e-3 ) newIdepth = 1e-3;
		if(newIdepth > 50) newIdepth = 50;
		pts[i].idepth_new = newIdepth;
	}

}
void CoarseInitializer::applyStep(int lvl)
{
	Pnt* pts = points[lvl];
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		if(!pts[i].isGood)
		{
			pts[i].idepth = pts[i].idepth_new = pts[i].iR;
			continue;
		}
		pts[i].energy = pts[i].energy_new;  // 更新当前层每一个kpt的energy, isGood, idepth, lastHessian
		pts[i].isGood = pts[i].isGood_new;
		pts[i].idepth = pts[i].idepth_new;
		pts[i].lastHessian = pts[i].lastHessian_new;
	}
	std::swap<Vec10f*>(JbBuffer, JbBuffer_new);
}

void CoarseInitializer::makeK(CalibHessian* HCalib)
{
	w[0] = wG[0];   // 640
	h[0] = hG[0];   // 480

	fx[0] = HCalib->fxl();
	fy[0] = HCalib->fyl();
	cx[0] = HCalib->cxl();  // 320
	cy[0] = HCalib->cyl();  // 240

	for (int level = 1; level < pyrLevelsUsed; ++ level)
	{
		w[level] = w[0] >> level;
		h[level] = h[0] >> level;
		fx[level] = fx[level-1] * 0.5;
		fy[level] = fy[level-1] * 0.5;
		cx[level] = (cx[0] + 0.5) / ((int)1<<level) - 0.5;
		cy[level] = (cy[0] + 0.5) / ((int)1<<level) - 0.5;
	}

	for (int level = 0; level < pyrLevelsUsed; ++ level)
	{
		K[level]  << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
		Ki[level] = K[level].inverse();
		fxi[level] = Ki[level](0,0);
		fyi[level] = Ki[level](1,1);
		cxi[level] = Ki[level](0,2);
		cyi[level] = Ki[level](1,2);
	}
}




void CoarseInitializer::makeNN()
{
	const float NNDistFactor=0.05;

	typedef nanoflann::KDTreeSingleIndexAdaptor<
			nanoflann::L2_Simple_Adaptor<float, FLANNPointcloud> ,
			FLANNPointcloud,2> KDTree;

	// build indices
	FLANNPointcloud pcs[PYR_LEVELS];    // 用pyrLevelsUsed也没问题吧，这里用PYR_LEVELS是因为算法最多只会处理PYR_LEVELS层，也就是6层
	KDTree* indexes[PYR_LEVELS];
	for(int i=0;i<pyrLevelsUsed;i++)
	{
		pcs[i] = FLANNPointcloud(numPoints[i], points[i]);  // 这一层上keypoints的数目，keypoints的指针
		indexes[i] = new KDTree(2, pcs[i], nanoflann::KDTreeSingleIndexAdaptorParams(5) );
		indexes[i]->buildIndex();
	}

	const int nn=10;

	// find NN & parents
	for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
	{
		Pnt* pts = points[lvl];
		int npts = numPoints[lvl];

		int ret_index[nn];
		float ret_dist[nn];
		nanoflann::KNNResultSet<float, int, int> resultSet(nn);
		nanoflann::KNNResultSet<float, int, int> resultSet1(1);

		for(int i=0;i<npts;i++)     // 遍历当前层的每一个keypoint
		{
			//resultSet.init(pts[i].neighbours, pts[i].neighboursDist );
			resultSet.init(ret_index, ret_dist);
			Vec2f pt = Vec2f(pts[i].u,pts[i].v);
			// 在当前层上找10个最近邻neighbors
			indexes[lvl]->findNeighbors(resultSet, (float*)&pt, nanoflann::SearchParams());
			int myidx=0;
			float sumDF = 0;
			for(int k=0;k<nn;k++)   // 遍历10个neighbor
			{
				pts[i].neighbours[myidx]=ret_index[k];      // 该neightbor在当前层keypoints序列中的id
				float df = expf(-ret_dist[k]*NNDistFactor); // pts[i]到该neighbor的像素距离
				sumDF += df;
				pts[i].neighboursDist[myidx]=df;
				assert(ret_index[k]>=0 && ret_index[k] < npts); // id当然大于等于0并且小于keypoints总数
				myidx++;
			}
			for(int k=0;k<nn;k++)
				pts[i].neighboursDist[k] *= 10/sumDF;   // pts[i]到每个neighbor的像素距离都乘上一个系数


			if(lvl < pyrLevelsUsed-1 )  // 如果不是最高层，再在上一层寻找离它最近的keypoint，记为该它的parent
			{
				resultSet1.init(ret_index, ret_dist);
				pt = pt*0.5f-Vec2f(0.25f,0.25f);    // 该keypoint在上一层的uv坐标
				indexes[lvl+1]->findNeighbors(resultSet1, (float*)&pt, nanoflann::SearchParams());

				pts[i].parent = ret_index[0];   // parent在上一层keypoints序列中的id
				pts[i].parentDist = expf(-ret_dist[0]*NNDistFactor);   // 在上一层图像空间上，parent到该keypoint的像素距离

				assert(ret_index[0]>=0 && ret_index[0] < numPoints[lvl+1]);
			}
			else
			{
				pts[i].parent = -1;     // 如果已经是最高层了，那么它没有parent
				pts[i].parentDist = -1;
			}
		}
	}



	// done.

	for(int i=0;i<pyrLevelsUsed;i++)
		delete indexes[i];  // delete声明的指向kdtree的指针
}
}

