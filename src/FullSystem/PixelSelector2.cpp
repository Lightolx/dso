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


#include "FullSystem/PixelSelector2.h"
 
// 



#include "util/NumType.h"
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include "FullSystem/HessianBlocks.h"
#include "util/globalFuncs.h"

namespace dso
{


PixelSelector::PixelSelector(int w, int h)
{
	randomPattern = new unsigned char[w*h];
	std::srand(3141592);	// want to be deterministic.
	for(int i=0;i<w*h;i++) randomPattern[i] = rand() & 0xFF;    // 取得低8位，也就是0~255

	currentPotential=3;


	gradHist = new int[100*(1+w/32)*(1+h/32)];
	ths = new float[(w/32)*(h/32)+100];
	thsSmoothed = new float[(w/32)*(h/32)+100];

	allowFast=false;
	gradHistFrame=nullptr;
}

PixelSelector::~PixelSelector()
{
	delete[] randomPattern;
	delete[] gradHist;
	delete[] ths;
	delete[] thsSmoothed;
}

// 返回直方图的平均梯度
int computeHistQuantil(int* hist, float below)
{
	int th = hist[0]*below+0.5f;    // 直方图所有bin内点的总数的一半
	for(int i=0;i<90;i++)
	{
		th -= hist[i+1];
		if(th<0) return i;  // 从第1个到第i个bin，包含的点数恰好过半，那也就说明了i是直方图的平均梯度
	}
	return 90;
}


void PixelSelector::makeHists(const FrameHessian* const fh)
{
	gradHistFrame = fh;
	float * mapmax0 = fh->absSquaredGrad[0];    // 第0层的图像梯度

	int w = wG[0];
	int h = hG[0];

	int w32 = w/32;     // w32 = 20
	int h32 = h/32;     // h32 = 15
	thsStep = w32;

	for(int y=0;y<h32;y++)          // 遍历每个格子
		for(int x=0;x<w32;x++)
		{
			float* map0 = mapmax0+32*x+32*y*w;      // 指向当前格子的左上角
			int* hist0 = gradHist;// + 50*(x+y*w32);
			memset(hist0,0,sizeof(int)*50);     // 初始化直方图，50个bin全部设置为0

			for(int j=0;j<32;j++) for(int i=0;i<32;i++)     // 遍历32*32的格子内的每一个像素，生成梯度直方图
			{
				int it = i+32*x;    // 在原图中的uv坐标
				int jt = j+32*y;
                // border设置为2个pixel，也就是边缘点不计算梯度。（注意这个边缘是原图的边缘，而不是该格子的边缘）
				if(it>w-2 || jt>h-2 || it<1 || jt<1) continue;
				int g = sqrtf(map0[i+j*w]);     // 该像素点的梯度
				if(g>48) g=48;  // 因为只有49个bin(有一个专门留出来用于统计numPixels),所以梯度最大为48（范围为0~48）
				hist0[g+1]++;   // 第1个到第49个bin统计处于该梯度范围的像素的个数
				hist0[0]++;     // 第一个bin专门留出来统计该格子内像素点的个数
			}

			// 被选为keypoint的梯度阈值, g+gth，g是格子内所有像素的梯度均值，gth是安全距离，一般设置为7
			ths[x+y*w32] = computeHistQuantil(hist0,setting_minGradHistCut) + setting_minGradHistAdd;
		}

	// 在上一步ths的基础上，求以该cell为中心，附近3*3的cell的阈值的均值，就想做了个kerner_size=3的高斯模糊
	for(int y=0;y<h32;y++)
		for(int x=0;x<w32;x++)
		{
			float sum=0,num=0;
			if(x>0)
			{
				if(y>0) 	{num++; 	sum+=ths[x-1+(y-1)*w32];}   // 左上角的cell的梯度阈值
				if(y<h32-1) {num++; 	sum+=ths[x-1+(y+1)*w32];}   // 左下角的cell的梯度阈值
				num++; sum+=ths[x-1+(y)*w32];   // 左边的cell的梯度阈值
			}

			if(x<w32-1)
			{
				if(y>0) 	{num++; 	sum+=ths[x+1+(y-1)*w32];}   // 右上角的cell的梯度阈值
				if(y<h32-1) {num++; 	sum+=ths[x+1+(y+1)*w32];}   // 右下角的cell的梯度阈值
				num++; sum+=ths[x+1+(y)*w32];   // 右边的cell的梯度阈值
			}

			if(y>0) 	{num++; 	sum+=ths[x+(y-1)*w32];}     // 上面的cell的梯度阈值
			if(y<h32-1) {num++; 	sum+=ths[x+(y+1)*w32];}     // 下面的cell的梯度阈值
			num++; sum+=ths[x+y*w32];   // 当前cell的keypoint梯度阈值

			thsSmoothed[x+y*w32] = (sum/num) * (sum/num);   // 注意这里还是ths的平方

		}





}
int PixelSelector::makeMaps(
		const FrameHessian* const fh,
		float* map_out, float density, int recursionsLeft, bool plot, float thFactor)
{
	float numHave=0;
	float numWant=density;
	float quotia;
	int idealPotential = currentPotential;


/*//	if(setting_pixelSelectionUseFast>0 && allowFast)
//	{
//		memset(map_out, 0, sizeof(float)*wG[0]*hG[0]);
//		std::vector<cv::KeyPoint> pts;
//		cv::Mat img8u(hG[0],wG[0],CV_8U);
//		for(int i=0;i<wG[0]*hG[0];i++)
//		{
//			float v = fh->dI[i][0]*0.8;
//			img8u.at<uchar>(i) = (!std::isfinite(v) || v>255) ? 255 : v;
//		}
//		cv::FAST(img8u, pts, setting_pixelSelectionUseFast, true);
//		for(unsigned int i=0;i<pts.size();i++)
//		{
//			int x = pts[i].pt.x+0.5;
//			int y = pts[i].pt.y+0.5;
//			map_out[x+y*wG[0]]=1;
//			numHave++;
//		}
//
//		printf("FAST selection: got %f / %f!\n", numHave, numWant);
//		quotia = numWant / numHave;
//	}
//	else*/
	{




		// the number of selected pixels behaves approximately as
		// K / (pot+1)^2, where K is a scene-dependent constant.
		// we will allow sub-selecting pixels by up to a quotia of 0.25, otherwise we will re-select.
        // 计算每个32*32的网格，提取keypoint的梯度的阈值
		if(fh != gradHistFrame) makeHists(fh);  // 这个指针判断没啥必要吧，新的帧肯定跟上一帧不一样

		// select! map_out表示最终输出、与原图size相同的4值图像，表示这个像素是否被选为了keypoint
		// currentPotential = 3, 表示d*d的网格的grid_size
        // thFactor = 2，本质上也是一个安全系数，表示某个像素的梯度只有大于阈值的两倍才被选为keypoint
		Eigen::Vector3i n = this->select(fh, map_out,currentPotential, thFactor);

		// sub-select!
		numHave = n[0]+n[1]+n[2];   // 第0,第1,第2层提取出的所有keypoints的总数
		quotia = numWant / numHave; // 想要几个 / 实际提取到了多少个keypoints
		// by default we want to over-sample by 40% just to be sure.
		float K = numHave * (currentPotential+1) * (currentPotential+1);
		idealPotential = sqrtf(K/numWant)-1;	// round down.
		if(idealPotential<1) idealPotential=1;

        // recursionsLeft是指剩余的迭代次数，一般为了性能考虑，我们只允许迭代1次去提取更多的keypoints
        // quotia是指numWant / numHave，想提取125个点，但实际只提取出了100个点，那么这时候就需要再提取一次了
        // currentPotential是指d*d的网格的grid_size，grid_size当然要大于1，要不最后就成了逐像素遍历了，从性能上来说这太慢了
		if( recursionsLeft>0 && quotia > 1.25 && currentPotential>1)
		{
			//re-sample to get more points!
			// potential needs to be smaller
			if(idealPotential>=currentPotential)    // 这一次迭代,d*d的网格的grid_size当然应该比上一次小，这样才能保证提取的keypoints更多
				idealPotential = currentPotential-1;

	//		printf("PixelSelector: have %.2f%%, need %.2f%%. RESAMPLE with pot %d -> %d.\n",
	//				100*numHave/(float)(wG[0]*hG[0]),
	//				100*numWant/(float)(wG[0]*hG[0]),
	//				currentPotential,
	//				idealPotential);
			currentPotential = idealPotential;
			// todo::这里不把map_out清零的话，会不会有什么问题
			// answer:不用清零，在PixelSelector::select()函数里面一进去就把map_out给清零了
			return makeMaps(fh,map_out, density, recursionsLeft-1, plot,thFactor);
		}
        // quotia是指numWant / numHave，想提取25个点，但实际只提取出了100个点，那么这时候就需要把网格划大点再提取一次了
		else if(recursionsLeft>0 && quotia < 0.25)
		{
			// re-sample to get less points!

			if(idealPotential<=currentPotential)    // d*d的网格的grid_size比上一轮迭代要大
				idealPotential = currentPotential+1;

	//		printf("PixelSelector: have %.2f%%, need %.2f%%. RESAMPLE with pot %d -> %d.\n",
	//				100*numHave/(float)(wG[0]*hG[0]),
	//				100*numWant/(float)(wG[0]*hG[0]),
	//				currentPotential,
	//				idealPotential);
			currentPotential = idealPotential;
			return makeMaps(fh,map_out, density, recursionsLeft-1, plot,thFactor);

		}
	}

	int numHaveSub = numHave;   // 提取出的keypoints的总数
    // 如果0.25 < quotia < 0.95，也就是虽然多提取了keypoints但多得不多，那么就随机删去一些keypoints
    // 如果0.95 < quotia < 1.25呢？那就说明差的不多，就不管了呀，不再新提取一轮，也不手动删除了
	if(quotia < 0.95)
	{
		int wh=wG[0]*hG[0];
		int rn=0;
        // 删除率，比如quotia = 0.90，那么删除率就应该是10%，这里的做法是如果一个0~255的随机数大于0.9*255，那么就认为应该删除
		unsigned char charTH = 255*quotia;
		for(int i=0;i<wh;i++)   // 遍历每个像素
		{
			if(map_out[i] != 0)     // 这个点被选为了keypoint，不管在哪一层
			{
				if(randomPattern[rn] > charTH )
				{
					map_out[i]=0;   // 把这个keypoint去除掉
					numHaveSub--;   // keypoints总数减1
				}
				rn++;
			}
		}
	}

//	printf("PixelSelector: have %.2f%%, need %.2f%%. KEEPCURR with pot %d -> %d. Subsampled to %.2f%%\n",
//			100*numHave/(float)(wG[0]*hG[0]),
//			100*numWant/(float)(wG[0]*hG[0]),
//			currentPotential,
//			idealPotential,
//			100*numHaveSub/(float)(wG[0]*hG[0]));
	currentPotential = idealPotential;      // 本帧比较理想的d*d的grid_size，下一帧提取keypoint也使用这个grid_size


	if(plot)
	{
		int w = wG[0];
		int h = hG[0];


		MinimalImageB3 img(w,h);

		for(int i=0;i<w*h;i++)
		{
			float c = fh->dI[i][0]*0.7;
			if(c>255) c=255;
			img.at(i) = Vec3b(c,c,c);
		}
		IOWrap::displayImage("Selector Image", &img);

		for(int y=0; y<h;y++)
			for(int x=0;x<w;x++)
			{
				int i=x+y*w;
				if(map_out[i] == 1)
					img.setPixelCirc(x,y,Vec3b(0,255,0));
				else if(map_out[i] == 2)
					img.setPixelCirc(x,y,Vec3b(255,0,0));
				else if(map_out[i] == 4)
					img.setPixelCirc(x,y,Vec3b(0,0,255));
			}
		IOWrap::displayImage("Selector Pixels", &img);
	}

	return numHaveSub;
}



Eigen::Vector3i PixelSelector::select(const FrameHessian* const fh,
		float* map_out, int pot, float thFactor)
{

	Eigen::Vector3f const * const map0 = fh->dI;

	// 金字塔底下3层的梯度，在3.2节step1中可以看到，首先在原图上提取keypoints，然后resize图像到1/2后再提取两次
	float * mapmax0 = fh->absSquaredGrad[0];
	float * mapmax1 = fh->absSquaredGrad[1];
	float * mapmax2 = fh->absSquaredGrad[2];


	int w = wG[0];      // 底下3层图像的宽
	int w1 = wG[1];
	int w2 = wG[2];
	int h = hG[0];

    // 为了让每个网格(d*d，或者2d*2d，或者4d*4d)内提取的keypoint的梯度方向尽可能均匀分布在0~2*pi，给每个网格随机分配一个主方向，
    // 然后选出该网格内与主方向的夹角最小的keypoint作为该网格最终的keypoint（注意一个网格只会提取一个keypoint）
	const Vec2f directions[16] = {
	         Vec2f(0,    1.0000),
	         Vec2f(0.3827,    0.9239),      // sin(22.5°)
	         Vec2f(0.1951,    0.9808),      // sin(11.0°)
	         Vec2f(0.9239,    0.3827),      // sin(67.5°)
	         Vec2f(0.7071,    0.7071),      // sin(45°)
	         Vec2f(0.3827,   -0.9239),
	         Vec2f(0.8315,    0.5556),      // sin(56.2°)
	         Vec2f(0.8315,   -0.5556),
	         Vec2f(0.5556,   -0.8315),
	         Vec2f(0.9808,    0.1951),
	         Vec2f(0.9239,   -0.3827),
	         Vec2f(0.7071,   -0.7071),
	         Vec2f(0.5556,    0.8315),
	         Vec2f(0.9808,   -0.1951),
	         Vec2f(1.0000,    0.0000),
	         Vec2f(0.1951,   -0.9808)};
    // status初始化，全部设置为非keypoint
	memset(map_out,0,w*h*sizeof(PixelSelectorStatus));  // sizeof(PixelSelectorStatus) = 4，表示4个字节，和1个int一样，其实感觉用1个bit就够了啊


	float dw1 = setting_gradDownweightPerLevel;     // 0.75
	float dw2 = dw1*dw1;                            // 0.5625

	int n3=0, n2=0, n4=0;
	for(int y4=0;y4<h;y4+=(4*pot)) for(int x4=0;x4<w;x4+=(4*pot))   //　将原图划分成大小为4d*4d的网格，遍历每个网格
	{
        // 表示这个4d*4d的格子的大小，一般都是4d，除非是走到了边缘而图像宽高不是4d的整数倍
		int my3 = std::min((4*pot), h-y4);
		int mx3 = std::min((4*pot), w-x4);
		int bestIdx4=-1; float bestVal4=0;
		Vec2f dir4 = directions[randomPattern[n2] & 0xF];   // 只取最低8位，也就是0~15的值
		// 将4d*4d的网格划分为4个2d*2d的网格，遍历每个2d*2d网格
		for(int y3=0;y3<my3;y3+=(2*pot)) for(int x3=0;x3<mx3;x3+=(2*pot))
		{
			int x34 = x3+x4;    // 这个2d*2d网格的左上角的uv坐标
			int y34 = y3+y4;
            // 这个2d*2d的格子的大小，一般都是2d，除非原先的4d*4d的网格走到了图像边缘导致grid_size小于4d
			int my2 = std::min((2*pot), h-y34);
			int mx2 = std::min((2*pot), w-x34);
			int bestIdx3=-1; float bestVal3=0;
			Vec2f dir3 = directions[randomPattern[n2] & 0xF];   // 这里也是只取低8位，也就是0~15的值
            // 将2d*2d的网格划分为4个d*d的网格，遍历每个d*d网格
			for(int y2=0;y2<my2;y2+=pot) for(int x2=0;x2<mx2;x2+=pot)   // d*d的区域遍历
			{
				int x234 = x2+x34;  // 这个d*d网格的左上角的uv坐标
				int y234 = y2+y34;
                // 这个d*d的格子的大小，一般都是d，除非原先的2d*2d的网格grid_size小于2d
				int my1 = std::min(pot, h-y234);
				int mx1 = std::min(pot, w-x234);
				int bestIdx2=-1; float bestVal2=0;
				Vec2f dir2 = directions[randomPattern[n2] & 0xF];
                // 在d*d的网格内逐像素遍历
				for(int y1=0;y1<my1;y1+=1) for(int x1=0;x1<mx1;x1+=1)
				{
					assert(x1+x234 < w);    // 该像素的u坐标
					assert(y1+y234 < h);    // 该像素的v坐标
					int idx = x1+x234 + w*(y1+y234);
					int xf = x1+x234;   // u
					int yf = y1+y234;   // v

					// border = 4，也就是在原图的4个pixel的边缘内不提取keypoints
					if(xf<4 || xf>=w-5 || yf<4 || yf>h-4) continue;

                    // 当前像素属于第几个32*32的网格，然后再取出该网格的keypoint的梯度阈值
					float pixelTH0 = thsSmoothed[(xf>>5) + (yf>>5) * thsStep];  // 右移5位表示除以32
                    // 在高层提取keypoint的时候，梯度阈值应该小一点，因为我们之所以要在高层提取keypoints就是为了利用那些弱纹理的区域
					float pixelTH1 = pixelTH0*dw1;
					float pixelTH2 = pixelTH1*dw2;


					float ag0 = mapmax0[idx];   // 在第0层上该点的梯度
					if(ag0 > pixelTH0*thFactor) // 如果梯度大于阈值，那么这个点就选为keypoint
					{
						Vec2f ag0d = map0[idx].tail<2>();   // 这个点的[du,dv]
						float dirNorm = fabsf((float)(ag0d.dot(dir2)));     // 计算这个点的梯度方向与该d*d的网格的主方向的夹角
						if(!setting_selectDirectionDistribution) dirNorm = ag0;

                        // 因为一个网格内只会提取出一个keypoint，所以找出这些keypoints中与d*d的网格的主方向夹角最小的keypoint
                        // （其实这个主方向是假的，是随机给定的，所以一般来说我们只需要找出梯度最大的点就可以了）
						if(dirNorm > bestVal2)
						{ bestVal2 = dirNorm; bestIdx2 = idx; bestIdx3 = -2; bestIdx4 = -2;}
					}
                    // 如果这个像素在d*d的网格内已经超过了keypoint的梯度阈值，说明它的梯度足够大，那么就不用在高层提取了
                    // 从这里可以看到，一旦该d*d的网格内的任何一个像素在第0层提取出了keypoint，那么就不在上层的2d*2d网格内提取第1层的keypoint了
                    // todo::可以看到这里会有漏网之鱼，也就是在第0层还没提取出keypoint的时候，还可能在第1层上尝试提取,而如果之后在第0层上又提取了出来，
                    //  那么按道理来说之前的在第1层提取的keypoint应该作废掉
                    // answer: 其实不会有漏网之鱼，一旦在2d*2d的网格内的任意一个像素处提取了第0层的keypoint，那么bestIdx3就会被置为-2，那么在
                    // 最后统计第1层的keypoint的时候就不会被计算进去
					if(bestIdx3==-2) continue;

                    // 在第1层上该点的梯度
					float ag1 = mapmax1[(int)(xf*0.5f+0.25f) + (int)(yf*0.5f+0.25f)*w1];
					if(ag1 > pixelTH1*thFactor)
					{
						Vec2f ag0d = map0[idx].tail<2>();   // 在第1层上这个点的[du,dv]
						float dirNorm = fabsf((float)(ag0d.dot(dir3)));
						if(!setting_selectDirectionDistribution) dirNorm = ag1;

						if(dirNorm > bestVal3)
						{ bestVal3 = dirNorm; bestIdx3 = idx; bestIdx4 = -2;}
					}
					if(bestIdx4==-2) continue;

					float ag2 = mapmax2[(int)(xf*0.25f+0.125) + (int)(yf*0.25f+0.125)*w2];
					if(ag2 > pixelTH2*thFactor)
					{
						Vec2f ag0d = map0[idx].tail<2>();
						float dirNorm = fabsf((float)(ag0d.dot(dir4)));
						if(!setting_selectDirectionDistribution) dirNorm = ag2;

						if(dirNorm > bestVal4)
						{ bestVal4 = dirNorm; bestIdx4 = idx; }
					}
				}

				if(bestIdx2>0)  // 这个d*d的格子内提取到了第0层的keypoint
				{
					map_out[bestIdx2] = 1;  // 这个2值图置为1，表示在该点提取keypoint
					bestVal3 = 1e10;
					n2++;
				}
			}

			if(bestIdx3>0)  // 说明4个d*d的网格都没有提取到第0层的keypoint，但该2d*2d的网格提取到了第1层的keypoint
			{
				map_out[bestIdx3] = 2;
				bestVal4 = 1e10;
				n3++;
			}
		}

		if(bestIdx4>0)  // 说明4个2d*2d的网格都没有提取到第1层的keypoint，但该4d*4d的网格提取到了第2层的keypoint
		{
			map_out[bestIdx4] = 4;
			n4++;
		}
	}


	return Eigen::Vector3i(n2,n3,n4);   // 分别从第0,1,2层提取出了多少个keypoint
}


}

