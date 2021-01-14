#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <random>
#include <string>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/videoio.hpp"
// !OpenCV

// Depth Camera
#include <OpenNI.h>
#include "OniSampleUtilities.h"
// !Depth Camera

// B-Spline
#include "BSplineBasisFunction.h"
// !B-Spline

using namespace cv;
using namespace std;
using namespace openni;

//
// Macors and Global Variables
//

// Depth Camera
#define MIN_DISTANCE 20  // unit : mm
#define MAX_DISTANCE 4000 // unit : mm
#define RESOULTION_X 640.0  // Resolution
#define RESOULTION_Y 480.0  // Resolution

#define SAMPLE_READ_WAIT_TIMEOUT 2000 //2000ms

// function main: 
int SizeSparseCloud_ = 20000;
int SizeSparseNormal_User_ = 10000;
// !function main 

// evaluation with multiple distances
int DistanceID_ = 0;
// !evaluation with multiple distances

// the RGB image 
Mat image_RGB_;
// the depth image 
Mat img3D_;
vector<Point3f> pointCloud_depthCam_; // original 3D point cloud from the depth camera
// !Depth Camera


// Normal Estiamtion
// a class of normal vector
class NorVec
{
	public:
		float x,y,z;
		float alpha;
	public:
		inline NorVec():x(0),y(0),z(0),alpha(0) {}
		inline NorVec(float x_,float y_,float z_,float alpha_){
			x		= x_;
			y 		= y_;
			z 		= z_;
			alpha	= alpha_;
		}
};
// the map of normal vectors
vector<NorVec> NorMap_RGBCam_;
vector<NorVec> NorMap_depthCam_;
// !Normal Estiamtion

// Normal Estiamtion: Control Group
vector<NorVec> NorMap_RGBCam_Control_;
// !Normal Estiamtion: Control Group

// Normal Vector type 2
struct NormalVector
{
	Point3f pt;
	Point3f normal;
	int flag;
};
// !Normal Vector type 2

vector<NormalVector> FullMap_;

// opencv taking points by mouse 
// mouse click
double mouseUse_width_  = 0;
double mouseUse_height_ = 0;
string WinName = "normalAnalysis";
// !mouse click

void On_mouse(int event, int x, int y, int flags, void*)
{
	/*if (event == EVENT_LBUTTONUP ||!( flags&EVENT_FLAG_LBUTTON))
	{
		previousPoint = Point(-1, -1);
	}*/
	if (event == EVENT_LBUTTONDOWN)
	{
		Point currentPoint = Point(x, y);

		string text = "(" + to_string(x) + ", " + to_string(y) + ")";

		int font_face = cv::FONT_HERSHEY_COMPLEX;
		double font_scale = 0.5;

		int thickness = 1;
		int lineType = 1;

		putText(image_RGB_, text, currentPoint, font_face, font_scale, Scalar(0, 0, 255), thickness, lineType);

		//
		// draw axes
		//
		int line_thickness = 1;

		int idx = y*image_RGB_.cols+x;
		cout<<image_RGB_.cols<<endl;

		double n_z1 = FullMap_[idx].normal.x;
		double n_z2 = FullMap_[idx].normal.y;
		double n_z3 = FullMap_[idx].normal.z;
		double alpha = FullMap_[idx].flag * 255;
		cout<<"n_z(original): "<<n_z1<<", "<<n_z2<<", "<<n_z3<<endl;

		Point3f n_z(n_z1, n_z2, n_z3);

		if(n_z1==0&&n_z2==0&&n_z3==0)
		{
			cout<<"No normal observed"<<endl;
			imshow(WinName, image_RGB_);
			return;
		}

		// n_z, which is the normal vector
		double scale_nz = 20;
		double norm_nz_2d = sqrt((n_z1*n_z1) + (n_z2*n_z2));
		if(norm_nz_2d<=0)
		{
			n_z1 = 0;
            n_z2 = 0;
		}
		else
		{
			n_z1 = n_z1/(norm_nz_2d)*scale_nz;
			n_z2 = n_z2/(norm_nz_2d)*scale_nz;
		}

		cout<<"n_z X, Y: "<<n_z1<<", "<<n_z2<<endl;

		Point pt_nz = Point(int(n_z1), int(n_z2));
		pt_nz.x += x;
		pt_nz.y += y;
		line(image_RGB_, currentPoint, pt_nz, Scalar(0, 0, 255), line_thickness);
		//cout<<"pt_nz: "<<pt_nz<<endl;


		// n_x
		double n_x1 = 1.;
		double n_x2 = 0.;
		double n_x3 = -1.*n_z1/n_z3;
		Point3f n_x(n_x1, n_x2, n_x3);

		double norm_nx_2d = sqrt((n_x1*n_x1) + (n_x2*n_x2));
		n_x1 = n_x1/norm_nx_2d*scale_nz;
		n_x2 = n_x2/norm_nx_2d*scale_nz;

		Point pt_nx = Point(int(n_x1), int(n_x2));
		pt_nx.x += x;
		pt_nx.y += y;
		line(image_RGB_, currentPoint, pt_nx, Scalar(0, 255, 0), line_thickness);

		// n_y
		Point3f n_y(0,0,0);
		n_y = n_z.cross(n_x);

		double n_y1 = n_y.x;
		double n_y2 = n_y.y;

		double norm_ny_2d = sqrt((n_y1*n_y1) + (n_y2*n_y2));
		n_y1 = n_y1/norm_ny_2d*scale_nz;
		n_y2 = n_y2/norm_ny_2d*scale_nz;

		Point pt_ny = Point(int(n_y1), int(n_y2));
		pt_ny.x += x;
		pt_ny.y += y;
		line(image_RGB_, currentPoint, pt_ny, Scalar(255, 0, 0), line_thickness);

		imshow(WinName, image_RGB_);
	}
}
// !opencv taking points by mouse 

// Normal Estiamtion
void GetNormal(int size, Point3f points[], Point3f &normal)
{
	//
	// method 24
	//

	// step 0 : if no point inputed, set the normal (0,0,0)
	if(size==0)
	{
		normal.x = 0;
		normal.y = 0;
		normal.z = 0;
		return;
	}

	// step 1 : get expectation of coordinates of the points
	float x_mean = 0;
	float y_mean = 0;
	float z_mean = 0;

	for(int i=0;i<size;i++)
	{
		x_mean += points[i].x;
		y_mean += points[i].y;
		z_mean += points[i].z;
	}
	x_mean /= float(size);
    y_mean /= float(size);
    z_mean /= float(size);

	//cout<<"size: "<<size<<endl;
	//cout<<"mean: "<<x_mean<<", "<<y_mean<<", "<<z_mean<<", "<<endl;

	//
	// step 2 : Covariance matrix
	//
	Mat Cov = Mat::zeros(3,3,CV_32F); //Covariance matrix
	//cout<<"Cov\n"<<Cov<<endl;

	for(int i=0;i<size;i++)
	{
		points[i].x -= x_mean;
    	points[i].y -= y_mean;
		points[i].z -= z_mean;
		//cout<<"points[i]: "<<points[i]<<endl;

		Mat m = Mat::zeros(3,1,CV_32F);
		m.ptr<float>(0)[0] = points[i].x;
		m.ptr<float>(1)[0] = points[i].y;
		m.ptr<float>(2)[0] = points[i].z;
		//cout<<"m: "<<m<<endl;

		Cov = Cov + m*m.t();
		//cout<<"Cov: "<<Cov<<endl;
	}
	//cout<<"Cov: "<<Cov<<endl;

	Cov /= float(size);
	//cout<<"Cov: "<<Cov<<endl;


	//
	// PCA
	//
	// A (Cov)=u*w*vt
	// A (Cov): the input and decomposed matrix - Cov: \n"<<Cov<<endl<<endl;
	// u : calculated singular values: \n"<<w<<endl<<endl;
	// w : calculated left singular vectors: \n"<<u<<endl<<endl;
	// vt: transposed matrix of right singular values: \n"<<vt<<endl<<endl;
	Mat w,u,vt;
	SVD::compute(Cov, w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
	//SVD::compute(Cov, w,u,vt, cv::SVD::FULL_UV);

	// get the minimum eigen value
	int Idx_minEigVal = 0;
	float minEigVal = w.ptr<float>(0)[0];
	for(int i=0;i<2;i++)
	{
		int idx = i+1;
		float curEigVal = w.ptr<float>(idx)[0];

		if(curEigVal<minEigVal) 
		{
			minEigVal=curEigVal;
			Idx_minEigVal = idx;
		}
	}

	//cout<<"Idx_minEigVal: "<<Idx_minEigVal<<endl;
	//cout<<"minEigVal: "<<minEigVal<<endl;

	// normal
	normal.x = vt.ptr<float>(Idx_minEigVal)[0];
	normal.y = vt.ptr<float>(Idx_minEigVal)[1];
	normal.z = vt.ptr<float>(Idx_minEigVal)[2];

	//
	// direction of the normal vector
	//
	Point3f normalCamera(0,0,-1);
	double cosAngle = normal.dot(normalCamera) / (norm(normal)*norm(normalCamera));
	if(cosAngle>1)			cosAngle = 1;
	else if(cosAngle<-1)	cosAngle = -1;
	double angle = acos(cosAngle);

	if(angle>0.5*M_PI)
	{
		normal *= -1;
	}

	return;
}
// !Normal Estiamtion

// search and fix holes in the original Point cloud,
void FixHolesFor3DMap_ISD(int nRows, int nCols, vector<NormalVector> SparseMap, vector<NormalVector> &FullMap)
{
	// search and fix holes in the original Point cloud 
	// An Interpolation method is used 
	// The inversed squared distance from the target to the pixel that is found is regearded as the weight.
	// We denote the method by "ISD".

	// nRows : row number
	// nCols : column number
	// SparseMap: the original 3D map
	// FullMap: the fixed 3D map

	//
	// step 0: determine the radius for search and interpolation
	//
	int IntRad = 8; // interpolation radius

	//
	//  step 2
	//
	int counter = 0;

	for (int v = 0; v < nRows; ++v)
	for (int u = 0; u < nCols; ++u)
	{
		// step (1) : flag 
		int flag = SparseMap[v*nCols+u].flag;

		// step (2) : copy the point whose location has been observed correctly 
		if(flag>0)
		{
			int idx = v*nCols+u;
			FullMap[idx] = SparseMap[idx];
			continue;
		}

		// step (3) : compute locations by interpolation
		double xMean = 0;
		double yMean = 0;
		double zMean = 0;

		//int IntRad = 8; // interpolation radius
		double TotalWeight = 0.;
		for(int vSub=v-IntRad;vSub<v+IntRad+1;vSub++)
		for(int uSub=u-IntRad;uSub<u+IntRad+1;uSub++)
		{
			if(uSub<0) continue;
			else if(uSub>=nCols) continue;

			if(vSub<0) continue;
			else if(vSub>=nRows) continue;

			int idx = vSub*nCols+uSub;
			if(SparseMap[idx].flag==0) 
			{
				continue;
			}
			else
			{
				double dis2 = (vSub-v)*(vSub-v) + (uSub-u)*(uSub-u);

				double weight = 1./dis2;
				TotalWeight += weight;

				xMean += (SparseMap[idx].pt.x*weight);
				yMean += (SparseMap[idx].pt.y*weight);
				zMean += (SparseMap[idx].pt.z*weight);
			}
		}

		double alphaMean = 0;
		if(TotalWeight!=0)
		{
			xMean /= TotalWeight; 
        	yMean /= TotalWeight;
        	zMean /= TotalWeight;
			alphaMean = 1;
		}
		else
		{
			xMean = 0; 
        	yMean = 0;
        	zMean = 0;
			alphaMean = 0;
		}

		FullMap[v*nCols+u].pt.x	= xMean;
		FullMap[v*nCols+u].pt.y	= yMean;
		FullMap[v*nCols+u].pt.z	= zMean;
		FullMap[v*nCols+u].flag	= alphaMean;

		counter ++;
	}

	// debug
	//cout<<"FixHolesFor3DMap_ISD, number of pixels(holes) that has been fixed: "<<counter<<endl;
	//cout<<"FixHolesFor3DMap_ISD, friction of pixels(holes) out of all pixels: "<<float(counter)/float(nCols*nRows)<<endl;

	return;
}
// !search and fix holes in the original Point cloud 

// search and fix holes in the original Point cloud 
void FixHolesFor3DMap_WoBS(int nRows, int nCols, vector<NormalVector> SparseMap, vector<NormalVector> &FullMap)
{
	// search and fix holes in the original Point cloud 
	// Interpolation method is used 
	// We propose an method in which weights from B-Spline are used.
	// We denote this method by Weights of BSpline. We abbreviate this name to WoBS.

	// nRows : row number
	// nCols : column number
	// SparseMap: the original 3D map
	// FullMap: the fixed 3D map

	//
	// step 0: determine the radius for search and interpolation
	//
	int IntRad = 8; // interpolation radius

	//
	// step 1: get weights by BSpline
	//
	int degree = IntRad*2+1;

	// build a BSpline Basis Function
	vector<double> V;
	int sizeHeight = 480;
	for(int v=0; v<sizeHeight; v++)
	{
		V.push_back(v);
	}
    BSplineBasisFunction basisFuncV(V);

	// generate weights
	vector<double> Weight_BSpline;

	double ty = 50; // any index that make the point have enough margin 
	int size  = degree*2+1;
	for(int i=0;i<IntRad+1;i++)
	{
		int knotv = ty - IntRad -1 + i;
		double Bv = basisFuncV.EvalBasis(knotv, degree, ty);
		//cout<<"ty: "<<ty<<"; knotv = "<<knotv<<"; Bv = "<<Bv<<endl;

		Weight_BSpline.push_back(Bv);
	}

	for(int i=0;i<Weight_BSpline.size();i++)
	{
		cout<<"i: "<<i<<"; Weight_BSpline: "<<Weight_BSpline[i]<<endl;
	}

	//
	//  step 2
	//
	int counter = 0;

	for (int v = 0; v < nRows; ++v)
	for (int u = 0; u < nCols; ++u)
	{
		// step (1) : flag 
		int flag = SparseMap[v*nCols+u].flag;

		// step (2) : copy the point whose location has been observed correctly 
		if(flag>0)
		{
			int idx = v*nCols+u;
			FullMap[idx] = SparseMap[idx];
			continue;
		}

		// step (3) : compute locations by interpolation
		double xMean = 0;
		double yMean = 0;
		double zMean = 0;

		//int IntRad = 8; // interpolation radius
		double TotalWeight = 0.;
		for(int vSub=v-IntRad;vSub<v+IntRad+1;vSub++)
		for(int uSub=u-IntRad;uSub<u+IntRad+1;uSub++)
		{
			if(uSub<0) continue;
			else if(uSub>=nCols) continue;

			if(vSub<0) continue;
			else if(vSub>=nRows) continue;

			int idx = vSub*nCols+uSub;
			if(SparseMap[idx].flag==0) 
			{
				continue;
			}
			else
			{
				int WIdx_v = abs(vSub-v);
				int WIdx_u = abs(uSub-u);
				double weight = 0;

				if(WIdx_v<Weight_BSpline.size())
				if(WIdx_u<Weight_BSpline.size())
				{
					weight = Weight_BSpline[WIdx_v]*Weight_BSpline[WIdx_u];
				}

				TotalWeight += weight;

				xMean += (SparseMap[idx].pt.x*weight);
				yMean += (SparseMap[idx].pt.y*weight);
				zMean += (SparseMap[idx].pt.z*weight);
			}
		}

		double alphaMean = 0;
		if(TotalWeight!=0)
		{
			xMean /= TotalWeight; 
        	yMean /= TotalWeight;
        	zMean /= TotalWeight;
			alphaMean = 1;
		}
		else
		{
			xMean = 0; 
        	yMean = 0;
        	zMean = 0;
			alphaMean = 0;
		}

		FullMap[v*nCols+u].pt.x	= xMean;
		FullMap[v*nCols+u].pt.y	= yMean;
		FullMap[v*nCols+u].pt.z	= zMean;
		FullMap[v*nCols+u].flag	= alphaMean;

		counter ++;
	}

	// debug
	//cout<<"FixHolesFor3DMap_WoBS, number of pixels(holes) that has been fixed: "<<counter<<endl;
	//cout<<"FixHolesFor3DMap_WoBS, friction of pixels(holes) out of all pixels: "<<float(counter)/float(nCols*nRows)<<endl;

	return;
}
// !search and fix holes in the original Point cloud 

// Generate a normal map that consists of a portion of all normal vectors
// Square search and selection would be used
void GetSparseNormalMap_SquareSearch(int SizeSparseNormal_User,int nRows, int nCols, vector<NormalVector> OriMap, vector<NormalVector> &SparseMap)
{
	// Generate a normal map that consists of a portion of all normal vectors

	//
	// step 1 : generate a sparse normal map
	//
	vector<int> NorMapV, NorMapU;

	// sparse method 2: definite method
	double scaleFullToSparse = double(nCols)*double(nRows) / SizeSparseNormal_User;
	int SizeSpaNorCol = floor(nCols/sqrt(scaleFullToSparse));
	int SizeSpaNorRow = floor(nRows/sqrt(scaleFullToSparse));
	cout<<"nCols: "<<nCols<<endl;
	cout<<"nRows: "<<nRows<<endl;
	cout<<"SizeSparseNormal_User: "<<SizeSparseNormal_User<<endl;
	cout<<"SizeSparseNormal_Real: "<<SizeSpaNorCol*SizeSpaNorRow<<endl;
	cout<<"SizeSpaNorCol: "<<SizeSpaNorCol<<", SizeSpaNorRow: "<<SizeSpaNorRow<<endl;
	for (int i = 0; i < SizeSpaNorRow; ++i)
	for (int j = 0; j < SizeSpaNorCol; ++j)
	{
		int v = 0 + double(nCols)/SizeSpaNorCol *i;
		int u = 0 + double(nRows)/SizeSpaNorRow *j;
		NorMapV.push_back(v);
		NorMapU.push_back(u);
	}
	//cout<<"NorMapV.size(): "<<NorMapV.size()<<endl;
	//cout<<"NorMapU.size(): "<<NorMapU.size()<<endl;


	//
	// step 2 : compute normal vectors for all points in the sparse normal map 
	//
	for(int i=0;i<NorMapV.size();i++)
	{
		int v = NorMapV[i];
		int u = NorMapU[i];

		// method 2 : knn
		vector<Point3f> points_vec;
		int KNNCounter = 0;
		int TotalSizeKNN = 80;

		int SeaRad = 1; // radius of search area
		int MaxSeaRad = 5;

		// keep the point where the normal vector is
		if(OriMap[v*nCols+u].flag!=0)
		{
			points_vec.push_back(OriMap[v*nCols+u].pt);
			KNNCounter ++;
		}

		while(SeaRad<MaxSeaRad)
		{
			//cout<<"\nSeaRad: "<<SeaRad<<endl;

			//// keep the point where the normal vector is
			//if(OriMap[v*nCols+u].flag!=0)
			//{
			//	points_vec.push_back(OriMap[v*nCols+u].pt);
			//	KNNCounter ++;
			//}

			// line 1 : search alone u-axis, top
			for(int uSub=u-SeaRad+1; uSub<=u+SeaRad; uSub++)
			{
				int vSub = v-SeaRad;
				if(OriMap[v*nCols+u].flag==0) continue;

				if(uSub<0) continue;
				else if(uSub>=nCols) continue;

				if(vSub<0) continue;
				else if(vSub>=nRows) continue;
				
				points_vec.push_back(OriMap[vSub*nCols+uSub].pt);

				KNNCounter ++;

				if(KNNCounter>=TotalSizeKNN) break;

				//cout<<"Top,    Sub u,v: "<<uSub<<", "<<vSub<<endl;
			}
			if(KNNCounter>=TotalSizeKNN) break;

			// line 2 : search alone v-axis, right side
			for(int vSub=v-SeaRad+1; vSub<=v+SeaRad; vSub++)
			{
				int uSub = u+SeaRad;
				if(OriMap[v*nCols+u].flag==0) continue;

				if(uSub<0) continue;
				else if(uSub>=nCols) continue;

				if(vSub<0) continue;
				else if(vSub>=nRows) continue;
				
				points_vec.push_back(OriMap[vSub*nCols+uSub].pt);

				KNNCounter ++;

				if(KNNCounter>=TotalSizeKNN) break;

				//cout<<"Right,  Sub u,v: "<<uSub<<", "<<vSub<<endl;
			}
			if(KNNCounter>=TotalSizeKNN) break;

			// line 3 : search alone u-axis, bottom
			for(int uSub=u+SeaRad-1; uSub>=u-SeaRad; uSub--)
			{
				int vSub = v+SeaRad;
				if(OriMap[v*nCols+u].flag==0) continue;

				if(uSub<0) continue;
				else if(uSub>=nCols) continue;

				if(vSub<0) continue;
				else if(vSub>=nRows) continue;
				
				points_vec.push_back(OriMap[vSub*nCols+uSub].pt);

				KNNCounter ++;

				if(KNNCounter>=TotalSizeKNN) break;

				//cout<<"Bottom, Sub u,v: "<<uSub<<", "<<vSub<<endl;
			}
			if(KNNCounter>=TotalSizeKNN) break;

			// line 4 : search alone v-axis, left side
			for(int vSub=v+SeaRad-1; vSub>=v-SeaRad; vSub--)
			{
				int uSub = u-SeaRad;

				if(OriMap[v*nCols+u].flag==0) continue;

				if(uSub<0) continue;
				else if(uSub>=nCols) continue;

				if(vSub<0) continue;
				else if(vSub>=nRows) continue;
				
				points_vec.push_back(OriMap[vSub*nCols+uSub].pt);

				KNNCounter ++;

				if(KNNCounter>=TotalSizeKNN) break;

				//cout<<"Left,   Sub u,v: "<<uSub<<", "<<vSub<<endl;
			}
			if(KNNCounter>=TotalSizeKNN) break;

			//
			SeaRad ++;
		}
		//cout<<"Selecting Point Set, size : "<<points_vec.size()<<endl;

		//
		// compute normal vector
		//
		// if the size of selecting point set is 0, we would not comput its normal 
		if(points_vec.size()==0)
		{
			SparseMap[v*nCols+u].normal = Point3f(0,0,-1);
			SparseMap[v*nCols+u].flag   = 0;
		}
		else
		{
			// generate a list of points for normal estimation
			Point3f points[points_vec.size()];
			for(int j=0;j<points_vec.size();j++)
			{
				points[j] = points_vec[j];
			}
	
			// estimate normal vectors
			Point3f normal;
			GetNormal(points_vec.size(), points, normal);
			//cout<<"normal: "<<normal<<"; u,v: "<<u<<", "<<v<<endl;
	
			SparseMap[v*nCols+u].normal = normal;
			SparseMap[v*nCols+u].flag   = 1;
		}
	}

	return;
}

// Generate a normal map that consists of a portion of all normal vectors
// cross search and selection would be used
void GetSparseNormalMap_CrossSearch(int SizeSparseNormal_User,int nRows, int nCols, vector<NormalVector> OriMap, vector<NormalVector> &SparseMap)
{
	// Generate a normal map that consists of a portion of all normal vectors

	//
	// step 1 : generate a sparse normal map
	//
	vector<int> NorMapV, NorMapU;

	// sparse method 2: definite method
	double scaleFullToSparse = double(nCols)*double(nRows) / SizeSparseNormal_User;
	int SizeSpaNorCol = floor(nCols/sqrt(scaleFullToSparse));
	int SizeSpaNorRow = floor(nRows/sqrt(scaleFullToSparse));
	cout<<"nCols: "<<nCols<<endl;
	cout<<"nRows: "<<nRows<<endl;
	cout<<"SizeSparseNormal_User: "<<SizeSparseNormal_User<<endl;
	cout<<"SizeSparseNormal_Real: "<<SizeSpaNorCol*SizeSpaNorRow<<endl;
	cout<<"SizeSpaNorCol: "<<SizeSpaNorCol<<", SizeSpaNorRow: "<<SizeSpaNorRow<<endl;
	for (int i = 0; i < SizeSpaNorRow; ++i)
	for (int j = 0; j < SizeSpaNorCol; ++j)
	{
		int v = 0 + double(nCols)/SizeSpaNorCol *i;
		int u = 0 + double(nRows)/SizeSpaNorRow *j;
		NorMapV.push_back(v);
		NorMapU.push_back(u);
	}
	//cout<<"NorMapV.size(): "<<NorMapV.size()<<endl;
	//cout<<"NorMapU.size(): "<<NorMapU.size()<<endl;


	//
	// step 2 : compute normal vectors for all points in the sparse normal map 
	//          cross search and selection would be used
	//
	for(int i=0;i<NorMapV.size();i++)
	{
		int v = NorMapV[i];
		int u = NorMapU[i];

		// method 2 : knn
		vector<Point3f> points_vec;
		int KNNCounter = 0;
		int TotalSizeKNN = 20;

		int SeaRad = 1; // radius of search area
		int MaxSeaRad = TotalSizeKNN/4;// maximum radius of search area

		// keep the point where the normal vector is
		if(OriMap[v*nCols+u].flag!=0)
		{
			points_vec.push_back(OriMap[v*nCols+u].pt);
			KNNCounter ++;
		}

		// searching rightwards
		for(int uSub=u+1; uSub<=u+MaxSeaRad; uSub++)
		{
			int vSub = v;
			if(OriMap[v*nCols+u].flag==0) continue;
	
			if(uSub<0) continue;
			else if(uSub>=nCols) continue;
	
			if(vSub<0) continue;
			else if(vSub>=nRows) continue;
	
			points_vec.push_back(OriMap[vSub*nCols+uSub].pt);
	
			KNNCounter ++;
	
			if(KNNCounter>=TotalSizeKNN) break;
		}
	
		// searching downward 
		for(int vSub=v+1; vSub<=v+MaxSeaRad; vSub++)
		{
			int uSub = u;
			if(OriMap[v*nCols+u].flag==0) continue;
	
			if(uSub<0) continue;
			else if(uSub>=nCols) continue;
	
			if(vSub<0) continue;
			else if(vSub>=nRows) continue;
	
			points_vec.push_back(OriMap[vSub*nCols+uSub].pt);
	
			KNNCounter ++;
	
			if(KNNCounter>=TotalSizeKNN) break;
		}
	
		// searching lefttwards
		for(int uSub=u-1; uSub>=u-MaxSeaRad; uSub--)
		{
			int vSub = v;
			if(OriMap[v*nCols+u].flag==0) continue;
	
			if(uSub<0) continue;
			else if(uSub>=nCols) continue;
	
			if(vSub<0) continue;
			else if(vSub>=nRows) continue;
	
			points_vec.push_back(OriMap[vSub*nCols+uSub].pt);
	
			KNNCounter ++;
	
			if(KNNCounter>=TotalSizeKNN) break;
		}
	
		// searching upwards 
		for(int vSub=v-1; vSub>=v-MaxSeaRad; vSub--)
		{
			int uSub = u;
			if(OriMap[v*nCols+u].flag==0) continue;
	
			if(uSub<0) continue;
			else if(uSub>=nCols) continue;
	
			if(vSub<0) continue;
			else if(vSub>=nRows) continue;
	
			points_vec.push_back(OriMap[vSub*nCols+uSub].pt);
	
			KNNCounter ++;
	
			if(KNNCounter>=TotalSizeKNN) break;
		}

		//cout<<"Selecting Point Set, size : "<<points_vec.size()<<endl;

		//
		// compute normal vector
		//
		// if the size of selecting point set is 0, we would not comput its normal 
		if(points_vec.size()==0)
		{
			SparseMap[v*nCols+u].normal = Point3f(0,0,-1);
			SparseMap[v*nCols+u].flag   = 0;
		}
		else
		{
			// generate a list of points for normal estimation
			Point3f points[points_vec.size()];
			for(int j=0;j<points_vec.size();j++)
			{
				points[j] = points_vec[j];
			}
	
			// estimate normal vectors
			Point3f normal;
			GetNormal(points_vec.size(), points, normal);
			//cout<<"normal: "<<normal<<"; u,v: "<<u<<", "<<v<<endl;
	
			SparseMap[v*nCols+u].normal = normal;
			SparseMap[v*nCols+u].flag   = 1;
		}
	}

	return;
}
// !Generate a normal map that consists of a portion of all normal vectors

// Normal Estiamtion for the point that does not belong to sparse normal map: Search and estimate
void FillNormalMapItp(int nRows, int nCols, vector<NormalVector> SparseMap, vector<NormalVector> &FullMap)
{
	// Normal Estiamtion for the point that does not belong to sparse normal map
	// Interpolation method is used 

	// nRows : row number
	// nCols : column number
	// SparseMap: the sparse normal map
	// FullMap: the complited normal map

	// step 1 : normal interpolation
	for (int v = 0; v < nRows; ++v)
	for (int u = 0; u < nCols; ++u)
	{
		// step (1) : alpha 
		int alpha = SparseMap[v*nCols+u].flag;

		// step (2) : copy the point whose normal has been computed 
		if(alpha>0)
		{
			int idx = v*nCols+u;
			FullMap[idx] = SparseMap[idx];
			//cout<<"the point whose normal has been computed"<<endl;
			continue;
		}

		// step (3) : compute normal by interpolation
		double xMean = 0;
		double yMean = 0;
		double zMean = 0;

		int IntRad = 8; // interpolation radius
		double TotalWeight = 0.;
		for(int vSub=v-IntRad;vSub<v+IntRad+1;vSub++)
		for(int uSub=u-IntRad;uSub<u+IntRad+1;uSub++)
		{
			if(uSub<0) continue;
			else if(uSub>=nCols) continue;

			if(vSub<0) continue;
			else if(vSub>=nRows) continue;

			int idx = vSub*nCols+uSub;
			if(SparseMap[idx].flag==0) 
			{
				continue;
			}
			else
			{
				double dis2 = (vSub-v)*(vSub-v) + (uSub-u)*(uSub-u);

				double weight = 1./dis2;
				TotalWeight += weight;

				xMean += (SparseMap[idx].normal.x*weight);
				yMean += (SparseMap[idx].normal.y*weight);
				zMean += (SparseMap[idx].normal.z*weight);
			}
		}

		double alphaMean = 0;
		if(TotalWeight!=0)
		{
			xMean /= TotalWeight; 
        	yMean /= TotalWeight;
        	zMean /= TotalWeight;
			alphaMean = 1;
		}
		else
		{
			xMean = 0; 
        	yMean = 0;
        	zMean = -1;
			alphaMean = 0;
		}

		FullMap[v*nCols+u].normal.x	= xMean;
		FullMap[v*nCols+u].normal.y	= yMean;
		FullMap[v*nCols+u].normal.z	= zMean;
		FullMap[v*nCols+u].flag	= alphaMean;
	}

	// step 2 : find the pixel that has no normal vector yet
	int NumberPixelWithoutNormal = 0;
	for (int v = 0; v < nRows; ++v)
	for (int u = 0; u < nCols; ++u)
	{
		double x = FullMap[v*nCols+u].normal.x;
		double y = FullMap[v*nCols+u].normal.y;
		double z = FullMap[v*nCols+u].normal.z;

		if(FullMap[v*nCols+u].flag == 0)
		{
			FullMap[v*nCols+u].normal.x	= 0;
			FullMap[v*nCols+u].normal.y	= 0;
			FullMap[v*nCols+u].normal.z	= -1;
			FullMap[v*nCols+u].flag		= 0;

			NumberPixelWithoutNormal ++;
		}
	}
	cout<<"Find the pixel that has no normal vector yet!"<<endl;
	cout<<"    NumberPixelWithoutNormal : "<<NumberPixelWithoutNormal<<endl;

	return;
}
// !Normal Estiamtion for the point that does not belong to sparse normal map: Search and estimate

// Depth Camera
typedef struct xnIntrinsic_Params
{
	xnIntrinsic_Params() :
		c_x(320.0), c_y(240.0), f_x(480.0), f_y(480.0)
	{}

	xnIntrinsic_Params(float c_x_, float c_y_, float f_x_, float f_y_) :
		c_x(c_x_), c_y(c_y_), f_x(f_x_),f_y(f_y_)
	{}
	
	float c_x; // focal length in X axis
	float c_y; // focal length in Y axis
	float f_x; // x coordinate of image plane's center  
	float f_y; // y coordinate of image plane's center  
}xIntrinsic_Params;

xIntrinsic_Params g_IntrinsicParam; //´æ´¢Ïà»úÄÚ²ÎµÄÈ«¾Ö±äÁ¿

void getCameraParams(openni::Device& Device, xIntrinsic_Params& IrParam)
{
	OBCameraParams cameraParam;
	int dataSize = sizeof(cameraParam);
	memset(&cameraParam, 0, sizeof(cameraParam));
	openni::Status rc = Device.getProperty(openni::OBEXTENSION_ID_CAM_PARAMS, (uint8_t *)&cameraParam, &dataSize);
	if (rc != openni::STATUS_OK)
	{
		std::cout << "Error:" << openni::OpenNI::getExtendedError() << std::endl;
		return;
	}
	IrParam.f_x = cameraParam.l_intr_p[0]; //uÖáÉÏµÄ¹éÒ»»¯½¹¾à
	IrParam.f_y = cameraParam.l_intr_p[1]; //vÖáÉÏµÄ¹éÒ»»¯½¹¾à
	IrParam.c_x = cameraParam.l_intr_p[2]; //Ö÷µãx×ø±ê
	IrParam.c_y = cameraParam.l_intr_p[3]; //Ö÷µãy×ø±ê

	std::cout << "IrParam.f_x = " << IrParam.f_x << std::endl;
	std::cout << "IrParam.f_y = " << IrParam.f_y << std::endl;
	std::cout << "IrParam.c_x = " << IrParam.c_x << std::endl;
	std::cout << "IrParam.c_y = " << IrParam.c_y << std::endl;

}

void convertDepthToPointCloud(const uint16_t *pDepth, int width, int height,const char *ply_filename)
{
	//cout<<"convertDepthToPointCloud, ply_filename: "<<ply_filename<<endl;

	// mouse click
	mouseUse_width_  = (double)(width) ;
    mouseUse_height_ = (double)(height);

	//cout<<"convertDepthToPointCloud- width, height: "<<width<<", "<<height<<endl;

	if (NULL == pDepth)
	{
		printf("depth frame is NULL!");
		return;
	}


	//½«Éî¶ÈÊý¾Ý×ª»»ÎªµãÔÆÊý¾Ý,²¢½«µãÔÆÊý¾Ý±£´æµ½ÎÄ¼þÖÐ
	//FILE *fp;
	
	//int res = fopen_s(&fp, ply_filename, "w");
	//fp = fopen(ply_filename, "w");
	
	int valid_count = 0;
	uint16_t max_depth = MAX_DISTANCE;
	uint16_t min_depth = MIN_DISTANCE;
	
	//Í³¼ÆÓÐÐ§µÄÊý¾ÝÌõÊý
	//int img_size = width * height;
	for (int v = 0; v < height; ++v)
	{
		for (int u = 0; u < width; ++u)
		{
			uint16_t depth = pDepth[v * width + u];
			if (depth <= 0 || depth < min_depth || depth > max_depth)
				continue;
	
			valid_count += 1;
		}
	}
	
	//plyµãÔÆÎÄ¼þµÄÍ·
	//fprintf(fp, "ply\n");
	//fprintf(fp, "format ascii 1.0\n");
	//fprintf(fp, "element vertex %d\n", valid_count);
	//fprintf(fp, "property float x\n");
	//fprintf(fp, "property float y\n");
	//fprintf(fp, "property float z\n");
	//fprintf(fp, "property uchar red\n");
	//fprintf(fp, "property uchar green\n");
	//fprintf(fp, "property uchar blue\n");
	//fprintf(fp, "end_header\n");

	// draw a photo
	Mat image(height,width,CV_8UC1,Scalar::all(0));  //创建一个高200，宽100的灰度图
	//Mat image(RESOULTION_Y,RESOULTION_X,CV_8UC1,Scalar::all(0));  //创建一个高200，宽100的灰度图
	Mat img3D(height,width,CV_8UC4,Scalar::all(0)); // build a 4 channel mat, in which the 3D cooordinate corresponding to a pixel is stored.
	int img3DChannels = img3D.channels();

	// the map of normal vectors
	//Mat imgNormal(height,width,CV_8UC4,Scalar::all(0)); // build a 4 channel mat, in which the normal vector corresponding to a pixel is stored.

	// build and clear the point cloud, which is the global variable
	pointCloud_depthCam_.clear();
	for (int v = 0; v < height; ++v)
	for (int u = 0; u < width; ++u)
	{
		pointCloud_depthCam_.push_back(Point3f(0,0,0));
	}


	float world_x, world_y, world_z;
	for (int v = 0; v < height; ++v)
	{

		for (int u = 0; u < width; ++u)
		{
			uint16_t depth = pDepth[v * width + u];
			if (depth <= 0 || depth < min_depth || depth > max_depth)
				continue;
	
			//·Ö±æÂÊËõ·Å£¬ÕâÀï¼ÙÉè±ê¶¨Ê±µÄ·Ö±æÂÊ·ÖRESOULTION_X£¬RESOULTION_Y
			float fdx = g_IntrinsicParam.f_x * ((float)(width) / RESOULTION_X);
			float fdy = g_IntrinsicParam.f_y * ((float)(height) / RESOULTION_Y);
			float u0 = g_IntrinsicParam.c_x * ((float)(width)/ RESOULTION_X);
			float v0 = g_IntrinsicParam.c_y * ((float)(height) / RESOULTION_Y);
	
			float tx = (u - u0) / fdx;
			float ty = (v - v0) / fdy;
			
			world_x = depth * tx;
			world_y = depth * ty;
			world_z = depth;
			//fprintf(fp, "%f %f %f 255 255 255\n", world_x, world_y, world_z);

			//
			// calculate coordiates with pixel unit in image plane
			//
			double px = g_IntrinsicParam.f_x * world_x + g_IntrinsicParam.c_x * world_z;
			double py = g_IntrinsicParam.f_y * world_y + g_IntrinsicParam.c_y * world_z;
			double pz = world_z;

			px /= pz;
			py /= pz;

			int pixelx = int(px * (double)(width)  / RESOULTION_X);
			int pixely = int(py * (double)(height) / RESOULTION_Y);

			//
			// draw a photo
			// 

			//
			// step 1 : should we mirror the depth image? 
			//          should    : 3D depth camera 1
			//          should NOT: 3D depth camera 2
			//
			// 3D depth camera 1
			int u_mir = width-u-1; // mirrored y axis

			// 3D depth camera 2
			//int u_mir = u; // mirrored y axis

			double xyMin = -2000.;
			double xyMax =  2000.;
			uchar x_normalized = (world_x-xyMin)/(xyMax-xyMin)*255;
			uchar y_normalized = (world_y-xyMin)/(xyMax-xyMin)*255;
			uchar z_normalized = (world_z-MIN_DISTANCE)/(MAX_DISTANCE-MIN_DISTANCE)*255;

			// method 1
			float isDepthTaken = 1;
			if(z_normalized<0) isDepthTaken=0;

			img3D.ptr<uchar>(v)[u_mir*img3DChannels+0] = x_normalized;
			img3D.ptr<uchar>(v)[u_mir*img3DChannels+1] = y_normalized;
			img3D.ptr<uchar>(v)[u_mir*img3DChannels+2] = z_normalized;
			img3D.ptr<uchar>(v)[u_mir*img3DChannels+3] = isDepthTaken;

			// method 2
			//uchar *p = image.ptr<uchar>(pixely);
			//int x_depth = width-u-1;
			//p[width-pixelx-1] = (world_z-MIN_DISTANCE)/(MAX_DISTANCE-MIN_DISTANCE)*255;

			//
			// set a point to the point cloud
			pointCloud_depthCam_[v * width + u_mir] = Point3f(world_x, world_y, world_z);
		}
	}

	// set global variables : the 3D depth image
	img3D_ = img3D;

	//fclose(fp);
}

int g_imageCount = 0;

void analyzeFrame(const VideoFrameRef& frame)
{
	//
	DepthPixel* pDepth;

	//¹¹ÔìµãÔÆÎÄ¼þÃû
	char plyFileName[256] = "";
	g_imageCount++;

	std::stringstream filename;
	filename << "pointcloud_";
	filename << g_imageCount;
	filename << ".ply";
	filename >> plyFileName;
	
	int middleIndex = (frame.getHeight() + 1)*frame.getWidth() / 2;
	
	switch (frame.getVideoMode().getPixelFormat())
	{
	case PIXEL_FORMAT_DEPTH_1_MM:
		pDepth = (DepthPixel*)frame.getData();
		//printf("[%08llu] %8d\n", (long long)frame.getTimestamp(),
		//	pDepth[middleIndex]);
	
		//½«Éî¶ÈÊý¾Ý×ª»»ÎªµãÔÆ²¢±£´æ³ÉplyÎÄ¼þ£¬Ã¿Ö¡Éî¶ÈÊý¾Ý¶ÔÓ¦Ò»¸öplyÎÄ¼þ
		convertDepthToPointCloud(pDepth, frame.getWidth(), frame.getHeight(), plyFileName);
		break;
	default:
		printf("Unknown format\n");
	}
}

class PrintCallback : public VideoStream::NewFrameListener
{
public:
	void onNewFrame(VideoStream& stream)
	{
		stream.readFrame(&m_frame);

		analyzeFrame(m_frame);
	}
private:
	VideoFrameRef m_frame;
};
// !Depth Camera

// Run Control Gruop: A 3D normal estimation
int Run_ControlGroup(int RunID, double &TimeInterval)
{
	cout<<"Hello"<<endl;

	// clear the global parameters
	pointCloud_depthCam_.clear();
	NorMap_RGBCam_.clear();
	NorMap_depthCam_.clear();

	//
	// fast normal estimation
	//

	//
	// step 1 : open the 3D Camera, get RGB images and depth images
	//

	//initialize openNI sdk
	Status rc = OpenNI::initialize();
	if (rc != STATUS_OK)
	{
		printf("Initialize failed\n%s\n", OpenNI::getExtendedError());
		return 1;
	}

	//open deivce
	Device device;
	rc = device.open(ANY_DEVICE);
	if (rc != STATUS_OK)
	{
		printf("Couldn't open device\n%s\n", OpenNI::getExtendedError());
		return 2;
	}

	// 
	//device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);
	
	VideoStream depth;

	//create depth stream
	if (device.getSensorInfo(SENSOR_DEPTH) != NULL)
	{
		rc = depth.create(device, SENSOR_DEPTH);
		if (rc != STATUS_OK)
		{
			printf("Couldn't create depth stream\n%s\n", OpenNI::getExtendedError());
		}
	}

	device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);

	
	//start depth stream
	rc = depth.start();
	if (rc != STATUS_OK)
	{
		printf("Couldn't start the depth stream\n%s\n", OpenNI::getExtendedError());
	}

	//
	// detecting and showing dpeth of the point at the center
	//
	VideoFrameRef frame;

	//
	// start taking a point cloud
	//
	PrintCallback depthPrinter;
	
	// Register frame listener
	depth.addNewFrameListener(&depthPrinter);
	
	//get intrinsic parameter from device
	getCameraParams(device, g_IntrinsicParam);

	// Open the RGB camera
    VideoCapture capture(0);

    if(!capture.isOpened())    // 判断是否打开成功
    {
        cout << "open camera failed. " << endl;
        return -1;
    }
	

	//namedWindow( "3DCameraImage", WINDOW_AUTOSIZE ); // Create a window for display.
	//namedWindow( "DepthImage", WINDOW_AUTOSIZE ); // Create a window for display.

	while (1)
	{
		// get RGB image 
        capture >> image_RGB_;
		int nRows = img3D_.rows;
		int nCols = img3D_.cols;
		int RGBChannels = image_RGB_.channels();
		int img3DChannels = img3D_.channels();

		Mat image_RGB_3D = image_RGB_.clone();
		if(image_RGB_3D.rows<nRows || image_RGB_3D.cols<nCols)
		{
			cout<<"The RGB image is missing."<<endl;
			continue;
		}

		double depthMin = 10;
		double depthMax = 4000;
		int uniDepthMin = (depthMin-MIN_DISTANCE)/(MAX_DISTANCE-MIN_DISTANCE)*255;
		int uniDepthMax = (depthMax-MIN_DISTANCE)/(MAX_DISTANCE-MIN_DISTANCE)*255;

		//
		// merge the RGB and depth images
		//
		for(int v=0;v<nRows;v++)
		{
			uchar *pDep = img3D_.ptr<uchar>(v);
			uchar *pRGB = image_RGB_3D.ptr<uchar>(v);
			for(int u=0;u<nCols;u++)
			{
				//cout<<"v u: "<<v<<", "<<u<<endl;
				if(pDep[u*img3DChannels+2]>uniDepthMin&&pDep[u*img3DChannels+2]<uniDepthMax)
				{
					// depth
					pRGB[u*RGBChannels+0] = abs(0  -pDep[u*img3DChannels+2]);
					pRGB[u*RGBChannels+1] = abs(127-pDep[u*img3DChannels+2]);
					pRGB[u*RGBChannels+2] = abs(255-pDep[u*img3DChannels+2]);

					// x, y, and z
					//pRGB[u*RGBChannels+0] = pDep[u*img3DChannels+0];
					//pRGB[u*RGBChannels+1] = pDep[u*img3DChannels+1];
					//pRGB[u*RGBChannels+2] = pDep[u*img3DChannels+2];
				}
			}
		}

		//
		// draw a box in the RGB image
		//
		//// a point that is interested by us
		//int pixelX_interested_ = 320;
		//int pixelY_interested_ = 240;
		//int length_box = 20;
		//for(int i=0;i<length_box+1;i++) // height, Y axis, vertical axis
		//{
		//	int v = pixelY_interested_ - length_box/2 + i; 
		//	uchar *pRGB = image_RGB_3D  .ptr<uchar>(v);
		//	
		//	for(int j=0;j<length_box;j++)
		//	{
		//		int u = pixelX_interested_ - length_box/2 + j; 
		//		//pRGB[u*RGBChannels+0] = 255/2;
		//		//pRGB[u*RGBChannels+1] = 255/2;
		//		//pRGB[u*RGBChannels+2] = 0;

		//		pRGB[u*RGBChannels+0] = 0;
		//		pRGB[u*RGBChannels+1] = 0;
		//		pRGB[u*RGBChannels+2] = 0;

		//		//pRGB[u*RGBChannels+0] = 255;
		//		//pRGB[u*RGBChannels+1] = 255;
		//		//pRGB[u*RGBChannels+2] = 255;
		//	}
		//}

		// 
		// take photoes manually
		// 
		imshow("3DCameraImage", image_RGB_3D);
		// shutting the camera down  
		if(waitKey(1)>-1)
		{
			cout<<"Input from keyboard has been detected."<<endl;
			cout<<"Take a snapshot and shut the camera down."<<endl;
			break;
		}

		// 
		// take photoes automatically, skip showing the RGB frame
		// 
		//sleep(1);
		//cout<<"Input from keyboard has been detected."<<endl;
		//cout<<"Take a snapshot and shut the camera down."<<endl;
		//break;

	}

	depth.removeNewFrameListener(&depthPrinter);
	
	//stop depth stream
	depth.stop();
	
	//destroy depth stream
	depth.destroy();
	
	//close device
	device.close();
	
	//shutdown OpenNI
	OpenNI::shutdown();

	//
	cout<<"3DCamera has been shut down"<<endl;

	//
	// step 2 :Fast Normal Estimation, compute the map of normal vectors
	//
	cout<<"start computing the map of normal vectors"<<endl;

	// time interval
	clock_t start, end;
	start = clock();

	//
	// step 2.1 : sparselization
	//
	// step 2.1.1 : define a margin which surround the depth image
	int marTop		= 45;
	int marBottom	= 0;
	int marLeft		= 40;
	int marRight	= 0;

	//
	// step 2.2 : generate groups of sparse points
	//

	// type 2 
	vector<NormalVector> SparseDepMap; // a map who stores original 3D points, may have holes
	vector<NormalVector> FullDepMap; // a map who stores original 3D points, do not have a hole

	// step 2.2.1 : initialization for the two maps
	for (int v = 0; v < image_RGB_.rows; ++v)
	for (int u = 0; u < image_RGB_.cols; ++u)
	{
		NormalVector normal;
		normal.pt = Point3f(0,0,0);
		normal.normal = Point3f(0,0,-1);
		normal.flag = 0;
		SparseDepMap.push_back(normal);

		NormalVector normal2;
		normal2.pt = Point3f(0,0,0);
		normal2.normal = Point3f(0,0,-1);
		normal2.flag = 0;
		FullDepMap  .push_back(normal2);
	}

	// step 2.2.2 : set 3D point locations  
	for (int v = 0; v < img3D_.rows; ++v)
	for (int u = 0; u < img3D_.cols; ++u)
	{
		int idx = v*image_RGB_.cols + u;

		// point 
		// generate a map to store 3D points in the origingal 3D map 
		float x = pointCloud_depthCam_[idx].x;
		float y = pointCloud_depthCam_[idx].y;
		float z = pointCloud_depthCam_[idx].z;

		int flag = 1;
		if(x==0&&y==0&&z==0) 
		{
			flag = 0;
		}

		SparseDepMap[idx].pt = Point3f(x,y,z); 
		SparseDepMap[idx].flag = flag;
		//cout<<"v,u: "<<v<<", "<<u<<"; pt: "<<SparseDepMap[idx].pt<<", flag: "<<SparseDepMap[idx].flag<<endl;
	}

	// step 2.2.3 : search and fix holes
	// fix holes in control group
	int nRow3D = image_RGB_.rows;
	int nCol3D = image_RGB_.cols;
	FixHolesFor3DMap_ISD(nRow3D, nCol3D, SparseDepMap, FullDepMap);
	//FixHolesFor3DMap_WoBS(nRow3D, nCol3D, SparseDepMap, FullDepMap);
	// !fix holes in control group

	// do not fix holes in control group
	//FullDepMap = SparseDepMap;
	// !do not fix holes in control group
	// !type 2 

	// step 2.2.4 : compute normal vectors for all points in the sparse normal map 
	// type 2
	vector<NormalVector> SparseNorMap; // a map who stores sparse normals and full 3D points
	// initialization for the two maps
	for (int v = 0; v < image_RGB_.rows; ++v)
	for (int u = 0; u < image_RGB_.cols; ++u)
	{
		NormalVector normal;
		normal.pt = Point3f(0,0,0);
		normal.normal = Point3f(0,0,-1);
		normal.flag = 0;
		SparseNorMap.push_back(normal);
	}

	int nRow3D_2 = image_RGB_.rows;
	int nCol3D_2 = image_RGB_.cols;
	//GetSparseNormalMap_SquareSearch(SizeSparseNormal_User_, nRow3D_2, nCol3D_2, FullDepMap, SparseNorMap);
	GetSparseNormalMap_CrossSearch(SizeSparseNormal_User_, nRow3D_2, nCol3D_2, FullDepMap, SparseNorMap);
	// ! type 2

	// step 2.2.5 : compute the left normal vectors that are not in the sparse normal map 
	// type 2 : initialization an empty normal map
	vector<NormalVector> FullMap;
	for (int v = 0; v < image_RGB_.rows; ++v)
	for (int u = 0; u < image_RGB_.cols; ++u)
	{

		NormalVector normal2;
		normal2.pt = Point3f(0,0,0);
		normal2.normal = Point3f(0,0,-1);
		normal2.flag = 0;
		FullMap  .push_back(normal2);
	}

	int nRow = image_RGB_.rows;
	int nCol = image_RGB_.cols;
	FillNormalMapItp(nRow, nCol, SparseNorMap, FullMap);

	// step 2.2.6 fill 3D coordinates to FullMap
	for(int i=0;i<FullDepMap.size();i++)
	{
		FullMap[i].pt = FullDepMap[i].pt;
	}


	end = clock();

	TimeInterval =  ((double)(end-start)/CLOCKS_PER_SEC);
	cout<<"Fast Normal Estimation, timeInterval_ per image: "<<TimeInterval<<" second"<<endl;

	//
	// step 3 : analysis
	//

	// analysis part 1
	string filename = "data_NorMap_RGBCam_Control_"+to_string(SizeSparseNormal_User_)+".txt";
	ofstream file(filename);

	for(int i=0;i<FullMap.size();i++)
	{
		file<<FullMap[i].pt.x<<" ";
		file<<FullMap[i].pt.y<<" ";
		file<<FullMap[i].pt.z<<" ";
		file<<FullMap[i].normal.x<<" ";
		file<<FullMap[i].normal.y<<" ";
		file<<FullMap[i].normal.z<<" ";
		file<<FullMap[i].flag<<endl;
	}

	file.close();
	cout<<"Nromal vector map has been written"<<endl;

	// analysis part 2
	// Draw a depth map with holes
	Mat imgSparseDepMap(image_RGB_.rows,image_RGB_.cols,CV_8UC4,Scalar::all(0)); // build a 4 channel mat, in which the normal vector corresponding to a pixel is stored.
	int Channels_Dep = imgSparseDepMap.channels();
	for (int v = 0; v < image_RGB_.rows; ++v)
	for (int u = 0; u < image_RGB_.cols; ++u)
	{
		float z = (SparseDepMap[v*image_RGB_.cols+u].pt.z-MIN_DISTANCE)/(MAX_DISTANCE-MIN_DISTANCE)*255;
		float alpha = SparseDepMap[v*image_RGB_.cols+u].flag * 255;

		if(alpha==0)
		{
			continue;
		}

		imgSparseDepMap.ptr<uchar>(v)[u*Channels_Dep+0] = int(abs(0-z));
		imgSparseDepMap.ptr<uchar>(v)[u*Channels_Dep+1] = int(abs(127-z));
		imgSparseDepMap.ptr<uchar>(v)[u*Channels_Dep+2] = int(abs(255-z));
		imgSparseDepMap.ptr<uchar>(v)[u*Channels_Dep+3] = int(alpha);

	}
	imwrite("figure_SparseDepMap_"+to_string(SizeSparseNormal_User_)+".png", imgSparseDepMap);
	imshow("SparseDepMap_depthCam_", imgSparseDepMap);

	// Draw a depth map in which holes are eliminated 
	Mat imgFullDepMap(image_RGB_.rows,image_RGB_.cols,CV_8UC4,Scalar::all(0)); // build a 4 channel mat, in which the normal vector corresponding to a pixel is stored.
	for (int v = 0; v < image_RGB_.rows; ++v)
	for (int u = 0; u < image_RGB_.cols; ++u)
	{
		float z = (FullDepMap[v*image_RGB_.cols+u].pt.z-MIN_DISTANCE)/(MAX_DISTANCE-MIN_DISTANCE)*255;
		float alpha = FullDepMap[v*image_RGB_.cols+u].flag * 255;

		if(alpha==0)
		{
			continue;
		}

		imgFullDepMap.ptr<uchar>(v)[u*Channels_Dep+0] = int(abs(0-z));
		imgFullDepMap.ptr<uchar>(v)[u*Channels_Dep+1] = int(abs(127-z));
		imgFullDepMap.ptr<uchar>(v)[u*Channels_Dep+2] = int(abs(255-z));
		imgFullDepMap.ptr<uchar>(v)[u*Channels_Dep+3] = int(alpha);

	}
	imwrite("figure_FullDepMap_"+to_string(SizeSparseNormal_User_)+".png", imgFullDepMap);
	imshow("FullDepMap_depthCam_", imgFullDepMap);


	// draw a sparse normal map 
	Mat imgSparseNormalMap(image_RGB_.rows,image_RGB_.cols,CV_8UC4,Scalar::all(0)); // build a 4 channel mat, in which the normal vector corresponding to a pixel is stored.

	int C4Channels_RGB = imgSparseNormalMap.channels();

	for (int v = 0; v < image_RGB_.rows; ++v)
	for (int u = 0; u < image_RGB_.cols; ++u)
	{
		float x = (SparseNorMap[v*image_RGB_.cols+u].normal.x+1.)/2.*255;
		float y = (SparseNorMap[v*image_RGB_.cols+u].normal.y+1.)/2.*255;
		float z = (SparseNorMap[v*image_RGB_.cols+u].normal.z+1.)/2.*255;
		float alpha = SparseNorMap[v*image_RGB_.cols+u].flag * 255;

		imgSparseNormalMap.ptr<uchar>(v)[u*C4Channels_RGB+0] = int(x);
		imgSparseNormalMap.ptr<uchar>(v)[u*C4Channels_RGB+1] = int(y);
		imgSparseNormalMap.ptr<uchar>(v)[u*C4Channels_RGB+2] = int(z);
		imgSparseNormalMap.ptr<uchar>(v)[u*C4Channels_RGB+3] = int(alpha);

	}
	imwrite("figure_SparseNormalMap_"+to_string(SizeSparseNormal_User_)+".png", imgSparseNormalMap);
	imshow("NormalMap_depthCam_", imgSparseNormalMap);


	// type 2 : analysis part 3
	// draw a map of NorMap_RGBCam_
	Mat imgNormal_Mapping_RGBCam_(image_RGB_.rows,image_RGB_.cols,CV_8UC4,Scalar::all(0)); // build a 4 channel mat, in which the normal vector corresponding to a pixel is stored.

	//int C4Channels_RGB = imgNormal_Mapping_RGBCam_.channels();

	for (int v = 0; v < image_RGB_.rows; ++v)
	for (int u = 0; u < image_RGB_.cols; ++u)
	{
		float x = (FullMap[v*image_RGB_.cols+u].normal.x+1.)/2.*255;
		float y = (FullMap[v*image_RGB_.cols+u].normal.y+1.)/2.*255;
		float z = (FullMap[v*image_RGB_.cols+u].normal.z+1.)/2.*255;
		float alpha = FullMap[v*image_RGB_.cols+u].flag * 255;

		//cout<<"NorMap_RGBCam_ - normal : "<<int(x)<<", "<<int(y)<<", "<<int(z)<<endl;

		if(x==0&&y==0&&z==0)
		{
			continue;
		}

		//uchar alpha	= NorMap_RGBCam_[v*image_RGB_.cols+u].alpha;

		imgNormal_Mapping_RGBCam_.ptr<uchar>(v)[u*C4Channels_RGB+0] = int(x);
		imgNormal_Mapping_RGBCam_.ptr<uchar>(v)[u*C4Channels_RGB+1] = int(y);
		imgNormal_Mapping_RGBCam_.ptr<uchar>(v)[u*C4Channels_RGB+2] = int(z);
		imgNormal_Mapping_RGBCam_.ptr<uchar>(v)[u*C4Channels_RGB+3] = int(alpha);

	}
	imwrite("figure_NormalMap_RGBCam_Size"+to_string(SizeSparseNormal_User_)+"RunID"+to_string(RunID)+".png", imgNormal_Mapping_RGBCam_);
	imshow("NormalMap_RGBCam_", imgNormal_Mapping_RGBCam_);

	// analysis part 4
	//
	// opencv taking points by mouse 
	//
	FullMap_ = FullMap;
	imwrite("figure_RGB_"+to_string(RunID)+".png", image_RGB_);
	imshow(WinName, image_RGB_);
	setMouseCallback(WinName, On_mouse, 0);


	// compute manually
	waitKey(0);

	// compute automatically
	//sleep(0.1);

	return 0;
}

// Run A 3D normal estimation
int Run(int RunID, double &TimeInterval)
{
	cout<<"Hello"<<endl;

	// clear the global parameters
	pointCloud_depthCam_.clear();
	NorMap_RGBCam_.clear();
	NorMap_depthCam_.clear();

	//
	// fast normal estimation
	//

	//
	// step 1 : open the 3D Camera, get RGB images and depth images
	//

	//initialize openNI sdk
	Status rc = OpenNI::initialize();
	if (rc != STATUS_OK)
	{
		printf("Initialize failed\n%s\n", OpenNI::getExtendedError());
		return 1;
	}

	//open deivce
	Device device;
	rc = device.open(ANY_DEVICE);
	if (rc != STATUS_OK)
	{
		printf("Couldn't open device\n%s\n", OpenNI::getExtendedError());
		return 2;
	}

	// 
	//device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);
	
	VideoStream depth;

	//create depth stream
	if (device.getSensorInfo(SENSOR_DEPTH) != NULL)
	{
		rc = depth.create(device, SENSOR_DEPTH);
		if (rc != STATUS_OK)
		{
			printf("Couldn't create depth stream\n%s\n", OpenNI::getExtendedError());
		}
	}

	device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);

	
	//start depth stream
	rc = depth.start();
	if (rc != STATUS_OK)
	{
		printf("Couldn't start the depth stream\n%s\n", OpenNI::getExtendedError());
	}

	//
	// detecting and showing dpeth of the point at the center
	//
	VideoFrameRef frame;

	//
	// start taking a point cloud
	//
	PrintCallback depthPrinter;
	
	// Register frame listener
	depth.addNewFrameListener(&depthPrinter);
	
	//get intrinsic parameter from device
	getCameraParams(device, g_IntrinsicParam);

	// Open the RGB camera
    VideoCapture capture(0);

    if(!capture.isOpened())    // 判断是否打开成功
    {
        cout << "open camera failed. " << endl;
        return -1;
    }
	

	//namedWindow( "3DCameraImage", WINDOW_AUTOSIZE ); // Create a window for display.
	//namedWindow( "DepthImage", WINDOW_AUTOSIZE ); // Create a window for display.

	while (1)
	{
		// get RGB image 
        capture >> image_RGB_;
		int nRows = img3D_.rows;
		int nCols = img3D_.cols;
		int RGBChannels = image_RGB_.channels();
		int img3DChannels = img3D_.channels();

		Mat image_RGB_3D = image_RGB_.clone();

		//
		// merge the RGB and depth images
		//
		for(int v=0;v<nRows;v++)
		{
			uchar *pDep = img3D_.ptr<uchar>(v);
			uchar *pRGB = image_RGB_3D.ptr<uchar>(v);
			for(int u=0;u<nCols;u++)
			{
				//cout<<"v u: "<<v<<", "<<u<<endl;
				if(pDep[u*img3DChannels+2]>10&&pDep[u*img3DChannels+2]<50)
				{
					pRGB[u*RGBChannels+0] = pDep[u*img3DChannels+2];
					pRGB[u*RGBChannels+1] = pDep[u*img3DChannels+2];
					pRGB[u*RGBChannels+2] = pDep[u*img3DChannels+2];
				}
			}
		}

		//
		// draw a box in the RGB image
		//
		// a point that is interested by us
		int pixelX_interested_ = 320;
		int pixelY_interested_ = 240;
		int length_box = 20;
		for(int i=0;i<length_box+1;i++) // height, Y axis, vertical axis
		{
			int v = pixelY_interested_ - length_box/2 + i; 
			uchar *pRGB = image_RGB_3D  .ptr<uchar>(v);
			
			for(int j=0;j<length_box;j++)
			{
				int u = pixelX_interested_ - length_box/2 + j; 
				//pRGB[u*RGBChannels+0] = 255/2;
				//pRGB[u*RGBChannels+1] = 255/2;
				//pRGB[u*RGBChannels+2] = 0;

				pRGB[u*RGBChannels+0] = 0;
				pRGB[u*RGBChannels+1] = 0;
				pRGB[u*RGBChannels+2] = 0;

				//pRGB[u*RGBChannels+0] = 255;
				//pRGB[u*RGBChannels+1] = 255;
				//pRGB[u*RGBChannels+2] = 255;
			}
		}

		// 
		// take photoes manually
		// 
		//imshow("3DCameraImage", image_RGB_3D);
		//// shutting the camera down  
		//if(waitKey(1)>-1)
		//{
		//	cout<<"Input from keyboard has been detected."<<endl;
		//	cout<<"Take a snapshot and shut the camera down."<<endl;
		//	break;
		//}

		// 
		// take photoes automatically, skip showing the RGB frame
		// 
		sleep(1);
		cout<<"Input from keyboard has been detected."<<endl;
		cout<<"Take a snapshot and shut the camera down."<<endl;
		break;

	}

	depth.removeNewFrameListener(&depthPrinter);
	
	//stop depth stream
	depth.stop();
	
	//destroy depth stream
	depth.destroy();
	
	//close device
	device.close();
	
	//shutdown OpenNI
	OpenNI::shutdown();

	//
	cout<<"3DCamera has been shut down"<<endl;

	//
	// step 2 :Fast Normal Estimation, compute the map of normal vectors
	//
	cout<<"start computing the map of normal vectors"<<endl;

	// time interval
	clock_t start, end;
	start = clock();

	//
	// step 2.1 : sparselization
	//
	// step 2.1.1 : define a margin which surround the depth image
	int marTop		= 45;
	int marBottom	= 0;
	int marLeft		= 40;
	int marRight	= 0;

	//
	// step 2.2 : generate groups of sparse points
	//

	// type 2 
	vector<NormalVector> SparseDepMap; // a map who stores original 3D points, may have holes
	vector<NormalVector> FullDepMap; // a map who stores original 3D points, do not have a hole

	// step 2.2.1 : initialization for the two maps
	for (int v = 0; v < image_RGB_.rows; ++v)
	for (int u = 0; u < image_RGB_.cols; ++u)
	{
		NormalVector normal;
		normal.pt = Point3f(0,0,0);
		normal.normal = Point3f(0,0,-1);
		normal.flag = 0;
		SparseDepMap.push_back(normal);

		NormalVector normal2;
		normal2.pt = Point3f(0,0,0);
		normal2.normal = Point3f(0,0,-1);
		normal2.flag = 0;
		FullDepMap  .push_back(normal2);
	}

	// step 2.2.2 : set 3D point locations  
	for (int v = 0; v < img3D_.rows; ++v)
	for (int u = 0; u < img3D_.cols; ++u)
	{
		int idx = v*image_RGB_.cols + u;

		// point 
		// generate a map to store 3D points in the origingal 3D map 
		float x = pointCloud_depthCam_[idx].x;
		float y = pointCloud_depthCam_[idx].y;
		float z = pointCloud_depthCam_[idx].z;

		int flag = 1;
		if(x==0&&y==0&&z==0) 
		{
			flag = 0;
		}

		SparseDepMap[idx].pt = Point3f(x,y,z); 
		SparseDepMap[idx].flag = flag;
		//cout<<"v,u: "<<v<<", "<<u<<"; pt: "<<SparseDepMap[idx].pt<<", flag: "<<SparseDepMap[idx].flag<<endl;
	}

	// step 2.2.3 : search and fix holes
	int nRow3D = image_RGB_.rows;
	int nCol3D = image_RGB_.cols;
	FixHolesFor3DMap_ISD(nRow3D, nCol3D, SparseDepMap, FullDepMap);
	// !type 2 

	// step 2.2.4 : compute normal vectors for all points in the sparse normal map 
	// type 2
	vector<NormalVector> SparseNorMap; // a map who stores sparse normals and full 3D points
	// initialization for the two maps
	for (int v = 0; v < image_RGB_.rows; ++v)
	for (int u = 0; u < image_RGB_.cols; ++u)
	{
		NormalVector normal;
		normal.pt = Point3f(0,0,0);
		normal.normal = Point3f(0,0,-1);
		normal.flag = 0;
		SparseNorMap.push_back(normal);
	}

	int nRow3D_2 = image_RGB_.rows;
	int nCol3D_2 = image_RGB_.cols;
	//GetSparseNormalMap_SquareSearch(SizeSparseNormal_User_, nRow3D_2, nCol3D_2, FullDepMap, SparseNorMap);
	GetSparseNormalMap_CrossSearch(SizeSparseNormal_User_, nRow3D_2, nCol3D_2, FullDepMap, SparseNorMap);
	// ! type 2

	// step 2.2.5 : compute the left normal vectors that are not in the sparse normal map 
	// type 2 : initialization an empty normal map
	vector<NormalVector> FullMap;
	for (int v = 0; v < image_RGB_.rows; ++v)
	for (int u = 0; u < image_RGB_.cols; ++u)
	{

		NormalVector normal2;
		normal2.pt = Point3f(0,0,0);
		normal2.normal = Point3f(0,0,-1);
		normal2.flag = 0;
		FullMap  .push_back(normal2);
	}

	int nRow = image_RGB_.rows;
	int nCol = image_RGB_.cols;
	FillNormalMapItp(nRow, nCol, SparseNorMap, FullMap);

	// step 2.2.6 fill 3D coordinates to FullMap
	for(int i=0;i<FullDepMap.size();i++)
	{
		FullMap[i].pt = FullDepMap[i].pt;
	}


	end = clock();

	TimeInterval =  ((double)(end-start)/CLOCKS_PER_SEC);
	cout<<"Fast Normal Estimation, timeInterval_ per image: "<<TimeInterval<<" second"<<endl;

	//
	// step 3 : analysis
	//

	// analysis part 1
	string filename = "data_NorMap_RGBCam_"+to_string(SizeSparseNormal_User_)+"_DisID_"+to_string(DistanceID_)+".txt";
	ofstream file(filename);

	for(int i=0;i<FullMap.size();i++)
	{
		file<<FullMap[i].pt.x<<" ";
		file<<FullMap[i].pt.y<<" ";
		file<<FullMap[i].pt.z<<" ";
		file<<FullMap[i].normal.x<<" ";
		file<<FullMap[i].normal.y<<" ";
		file<<FullMap[i].normal.z<<" ";
		file<<FullMap[i].flag<<endl;
	}

	file.close();
	cout<<"Nromal vector map has been written"<<endl;

	// analysis part 1.2: evaluate a normal by all 3D points
	// We assume that all points are located at a plane
	int PlainMarTop		= 100;
	int PlainMarBottom	= 100;
	int PlainMarLeft	= 100;
	int PlainMarRight	= 100;

	cout<<"image_RGB_.rows: "<<image_RGB_.rows<<endl;
	cout<<"image_RGB_.cols: "<<image_RGB_.cols<<endl;
	cout<<"image_RGB_.rows-PlainMarRight: " <<image_RGB_.rows-PlainMarRight<<endl;
	cout<<"image_RGB_.cols-PlainMarBottom: "<<image_RGB_.cols-PlainMarBottom<<endl;
	vector<Point3f> planePoints_vec;
	for(int v=PlainMarLeft;v<image_RGB_.rows-PlainMarRight;  v++)
	for(int u=PlainMarTop; u<image_RGB_.cols-PlainMarBottom; u++)
	{
		int idx = v*image_RGB_.cols + u;

		if(FullMap[idx].flag==0) continue;

		if(FullMap[idx].pt.x==0&&FullMap[idx].pt.y==0&&FullMap[idx].pt.z==0)
			continue;

		//cout<<FullMap[idx].pt<<", flag: "<<FullMap[idx].flag<<endl;
		planePoints_vec.push_back(FullMap[idx].pt);
	}

	Point3f planePoints[planePoints_vec.size()];
	Point3f planeNormal(0,0,0);
	for(int i=0;i<planePoints_vec.size();i++)
	{
		planePoints[i] = planePoints_vec[i];
		//cout<<"i"<<i<<", planePoints[i]: "<<planePoints[i]<<endl;
	}
	GetNormal(planePoints_vec.size(), planePoints, planeNormal);
	cout<<"planeNormal: "<<planeNormal<<endl;

	string filename2 = "data_PlainNorMap_RGBCam_"+to_string(SizeSparseNormal_User_)+"_DisID_"+to_string(DistanceID_)+".txt";
	ofstream file2(filename2);
	file2<<planeNormal.x<<" "<<planeNormal.y<<" "<<planeNormal.z<<endl;
	file2.close();


	// analysis part 2
	// draw a sparse normal map 
	Mat imgSparseNormalMap(image_RGB_.rows,image_RGB_.cols,CV_8UC4,Scalar::all(0)); // build a 4 channel mat, in which the normal vector corresponding to a pixel is stored.

	int C4Channels_RGB = imgSparseNormalMap.channels();

	for (int v = 0; v < image_RGB_.rows; ++v)
	for (int u = 0; u < image_RGB_.cols; ++u)
	{
		float x = (SparseNorMap[v*image_RGB_.cols+u].normal.x+1.)/2.*255;
		float y = (SparseNorMap[v*image_RGB_.cols+u].normal.y+1.)/2.*255;
		float z = (SparseNorMap[v*image_RGB_.cols+u].normal.z+1.)/2.*255;
		float alpha = SparseNorMap[v*image_RGB_.cols+u].flag * 255;

		imgSparseNormalMap.ptr<uchar>(v)[u*C4Channels_RGB+0] = int(x);
		imgSparseNormalMap.ptr<uchar>(v)[u*C4Channels_RGB+1] = int(y);
		imgSparseNormalMap.ptr<uchar>(v)[u*C4Channels_RGB+2] = int(z);
		imgSparseNormalMap.ptr<uchar>(v)[u*C4Channels_RGB+3] = int(alpha);

	}
	imwrite("figure_SparseNormalMap_"+to_string(SizeSparseNormal_User_)+"_DisID_"+to_string(DistanceID_)+".png", imgSparseNormalMap);
	imshow("NormalMap_depthCam_", imgSparseNormalMap);


	// type 2 : analysis part 3
	// draw a map of NorMap_RGBCam_
	Mat imgNormal_Mapping_RGBCam_(image_RGB_.rows,image_RGB_.cols,CV_8UC4,Scalar::all(0)); // build a 4 channel mat, in which the normal vector corresponding to a pixel is stored.

	//int C4Channels_RGB = imgNormal_Mapping_RGBCam_.channels();

	for (int v = 0; v < image_RGB_.rows; ++v)
	for (int u = 0; u < image_RGB_.cols; ++u)
	{
		float x = (FullMap[v*image_RGB_.cols+u].normal.x+1.)/2.*255;
		float y = (FullMap[v*image_RGB_.cols+u].normal.y+1.)/2.*255;
		float z = (FullMap[v*image_RGB_.cols+u].normal.z+1.)/2.*255;
		float alpha = FullMap[v*image_RGB_.cols+u].flag * 255;

		//cout<<"NorMap_RGBCam_ - normal : "<<int(x)<<", "<<int(y)<<", "<<int(z)<<endl;

		if(x==0&&y==0&&z==0)
		{
			continue;
		}

		//uchar alpha	= NorMap_RGBCam_[v*image_RGB_.cols+u].alpha;

		imgNormal_Mapping_RGBCam_.ptr<uchar>(v)[u*C4Channels_RGB+0] = int(x);
		imgNormal_Mapping_RGBCam_.ptr<uchar>(v)[u*C4Channels_RGB+1] = int(y);
		imgNormal_Mapping_RGBCam_.ptr<uchar>(v)[u*C4Channels_RGB+2] = int(z);
		imgNormal_Mapping_RGBCam_.ptr<uchar>(v)[u*C4Channels_RGB+3] = int(alpha);

	}
	imwrite("figure_NormalMap_RGBCam_Size"+to_string(SizeSparseNormal_User_)+"_DisID_"+to_string(DistanceID_)+"RunID"+to_string(RunID)+".png", imgNormal_Mapping_RGBCam_);
	imshow("NormalMap_RGBCam_", imgNormal_Mapping_RGBCam_);

	// draw a map of depth
	Mat imgDepth_Mapping_RGBCam_(image_RGB_.rows,image_RGB_.cols,CV_8UC1,Scalar::all(0)); // build a 4 channel mat, in which the normal vector corresponding to a pixel is stored.

	//int C4Channels_RGB = imgNormal_Mapping_RGBCam_.channels();

	for (int v = 0; v < image_RGB_.rows; ++v)
	for (int u = 0; u < image_RGB_.cols; ++u)
	{
		float x = SparseDepMap[v*image_RGB_.cols+u].pt.x;
		float y = SparseDepMap[v*image_RGB_.cols+u].pt.y;
		float z = SparseDepMap[v*image_RGB_.cols+u].pt.z;
		float alpha = SparseDepMap[v*image_RGB_.cols+u].flag * 255;

		//cout<<"NorMap_RGBCam_ - normal : "<<int(x)<<", "<<int(y)<<", "<<int(z)<<endl;

		//if(x==0&&y==0&&z==0)
		//{
		//	continue;
		//}

		//uchar alpha	= NorMap_RGBCam_[v*image_RGB_.cols+u].alpha;
		float pz = (z-MIN_DISTANCE)/MAX_DISTANCE*255;

		imgDepth_Mapping_RGBCam_.ptr<uchar>(v)[u*imgDepth_Mapping_RGBCam_.channels()+0] = int(pz);
	}
	imwrite("figure_DepthMap_RGBCam_Size"+to_string(SizeSparseNormal_User_)+"_DisID_"+to_string(DistanceID_)+"RunID"+to_string(RunID)+".png", imgDepth_Mapping_RGBCam_);
	imshow("DepthMap_RGBCam_", imgDepth_Mapping_RGBCam_);

	// analysis part 4
	//
	// opencv taking points by mouse 
	//
	FullMap_ = FullMap;
	imshow(WinName, image_RGB_);
	setMouseCallback(WinName, On_mouse, 0);
	imwrite("figure_RGB_"+to_string(RunID)+".png", image_RGB_);


	// compute manually
	//waitKey(0);

	// compute automatically
	sleep(0.1);

	return 0;
}

//
// Analysis for situations of SizeSparseCloud
//
void Analysis_SizeSparseCloud(double &meanTime, double &stdDev)
{
	//
	// analysis
	//

	//// control group
	//double TimeInterval_control = 0;
	//int isRunGood1 = Run_ControlGroup(0, TimeInterval_control);
	//WriteNormalMap("data_NorMap_RGBCam_Control.txt",NorMap_RGBCam_Control_, image_RGB_.rows, image_RGB_.cols);

	//  write file

	vector<double> TimeIntervalList;
	for(int i=0;i<5;i++)
	{
		double TimeInterval = 0;
		int isRunGood2 = Run(i, TimeInterval);
		//cout<<"RunID "<<i<<", Fast Normal Estimation, timeInterval_ per image: "<<TimeInterval<<" second"<<endl;

		TimeIntervalList.push_back(TimeInterval);
	}

	//  mean 
	meanTime = 0;
	for(int i=0;i<TimeIntervalList.size();i++)
	{
		meanTime += (TimeIntervalList[i]/double(TimeIntervalList.size()));
	}

	// stdDev
	stdDev = 0;
	for(int i=0;i<TimeIntervalList.size();i++)
	{
		double part1 = (TimeIntervalList[i]-meanTime)*(TimeIntervalList[i]-meanTime);
		stdDev += part1;
	}
	stdDev = sqrt(stdDev/double(TimeIntervalList.size()));

	cout<<"Run Time of the fast nromal estimation: (mean +- stdDev) "<<meanTime<<" +- "<<stdDev<<endl;
}

int main( int argc, char** argv )
{
	cout<<"Hello"<<endl;


	vector<int> ListSizeSparseNormal;
	ListSizeSparseNormal.push_back(1e5);

	// write data
	ofstream file("data_timeInterval_sparseNormal.txt"); // margin was not eliminated
	//Analysis_SizeSparseCloud(meanTime, stdDev);

	// step 0 : control group
	double TimeInterval_control = 0;

	SizeSparseNormal_User_ = 3.5e5;
	//SizeSparseNormal_User_ = 1e4;
	int isRunGood1 = Run_ControlGroup(0, TimeInterval_control);
	//WriteNormalMap("data_NorMap_RGBCam_Control.txt",NorMap_RGBCam_Control_, image_RGB_.rows, image_RGB_.cols);

//	for(int i=0;i<ListSizeSparseNormal.size();i++)
//	{
//		cout<<"\n----"<<endl;
//
//		// step 1 : set size of the sparse normal 
//		SizeSparseNormal_User_ = ListSizeSparseNormal[i];
//		cout<<"SizeSparseNormal_User_ has been set to "<<SizeSparseNormal_User_<<endl;
//
//		// step 2 : analysis
//		double meanTime = 0;
//		double stdDev = 0;
//		Analysis_SizeSparseCloud(meanTime, stdDev);
//
//		file<<SizeSparseNormal_User_<<" "<<meanTime<<" "<<stdDev<<endl;
//	}
//	file.close();





//	// test
//	SizeSparseNormal_User_ = 1e5;
//	cout<<"SizeSparseNormal_User_ : "<<SizeSparseNormal_User_<<endl;
//
//	double meanTime;
//	double stdDev;
//	Analysis_SizeSparseCloud(meanTime, stdDev);

//	// test : evaluation with multiple distances
//	double meanTime;
//	double stdDev;
//	SizeSparseNormal_User_ = 1e5;
//	for(int i=0;i<1;i++)
//	{
//		cout<<"\n----"<<endl;
//		cout<<"----------------"<<endl;
//		cout<<"Distance ID: "<<i<<endl;
//		DistanceID_ = i;
//		Analysis_SizeSparseCloud(meanTime, stdDev);
//	}


//	// BSpline test
//	vector<double> V;
//	int sizeWidth = 640;
//	int sizeHeight = 480;
//	for(int v=0; v<sizeHeight; v++)
//	{
//		V.push_back(v);
//	}
//    BSplineBasisFunction basisFuncV(V);
//
//	vector<double> Weight_BSpline;
//	int IntRad = 8; // interpolation radius
//	int degree = IntRad*2+1;
//
//	double ty = 50;
//	int size  = degree*2+1;
//	for(int i=0;i<IntRad+1;i++)
//	{
//		int knotv = ty - IntRad -1 + i;
//		double Bv = basisFuncV.EvalBasis(knotv, degree, ty);
//		//cout<<"ty: "<<ty<<"; knotv = "<<knotv+ceil(degree/2.)<<"; Bv = "<<Bv<<endl;
//		cout<<"ty: "<<ty<<"; knotv = "<<knotv<<"; Bv = "<<Bv<<endl;
//
//		Weight_BSpline.push_back(Bv);
//	}
//
//	for(int i=0;i<Weight_BSpline.size();i++)
//	{
//		cout<<"i: "<<i<<"; Weight_BSpline: "<<Weight_BSpline[i]<<endl;
//	}
//

    return 0;
}
