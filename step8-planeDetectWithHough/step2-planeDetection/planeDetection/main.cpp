#include <iostream>
#include <fstream>
#include <string>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/videoio.hpp"
// !OpenCV

using namespace std;
using namespace cv;

//
// Class Histogram
//
class Histogram3D
{
	public:
		Histogram3D(int binx, double minx, double maxx,
					int biny, double miny, double maxy,
					int binz, double minz, double maxz);
		~Histogram3D();

	public:
		void	Fill(double x, double y, double z, double weight);
		void	GetBinCentor(int IDx, int IDy, int IDz, double &x, double &y, double &z);
		double	GetBinCounts(int IDx, int IDy, int IDz);
		int		GetID(int IDx, int IDy, int IDz);
		void	Show();
	
	private:
		int binx_;
		int biny_;
		int binz_;

		double minx_, maxx_;
		double miny_, maxy_;
		double minz_, maxz_;

		double binWidthx_;
		double binWidthy_;
		double binWidthz_;

		vector<double> Hist_; // the 3D histogram

		int CountsLower_;
		int CountsUpper_;

};

Histogram3D::Histogram3D(int binx, double minx, double maxx,
					int biny, double miny, double maxy,
					int binz, double minz, double maxz)
{
	binx_ =  binx;
    biny_ =  biny;
    binz_ =  binz;

	minx_ =  minx;
    miny_ =  miny;
    minz_ =  minz;

	maxx_ =  maxx;
    maxy_ =  maxy;
    maxz_ =  maxz;

	binWidthx_ = (maxx_-minx_)/double(binx_);
    binWidthy_ = (maxy_-miny_)/double(biny_);
    binWidthz_ = (maxz_-minz_)/double(binz_);

	CountsLower_ = 0;
	CountsUpper_ = 0;

	// debug
	cout<<"Histogram: "<<endl;
	cout<<"X - bin, min, max, binWidth: "<<binx_<<", "<<minx_<<", "<<maxx_<<", "<<binWidthx_<<endl;
	cout<<"Y - bin, min, max, binWidth: "<<biny_<<", "<<miny_<<", "<<maxy_<<", "<<binWidthy_<<endl;
	cout<<"Z - bin, min, max, binWidth: "<<binz_<<", "<<minz_<<", "<<maxz_<<", "<<binWidthz_<<endl;

	// initialize the 3D histogram
	int size = binx_*biny_*binz_;
	for(int i=0;i<size;i++)
	{
		Hist_.push_back(0);
	}

}

Histogram3D::~Histogram3D()
{}

void Histogram3D::Fill(double x, double y, double z, double weight)
{
	//
	// ID = IDx*biny_*binz_ + IDy*binz_ + IDz

	if(x<=minx_||y<=miny_||z<=minz_) 
	{
		CountsLower_ ++;
		return;
	}

	if(x>=maxx_||y>=maxy_||z>=maxz_) 
	{
		CountsUpper_ ++;
		return;
	}

	int IDx = ceil((x-minx_)/binWidthx_);
	int IDy = ceil((y-miny_)/binWidthy_);
	int IDz = ceil((z-minz_)/binWidthz_);

	int ID = GetID(IDx, IDy, IDz);

	Hist_[ID] += weight;
}

void Histogram3D::GetBinCentor(int IDx, int IDy, int IDz, double &x, double &y, double &z)
{
	x = minx_ + double(IDx+0.5)*binWidthx_;
	y = miny_ + double(IDy+0.5)*binWidthy_;
	z = minz_ + double(IDz+0.5)*binWidthz_;
}

double	Histogram3D::GetBinCounts(int IDx, int IDy, int IDz)
{
	int ID = GetID(IDx, IDy, IDz);
	double counts = Hist_[ID];

	return counts;
}

int Histogram3D::GetID(int IDx, int IDy, int IDz)
{
	int ID = IDx*biny_*binz_ + IDy*binz_ + IDz;
	return ID;
}

void Histogram3D::Show()
{
	for(int i=0;i<binx_;i++)
	for(int j=0;j<biny_;j++)
	for(int k=0;k<binz_;k++)
	{
		double x = 0;
		double y = 0;
		double z = 0;
		GetBinCentor(i, j, k, x, y, z);

		double counts = GetBinCounts(i, j, k);

		cout<<"ID(x,y,z): "<<i<<", "<<j<<", "<<k
			<<"; BinCentor(x,y,z): "<<x<<", "<<y<<", "<<z
			<<"; Counts: "<<counts<<endl;
	}
}

//
// !Class Histogram
//

int ReadPointCloud(string filename, vector<Point3f> &points)
{
	points.clear();

	//
	// import 3D point cloud
	//
	ifstream file(filename);

	if(file.fail())
	{
		cout<<"Can not find the file \" "<<filename<<" \""<<endl;
		return 0;
	}

	double x,y,z;
	double nx,ny,nz;
	double flag;
	while(!file.eof())
	{
		file>>x>>y>>z>>nx>>ny>>nz>>flag;

		if(file.eof()) break;

		if(flag==0) continue;

		if(x==0&&y==0&&z==0) continue;

		if(z>1500) continue;

		Point3f	p(x,y,z);
		points.push_back(p);
	}

	return 1;
}

int main(int argc, char *argv[])
{
	cout<<"Hello"<<endl;

	vector<Point3f> points;
	string filename = "../../step1-pointCloud/build-3DCameraNormalEstimation-Desktop_Qt_5_14_2_GCC_64bit-Debug/data_NorMap_RGBCam_Control_350000.txt";
	int IsFileGood = ReadPointCloud(filename, points);

	// debug
	cout<<"points.size(): "<<points.size()<<endl;
	// !debug

	// get the maximum value of distances
	double maxDist = 0;
	for(int i=0;i<points.size();i++)
	{
		double x = points[i].x;
		double y = points[i].y;
		double z = points[i].z;
		double dist = sqrt(x*x + y*y + z*z);

		if(maxDist<dist) maxDist=dist;
	}
	cout<<"maxDist : "<<maxDist<<endl;
	// !get the maximum value of distances

	int N_bin = 10;
	double min = 0.;
	double maxx = M_PI; // theta
	double maxy = 2.*M_PI; // phi
	double maxz = maxDist; // r, maximum distance from the point to the origin

	Histogram3D h(	N_bin, min, maxx, 
					N_bin, min, maxy,
					N_bin, min, maxz
				);

	// test
	//double mint = 10;
	//double binWidth = 3;
	//double t1 = 9;
	//double t2 = ((t1 - mint)/binWidth);
	//cout<<"t2: "<<t2<<", ceil "<<ceil(t2)<<endl;
	// !test

	// fill histogram
	for(int i=0;i<points.size(); i++)
	{
		for(int hi=0;hi<N_bin;hi++)
		for(int hj=0;hj<N_bin;hj++)
		{
			double x = points[i].x;
			double y = points[i].y;
			double z = points[i].z;

			double theta = min + double(hi+0.5)*maxx;
			double phi   = min + double(hj+0.5)*maxy;
			double r =	x*sin(theta)*cos(phi) 
					  + y*sin(theta)*sin(phi)
					  + z*cos(theta);
			h.Fill(theta, phi, r, 1);
		}
	}

	h.Show();



    return 1;
}
