#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <cmath>

#include "Eigen/Eigen"

#include <nlopt.h>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/videoio.hpp"
// !OpenCV

using namespace std;
using namespace Eigen;
using namespace cv;

using std::default_random_engine;
using std::uniform_real_distribution;
using std::normal_distribution;

vector<Point3f> Points_;

default_random_engine engine_(time(0));

// NLOPT
double myfunc(unsigned n, const double *x, double *grad, void *my_func_data)
{
	// x[0] : x_0
	// x[1] : y_0
	// x[2] : r

    if (grad) {
        //grad[0] = 0.0;
        //grad[1] = 0.5 / sqrt(x[1]);

        grad[0] = 0.0;
        grad[1] = 0.0;
        grad[2] = 0.0;
		for(int i=0;i<Points_.size();i++)
		{
			Point3f pt = Points_[i];
			double difference =	  (x[0]-pt.x)*(x[0]-pt.x) 
								+ (x[1]-pt.y)*(x[1]-pt.y) 
								- x[2]*x[2];
			grad[0] += (4.*(x[0]-pt.x)*difference); 
			grad[1] += (4.*(x[1]-pt.y)*difference); 
			grad[2] += (-4.*x[2]*difference); 

		}
    }

	double target = 0;
	for(int i=0;i<Points_.size();i++)
	{
		Point3f pt = Points_[i];
		double difference =	  (x[0]-pt.x)*(x[0]-pt.x) 
							+ (x[1]-pt.y)*(x[1]-pt.y) 
							- x[2]*x[2];
		target += (difference*difference);
	}

    return target;
}

typedef struct {
    double a, b;
} my_constraint_data;

double myconstraint(unsigned n, const double *x, double *grad, void *data)
{
    my_constraint_data *d = (my_constraint_data *) data;
    double a = d->a, b = d->b;
    if (grad) {
        grad[0] = 3 * a * (a*x[0] + b) * (a*x[0] + b);
        grad[1] = -1.0;
    }
    return ((a*x[0] + b) * (a*x[0] + b) * (a*x[0] + b) - x[1]);
}

// !NLOPT

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

void CylinderEstimationEvaluation(vector<double> paraCyl, vector<Point3f> points, double &difference)
{
	// paraCyl[0] : x_0
	// paraCyl[1] : y_0
	// paraCyl[2] : z_0
	// paraCyl[3] : a
	// paraCyl[4] : b
	// paraCyl[5] : c
	// paraCyl[6] : r

	double x[7];
	for(int i=0;i<7;i++)
	{
		x[i] = paraCyl[i];
	}

	// calculate difference 
	difference = 0;
	for(int i=0;i<points.size();i++)
	{
		Point3f pt = points[i];

		double part1 =	  x[3]*(pt.x-x[0])
						+ x[4]*(pt.y-x[1])
						+ x[5]*(pt.z-x[2]);

		double part3 =    x[3]*x[3]
						+ x[4]*x[4]
						+ x[5]*x[5];

		double part2 =	  (x[0]-pt.x)*(x[0]-pt.x) 
						+ (x[1]-pt.y)*(x[1]-pt.y) 
						+ (x[2]-pt.z)*(x[2]-pt.z) 
						- part1*part1/part3
						- x[6]*x[6];

		difference += sqrt(abs(part2));
	}

	difference = difference/double(points.size());

	return;
}



void CylinderEstimation(vector<Point3f> &points, double GaussNoise_StdDev_, double &difference)
{
	cout<<"\n----"<<endl;
	cout<<"Cylinder Estimation"<<endl;
	cout<<"GaussNoise_StdDev_: "<<GaussNoise_StdDev_<<endl;
	//
	// step 1 : generate a piont cloud in which the point are made by the cylinder formula
	//

	// build a sphere
	// cylinder function
	// x = x0 + r0*cos(theta)
	// y = y0 + r0*sin(theta)
	// z ~ (-2,2) + z0
	//vector<Point3f> points;

	points.clear();

	double x0 = 120;
	double y0 = 0;
	double z0 = 0;
	double r0 = 100;

	default_random_engine engine_(time(0));
    uniform_real_distribution<double> RanTheta(0, 2.*M_PI);
    uniform_real_distribution<double> RanHeight(-2,2);

	normal_distribution<double> noise_Gauss(0, GaussNoise_StdDev_);

	int size = 10000;
	for(int i=0;i<size;i++)
	{
		double theta = RanTheta(engine_);
		double height = RanHeight(engine_);

		//p.x = x0 + r0*cos(theta);
		//p.y = y0 + r0*sin(theta);
		//p.z = z0 + 10.*height;

		//p.x = x0 + 10.*height;
		//p.y = y0 + r0*sin(theta);
		//p.z = z0 + r0*cos(theta);

		MatrixXd pOri(3,1);
		pOri(0,0) = x0 + noise_Gauss(engine_) + r0*cos(theta);
		pOri(1,0) = y0 + noise_Gauss(engine_) + r0*sin(theta);
		pOri(2,0) = z0 + noise_Gauss(engine_) + 10.*height;   

		// rotate
		double angle = 40./180.*M_PI;
		AngleAxisd R(angle, Vector3d(0,1,0));

		MatrixXd pRot = MatrixXd::Zero(3,1); // the rotated point
		pRot = R.matrix()*pOri;
		
		// save the point
		Point3f p;
		p.x = pRot(0,0);
		p.y = pRot(1,0);
		p.z = pRot(2,0);

		points.push_back(p);
	}

	//
	// output points of the cylinder 
	//
	string filename_Ori = "cylinder_origin_std"+to_string(int(GaussNoise_StdDev_))+".txt";
	ofstream file(filename_Ori);
	for(int i=0;i<points.size();i++)
	{
		Point3f pt = points[i];
		file<<pt.x<<" "<<pt.y<<" "<<pt.z<<" "<<1<<endl;
	}
	file.close();

	//
	// step 2 : estimate the axis of the cylinder
	//
	cout<<"step 2 : estimate the axis of the cylinder"<<endl;
	vector<Point3f> normalVectors;

	// step 2.1 : get normal vectors
	double numberPointsUsedByANormal = 0;
	for(int i=0;i<size;i++)
	{
		//cout<<"\nstep 2 : estimate the axis of the cylinder; Point ID: "<<i<<endl;

		vector<Point3f> points_cur;
		for(int j=0;j<size;j++)
		{
			//double distance2 =  norm(points[i]-points[j]);

			double distance2 =	  pow(points[i].x-points[j].x, 2)
								+ pow(points[i].y-points[j].y, 2)
								+ pow(points[i].z-points[j].z, 2);
			double threshold = 550;
			if(distance2<threshold)
			{
				points_cur.push_back(points[j]);
			}
		}

		// debug
		//cout<<"size of the points : "<<points_cur.size()<<endl;
		numberPointsUsedByANormal += (points_cur.size()/double(points.size()));

		// estimate the normal of the current point  
		Point3f planePoints[points_cur.size()];
		Point3f planeNormal(0,0,0);
		for(int i=0;i<points_cur.size();i++)
		{
			planePoints[i] = points_cur[i];
		}
		GetNormal(points_cur.size(), planePoints, planeNormal);
		//cout<<"planeNormal: "<<planeNormal<<endl;

		normalVectors.push_back(planeNormal);
	}

	//debug
	cout<<"numberPointsUsedByANormal: "<<numberPointsUsedByANormal<<endl;
	if(numberPointsUsedByANormal<8)
	{
		cout<<"!!!!!!!!!!!!!"<<endl;
		cout<<"!!!! A Warning should be noticed: number of the points that were used to estimate the normal for each point of the cylinder is not enough."<<endl;
		cout<<"!!!! We should increase the threshold."<<endl;
		cout<<"!!!!!!!!!!!!!"<<endl;
	}
	//!debug

	// step 2.2 : estimate cylinder axis, which is NOT the rotation axis 
	Point3f planePoints[normalVectors.size()];
	Point3f cylinderAxis(0,0,0);
	for(int i=0;i<normalVectors.size();i++)
	{
		planePoints[i] = normalVectors[i];
	}
	GetNormal(normalVectors.size(), planePoints, cylinderAxis);
	cout<<"cylinderAxis: "<<cylinderAxis<<endl;

	// step 2.3 : calculate rotation axis and angle 
	Point3f ZAxis(0,0,1);
	//Point3f rotatoinAxis = ZAxis.cross(cylinderAxis);
	Point3f rotatoinAxis = cylinderAxis.cross(ZAxis);
	rotatoinAxis /= norm(rotatoinAxis); // normalization

	double cosAngle = ZAxis.dot(cylinderAxis)/(norm(ZAxis)*norm(cylinderAxis));
	double angle = acos(cosAngle);
	cout<<"angle between ZAxis and cylinderAxis: "<<angle*180./M_PI<<" degrees"<<endl;
	cout<<"Rotation axis: "<<rotatoinAxis<<endl;

	// step 2.4 : calculate rotation matrix according to the angle-axis 
	// (1) : Eigen is used
	AngleAxisd R_AngleAxis(angle, Vector3d(rotatoinAxis.x, rotatoinAxis.y, rotatoinAxis.z));
	MatrixXd C_OR = R_AngleAxis.matrix(); // C_OR
	MatrixXd C_RO = C_OR.transpose(); // C_RO

	// debug
	//cout<<"Rotation Matrix of the cylinder coordinate frame with respected to the world coordinate system: \n"<<C_OR<<endl; 
	//cout<<"Rotation Matrix of the world coordinate frame with respected to the cylinder coordinate system: \n"<<C_RO<<endl; 

	// step 2.5 : translate the point cloud to the coordinate frame of cylinder 
	Points_.clear();
	for(int i=0;i<points.size();i++)
	{
		MatrixXd p_O(3,1);
		p_O(0,0) = points[i].x;
		p_O(1,0) = points[i].y;
		p_O(2,0) = points[i].z;

		//MatrixXd p_R = C_RO*p_O;
		MatrixXd p_R = C_OR*p_O;

		// save the point
		Point3f p;
		p.x = p_R(0,0);
		p.y = p_R(1,0);
		p.z = p_R(2,0);
		Points_.push_back(p);
	}

	//
	// output points of the cylinder with respect to the cylinder coordinate system
	//
	string filename_Rot = "cylinder_rotated_std"+to_string(int(GaussNoise_StdDev_))+".txt";
	ofstream file_Rot(filename_Rot);
	for(int i=0;i<Points_.size();i++)
	{
		Point3f pt = Points_[i];
		file_Rot<<pt.x<<" "<<pt.y<<" "<<pt.z<<" "<<1<<endl;
	}
	file_Rot.close();

	//
	// step 3 : Optimization
	//
	cout<<"\nstep 3 : Optimization"<<endl;
	// step 3.1 : calculate radius and position of the origin of the circle
	double lb[3] = { -HUGE_VAL, -HUGE_VAL, 0 }; /* lower bounds */
	nlopt_opt opt;

	// generate an optimum problem
	opt = nlopt_create(NLOPT_LD_MMA, 3); /* algorithm and dimensionality */
	nlopt_set_lower_bounds(opt, lb);
	nlopt_set_min_objective(opt, myfunc, NULL);

	nlopt_set_xtol_rel(opt, 1e-4);

	// start optimization 
	double x[3] = { 0,0, 1 };  /* `*`some` `initial` `guess`*` */

	double minf; /* `*`the` `minimum` `objective` `value,` `upon` `return`*` */
	nlopt_optimize(opt, x, &minf);
	if(minf<0)
	{
		cout<<"nlopt failed!"<<endl;
	}
	else {
	    //printf("found minimum at f(%g,%g) = %0.10g\n", x[0], x[1], minf);
		cout<<"found minimum at: "<<x[0]<<", "<<x[1]<<", "<<x[2]<<"; minf: "<<minf<<endl;
	}

	// end
	nlopt_destroy(opt);

	
	//
	// step 4 : generate a cylinder corresponding to the estimated cylinder parameters 
	//
	double zmin = 0;
	double zmax = 0;
	for(int i=0;i<Points_.size();i++)
	{
		double z = Points_[i].z;
		if(z<zmin) zmin=z;
		if(z>zmax) zmax=z;
	}
	cout<<"zmin and zmax"<<zmin<<", "<<zmax<<endl;

    uniform_real_distribution<double> RanTheta_est(0, 2.*M_PI);
    uniform_real_distribution<double> RanHeight_est(zmin,zmax);

	vector<Point3f> points_est_rot; // the point cloud that is estimated and is represented in the cylinder coordinate system
	vector<Point3f> points_est;// the point cloud that is estimated and is represented in the world coordinate system
	int size_est = size;
	for(int i=0;i<size_est;i++)
	{
		double theta = RanTheta_est(engine_);
		double height = RanHeight_est(engine_);

		MatrixXd pR(3,1);
		pR(0,0) = x[0] + x[2]*cos(theta);
		pR(1,0) = x[1] + x[2]*sin(theta);
		pR(2,0) = height;   

		// rotate
		MatrixXd pO = C_RO*pR;
		//MatrixXd pO = C_OR*pR;

		// save the point 
		Point3f pCyl;
		pCyl.x = pR(0,0);
		pCyl.y = pR(1,0);
		pCyl.z = pR(2,0);
		points_est_rot.push_back(pCyl);

		// save the point
		Point3f p;
		p.x = pO(0,0);
		p.y = pO(1,0);
		p.z = pO(2,0);
		points_est.push_back(p);
	}
	//
	// output points of the estimated cylinder 
	//
	string filename_EstCyl = "cylinder_esti_cyl_std"+to_string(int(GaussNoise_StdDev_))+".txt";
	ofstream file_EstCyl(filename_EstCyl);
	for(int i=0;i<points_est_rot.size();i++)
	{
		Point3f pt = points_est_rot[i];
		file_EstCyl<<pt.x<<" "<<pt.y<<" "<<pt.z<<" "<<1<<endl;
	}
	file_EstCyl.close();

	string filename_Est = "cylinder_esti_std"+to_string(int(GaussNoise_StdDev_))+".txt";
	ofstream file_Est(filename_Est);
	for(int i=0;i<points_est.size();i++)
	{
		Point3f pt = points_est[i];
		file_Est<<pt.x<<" "<<pt.y<<" "<<pt.z<<" "<<1<<endl;
	}
	file_Est.close();

	//
	// step 5 : evaluation
	//
	// step 5.1 : tranform the origin of the circle to the world coordinate system
	MatrixXd circleOri_R(3,1);
	circleOri_R(0,0) = x[0];
	circleOri_R(1,0) = x[1];
	circleOri_R(2,0) = 0;
	MatrixXd circleOri_O = C_RO*circleOri_R;

	MatrixXd clyAxis_R(3,1);
	clyAxis_R(0,0) = 0;
	clyAxis_R(1,0) = 0;
	clyAxis_R(2,0) = 1;
	MatrixXd clyAxis_O = C_RO*clyAxis_R;


	vector<double> paraCyl;
	paraCyl.push_back(circleOri_O(0,0)); // x_0
	paraCyl.push_back(circleOri_O(1,0)); // y_0
	paraCyl.push_back(circleOri_O(2,0)); // z_0
	paraCyl.push_back(clyAxis_O(0,0)); // a
	paraCyl.push_back(clyAxis_O(1,0)); // b
	paraCyl.push_back(clyAxis_O(2,0)); // c
	paraCyl.push_back(x[2]); // r

	CylinderEstimationEvaluation(paraCyl, points, difference);
}

void planeGenerator(int ID, double GaussNoise_StdDev_, vector<Point3f> &points, vector<Point3f> &normalVectors)
{
	//
	// step 1 : prepare parameters
	//
	// clear the vectors
	points.clear();
	normalVectors.clear();

	//
	// step 2 : generate a plane
	//
    uniform_real_distribution<double> RanLocation(-100, 100);

	int size = 10000;
	for(int i=0;i<size;i++)
	{
		double x = RanLocation(engine_);
		double y = RanLocation(engine_);
		double z = -300;

		// save the point
		Point3f p;
		p.x = x;
		p.y = y;
		p.z = z;

		points.push_back(p);
	}

	//
	// step 3 : estimate the normal of the plane 
	//
	cout<<"step 3 : estimate the normal of the plane"<<endl;
	double numberPointsUsedByANormal = 0;
	for(int i=0;i<size;i++)
	{
		vector<Point3f> points_cur;
		for(int j=0;j<size;j++)
		{
			//double distance2 =  norm(points[i]-points[j]);

			double distance2 =	  pow(points[i].x-points[j].x, 2)
								+ pow(points[i].y-points[j].y, 2)
								+ pow(points[i].z-points[j].z, 2);
			double threshold = 25;
			if(distance2<threshold)
			{
				points_cur.push_back(points[j]);
			}
		}

		// debug
		//cout<<"size of the points : "<<points_cur.size()<<endl;
		numberPointsUsedByANormal += (points_cur.size()/double(points.size()));

		// estimate the normal of the current point  
		Point3f planePoints[points_cur.size()];
		Point3f planeNormal(0,0,0);
		for(int i=0;i<points_cur.size();i++)
		{
			planePoints[i] = points_cur[i];
		}
		GetNormal(points_cur.size(), planePoints, planeNormal);
		//cout<<"planeNormal: "<<planeNormal<<endl;

		normalVectors.push_back(planeNormal);
	}

	//debug
	cout<<"numberPointsUsedByANormal: "<<numberPointsUsedByANormal<<endl;
	if(numberPointsUsedByANormal<8)
	{
		cout<<"!!!!!!!!!!!!!"<<endl;
		cout<<"!!!! A Warning should be noticed: number of the points that were used to estimate the normal for each point of the cylinder is not enough."<<endl;
		cout<<"!!!! We should increase the threshold."<<endl;
		cout<<"!!!!!!!!!!!!!"<<endl;
	}
	//!debug

	return;
}


void cylinderGenerator(int ID, double GaussNoise_StdDev_, vector<Point3f> &points, vector<Point3f> &normalVectors)
{
	//
	// step 1 : prepare parameters
	//
	// random number generators
	// x_0, y_0, z_0
    uniform_real_distribution<double> RanLocation(-100, 100);
    uniform_real_distribution<double> RanRadius(50, 100);

    uniform_real_distribution<double> RanTheta(0, 2.*M_PI);
    uniform_real_distribution<double> RanHeight(-2,2);

    uniform_real_distribution<double> RanRotationAngle(0, 2.*M_PI);
    uniform_real_distribution<double> RanRotationAxis(0, 1.);


	normal_distribution<double> noise_Gauss(0, GaussNoise_StdDev_);

	// clear the vectors
	points.clear();
	normalVectors.clear();


	//
	// step 2 :
	//
	// cylinder function
	// x = x0 + r0*cos(theta)
	// y = y0 + r0*sin(theta)
	// z ~ (-2,2) + z0

	double x0 = RanLocation(engine_);
	double y0 = RanLocation(engine_);
	double z0 = RanLocation(engine_);
	double r0 = RanRadius(engine_);

	// rotate
	double angle = RanRotationAngle(engine_);
	MatrixXd rotAxis(3,1);
	rotAxis(0,0) = RanRotationAxis(engine_);
	rotAxis(1,0) = RanRotationAxis(engine_);
	rotAxis(2,0) = RanRotationAxis(engine_);
	rotAxis /= rotAxis.norm();
	AngleAxisd R(angle, Vector3d(rotAxis(0,0),rotAxis(1,0),rotAxis(2,0)));

	cout<<"rotAxis: "<<rotAxis<<endl;
	cout<<"angle: "<<angle/M_PI*180<<endl;

	int size = 10000;
	for(int i=0;i<size;i++)
	{
		double theta = RanTheta(engine_);
		double height = RanHeight(engine_);

		//p.x = x0 + r0*cos(theta);
		//p.y = y0 + r0*sin(theta);
		//p.z = z0 + 10.*height;

		//p.x = x0 + 10.*height;
		//p.y = y0 + r0*sin(theta);
		//p.z = z0 + r0*cos(theta);

		MatrixXd pOri(3,1);
		pOri(0,0) = x0 + noise_Gauss(engine_) + r0*cos(theta);
		pOri(1,0) = y0 + noise_Gauss(engine_) + r0*sin(theta);
		pOri(2,0) = z0 + noise_Gauss(engine_) + 10.*height;   

		MatrixXd pRot = MatrixXd::Zero(3,1); // the rotated point
		pRot = R.matrix()*pOri;
		
		// save the point
		Point3f p;
		p.x = pRot(0,0);
		p.y = pRot(1,0);
		p.z = pRot(2,0);

		points.push_back(p);
	}

	//
	// step 3 : estimate the axis of the cylinder
	//
	cout<<"step 3 : estimate the axis of the cylinder"<<endl;
	double numberPointsUsedByANormal = 0;
	for(int i=0;i<size;i++)
	{
		vector<Point3f> points_cur;
		for(int j=0;j<size;j++)
		{
			//double distance2 =  norm(points[i]-points[j]);

			double distance2 =	  pow(points[i].x-points[j].x, 2)
								+ pow(points[i].y-points[j].y, 2)
								+ pow(points[i].z-points[j].z, 2);
			double threshold = 25;
			if(distance2<threshold)
			{
				points_cur.push_back(points[j]);
			}
		}

		// debug
		//cout<<"size of the points : "<<points_cur.size()<<endl;
		numberPointsUsedByANormal += (points_cur.size()/double(points.size()));

		// estimate the normal of the current point  
		Point3f planePoints[points_cur.size()];
		Point3f planeNormal(0,0,0);
		for(int i=0;i<points_cur.size();i++)
		{
			planePoints[i] = points_cur[i];
		}
		GetNormal(points_cur.size(), planePoints, planeNormal);
		//cout<<"planeNormal: "<<planeNormal<<endl;

		normalVectors.push_back(planeNormal);
	}

	//debug
	cout<<"numberPointsUsedByANormal: "<<numberPointsUsedByANormal<<endl;
	if(numberPointsUsedByANormal<8)
	{
		cout<<"!!!!!!!!!!!!!"<<endl;
		cout<<"!!!! A Warning should be noticed: number of the points that were used to estimate the normal for each point of the cylinder is not enough."<<endl;
		cout<<"!!!! We should increase the threshold."<<endl;
		cout<<"!!!!!!!!!!!!!"<<endl;
	}
	//!debug


	return;
}

int main( int argc, char** argv )
{
	cout<<"hello"<<endl;

	// test 
	//Point3f t1(0,1,2);
	//Point3f t2(3,1,2);
	//Point3f t3 = t2-t1;
	//cout<<t1<<", norm: "<<norm(t1)<<endl;
	//double distance2 = pow(t1.x-t2.x, 2);
	//cout<<"distance2: "<<distance2<<endl;
	//Point3f t4 = t2.cross(t1);
	//t4 = t4/norm(t4);
	//cout<<"t4=t2 cross t1: "<<t4<<"; its norm: "<<norm(t4)<<endl;
	//double cosAngle = t2.dot(t1)/(norm(t1)*norm(t2));
	//double angle = acos(cosAngle);
	//cout<<"angle between t1 and t2: "<<angle*180./M_PI<<endl;


	// step 0 : prepare functions
	void GetNormal(int size, Point3f points[], Point3f &normal);

	//void CylinderEstimation(vector<Point3f> &points, double GaussNoise_StdDev_, double &difference);

	// step 1 : cylinder estimation
	//double GaussNoise_StdDev_ = 1.;
	//double difference = 0;
	//vector<Point3f> points;
	//CylinderEstimation(points, GaussNoise_StdDev_, difference);

	// debug
	//cout<<"size of the point cloud: "<<points.size()<<endl;
	//for(int i=0;i<points.size();i++)
	//{
	//	cout<<"Point ID: "<<i<<"; loc: "<<points[i]<<endl;
	//}
	// !debug


	//ofstream file_accuracy("cylinderEstimationAccuracy.txt");
	//int numberOfTests = 4;
	//for(int i=0;i<numberOfTests;i++)
	//{
	//	double GaussNoise_StdDev_ = 0. + 2.*i;
	//	double difference = 0;
	//	vector<Point3f> points;
	//	CylinderEstimation(points, GaussNoise_StdDev_, difference);
	//	file_accuracy<<int(GaussNoise_StdDev_)<<" "<<difference<<endl;
	//}
	//file_accuracy.close();

	//
	// generate cylinders
	//
	int numberCylinders = 20;
	double GaussNoise_StdDev_ = 1.;
	vector<Point3f> points;
	vector<Point3f> normalVectors;

	string filename = "cylinders.txt";
	ofstream file(filename);

	for(int i=0;i<numberCylinders;i++)
	{
		cylinderGenerator(i, GaussNoise_StdDev_, points, normalVectors);

		// output points of the cylinder 
		for(int i=0;i<points.size();i++)
		{
			Point3f pt  = points[i];
			Point3f nor = normalVectors[i];

			file<<pt.x<<" "<<pt.y<<" "<<pt.z<<" ";
			file<<nor.x<<" "<<nor.y<<" "<<nor.z<<endl;
		}
	}

	//
	// generate a plane
	//
	planeGenerator(0, GaussNoise_StdDev_, points, normalVectors);

	// output points of the cylinder 
	for(int i=0;i<points.size();i++)
	{
		Point3f pt  = points[i];
		Point3f nor = normalVectors[i];

		file<<pt.x<<" "<<pt.y<<" "<<pt.z<<" ";
		file<<nor.x<<" "<<nor.y<<" "<<nor.z<<endl;
	}


	file.close();


	return 1;
}
