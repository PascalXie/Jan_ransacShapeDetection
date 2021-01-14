#include <fstream>
#include <iostream>
#include <string>

#include <CGAL/Timer.h>
#include <CGAL/number_utils.h>
#include <CGAL/property_map.h>
#include <CGAL/IO/read_xyz_points.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

#include <CGAL/Shape_detection/Efficient_RANSAC.h>

// Type declarations.
typedef CGAL::Exact_predicates_inexact_constructions_kernel	Kernel;
typedef Kernel::FT											FT;
typedef std::pair<Kernel::Point_3, Kernel::Vector_3>		Point_with_normal;
typedef std::vector<Point_with_normal>						Pwn_vector;
typedef CGAL::First_of_pair_property_map<Point_with_normal>	Point_map;
typedef CGAL::Second_of_pair_property_map<Point_with_normal> Normal_map;

typedef CGAL::Shape_detection::Efficient_RANSAC_traits
<Kernel, Pwn_vector, Point_map, Normal_map>					Traits;
typedef CGAL::Shape_detection::Efficient_RANSAC<Traits>		Efficient_ransac;
typedef CGAL::Shape_detection::Plane<Traits>				Plane;
typedef CGAL::Shape_detection::Sphere<Traits>				Sphere;
typedef CGAL::Shape_detection::Cylinder<Traits>				Cylinder;

using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::string;
using std::to_string;

int ReadPointCloud(string filename, Pwn_vector &points)
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

		Kernel::Point_3		p(x,y,z);
		Kernel::Vector_3	v(nx,ny,nz);

		Point_with_normal PAndN(p,v);
		points.push_back(PAndN);
	}

	return 1;
}

int main(int argc, char** argv) {

	// Points with normals.
	Pwn_vector points;

	string filename_read = "../../step1-pointCloud/build-3DCameraNormalEstimation-Desktop_Qt_5_14_2_GCC_64bit-Debug/data_NorMap_RGBCam_Control_350000.txt";
	int isFileGood = ReadPointCloud(filename_read, points);
	cout<<"Size of the point cloud: "<<points.size()<<endl;


//	// Load point set from a file.
//	std::ifstream stream((argc > 1) ? argv[1] : "../../step1-generateCylinders/build-cylinderGenerator-Desktop_Qt_5_14_2_GCC_64bit-Debug/cylinders.txt");
//
//	if (!stream ||
//		!CGAL::read_xyz_points(
//			stream,
//			std::back_inserter(points),
//			CGAL::parameters::point_map(Point_map()).
//			normal_map(Normal_map()))) {
//
//		std::cerr << "Error: cannot read file cube.pwn!" << std::endl;
//		return EXIT_FAILURE;
//	}

	// point cloud check
	cout<<"points.size(): "<<points.size()<<endl;

	//// debug
	//for(int i=0;i<points.size();i++)
	//{
	//	Kernel::Point_3		p = points[i].first;
	//	Kernel::Vector_3	v = points[i].second;
	//	cout<<"Point ID: "<<i<<"; point: "<<p.x()<<", "<<p.y()<<", "<<p.z()<<"; ";
	//	cout<<"Normal: "<<v.x()<<", "<<v.y()<<", "<<v.z()<<endl;
	//}


	// Instantiate shape detection engine.
	Efficient_ransac ransac;

	// Provide input data.
	ransac.set_input(points);

	// Register detection of planes.
	ransac.add_shape_factory<Plane>();
	ransac.add_shape_factory<Sphere>();
	ransac.add_shape_factory<Cylinder>();

	// Measure time before setting up the shape detection.
	CGAL::Timer time;
	time.start();

	// Build internal data structures.
	ransac.preprocess();

	// Measure time after preprocessing.
	time.stop();

	std::cout << "preprocessing took: " << time.time() * 1000 << "ms" << std::endl;

	// Perform detection several times and choose result with the highest coverage.
	Efficient_ransac::Shape_range shapes = ransac.shapes();

	FT best_coverage = 0;
	for (std::size_t i = 0; i < 1; ++i) {

		// Reset timer.
		time.reset();
		time.start();

		// Detect shapes.
		ransac.detect();

		// Measure time after detection.
		time.stop();

		// Compute coverage, i.e. ratio of the points assigned to a shape.
		FT coverage =
		FT(points.size() - ransac.number_of_unassigned_points()) / FT(points.size());

		// Print number of assigned shapes and unassigned points.
		std::cout << "time: " << time.time() * 1000 << "ms" << std::endl;
		std::cout << ransac.shapes().end() - ransac.shapes().begin()
		<< " primitives, " << coverage << " coverage" << std::endl;

		// Choose result with the highest coverage.
		if (coverage > best_coverage) {

			best_coverage = coverage;

			// Efficient_ransac::shapes() provides
			// an iterator range to the detected shapes.
			shapes = ransac.shapes();
		}
	}

	//
	// output shapes
	//
	int ShapeCounter = 0;
	string filename = "data_pointsWithShapes.txt";
	ofstream file(filename);

	Efficient_ransac::Shape_range::iterator it = shapes.begin();
	while (it != shapes.end()) {

		string shapeType = "Shape";
		// Get specific parameters depending on the detected shape.
		if (Plane* plane = dynamic_cast<Plane*>(it->get())) {

			//Kernel::Vector_3 normal = plane->plane_normal();
			//std::cout << "Plane with normal " << normal << std::endl;
			
			// Plane shape can also be converted to the Kernel::Plane_3.
			//std::cout << "Kernel::Plane_3: " <<
			//static_cast<Kernel::Plane_3>(*plane) << std::endl;

			// determine the shape type
			shapeType = "Pla";
		}

		else if (Sphere* sph = dynamic_cast<Sphere*>(it->get())) {
			//FT radius = cyl->radius();
			double r = sph->radius();
			
			// determine the shape type
			shapeType = "Sph-" + to_string(int(r))+"mm";
		}

		else if (Cylinder* cyl = dynamic_cast<Cylinder*>(it->get())) {
			Kernel::Line_3 axis = cyl->axis();
			//FT radius = cyl->radius();
			double r = cyl->radius();
			
			//std::cout << "Cylinder with axis "
			//<< axis << " and radius " << radius << std::endl;

			// determine the shape type
			shapeType = "Cyl-" + to_string(int(r))+"mm";
		}

		// Iterate through point indices assigned to each detected shape.
		std::vector<std::size_t>::const_iterator
		index_it = (*it)->indices_of_assigned_points().begin();

		while (index_it != (*it)->indices_of_assigned_points().end()) {
			// Retrieve point.
			const Point_with_normal& p = *(points.begin() + (*index_it));
			//std::cout<<"point location : "<<p.first<<"; "<<p.first.x()<<", "<<p.first.y()<<", "<<p.first.z()<<"\n";

			// Adds Euclidean distance between point and shape.
			//sum_distances += CGAL::sqrt((*it)->squared_distance(p.first));

			// Proceed with the next point.
			index_it++;

			// output points of the current shape into a file
			string shapeName = to_string(ShapeCounter)+"-"+shapeType;
			file<<shapeName<<" "<<p.first.x()<<" "<<p.first.y()<<" "<<p.first.z()<<endl;
		}

		// Proceed with the next detected shape.
		it++;

		// shape counter for output file
		ShapeCounter ++;
	}

	//Efficient_ransac::Shape_range::iterator it = shapes.begin();
	//while (it != shapes.end()) {

	//	boost::shared_ptr<Efficient_ransac::Shape> shape = *it;

	//	// Use Shape_base::info() to print the parameters of the detected shape.
	//	std::cout << (*it)->info();

	//	// Sums distances of points to the detected shapes.
	//	FT sum_distances = 0;

	//	// Iterate through point indices assigned to each detected shape.
	//	std::vector<std::size_t>::const_iterator
	//	index_it = (*it)->indices_of_assigned_points().begin();

	//	while (index_it != (*it)->indices_of_assigned_points().end()) {

	//		// Retrieve point.
	//		const Point_with_normal& p = *(points.begin() + (*index_it));
	//		//std::cout<<"point location : "<<p.first<<"; "<<p.first.x()<<", "<<p.first.y()<<", "<<p.first.z()<<"\n";

	//		// Adds Euclidean distance between point and shape.
	//		sum_distances += CGAL::sqrt((*it)->squared_distance(p.first));

	//		// Proceed with the next point.
	//		index_it++;

	//		// output points of the current shape into a file
	//		string shapeName = "Shape" + to_string(ShapeCounter);
	//		file<<shapeName<<" "<<p.first.x()<<" "<<p.first.y()<<" "<<p.first.z()<<endl;
	//	}

	//	// Compute and print the average distance.
	//	FT average_distance = sum_distances / shape->indices_of_assigned_points().size();
	//	std::cout << " average distance: " << average_distance << std::endl;

	//	// Proceed with the next detected shape.
	//	it++;

	//// shape counter for output file
	//ShapeCounter ++;
	//}

	// close the file
	file.close();


	return EXIT_SUCCESS;
}

