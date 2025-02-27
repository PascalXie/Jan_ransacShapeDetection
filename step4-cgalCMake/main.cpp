#include <fstream>
#include <iostream>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/property_map.h>
#include <CGAL/IO/read_xyz_points.h>
#include <CGAL/Point_with_normal_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

#include <CGAL/Shape_detection/Efficient_RANSAC.h>

// Type declarations.
typedef CGAL::Exact_predicates_inexact_constructions_kernel  Kernel;
typedef std::pair<Kernel::Point_3, Kernel::Vector_3>         Point_with_normal;
typedef std::vector<Point_with_normal>                       Pwn_vector;
typedef CGAL::First_of_pair_property_map<Point_with_normal>  Point_map;
typedef CGAL::Second_of_pair_property_map<Point_with_normal> Normal_map;

typedef CGAL::Shape_detection::Efficient_RANSAC_traits
<Kernel, Pwn_vector, Point_map, Normal_map>             Traits;
typedef CGAL::Shape_detection::Efficient_RANSAC<Traits> Efficient_ransac;
typedef CGAL::Shape_detection::Plane<Traits>            Plane;

int main (int argc, char** argv) {
	std::cout<<"Hello"<<std::endl;

	std::cout << "Efficient RANSAC" << std::endl;
	const char* filename = (argc > 1) ? argv[1] : "../data/cube.pwn";
	
	// Points with normals.
	Pwn_vector points;
	
	// Load point set from a file.
	std::ifstream stream(filename);
	
	if (!stream ||
	  !CGAL::read_xyz_points(
	    stream,
	    std::back_inserter(points),
	    CGAL::parameters::point_map(Point_map()).
	    normal_map(Normal_map()))) {
	
	  std::cerr << "Error: cannot read file cube.pwn!" << std::endl;
	  return EXIT_FAILURE;
	}
	
	// Instantiate shape detection engine.
	Efficient_ransac ransac;
	
	// Provide input data.
	ransac.set_input(points);
	
	// Register planar shapes via template method.
	ransac.add_shape_factory<Plane>();
	
	// Detect registered shapes with default parameters.
	ransac.detect();
	
	// Print number of detected shapes.
	std::cout << ransac.shapes().end() - ransac.shapes().begin()
	<< " shapes detected." << std::endl;

    return 0;
}
