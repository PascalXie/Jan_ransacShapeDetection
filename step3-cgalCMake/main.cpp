#include <iostream>
#include <CGAL/Simple_cartesian.h>

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_2 Point_2;
typedef Kernel::Segment_2 Segment_2;

int main() {
    Point_2 p(1, 1), q(10, 10), m(5, 9);
    Segment_2 s(p, q);



    std::cout << "p位置:" << p << std::endl;
    std::cout << "q位置:" << q.x() << " " << q.y() << std::endl;
    std::cout << "m位置:" << m << std::endl;

    std::cout << "---------计算欧几里德距离的平方----------- " << std::endl;
    std::cout << "平方距离(p,q) = "
              << CGAL::squared_distance(p, q) << std::endl;

    std::cout << "---------计算欧几里德距离的平方----------- " << std::endl;
    std::cout << "平方距离(线段(p,q), m) = "
              << CGAL::squared_distance(s, m) << std::endl;

    std::cout << "---------判断共线----------- " << std::endl;
    std::cout << "p, q, m ";
    switch (CGAL::orientation(p, q, m)) {
    case CGAL::COLLINEAR:
        std::cout << "共线\n";
        break;
    case CGAL::LEFT_TURN:
        std::cout << "左侧\n";
        break;
    case CGAL::RIGHT_TURN:
        std::cout << "右侧\n";
        break;
    }

    std::cout << "---------计算中点----------- " << std::endl;
    std::cout << " 中点(p,q) = " << CGAL::midpoint(p, q) << std::endl;
    return 0;
}
