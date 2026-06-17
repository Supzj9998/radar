#include "radar_utils.h"

#define ARMOR_HEIGHT 0.15
namespace tdt_radar {
bool isPointInsideScreen(cv::Point2f point, int screenWidth,
                         int screenHeight)
{
    return point.x >= 0 && point.x <= screenWidth && point.y >= 0 &&
           point.y <= screenHeight;
}
parser::parser()
{
    //打开文件读取相机外参 即相机坐标系与世界坐标系之间的几何变换关系
    cv::FileStorage fs;
    fs.open("./config/out_matrix.yaml", cv::FileStorage::READ);
    fs["world_tvec"] >> this->world_tvec;
    fs["world_rvec"] >> this->world_rvec;
    std::cout << world_tvec << std::endl;
    std::cout << world_rvec << std::endl;
    fs.release();

    //打开文件读取相机内参与畸变参数
    cv::FileStorage fs1;
    fs1.open("./config/camera_params.yaml", cv::FileStorage::READ);
    fs1["camera_matrix"] >> this->camera_matrix;
    fs1["dist_coeffs"] >> this->dist_coeffs;
    fs1.release();
    std::cout << "Resolve camera" << camera_matrix << std::endl;
    std::cout << "Resolve dist" << dist_coeffs << std::endl;

    //用map容器维护所有地图关键区域的所有点
    points_map["Middle_Line"] = new Parser_Points("Middle_Line");
    points_map["Left_Road"] = new Parser_Points("Left_Road");
    points_map["Right_Road"] = new Parser_Points("Right_Road");
    points_map["Enemy_Buff"] = new Parser_Points("Enemy_Buff");
    points_map["Self_Fortress"] = new Parser_Points("Self_Fortress");
    points_map["Enemy_Fortress"] = new Parser_Points("Enemy_Fortress");

    //设置每个地图关键区域的高度
    points_map["Middle_Line"]->Height = 0.3;
    points_map["Left_Road"]->Height = 0.2;
    points_map["Right_Road"]->Height = 0.2;
    points_map["Enemy_Buff"]->Height = 0.6;
    points_map["Self_Fortress"]->Height = 0.15;
    points_map["Enemy_Fortress"]->Height = 0.15;
}
//从 YAML 文件更新世界坐标变换矩阵 刷新所有点对象的位置或状态
void parser::Change_Matrix()
{
    //读取相机外参
    cv::FileStorage fs;
    fs.open("./config/out_matrix.yaml", cv::FileStorage::READ);
    fs["world_tvec"] >> this->world_tvec;
    fs["world_rvec"] >> this->world_rvec;
    fs.release();

    //刷新
    for (auto& points : points_map) {
        points.second->Update();
    }
}
//将图片img中与地图关键区域对应的区域用多边形框起来
void parser::draw_ui(cv::Mat& img)
{
    for (auto& points : points_map) {
        cv::polylines(img, points.second->Points_2D, true,
                      cv::Scalar(255, 255, 255));
    }
}
//返回input_point在2D地图坐标上的x,y
cv::Point2f parser::parse(cv::Point2f& input_point)
{
    //获得input_point所在区域的高度
    float temp_height = get_height(input_point);
    //如果高度大于0.79说明input_point在特定的位置 直接返回此位置的坐标 
    if (temp_height > 0.79) {
        return cv::Point2f(19.322, -1.915);
    }
    //如果不大于0.79返回input_point在2D地图坐标上的x,y
    return get_2d(input_point, temp_height);
}
//找到input_point所在的关键区域后返回该区域的高度
float parser::get_height(cv::Point2f& input_point)
{
    for (auto& points : points_map) {
        if (points.second->return_height(input_point)) {
            return points.second->Height;
        }
    }
    return 0;
}
//将input_point从相机坐标系转换到2D地图平面
cv::Point2f parser::get_2d(cv::Point2f& input_point, float height)
{
    //定义了四个相同高度的世界坐标点
    std::vector<cv::Point3f> world_points;
    world_points.push_back(cv::Point3f(12, -6, ARMOR_HEIGHT + height));
    world_points.push_back(cv::Point3f(16, -6, ARMOR_HEIGHT + height));
    world_points.push_back(cv::Point3f(16, -8, ARMOR_HEIGHT + height));
    world_points.push_back(cv::Point3f(12, -8, ARMOR_HEIGHT + height));
    //将3D世界坐标点投影到相机图像平面上
    std::vector<cv::Point2f> image_points;
    cv::projectPoints(world_points, world_rvec, world_tvec, camera_matrix,
                      dist_coeffs, image_points);
    
    for (auto& point : image_points) {
    }
    //定义四个世界2D坐标
    std::vector<cv::Point2f> world_points2D;
    world_points2D.push_back(cv::Point2f(12, -6));
    world_points2D.push_back(cv::Point2f(16, -6));
    world_points2D.push_back(cv::Point2f(16, -8));
    world_points2D.push_back(cv::Point2f(12, -8));
    //算出相机坐标点与2D地图坐标点的透视变换矩阵Perspective_matrix
    cv::Mat Perspective_matrix =
        cv::getPerspectiveTransform(image_points, world_points2D);
    //将输入的点input_point放入矩阵srcPointMa
    cv::Mat srcPointMat(1, 1, CV_32FC2);
    srcPointMat.at<cv::Point2f>(0, 0) = input_point;
    //利用透视变换将输入的input_point点从相机图像平面映射到2D地图平面
    cv::perspectiveTransform(srcPointMat, srcPointMat, Perspective_matrix);
    //返会input_point点在2D地图平面的x,y坐标
    return srcPointMat.at<cv::Point2f>(0, 0);
}
//从YAML文件中读取points_name节点的三维点
std::vector<cv::Point3f>
Parser_Points::ReadPoints(const std::string& points_name)
{
    cv::FileStorage fs("./config/RM2025_Points.yaml",
                       cv::FileStorage::READ);  // 打开YAML文件
    //判断文件是否已经被打开
    if (!fs.isOpened()) {
        std::cout << "无法打开文件" << std::endl;
        exit(-1);
    }

    std::vector<cv::Point3f> points;

    //读取文件中的points_name节点并赋值给pointsNode
    cv::FileNode pointsNode = fs[points_name];
    //判断如果节点point_name不是序列那就是说明YAML写错了
    if (pointsNode.type() != cv::FileNode::SEQ) {
        std::cout << "points节点不是序列" << std::endl;
        exit(-1);
    }
    //将pointsNode中的每一个三维点的x,y,z坐标分别记录在point中并把所有point依次放入vector容器points中
    for (auto&& it : pointsNode) {
        cv::Point3f point;
        it["x"] >> point.x;
        it["y"] >> point.y;
        it["z"] >> point.z;

        points.push_back(point);
    }
    //返回记录这个节点所有三维点坐标的vector容器points
    return points;
}
//将浮点型二维点坐标转换成整型
std::vector<cv::Point>
Parser_Points::Float2Int(std::vector<cv::Point2f>& FloatPoint)
{
    std::vector<cv::Point> dstPoint;
    for (auto& i : FloatPoint) {
        dstPoint.emplace_back(int(i.x), int(i.y));
    }
    return dstPoint;
}
//将3D世界坐标点（Parser_Points成员变量Points_3D）投影成2D图像平面坐标（Parser_Points成员变量Points_2D）
void Parser_Points::World2Camera()
{
    std::vector<cv::Point2f> temp_2D;
    cv::projectPoints(Points_3D, world_rvec, world_tvec, camera_matrix,
                      dist_coeffs, temp_2D);
    Points_2D = Float2Int(temp_2D);
}
//读取points_name节点(地图关键区域)的三维点和二维点
Parser_Points::Parser_Points(const std::string& points_name)
{
    //读取相机内参
    cv::FileStorage fs;
    fs.open("./config/camera_params.yaml", cv::FileStorage::READ);
    fs["camera_matrix"] >> this->camera_matrix;
    fs["dist_coeffs"] >> this->dist_coeffs;
    fs.release();
    //读取相机外参
    fs.open("./config/out_matrix.yaml", cv::FileStorage::READ);
    fs["world_tvec"] >> this->world_tvec;
    fs["world_rvec"] >> this->world_rvec;
    fs.release();
    //使用temp_3d接受ReadPoints()返回points_name节点序列的所有点坐标
    std::vector<cv::Point3f> temp_3d = ReadPoints(points_name);
    //将temp_3d赋值给成员变量Points_3D
    this->Points_3D = temp_3d;
    World2Camera();
}
//判断input_point是否在地图关键区域内 如果在就返回这个地图关键区域在三维世界的高度
float Parser_Points::return_height(cv::Point2f& input_point)
{
    bool inside = false;
    if (cv::pointPolygonTest(
            Points_2D, cv::Point((int)input_point.x, (int)input_point.y),
            false) > 0) {
        return this->Height;
    } else {
        return 0;
    }
}
//重新读取相机外参并重新计算投影平面坐标
void Parser_Points::Update()
{
    cv::FileStorage fs;
    fs.open("./config/out_matrix.yaml", cv::FileStorage::READ);
    fs["world_tvec"] >> this->world_tvec;
    fs["world_rvec"] >> this->world_rvec;
    fs.release();
    World2Camera();
}
}  // namespace tdt_radar
