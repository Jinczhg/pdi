
// This is No Warranty No Copyright Software.
// Jincheng Zhang
// Dec 4, 2023

#include <GL/gl.h>
#include <opencv2/opencv.hpp>
#include <opencv2/ccalib/omnidir.hpp>
#include "glwindow/scenewindow.hpp"
#include <iostream>
#include <fstream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

void writePFM(std::string filename, const cv::Mat& img){
    double scale = -1.0;
    std::ofstream myfile(filename, std::ios::out | std::ios::trunc | std::ios::binary);
    int numberOfComponents(img.channels());
    if(numberOfComponents == 3) {
        myfile << "PF\n" << img.cols << " " << img.rows << "\n" << scale << "\n";
    }
    else if(numberOfComponents == 1) {
        myfile << "Pf\n" << img.cols << " " << img.rows << "\n" << scale << "\n";
    }
    myfile.write((char*) img.data, img.rows*img.cols*numberOfComponents*sizeof(float));
    myfile.close();
}


/**
 * Loads a PFM image stored in little endian and returns the image as an OpenCV Mat.
 * @brief loadPFM
 * @param filePath
 * @return
 */
cv::Mat loadPFM(const std::string filePath)
{
    //Open binary file
    std::ifstream file(filePath.c_str(),  std::ios::in | std::ios::binary);
    cv::Mat imagePFM;

    //If file correctly openened
    if (file) {
        //Read the type of file plus the 0x0a UNIX return character at the end
        char type[3];
        file.read(type, 3*sizeof(char));
        //Read the width and height
        unsigned int width(0), height(0);
        file >> width >> height;
        //Read the 0x0a UNIX return character at the end
        char endOfLine;
        file.read(&endOfLine, sizeof(char));
        int numberOfComponents(0);
        //The type gets the number of color channels
        if (type[1] == 'F') {
            imagePFM = cv::Mat(height, width, CV_32FC3);
            numberOfComponents = 3;
        }
        else if (type[1] == 'f') {
            imagePFM = cv::Mat(height, width, CV_32FC1);
            numberOfComponents = 1;
        }
        //TODO Read correctly depending on the endianness
        //Read the endianness plus the 0x0a UNIX return character at the end
        //Byte Order contains -1.0 or 1.0
        char byteOrder[2];
        file.read(byteOrder, 2*sizeof(char));
        //Find the last line return 0x0a before the pixels of the image
        char findReturn = ' ';
        while (findReturn != 0x0a)
        {
            file.read(&findReturn, sizeof(char));
        }
        //Read each RGB colors as 3 floats and store it in the image.
        auto *color = new float[numberOfComponents];
        for (int i = 0 ; i<height ; ++i) {
            for(int j = 0 ; j<width ; ++j) {
                file.read((char*) color, numberOfComponents*sizeof(float));
                //In the PFM format the image is upside down
                if (numberOfComponents == 3) {
                    //OpenCV stores the color as BGR
                    imagePFM.at<cv::Vec3f>(i,j) = cv::Vec3f(color[2], color[1], color[0]);
                }
                else if (numberOfComponents == 1) {
                    //OpenCV stores the color as BGR
                    imagePFM.at<float>(i,j) = color[0];
                }
            }
        }
        delete[] color;
        //Close file
        file.close();
    }
    else {
        std::cerr << "Could not open the file : " << filePath << std::endl;
    }
    return imagePFM;
}

////////////////////////////////////////////////////////////////////////////////

int ndisp_bar = 4,   wsize_bar = 2, thr_bar = 0;
int ndisp_max = 6,   wsize_max = 3, thr_max = 30;
int ndisp_now = 76,  wsize_now = 9, thr_now = 70;

double MAX_DEPTH = 200;
////////////////////////////////////////////////////////////////////////////////

void OnTrackNdisp(int, void*) {
    ndisp_now = 16 + 16 * ndisp_bar;
}

////////////////////////////////////////////////////////////////////////////////

void OnTrackWsize(int, void*) {
    wsize_now = 5 + 2 * wsize_bar;
}

////////////////////////////////////////////////////////////////////////////////

void OnTrackThreshold(int, void*) {
    thr_now = 70 + thr_bar;
}

////////////////////////////////////////////////////////////////////////////////

int     cap_cols, cap_rows, img_width;
cv::Mat T, Kl, Kr, Dl, Dr, xil, xir, Rl, Rr;
cv::Mat lmap[2][2];

void LoadParameters(std::string file_name, bool half_size) {
    cv::FileStorage fs(file_name, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cout << "Failed to open ini parameters" << std::endl;
        exit(-1);
    }

    cv::Size cap_size;
    fs["cap_size" ] >> cap_size;
    fs["Kl"       ] >> Kl;
    fs["Dl"       ] >> Dl;
    fs["xil"      ] >> xil;
    fs["Rl"       ] >> Rl;
    fs["Kr"       ] >> Kr;
    fs["Dr"       ] >> Dr;
    fs["xir"      ] >> xir;
    fs["Rr"       ] >> Rr;
    fs["T"        ] >> T;
    fs.release();

    if (half_size) {
        cap_size  = cap_size / 2;
        Kl.row(0) = Kl.row(0) / 2.;
        Kl.row(1) = Kl.row(1) / 2.;
        Kr.row(0) = Kr.row(0) / 2.;
        Kr.row(1) = Kr.row(1) / 2.;
    }

    cap_cols  = cap_size.width;
    cap_rows  = cap_size.height;
    img_width = cap_size.width / 2;
}

////////////////////////////////////////////////////////////////////////////////

inline double MatRowMul(cv::Mat m, double x, double y, double z, int r) {
    return m.at<double>(r,0) * x + m.at<double>(r,1) * y + m.at<double>(r,2) * z;
}

////////////////////////////////////////////////////////////////////////////////

enum RectMode {
    RECT_PERSPECTIVE,
    RECT_FISHEYE,
    RECT_LONGLAT
};

void InitRectifyMap(cv::Mat K,
                    cv::Mat D,
                    cv::Mat R,
                    cv::Mat Knew,
                    double xi0,
                    cv::Size size,
                    RectMode mode,
                    cv::Mat& map1,
                    cv::Mat& map2) {
    map1.create(size, CV_32F);
    map2.create(size, CV_32F);

    double fx = K.at<double>(0,0);
    double fy = K.at<double>(1,1);
    double cx = K.at<double>(0,2);
    double cy = K.at<double>(1,2);
    double s  = K.at<double>(0,1);

    double k1 = D.at<double>(0,0);
    double k2 = D.at<double>(0,1);
    double p1 = D.at<double>(0,2);
    double p2 = D.at<double>(0,3);

    cv::Mat Ki  = Knew.inv();
    cv::Mat Ri  = R.inv();
    cv::Mat KRi = (Knew * R).inv();

    for (int r = 0; r < size.height; ++r) {
        for (int c = 0; c < size.width; ++c) {
            double xc = 0.;
            double yc = 0.;
            double zc = 0.;

            if (mode == RECT_PERSPECTIVE) {
                xc = MatRowMul(KRi, c, r, 1., 0);
                yc = MatRowMul(KRi, c, r, 1., 1);
                zc = MatRowMul(KRi, c, r, 1., 2);
            }

            if (mode == RECT_LONGLAT) {
                double tt = MatRowMul(Ki, c, r, 1., 0);
                double pp = MatRowMul(Ki, c, r, 1., 1);

                double xn = -cos(tt);
                double yn = -sin(tt) * cos(pp);
                double zn =  sin(tt) * sin(pp);

                xc = MatRowMul(Ri, xn, yn, zn, 0);
                yc = MatRowMul(Ri, xn, yn, zn, 1);
                zc = MatRowMul(Ri, xn, yn, zn, 2);
            }

            if (mode == RECT_FISHEYE) {
                double ee = MatRowMul(Ki, c, r, 1., 0);
                double ff = MatRowMul(Ki, c, r, 1., 1);
                double zz = 2. / (ee * ee + ff * ff + 1.);

                double xn = zz * ee;
                double yn = zz * ff;
                double zn = zz - 1.;

                xc = MatRowMul(Ri, xn, yn, zn, 0);
                yc = MatRowMul(Ri, xn, yn, zn, 1);
                zc = MatRowMul(Ri, xn, yn, zn, 2);
            }

            double rr = sqrt(xc * xc + yc * yc + zc * zc);
            double xs = xc / rr;
            double ys = yc / rr;
            double zs = zc / rr;

            double xu = xs / (zs + xi0);
            double yu = ys / (zs + xi0);

            double r2 = xu * xu + yu * yu;
            double r4 = r2 * r2;
            double xd = (1+k1*r2+k2*r4)*xu + 2*p1*xu*yu + p2*(r2+2*xu*xu);
            double yd = (1+k1*r2+k2*r4)*yu + 2*p2*xu*yu + p1*(r2+2*yu*yu);

            double u = fx * xd + s * yd + cx;
            double v = fy * yd + cy;

            map1.at<float>(r,c) = (float) u;
            map2.at<float>(r,c) = (float) v;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

int rect_cols = 640, rect_rows = 640;

void InitRectifyMap() {
    cv::Size img_size(rect_cols, rect_rows);
    cv::Mat  Kll = cv::Mat::eye(3, 3, CV_64F);
    Kll.at<double>(0,0) = (img_size.width - 1.) / CV_PI;
    Kll.at<double>(1,1) = (img_size.height - 1.) / CV_PI;

    InitRectifyMap(Kl, Dl, Rl, Kll, xil.at<double>(0,0),
                   img_size, RECT_LONGLAT, lmap[0][0], lmap[0][1]);
    InitRectifyMap(Kr, Dr, Rr, Kll, xir.at<double>(0,0),
                   img_size, RECT_LONGLAT, lmap[1][0], lmap[1][1]);
}

////////////////////////////////////////////////////////////////////////////////

void DisparityImage(const cv::Mat& recl, const cv::Mat& recr, cv::Mat& dispf) {
    cv::Mat disps;
    int N = ndisp_now, W = wsize_now, C = recl.channels();
    if (1) {
        cv::Ptr<cv::StereoSGBM> sgbm =
                cv::StereoSGBM::create(0, N, W, 8 * C * W * W, 32 * C * W * W);
        sgbm->compute(recl, recr, disps);
    } else {
        cv::Mat grayl, grayr;
        cv::cvtColor(recl, grayl, cv::COLOR_BGR2GRAY);
        cv::cvtColor(recr, grayr, cv::COLOR_BGR2GRAY);

        cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create(N, W);
        sbm->setPreFilterCap(31);
        sbm->setMinDisparity(0);
        sbm->setTextureThreshold(10);
        sbm->setUniquenessRatio(15);
        sbm->setSpeckleWindowSize(100);
        sbm->setSpeckleRange(32);
        sbm->setDisp12MaxDiff(1);
        sbm->compute(grayl, grayr, disps);
    }

    disps.convertTo(dispf, CV_32F, 1.f / 16.f);
}

////////////////////////////////////////////////////////////////////////////////

void computeEquirectangleMaps(
        cv::Mat K, const cv::Size& img_sz,
        cv::Mat &eqrec_map_x,
        cv::Mat &eqrec_map_y)
{
    eqrec_map_x = cv::Mat(img_sz, CV_32FC1);
    eqrec_map_y = cv::Mat(img_sz, CV_32FC1);
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);
    double rad_x = std::atan(static_cast<double>(img_sz.width) / fx);
    double rad_y = std::atan(static_cast<double>(img_sz.height) / fy);
    for (int y = 0; y < img_sz.height; y++)
    {
        for (int x = 0; x < img_sz.width; x++)
        {
            double lamb = (1.0 - y / (img_sz.height / 2.0)) * rad_y;
            double phi = (x / (img_sz.width / 2.0) - 1.0) * rad_x;
            double rec_x = cx + fx * tan(phi) / cos(lamb);
            double rec_y = cy - fy * tan(lamb);
            eqrec_map_x.at<float>(y, x) = static_cast<float>(rec_x);
            eqrec_map_y.at<float>(y, x) = static_cast<float>(rec_y);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

void computeFisheyeMap(
        const cv::Mat& K, const cv::Mat& D, const cv::Mat& Rotation, const cv::Mat& P,
        const cv::Size& img_sz, int dtype,
        cv::Mat &map_x, cv::Mat &map_y)
{

    cv::Mat rect_map_x(img_sz, CV_32FC1);
    cv::Mat rect_map_y(img_sz, CV_32FC1);
    cv::Mat rect_to_eqrec_map_x(img_sz, CV_32FC1);
    cv::Mat rect_to_eqrec_map_y(img_sz, CV_32FC1);
    cv::fisheye::initUndistortRectifyMap(K, D, Rotation, P, img_sz, dtype,
                                         rect_map_x, rect_map_y);
    computeEquirectangleMaps(P, img_sz,
                             rect_to_eqrec_map_x, rect_to_eqrec_map_y);

    cv::remap(rect_map_x, map_x, rect_to_eqrec_map_x,
              rect_to_eqrec_map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    cv::remap(rect_map_y, map_y, rect_to_eqrec_map_x,
              rect_to_eqrec_map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
}

////////////////////////////////////////////////////////////////////////////////

void fisheyeToEquirect(const cv::Size &srcImgSize, const cv::Size &dstImgSize, cv::Mat &map_x, cv::Mat &map_y, float FOV)
{
    int Hs = srcImgSize.height;
    int Ws = srcImgSize.width;
    int Hd = dstImgSize.height;
    int Wd = dstImgSize.width;

    int radius = int(std::min(Hs, Ws) * 0.5);
    float cxs = 0.5f * (Ws - 1);     // col: 0 ~ (Ws-1)
    float cys = 0.5f * (Hs - 1);

    float x_d_n, y_d_n;     // normalized coordinates in the destination image
    float x_s_n, y_s_n;     // normalized coordinates in the source image
    float longitude, latitude;
    float Px, Py, Pz, r, theta;

    map_x.create(dstImgSize, CV_32F);
    map_y.create(dstImgSize, CV_32F);

    // http://paulbourke.net/dome/dualfish2sphere/
    // https://www.paul-reed.co.uk/programming.html
    // https://kaustubh-sadekar.github.io/OmniCV-Lib/Equirectangular-to-perspective.html
    for(int i = 0; i < Hd; i++) {
        auto* dataX = map_x.ptr<float>(i);
        auto* dataY = map_y.ptr<float>(i);

        for(int j = 0; j < Wd; j++) {
            x_d_n = 1.0f - (float) j / (Wd - 1);
            y_d_n = 0.5f - (float) i / (Hd - 1);
            longitude = x_d_n * CV_PI;
            latitude = y_d_n * CV_PI;

            Px = std::cos(latitude)*std::cos(longitude);
            Py = std::cos(latitude)*std::sin(longitude);
            Pz = std::sin(latitude);

            r = 2.0 * (std::atan2(std::sqrt(Px*Px + Pz*Pz), Py))/(FOV*CV_PI/180);
            theta = std::atan2(Pz, Px);

            x_s_n = r * std::cos(theta);
            y_s_n = r * std::sin(theta);

            dataX[j] = radius * x_s_n + (cxs - 1);
            dataY[j] = (cys - 1) - radius * y_s_n;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

struct PCL {
    cv::Vec3f pts;
    cv::Vec3b clr;
};

void PointClouds(const cv::Mat& disp_img,
                 const cv::Mat& color_img,
                 cv::Mat& est_depth_img,
                 std::vector<PCL>& pcl_vec) {
    double bl = cv::norm(T);
    double pi_w = CV_PI / (rect_cols - 1.);

    for (int r = 0; r < color_img.rows; ++r) {
        for (int c = 0; c < color_img.cols; ++c) {
            float disp = disp_img.at<float>(r, c);
            if (disp <= 0.f or isnan(disp)) {
                continue;
            }

            double tt = (c / (color_img.cols - 1.) - 0.5) * CV_PI;
            double pp = (r / (color_img.rows - 1.) - 0.5) * CV_PI;

            double cx = std::sin(tt);
            double cy = std::cos(tt) * std::sin(pp);
            double cz = std::cos(tt) * std::cos(pp);

            double cr = hypot(hypot(cx, cy), cz);
            double ct = std::acos(cz / cr);

            double diff = pi_w * disp;
            double mgnt = bl * sin(c * pi_w - diff) / sin(diff);
            est_depth_img.at<float>(r, c) = float(cz * mgnt);

            if (ct > (CV_PI / 2.) * (thr_now / 100.)) {
                continue;
            }

//            cv::Mat pt2D = (cv::Mat_<double>(3, 1) << r, c, 1.0);
//            cv::Mat pt2D_n = Kl.inv() * pt2D;
//            double radius_sq = pt2D_n.at<double>(0, 0) * pt2D_n.at<double>(0, 0) + pt2D_n.at<double>(1, 0) * pt2D_n.at<double>(1, 0);
//            double param = 1.0f + (1-xil.at<double>(0, 0)*xil.at<double>(0, 0))*radius_sq;
//            double factor_l = (xil.at<double>(0, 0) + std::sqrt(param)) /( radius_sq + 1);
//            cv::Mat spherecial_pt = factor_l * pt2D_n;
//            spherecial_pt.at<double>(0, 2) = spherecial_pt.at<double>(0, 2) - xil.at<double>(0, 0);
//            double rl = sqrt(spherecial_pt.at<double>(0, 0) *  spherecial_pt.at<double>(0, 0)
//                    + spherecial_pt.at<double>(0, 1) *  spherecial_pt.at<double>(0, 1)
//                    + spherecial_pt.at<double>(0, 2) *  spherecial_pt.at<double>(0, 2));
//            double theta_l = CV_PI - std::acos(spherecial_pt.at<double>(0, 1) / rl);

//            double mgnt = bl * sin(theta_l - diff) / sin(diff);

            if (mgnt * cz > MAX_DEPTH) continue;
            cv::Vec3b color = color_img.at<cv::Vec3b>(r,c);
            est_depth_img.at<float>(r, c) = float(cz * mgnt);
            PCL pcl;
            pcl.pts = cv::Vec3f(cx, cy, cz) * mgnt;
            pcl.clr = cv::Vec3b(color(2), color(1), color(0));
            pcl_vec.push_back(pcl);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

void PointCloudsFromDepth(const cv::Mat &depth_img,
                          const cv::Mat &color_img,
                          std::vector<PCL> &pcl_vec) {
    for (int r = 0; r < color_img.rows; ++r) {
        for (int c = 0; c < color_img.cols; ++c) {
            float depth = depth_img.at<float>(r, c);
            if (depth > MAX_DEPTH) continue;
            if (depth <= 0) {
//                std::cout << "Invalid depth." << std::endl;
                continue;
            }
            double tt = (c / (color_img.cols - 1.) - 0.5) * CV_PI;
            double pp = (r / (color_img.rows - 1.) - 0.5) * CV_PI;

            double cx = std::sin(tt);
            double cy = std::cos(tt) * std::sin(pp);
            double cz = std::cos(tt) * std::cos(pp);

            double cr = std::sqrt(cx * cx + cy * cy + cz * cz);
            double ct = std::acos(cz / cr); // theta [0, PI]

            if (ct > (CV_PI / 2.) * (thr_now / 100.)) {
                continue;
            }
            double mgnt = depth / cr;

            cv::Vec3b color = color_img.at<cv::Vec3b>(r, c);

            PCL pcl;
            pcl.pts = cv::Vec3f(cx, cy, cz) * mgnt;
            pcl.clr = cv::Vec3b(color(2), color(1), color(0));
            pcl_vec.push_back(pcl);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

void undistortedDisparityFromDepth(const cv::Mat &depth_img,
                                   cv::Mat& disp_img_gt) {
    disp_img_gt = cv::Mat::zeros(cv::Size(depth_img.rows, depth_img.cols), CV_32FC1);
    double bl = cv::norm(T);
    double pi_w = CV_PI / (rect_cols - 1.);
    for (int r = 0; r < depth_img.rows; ++r) {
        for (int c = 0; c < depth_img.cols; ++c) {
            float depth = depth_img.at<float>(r, c);
//            if (depth > MAX_DEPTH) continue;
//            if (depth <= 0) {
////                std::cout << "Invalid depth." << std::endl;
//                continue;
//            }
            double tt = (c / (depth_img.cols - 1.) - 0.5) * CV_PI;
            double pp = (r / (depth_img.rows - 1.) - 0.5) * CV_PI;

            double cx = std::sin(tt);
            double cy = std::cos(tt) * std::sin(pp);
            double cz = std::cos(tt) * std::cos(pp);

            double cr = std::sqrt(cx * cx + cy * cy + cz * cz);
            double ct = std::acos(cz / cr); // theta [0, PI]

            if (ct > (CV_PI / 2.) * (100 / 100.)) {
                continue;
            }

//            cv::Mat pt2D = (cv::Mat_<double>(3, 1) << r, c, 1.0);
//            cv::Mat pt2D_n = Kl.inv() * pt2D;
//            double radius_sq = pt2D_n.at<double>(0, 0) * pt2D_n.at<double>(0, 0) + pt2D_n.at<double>(1, 0) * pt2D_n.at<double>(1, 0);
//            double param = 1.0f + (1-xil.at<double>(0, 0)*xil.at<double>(0, 0))*radius_sq;
//            double factor_l = (xil.at<double>(0, 0) + std::sqrt(param)) /( radius_sq + 1);
//            cv::Mat spherecial_pt = factor_l * pt2D_n;
//            spherecial_pt.at<double>(0, 2) = spherecial_pt.at<double>(0, 2) - xil.at<double>(0, 0);
//            double theta_l = CV_PI / 2 + std::atan2(spherecial_pt.at<double>(0, 0), spherecial_pt.at<double>(0, 2));

            double mgnt = depth / cr;   // divided by cr (not cz) because depth here is "distance" (or perspective depth) not "planar depth".
            double diff = atan(bl * sin(c * pi_w) / (mgnt + bl * cos(c * pi_w)));
//            double diff = atan(bl * sin(theta_l) / (mgnt + bl * cos(theta_l)));

            float disp = diff / pi_w;

            disp_img_gt.at<float>(r, c) = disp;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

void fisheyeDisparityFromDepth(const cv::Mat &depth_img,
                        cv::Mat& fisheye_disp_img_gt) {
    // fisheye disparity is angular disparity defined in "Binocular Spherical Stereo" (https://ieeexplore.ieee.org/document/4667675)
    fisheye_disp_img_gt = cv::Mat::zeros(cv::Size(depth_img.rows, depth_img.cols), CV_32FC1);
    float bl = cv::norm(T);
    float pi_w = CV_PI / (depth_img.cols - 1.);
    for (int r = 0; r < depth_img.rows; ++r) {
        for (int c = 0; c < depth_img.cols; ++c) {
            float depth = depth_img.at<float>(r, c);

            if (depth <= 0) continue;

            float mgnt = depth;
            float param = bl * sin(c * pi_w) / (mgnt + bl * cos(c * pi_w));
            float diff = atan(param);

            float disp = diff / pi_w;

            fisheye_disp_img_gt.at<float>(r, c) = disp;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

void evalEstDepth(cv::Mat &estDepth, cv::Mat &gtDepth, cv::Mat &errorDepth) {
    errorDepth = -1.0 * cv::Mat::ones(estDepth.rows, estDepth.cols, CV_32F);
    for (int r=0; r<estDepth.rows; r++) {
        for (int c=0; c<estDepth.cols; c++) {
            float gtVal = gtDepth.at<float>(r, c);
            float estVal = estDepth.at<float>(r, c);
            if (isnan(gtVal) or isnan(estVal) or gtVal > 75 or estVal > 75 or gtVal <= 0 or estVal <= 0)
                continue;
            errorDepth.at<float>(r, c) = abs(estVal - gtVal);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

void DrawScene(const std::vector<PCL>& pcl_vec) {
    glBegin(GL_POINTS);

    for (uint i = 0; i < pcl_vec.size(); ++i) {
        PCL pcl = pcl_vec[i];
        glColor3ub(pcl.clr(0), pcl.clr(1), pcl.clr(2));
        glVertex3f(pcl.pts(0), pcl.pts(1), pcl.pts(2));
    }

    glEnd();
}

////////////////////////////////////////////////////////////////////////////////

void writePCD(std::vector<PCL> &pcl_vec) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcd(new pcl::PointCloud<pcl::PointXYZRGB>);
    std::string save_pcd_path = "test_3d.pcd";
    for (auto pcl : pcl_vec) {
        pcl::PointXYZRGB point;
        point.x = pcl.pts[0];
        point.y = pcl.pts[1];
        point.z = pcl.pts[2];
        uint32_t rgb = ((uint32_t) pcl.clr[0] << 16 | (uint32_t) pcl.clr[1] << 8 | (uint32_t) pcl.clr[2]);
        point.rgb = *reinterpret_cast<float *>(&rgb);
        pcd->points.push_back(point);
    }
    pcd->width = pcd->points.size();
    pcd->height = 1;
    pcd->is_dense = false;
    try {
        if (boost::filesystem::remove(save_pcd_path))
            std::cout << "file " << save_pcd_path << " deleted.\n";
        else
            std::cout << "file " << save_pcd_path << " not found.\n";
    }
    catch(const boost::filesystem::filesystem_error& err) {
        std::cout << "filesystem error: " << err.what() << '\n';
    }
    pcl::io::savePCDFileASCII(save_pcd_path, *pcd);
    std::cout << "Point Cloud saved >> " << save_pcd_path << std::endl;
}

////////////////////////////////////////////////////////////////////////////////

// CaliCam stereo is calibrated at 2560x960
// If you want to process image at 1280x480, set half_size to true.
bool half_size = false;
int main() {
    bool airsimImage = false, astarImage = false, astarVideo = false, astarLive = false; // To run live mode, you need a CaliCam from www.astar.ai
    int experiment = 0;
    switch (experiment) {
        case 0:
            airsimImage = true;
            break;
        case 1:
            astarImage = true;
            break;
        case 2:
            astarVideo = true;
            half_size = true;     // for now, astarVideo mode can only record at half resolution 1280x480
            break;
        case 3:
            astarLive = true;
            break;
        default:
            std::cout << "Please choose an experiment." << std::endl;
            return 0;
            break;
    }

    std::string image_name, input_folder, output_folder, param_name;
    cv::Mat raw_imgl, raw_imgr;
    bool gtDepth = true;
    if (airsimImage){
        gtDepth = true;
        input_folder = "../data/images/outdoors/input/";
        output_folder = "../data/images/outdoors/output/";
        param_name = "../calibration/calicam_airsim.yml";
        raw_imgl = cv::imread(input_folder + "left_fisheye.png", cv::IMREAD_COLOR);
        raw_imgr = cv::imread(input_folder + "right_fisheye.png", cv::IMREAD_COLOR);
    }
    else if (astarImage) {
        input_folder = "../data/images/pdi/input/";
        output_folder = "../data/images/pdi/output/";
        param_name = "../calibration/calicam_pdi.yml";
        raw_imgl = cv::imread(input_folder + "wm_garden_left.jpg", cv::IMREAD_COLOR);
        raw_imgr = cv::imread(input_folder + "wm_garden_right.jpg", cv::IMREAD_COLOR);
    }
    else if (astarVideo) {
        input_folder = "../data/videos/9_15/input/";
        output_folder = "../data/videos/9_15/output/";
        param_name = "../calibration/calicam_roboticslab.yml";
    }
    else if (astarLive) {
        param_name = "../calibration/calicam_visionlab.yml";  // live camera
    }
    else {
        std::cout << "Invalid data source type name." << std::endl;
    }

    cv::VideoCapture vcapture;
    if (astarLive) {
        vcapture.open(2);
        if (!vcapture.isOpened()) {
            std::cout << "Camera doesn't work" << std::endl;
            exit(-1);
        }
        vcapture.set(cv::CAP_PROP_FRAME_WIDTH,  cap_cols);
        vcapture.set(cv::CAP_PROP_FRAME_HEIGHT, cap_rows);
        vcapture.set(cv::CAP_PROP_FPS, 30);
    }

    std::vector<std::string> fn;
    if (astarVideo)
        cv::glob(input_folder + "*.png", fn);

    cv::Mat raw_img, ll_imgl, ll_imgr, ll_depth;

    LoadParameters(param_name, half_size);
    InitRectifyMap();

    glwindow::SceneWindow scene(960, 720, "Panorama 3D Scene");

    std::string win_name = "Fisheye Image";
    cv::namedWindow(win_name);
    cv::createTrackbar("Num Disp:  16 + 16 *", win_name,
                       &ndisp_bar,  ndisp_max,   OnTrackNdisp);
    cv::createTrackbar("Blk   Size :     5  +  2 * ", win_name,
                       &wsize_bar,  wsize_max,  OnTrackWsize);
    cv::createTrackbar("Threshold :     70 + ", win_name,
                       &thr_bar,    thr_max,    OnTrackThreshold);

    int idx = 0;
    while (true) {
        if (astarLive) {
            vcapture >> raw_img;
            raw_img(cv::Rect(0, 0, img_width, cap_rows)).copyTo(raw_imgl);
            raw_img(cv::Rect(img_width, 0, img_width, cap_rows)).copyTo(raw_imgr);
            if (raw_img.total() == 0) {
                std::cout << "Image capture error" << std::endl;
                exit(-1);
            }
        }

        std::string imgName;
        if (astarVideo) {
            raw_img = cv::imread(fn[idx]);
            const size_t file_slash_idx = fn[idx].rfind('/');
            if (std::string::npos != file_slash_idx) {
                imgName = fn[idx].substr(file_slash_idx+1);     // copy string after a position
                const size_t file_ext_idx = imgName.rfind('.');
                if (std::string::npos != file_ext_idx) {
                    imgName = imgName.substr(0, file_ext_idx);
                }
            }
            raw_img(cv::Rect(0, 0, img_width, cap_rows)).copyTo(raw_imgl);
            raw_img(cv::Rect(img_width, 0, img_width, cap_rows)).copyTo(raw_imgr);
        }

        if (!astarVideo and half_size) {    // the recorded videos/images are already half size and don't need to resize
            cv::resize(raw_imgl, raw_imgl, cv::Size(), 0.5, 0.5);
            cv::resize(raw_imgr, raw_imgr, cv::Size(), 0.5, 0.5);
        }

        cv::remap(raw_imgl, ll_imgl, lmap[0][0], lmap[0][1], cv::INTER_LINEAR);
        cv::remap(raw_imgr, ll_imgr, lmap[1][0], lmap[1][1], cv::INTER_LINEAR);
        cv::imshow("rectified_left", ll_imgl);
        cv::imshow("rectified_right", ll_imgr);
//        cv::imwrite(output_folder + "/rectified_left_fisheye.png", ll_imgl);
//        cv::imwrite(output_folder + "/rectified_right_fisheye.png", ll_imgr);

        if (gtDepth) {
            cv::Mat raw_depth = loadPFM(std::string(input_folder + "/left_depth_fisheye.pfm"));
            cv::Mat raw_depth_norm;
            cv::normalize(raw_depth, raw_depth_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
//            cv::imshow("Raw depth PFM image", raw_depth_norm);
            cv::remap(raw_depth, ll_depth, lmap[0][0], lmap[0][1], cv::INTER_LINEAR);
//            writePFM(output_folder + "/rectified_depth.pfm", ll_depth);
            cv::Mat ll_depth_norm;
            cv::normalize(ll_depth, ll_depth_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
//            applyColorMap(ll_depth_norm, ll_depth_norm, cv::COLORMAP_JET);
//            cv::imshow("rectified_depth", ll_depth_norm);
            cv::Mat disp_img_gt;
            undistortedDisparityFromDepth(ll_depth, disp_img_gt);
            cv::Mat disp_img_gt_norm;
            cv::normalize(disp_img_gt, disp_img_gt_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::imshow("disparity_gt", disp_img_gt_norm);
//            writePFM(output_folder + "/disparity_gt.pfm", disp_img_gt);
            std::vector<PCL> pcl_vec_gt;
            PointCloudsFromDepth(ll_depth, ll_imgl, pcl_vec_gt);
        }

        cv::Mat          disp_img;
        std::vector<PCL> pcl_vec;

        DisparityImage(ll_imgl, ll_imgr, disp_img);
//        writePFM(output_folder + "/disparity.pfm", disp_img);

        cv::Mat disp_norm;
        cv::normalize(disp_img, disp_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
//        cv::imshow("disparity", disp_img);

        cv::Mat est_depth = -1.0 * cv::Mat::ones(ll_imgl.rows, ll_imgl.cols, CV_32FC1);
        PointClouds(disp_img, ll_imgl, est_depth, pcl_vec);
//        writePFM(std::string(output_folder + imgName + "_depth.pfm"), est_depth);
        if (gtDepth) {
            cv::Mat err_depth;
            evalEstDepth(est_depth, ll_depth, err_depth);
            cv::normalize(err_depth, err_depth, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::Mat err_depth_color;
            cv::applyColorMap(err_depth, err_depth_color, cv::COLORMAP_JET);
            cv::imshow("err_depth_color", err_depth_color);
            cv::waitKey(0);
        }
        cv::Mat est_depth_img;
        cv::normalize(est_depth, est_depth_img, 0, 255, cv::NORM_MINMAX, CV_8U);
//        cv::imshow("Rectified estimated depth PFM image", est_depth_img);

        if (scene.win.alive()) {
            if (scene.start_draw()) {
                DrawScene(pcl_vec);
                scene.finish_draw();
            }
        }

        imshow(win_name, raw_imgl);

        idx++;
        if (!fn.empty() and idx >= fn.size()) {
            break;
        }

        char key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27)
            break;
    }

    return 0;
}

////////////////////////////////////////////////////////////////////////////////



