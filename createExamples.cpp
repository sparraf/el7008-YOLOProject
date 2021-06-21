/* Universidad de Chile - FCFM
 * EL7008 - Advanced Image Processing
 * Course Project: YOLO Fine-tuning
 *
 * Author: Sebastian Parra
 * 2018
 */

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <chrono>



/**
 * Puts a determined object image in a random place inside a background image. Background image
 * must be at least as big as object image. This function returns a string with 5 fields:
 * <object-class> <x> <y> <width> <height> so that in can be compatible with YOLO training
 *
 * @param obj object image to put inside the background
 * @param bg background image. Must be at least as big as object image
 * @param objClass integer indicating object class
 * @param rng random state
 * @return string signaling object class and location inside background, compatible with YOLO format
 */
cv::String putObjInBg(cv::Mat obj, cv::Mat bg, int objClass, cv::RNG rng)
{
    cv::String objDescr;

    // Choose a random point to put object in
    int randX = rng.uniform(0, bg.cols - obj.cols);
    int randY = rng.uniform(0, bg.rows - obj.rows);

    // Get bg region where obj will be put
    cv::Mat region = bg(cv::Rect(randX, randY, obj.cols, obj.rows));

    // Choose if brightness+saturation modification or gamma correction will be applied to obj
    int linearOrGamma = rng.uniform(0, 2);

    double randAlpha, randGamma;
    int randBeta;
    cv::Mat lookupTable(1, 256, CV_8U);
    cv::Mat gammaCorrected;

    if(linearOrGamma == 0)
    {
        randAlpha = rng.uniform((double) 0.5, (double) 1.5);
        randBeta = rng.uniform(-40, 41);

    } else {
        randGamma = rng.uniform((double) 0.5, (double) 1.5);
        uchar *p = lookupTable.ptr();

        for (int i = 0; i < 256; i++)
        {
            p[i] = cv::saturate_cast<uchar>(cv::pow(i / 255.0, randGamma) * 255.0);
        }
        cv::LUT(obj, lookupTable, gammaCorrected);
    }
    // Copy obj image pixels into bg
    for(int r = 0; r < region.rows; ++r)
    {
        for(int c = 0; c < region.cols; ++c)
        {
            // Do not count background pixels of obj image
            cv::Vec3b objPixel = obj.at<cv::Vec3b>(r,c);
            if(objPixel != cv::Vec3b(255, 255, 255))
            {
                // Apply brightness+contrast or gamma correction
                if(linearOrGamma == 0)
                {
                    for (int channel = 0; channel < objPixel.channels; channel++)
                    {
                        region.at<cv::Vec3b>(r, c)[channel] =
                                cv::saturate_cast<uchar>(randAlpha * objPixel[channel] + randBeta);
                    }
                } else {
                    region.at<cv::Vec3b>(r,c) = gammaCorrected.at<cv::Vec3b>(r,c);
                }
            }
        }
    }

    float xCenterAbs = (2 * randX + obj.cols) / 2.f;
    float yCenterAbs = (2 * randY + obj.rows) / 2.f;

    float xCenterRel = xCenterAbs / bg.cols;
    float yCenterRel = yCenterAbs / bg.rows;
    float widthRel = (float) obj.cols / bg.cols;
    float heightRel = (float) obj.rows / bg.rows;

    std::stringstream ss;
    ss << objClass << " " << xCenterRel << " " << yCenterRel << " " << widthRel << " " << heightRel;
    objDescr = ss.str();

    return objDescr;
}


int main(int argc, char *argv[])
{
    cv::String keys =
            "{@nExamples | 1500 | set how many examples will be generated}"
            "{@minObjPerImg | 0 | set the minimum amount of objects that will be put in a background image}"
            "{@maxObjPerImg | 4 | set the maximum amount of objects that will be put in a background image}"
            "{@objFolder | ../db_project/obj_cropped | cropped object image folder}"
            "{@bgFolder | ../db_project/bg | background image folder}"
            "{@outFolder | ../examples | folder where artificial examples will be saved}"
            "{help h ? |      | show help message}";

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 1;
    }

    // Get vector of all cropped object image paths
    cv::String objFolder = parser.get<cv::String>("@objFolder");
    std::vector<cv::String> objPaths;
    cv::glob(objFolder, objPaths, true);

    // Get vector of all background image paths
    cv::String bgFolder = parser.get<cv::String>("@bgFolder");
    std::vector<cv::String> bgPaths;
    cv::glob(bgFolder, bgPaths, true);

    // Get out folder path
    cv::String outFolder = parser.get<cv::String>("@outFolder");

    auto nExamples = parser.get<int>("@nExamples");
    auto minObjPerImg = parser.get<int>("@minObjPerImg");
    auto maxObjPerImg = parser.get<int>("@maxObjPerImg");

    cv::RNG rng = cv::RNG(
            static_cast<uint64>(std::chrono::duration_cast<std::chrono::milliseconds>
                    (std::chrono::system_clock::now().time_since_epoch()).count()));

    // Create nExamples examples
    for(int i = 0; i < nExamples; ++i)
    {
        // Create vector of object descriptions to be saved into .txt file
        std::vector<std::string> descriptions;

        // Choose a random background image
        int randBg = rng.uniform(0, static_cast<int>(bgPaths.size()));

        cv::String chosenBg = bgPaths[randBg];
        cv::Mat bg = cv::imread(chosenBg);

        // Set how many objects will be put in the image (random number between minObjPerImg and maxObjPerImg)
        int nObj = rng.uniform(minObjPerImg, maxObjPerImg+1);
        for(int j = 0; j < nObj; ++j)
        {
            // Choose a random object image
            int randObj = rng.uniform(0, static_cast<int>(objPaths.size()));

            cv::String chosenObj = objPaths[randObj];

            cv::Mat obj = cv::imread(chosenObj);
            int objClass = std::stoi(chosenObj.substr(chosenObj.size() - 6, 2), nullptr, 10);

            // Apply random scaling to biggest object dimension, equivalent to 0.1-0.5 times the smallest dimension
            // of background image, while keeping aspect ratio
            double randScale = rng.uniform((double)0.1, (double)0.5);
            auto newMaxDim = static_cast<int>(round(randScale * std::min(bg.cols, bg.rows)));
            cv::Size newSize;
            if (obj.rows > obj.cols)
            {
                double scaleRatio = (double) newMaxDim / obj.rows;
                auto newCols = static_cast<int>(round(scaleRatio * obj.cols));
                newSize = cv::Size(newCols, newMaxDim);
            } else {
                double scaleRatio = (double) newMaxDim / obj.cols;
                auto newRows = static_cast<int>(round(scaleRatio * obj.rows));
                newSize = cv::Size(newMaxDim, newRows);
            }

            int interpolation;
            if(randScale < 0)
            {
                interpolation = cv::INTER_AREA;
            } else
            {
                interpolation = cv::INTER_CUBIC;
            }
            cv::resize(obj, obj, newSize, 0, 0, interpolation);

            cv::String objDescr = putObjInBg(obj, bg, objClass, rng);
            descriptions.push_back(objDescr);
        }

        std::stringstream ss;
        ss << outFolder << "/example_" << i;
        cv::String outPath = ss.str();

        std::stringstream ssImg;
        ssImg << outPath << ".png";
        cv::String outImgFilename = ssImg.str();

        std::stringstream ssTxt;
        ssTxt << outPath << ".txt";
        cv::String outTxtFilename = ssTxt.str();

        cv::imwrite(outImgFilename, bg);

        std::ofstream txtFile(outTxtFilename);
        for(const auto &s : descriptions)
        {
            txtFile << s << "\n";
        }
    }
    return 0;
}