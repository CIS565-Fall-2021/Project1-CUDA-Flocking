/**
 * @file      utilityCore.hpp
 * @brief     UTILITYCORE: A collection/kitchen sink of generally useful
 * functions
 * @authors   Yining Karl Li
 * @date      2012
 * @copyright Yining Karl Li
 */

#pragma once

#include <algorithm>
#include <glm/glm.hpp>
#include <istream>
#include <iterator>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include "cudaMat4.hpp"

#define PI                      3.1415926535897932384626422832795028841971
#define TWO_PI                  6.2831853071795864769252867665590057683943
#define SQRT_OF_ONE_THIRD       0.5773502691896257645091487805019574556476
#define E                       2.7182818284590452353602874713526624977572
#define G                       6.67384e-11
#define EPSILON                 .000000001
#define ZERO_ABSORPTION_EPSILON 0.00001

namespace utilityCore {
float clamp(float f, float min, float max);
bool replaceString(std::string& str, const std::string& from,
                   const std::string& to);
glm::vec3 clampRGB(glm::vec3 color);
bool epsilonCheck(float a, float b);
std::vector<std::string> tokenizeString(std::string str);
cudaMat4 glmMat4ToCudaMat4(const glm::mat4& a);
glm::mat4 cudaMat4ToGlmMat4(const cudaMat4& a);
glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation,
                                    glm::vec3 scale);
void printCudaMat4(const cudaMat4& m);
std::string convertIntToString(int number);
std::istream& safeGetline(
    std::istream& is,
    std::string& t);  // Thanks to http://stackoverflow.com/a/6089413

//-----------------------------
//-------GLM Printers----------
//-----------------------------
void printMat4(const glm::mat4&);
void printVec4(const glm::vec4&);
void printVec3(const glm::vec3&);
}  // namespace utilityCore
