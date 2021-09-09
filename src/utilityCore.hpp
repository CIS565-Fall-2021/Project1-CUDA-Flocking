/**
 * @file      utilityCore.hpp
 * @brief     UTILITYCORE: A collection/kitchen sink of generally useful functions
 * @authors   Yining Karl Li
 * @date      2012
 * @copyright Yining Karl Li
 */

#pragma once

#include <algorithm>
#include <istream>
#include <ostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include "cudaMat4.hpp"

#define PI                          3.1415926535897932384626422832795028841971
#define TWO_PI                      6.2831853071795864769252867665590057683943
#define SQRT_OF_ONE_THIRD           0.5773502691896257645091487805019574556476
#define E                           2.7182818284590452353602874713526624977572
#define G                           6.67384e-11
#define EPSILON                     .000000001
#define ZERO_ABSORPTION_EPSILON     0.00001

namespace utilityCore {
extern float clamp(float f, float min, float max);
extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
extern glm::vec3 clampRGB(glm::vec3 color);
extern bool epsilonCheck(float a, float b);
extern std::vector<std::string> tokenizeString(std::string str);
extern cudaMat4 glmMat4ToCudaMat4(const glm::mat4 &a);
extern glm::mat4 cudaMat4ToGlmMat4(const cudaMat4 &a);
extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
extern void printCudaMat4(const cudaMat4 &m);
extern std::string convertIntToString(int number);
extern std::istream& safeGetline(std::istream& is, std::string& t); //Thanks to http://stackoverflow.com/a/6089413

//-----------------------------
//-------GLM Printers----------
//-----------------------------
extern void printMat4(const glm::mat4 &);
extern void printVec4(const glm::vec4 &);
extern void printVec3(const glm::vec3 &);
}


/*********************
* Start Profile Log *
*********************/

#define ENABLE_PROFILE_LOG 0//0

#if ENABLE_PROFILE_LOG
#include <fstream>
#include <unordered_map>
namespace utilityCore {
class ProfileLog {
private:
  std::ofstream m_fout;
  std::string m_filePath;
  std::string m_endEventName;

  bool m_recording = false;
  bool m_profiling = false;

  std::vector<std::string> m_args;
  std::vector<std::string> m_kwargs_keys;
  std::unordered_map<std::string, int> m_kwargs;

  std::ios_base::openmode m_mode = std::ios_base::out;
  float m_timeStamp = 0.f;
  int m_frameCount = 0, m_dataSizeToCount = 2048;
  int m_startFrame = -1, m_endFrame = -1;
  float m_startTime = 2000.f;
  int m_frameInterval = 1;

  std::vector<float> m_frameRates;
  std::vector<float> m_frameDurations;
  std::vector<std::unordered_map<size_t, float>> m_eventDurations;

  std::unordered_map<std::string, size_t> m_cudaEventIndices;
  std::vector<std::string> m_cudaEventNames;
  std::vector<cudaEvent_t> m_cudaEvents;

private:
  void cacheInStep(float avgFPS, float timeElapsed);

public:
  static ProfileLog& get();
  bool isRecordingEvent() const;
  bool isProfiling() const;
  const std::string& getEndEventName() const;
  
  void clearArgs();
  void addArg(const std::string& arg);
  void addKwArg(const std::string& key, int value);

  void initProfile(const std::string& prefix, const std::string& endEventName = "end", int startFrame = 256, int frameCount = 2048, int frameInterval = 1, std::ios_base::openmode mode = std::ios_base::out);
  size_t registerEvent(const std::string& eventName);
  void recordEvent(const std::string& eventName);

  // Call before runCUDA, timeStamp in ms
  void step(float avgFPS, float timeStamp);

  void writeToFile(std::ios_base::openmode mode = std::ios_base::out);
  void unregisterEvents();
};
}

#define callCUDA_Profile(funcName) \
  utilityCore::ProfileLog::get().recordEvent(#funcName "-" + std::to_string(__LINE__)); \
  funcName

#define callCUDA_ProfileWithAlias(alias, funcName) \
  utilityCore::ProfileLog::get().recordEvent(#alias); \
  funcName

#define callCUDA_ProfileEnd() utilityCore::ProfileLog::get().recordEvent(utilityCore::ProfileLog::get().getEndEventName())

#else // ENABLE_PROFILE_LOG
#define callCUDA_Profile(funcName) funcName
#define callCUDA_ProfileWithAlias(alias, funcName) funcName
#define callCUDA_ProfileEnd()
#endif // ENABLE_PROFILE_LOG

/*******************
* End Profile Log *
*******************/
