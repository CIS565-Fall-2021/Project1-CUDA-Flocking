/**
 * @file      utilityCore.cpp
 * @brief     UTILITYCORE: A collection/kitchen sink of generally useful functions
 * @authors   Yining Karl Li
 * @date      2012
 * @copyright Yining Karl Li
 */

#include <iostream>
#include <cstdio>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include "utilityCore.hpp"

float utilityCore::clamp(float f, float min, float max) {
    if (f < min) {
        return min;
    } else if (f > max) {
        return max;
    } else {
        return f;
    }
}

bool utilityCore::replaceString(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    if (start_pos == std::string::npos) {
        return false;
    }
    str.replace(start_pos, from.length(), to);
    return true;
}

std::string utilityCore::convertIntToString(int number) {
    std::stringstream ss;
    ss << number;
    return ss.str();
}

glm::vec3 utilityCore::clampRGB(glm::vec3 color) {
    if (color[0] < 0) {
        color[0] = 0;
    } else if (color[0] > 255) {
        color[0] = 255;
    }
    if (color[1] < 0) {
        color[1] = 0;
    } else if (color[1] > 255) {
        color[1] = 255;
    }
    if (color[2] < 0) {
        color[2] = 0;
    } else if (color[2] > 255) {
        color[2] = 255;
    }
    return color;
}

bool utilityCore::epsilonCheck(float a, float b) {
    if (fabs(fabs(a) - fabs(b)) < EPSILON) {
        return true;
    } else {
        return false;
    }
}

void utilityCore::printCudaMat4(const cudaMat4 &m) {
    utilityCore::printVec4(m.x);
    utilityCore::printVec4(m.y);
    utilityCore::printVec4(m.z);
    utilityCore::printVec4(m.w);
}

glm::mat4 utilityCore::buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale) {
    glm::mat4 translationMat = glm::translate(glm::mat4(), translation);
    glm::mat4 rotationMat = glm::rotate(glm::mat4(), rotation.x, glm::vec3(1, 0, 0));
    rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.y, glm::vec3(0, 1, 0));
    rotationMat = rotationMat * glm::rotate(glm::mat4(), rotation.z, glm::vec3(0, 0, 1));
    glm::mat4 scaleMat = glm::scale(glm::mat4(), scale);
    return translationMat * rotationMat * scaleMat;
}

cudaMat4 utilityCore::glmMat4ToCudaMat4(const glm::mat4 &a) {
    cudaMat4 m;
    glm::mat4 aTr = glm::transpose(a);
    m.x = aTr[0];
    m.y = aTr[1];
    m.z = aTr[2];
    m.w = aTr[3];
    return m;
}

glm::mat4 utilityCore::cudaMat4ToGlmMat4(const cudaMat4 &a) {
    glm::mat4 m;
    m[0] = a.x;
    m[1] = a.y;
    m[2] = a.z;
    m[3] = a.w;
    return glm::transpose(m);
}

std::vector<std::string> utilityCore::tokenizeString(std::string str) {
    std::stringstream strstr(str);
    std::istream_iterator<std::string> it(strstr);
    std::istream_iterator<std::string> end;
    std::vector<std::string> results(it, end);
    return results;
}

std::istream& utilityCore::safeGetline(std::istream& is, std::string& t) {
    t.clear();

    // The characters in the stream are read one-by-one using a std::streambuf.
    // That is faster than reading them one-by-one using the std::istream.
    // Code that uses streambuf this way must be guarded by a sentry object.
    // The sentry object performs various tasks,
    // such as thread synchronization and updating the stream state.

    std::istream::sentry se(is, true);
    std::streambuf* sb = is.rdbuf();

    for (;;) {
        int c = sb->sbumpc();
        switch (c) {
        case '\n':
            return is;
        case '\r':
            if (sb->sgetc() == '\n') {
                sb->sbumpc();
            }
            return is;
        case EOF:
            // Also handle the case when the last line has no line ending
            if (t.empty()) {
                is.setstate(std::ios::eofbit);
            }
            return is;
        default:
            t += (char)c;
        }
    }

    return is;
}
//-----------------------------
//-------GLM Printers----------
//-----------------------------

void utilityCore::printMat4(const glm::mat4 &m) {
    std::cout << m[0][0] << " " << m[1][0] << " " << m[2][0] << " " << m[3][0] << std::endl;
    std::cout << m[0][1] << " " << m[1][1] << " " << m[2][1] << " " << m[3][1] << std::endl;
    std::cout << m[0][2] << " " << m[1][2] << " " << m[2][2] << " " << m[3][2] << std::endl;
    std::cout << m[0][3] << " " << m[1][3] << " " << m[2][3] << " " << m[3][3] << std::endl;
}

void utilityCore::printVec4(const glm::vec4 &m) {
    std::cout << m[0] << " " << m[1] << " " << m[2] << " " << m[3] << std::endl;
}

void utilityCore::printVec3(const glm::vec3 &m) {
    std::cout << m[0] << " " << m[1] << " " << m[2] << std::endl;
}

#if ENABLE_PROFILE_LOG

void utilityCore::ProfileLog::cacheInStep(float avgFPS, float timeElapsed) {
  //m_frameRates.push_back(avgFPS);
  m_frameRates.push_back(1000.f / timeElapsed);
  m_frameDurations.push_back(timeElapsed);
  for (size_t i = 0; i < m_cudaEvents.size(); ++i) {
    cudaEventSynchronize(m_cudaEvents[i]);
  }
  m_eventDurations.emplace_back();
  std::unordered_map<size_t, float>& eventDurationMap = m_eventDurations.back();
  eventDurationMap.reserve(m_cudaEvents.size() - 1);
  for (size_t i = 0; i + 1 < m_cudaEvents.size(); ++i) {
    eventDurationMap[i] = 0.f;
    cudaEventElapsedTime(&eventDurationMap[i], m_cudaEvents[i], m_cudaEvents[i + 1]);
  }
}

utilityCore::ProfileLog& utilityCore::ProfileLog::get() {
  static ProfileLog profileLog;
  return profileLog;
}

bool utilityCore::ProfileLog::isRecordingEvent() const {
  return m_recording;
}

bool utilityCore::ProfileLog::isProfiling() const {
  return m_profiling;
}

const std::string& utilityCore::ProfileLog::getEndEventName() const {
  return m_endEventName;
}

void utilityCore::ProfileLog::clearArgs() {
  m_args.empty();
  m_kwargs.empty();
}

void utilityCore::ProfileLog::addArg(const std::string& arg) {
  m_args.push_back(arg);
}

void utilityCore::ProfileLog::addKwArg(const std::string& key, int value) {
  if (m_kwargs.find(key) == m_kwargs.end()) {
    m_kwargs_keys.push_back(key);
  }
  m_kwargs[key] = value;
}

void utilityCore::ProfileLog::initProfile(const std::string& prefix, const std::string& endEventName, int startFrame, int frameCount, int frameInterval, std::ios_base::openmode mode) {
  m_mode = mode;
  m_endEventName = endEventName;
  m_startFrame = startFrame;
  m_endFrame = startFrame + frameCount * frameInterval;

  //m_startTime = startTime;
  m_frameInterval = frameInterval;

  //m_dataSizeToCount = (m_endFrame - m_startFrame) / m_frameInterval;
  m_dataSizeToCount = frameCount;

  m_frameRates.reserve(m_dataSizeToCount);
  m_frameDurations.reserve(m_dataSizeToCount);
  m_eventDurations.reserve(m_dataSizeToCount);

  m_filePath = prefix;
  //for (const std::pair<std::string, int>& kw : m_kwargs) {
  //  m_filePath.append("," + kw.first + "_" + std::to_string(kw.second));
  for(const std::string& arg : m_kwargs_keys) {
    m_filePath.append("," + arg + "_" + std::to_string(m_kwargs[arg]));
  }
  for (const std::string& arg : m_args) {
    m_filePath.append("," + arg);
  }
  m_filePath.append(".csv");
  m_recording = true;
}

size_t utilityCore::ProfileLog::registerEvent(const std::string& eventName) {
  auto it = m_cudaEventIndices.find(eventName);
  if (it != m_cudaEventIndices.end()) {
    return it->second;
  }
  size_t result = m_cudaEvents.size();
  m_cudaEventNames.push_back(eventName);
  m_cudaEventIndices[eventName] = result;
  m_cudaEvents.emplace_back();
  cudaEvent_t& newEvent = m_cudaEvents.back();
  cudaEventCreate(&newEvent);
  return result;
}

void utilityCore::ProfileLog::recordEvent(const std::string& eventName) {
  size_t index = registerEvent(eventName);
  if (!m_recording) {
    return;
  }
  cudaEvent_t& event = m_cudaEvents[index];
  cudaEventRecord(event);
}

void utilityCore::ProfileLog::step(float avgFPS, float timeStamp){
  float timeElapsed = timeStamp - m_timeStamp;
  m_timeStamp = timeStamp;
  m_profiling = false;

  if (m_frameCount >= m_startFrame && m_frameCount < m_endFrame) {
  //if(timeStamp >= m_startTime) {
  //  if (m_startFrame == -1) {
  //    m_startFrame = m_frameCount;
  //    m_endFrame = m_frameCount + m_dataSizeToCount * m_frameInterval;
  //  }
  //  if (m_frameCount < m_endFrame) {
      if (m_frameInterval <= 1 || (m_frameCount - m_startFrame) % m_frameInterval == 0) {
        cacheInStep(avgFPS, timeElapsed);
      }
      m_profiling = true;
  //  }
  }

  if(!m_profiling) {
    if (m_recording && m_frameCount >= m_endFrame) {
      writeToFile(m_mode);
      unregisterEvents();
    }
  }
  ++m_frameCount;
}
void utilityCore::ProfileLog::writeToFile(std::ios_base::openmode mode) {
  if (m_eventDurations.size() == 0) {
    return;
  }
  
  m_fout.open(m_filePath, mode);
  m_fout << ",fps,duration";
  m_fout << ",cudaTotal";
  for (size_t i = 0; i + 1 < m_cudaEventNames.size(); ++i) {
    const std::string& header = m_cudaEventNames[i];
    m_fout << ',' << header;
  }
  m_fout << std::endl;
  int frameCount = m_startFrame;
  
  double avgFPS = 0., avgDuration = 0., avgTotal = 0.;
  double maxFPS = 0., maxDuration = 0., maxTotal = 0.;
  double minFPS = std::numeric_limits<double>::max(), minDuration = std::numeric_limits<double>::max(), minTotal = std::numeric_limits<double>::max();

  std::unordered_map<size_t, double> avgEventDurations;
  std::unordered_map<size_t, double> maxEventDurations;
  std::unordered_map<size_t, double> minEventDurations;

  for (std::pair<const size_t, float>& eventPair : m_eventDurations[0]) {
    avgEventDurations[eventPair.first] = 0.;
    maxEventDurations[eventPair.first] = 0.;
    minEventDurations[eventPair.first] = std::numeric_limits<double>::max();
  }
  double invCount = 1. / m_eventDurations.size();

  std::stringstream ss;

  for (size_t i = 0; i < m_frameRates.size(); ++i) {
    double cudaTotalDuration = 0.;

    avgFPS += m_frameRates[i] * invCount;
    avgDuration += m_frameDurations[i] * invCount;

    maxFPS = std::max<double>(maxFPS, m_frameRates[i]);
    minFPS = std::min<double>(minFPS, m_frameRates[i]);

    maxDuration = std::max<double>(maxDuration, m_frameDurations[i]);
    minDuration = std::min<double>(minDuration, m_frameDurations[i]);

    //m_fout << frameCount << ',' << m_frameRates[i] << ',' << m_frameDurations[i];

    std::unordered_map<size_t, float>& eventDurationMap = m_eventDurations[i];
    for (std::pair<const size_t, float>& eventPair : eventDurationMap) {
      avgEventDurations[eventPair.first] += eventPair.second * invCount;
      maxEventDurations[eventPair.first] = std::max<double>(maxEventDurations[eventPair.first], eventPair.second);
      minEventDurations[eventPair.first] = std::min<double>(minEventDurations[eventPair.first], eventPair.second);
      cudaTotalDuration += eventPair.second;
      //m_fout << ',' << eventPair.second;
    }
    avgTotal += cudaTotalDuration * invCount;
    maxTotal = std::max(maxTotal, static_cast<double>(cudaTotalDuration));
    minTotal = std::min(minTotal, static_cast<double>(cudaTotalDuration));
    //m_fout << ',' << cudaTotalDuration << std::endl;
  }
  m_fout << "avg_" << m_frameRates.size() << ',' << avgFPS << ',' << avgDuration;
  m_fout << ',' << avgTotal;
  for (std::pair<const size_t, double>& eventPair : avgEventDurations) {
    m_fout << ',' << eventPair.second;
  }
  m_fout << std::endl;

  m_fout << "max_" << m_frameRates.size() << ',' << maxFPS << ',' << maxDuration;
  m_fout << ',' << maxTotal;
  for (std::pair<const size_t, double>& eventPair : maxEventDurations) {
    m_fout << ',' << eventPair.second;
  }
  m_fout << std::endl;

  m_fout << "min_" << m_frameRates.size() << ',' << minFPS << ',' << minDuration;
  m_fout << ',' << minTotal;
  for (std::pair<const size_t, double>& eventPair : minEventDurations) {
    m_fout << ',' << eventPair.second;
  }
  m_fout << std::endl;

  for (size_t i = 0; i < m_frameRates.size(); ++i) {
    double cudaTotalDuration = 0.;

    m_fout << frameCount << ',' << m_frameRates[i] << ',' << m_frameDurations[i];

    std::unordered_map<size_t, float>& eventDurationMap = m_eventDurations[i];
    for (std::pair<const size_t, float>& eventPair : eventDurationMap) {
      avgEventDurations[eventPair.first] += eventPair.second * invCount;
      cudaTotalDuration += eventPair.second;
      //m_fout << ',' << eventPair.second;
    }
    m_fout << ',' << cudaTotalDuration;
    for (std::pair<const size_t, float>& eventPair : eventDurationMap) {
      m_fout << ',' << eventPair.second;
    }
    m_fout << std::endl;
    frameCount += m_frameInterval;
  }

  m_fout.close();

  std::cout << "Write profile at " << m_filePath << std::endl;
}

void utilityCore::ProfileLog::unregisterEvents() {
  m_recording = false;
  m_cudaEventIndices.empty();
  m_cudaEventNames.empty();
  for (size_t i = 0; i < m_cudaEvents.size(); ++i) {
    cudaEventDestroy(m_cudaEvents[i]);
  }
  m_cudaEvents.empty();
}
#endif // ENABLE_PROFILE_LOG
