#ifndef __PRINT_GLM_HPP__
#define __PRINT_GLM_HPP__

#define GLM_FORCE_RADIANS

#include <glm/glm.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtc/quaternion.hpp>
#include <iomanip>
#include <iostream>

std::ostream &operator<<(std::ostream &o, const glm::vec4 &v);
std::ostream &operator<<(std::ostream &o, const glm::vec3 &v);
std::ostream &operator<<(std::ostream &o, const glm::mat4 &m);
std::ostream &operator<<(std::ostream &o, const glm::quat &q);

#endif  // __PRINT_GLM_HPP__
