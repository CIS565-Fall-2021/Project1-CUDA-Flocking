#include "print_glm.hpp"

std::ostream &operator<<(std::ostream &o, const glm::vec4 &v) {
  o << std::setprecision(4) << "[" << std::setw(7) << v[0] << ", "
    << std::setw(7) << v[1] << ", " << std::setw(7) << v[2] << ", "
    << std::setw(7) << v[3] << "]";
  return o;
}

std::ostream &operator<<(std::ostream &o, const glm::vec3 &v) {
  o << std::setprecision(4) << "[" << std::setw(7) << v[0] << ", "
    << std::setw(7) << v[1] << ", " << std::setw(7) << v[2] << "]";
  return o;
}

std::ostream &operator<<(std::ostream &o, const glm::mat4 &m) {
  o << "\n[" << glm::row(m, 0) << "\n " << row(m, 1) << "\n " << row(m, 2)
    << "\n " << row(m, 3) << "]";

  return o;
}

std::ostream &operator<<(std::ostream &o, const glm::quat &q) {
  o << std::setprecision(4) << "[" << std::setw(7) << q[0] << ", "
    << std::setw(7) << q[1] << ", " << std::setw(7) << q[2] << ", "
    << std::setw(7) << q[3] << "]";
  return o;
}
