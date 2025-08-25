/*!
 * \file tl/target/utils.cc
 * \brief helper functions for target attributes.
 */

#include "utils.h"

namespace tvm {
namespace tl {

bool TargetIsCuda(Target target) {
  return target->GetTargetDeviceType() == kDLCUDA;
}
bool TargetIsRocm(Target target) {
  return target->GetTargetDeviceType() == kDLROCM;
}

int GetArchInt(Target target) {
  auto s = target->GetAttr<String>("arch");
  ICHECK(s.defined());
  const char *arch_str = s.value().c_str();
  ICHECK_EQ(arch_str[0], 's');
  ICHECK_EQ(arch_str[1], 'm');
  ICHECK_EQ(arch_str[2], '_');
  return atoi(&arch_str[3]);
}

bool TargetIsVolta(Target target) {
  if (!TargetIsCuda(target))
    return false;
  int arch = GetArchInt(target);
  return arch >= 70 && arch < 75;
}

bool TargetIsTuring(Target target) {
  if (!TargetIsCuda(target))
    return false;
  int arch = GetArchInt(target);
  return arch >= 75 && arch < 80;
}

bool TargetIsAmpere(Target target) {
  if (!TargetIsCuda(target))
    return false;
  int arch = GetArchInt(target);
  return arch >= 80 && arch < 90;
}

bool TargetIsHopper(Target target) {
  if (!TargetIsCuda(target))
    return false;
  int arch = GetArchInt(target);
  return arch >= 90 && arch < 100;
}

bool TargetIsSM120(Target target) {
  if (!TargetIsCuda(target))
    return false;
  int arch = GetArchInt(target);
  return arch >= 120 && arch < 130;
}

bool TargetIsCDNA(Target target) {
  if (!TargetIsRocm(target))
    return false;
  if (target->attrs.count("mcpu")) {
    std::string mcpu = Downcast<String>(target->attrs.at("mcpu"));
    // if mcpu start with "gfx9", it is CDNA
    return mcpu.find("gfx9") == 0;
  }
  return false;
}

bool TargetHasAsyncCopy(Target target) {
  if (TargetIsCuda(target)) {
    int arch = GetArchInt(target);
    return arch >= 80;
  } else if (TargetIsCDNA(target)) {
    if (target->attrs.count("mcpu")) {
      std::string mcpu = Downcast<String>(target->attrs.at("mcpu"));
      if (mcpu.rfind("gfx9", 0) == 0) {
        int gfx_version = std::stoi(mcpu.substr(3, 2));
        return gfx_version >= 94;
      }
      return false;
    } else {
      return false;
    }
  }

  return false;
}
bool TargetHasLdmatrix(Target target) {
  if (!TargetIsCuda(target))
    return false;
  int arch = GetArchInt(target);
  return arch >= 75;
}

bool TargetHasStmatrix(Target target) {
  if (!TargetIsCuda(target))
    return false;
  int arch = GetArchInt(target);
  return arch >= 90;
}

bool TargetHasBulkCopy(Target target) {
  if (!TargetIsCuda(target))
    return false;
  int arch = GetArchInt(target);
  return arch >= 90;
}

int TargetGetWarpSize(Target target) {
  int res = 32;
  if (TargetIsCDNA(target))
    res = 64;
  return res;
}

} // namespace tl
} // namespace tvm
