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
  const std::string arch_str = s.value();
  ICHECK(arch_str.size() >= 3);
  ICHECK_EQ(arch_str.compare(0, 3, "sm_"), 0)
      << "arch string must start with sm_";
  return std::stoi(arch_str.substr(3));
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

bool TargetIsSm100(Target target) {
  if (!TargetIsCuda(target))
    return false;
  int arch = GetArchInt(target);
  return arch >= 100 & arch <= 110;
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

bool TargetHasTmem(Target target) {
  if (!TargetIsCuda(target))
    return false;
  return TargetIsSm100(target);
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

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.TargetIsCuda",
           [](Target target) { return TargetIsCuda(target); })
      .def("tl.TargetIsRocm",
           [](Target target) { return TargetIsRocm(target); })
      .def("tl.TargetIsVolta",
           [](Target target) { return TargetIsVolta(target); })
      .def("tl.TargetIsTuring",
           [](Target target) { return TargetIsTuring(target); })
      .def("tl.TargetIsAmpere",
           [](Target target) { return TargetIsAmpere(target); })
      .def("tl.TargetIsHopper",
           [](Target target) { return TargetIsHopper(target); })
      .def("tl.TargetIsSM120",
           [](Target target) { return TargetIsSM120(target); })
      .def("tl.TargetIsCDNA",
           [](Target target) { return TargetIsCDNA(target); })
      .def("tl.TargetHasAsyncCopy",
           [](Target target) { return TargetHasAsyncCopy(target); })
      .def("tl.TargetHasLdmatrix",
           [](Target target) { return TargetHasLdmatrix(target); })
      .def("tl.TargetHasStmatrix",
           [](Target target) { return TargetHasStmatrix(target); })
      .def("tl.TargetHasBulkCopy",
           [](Target target) { return TargetHasBulkCopy(target); })
      .def("tl.TargetGetWarpSize",
           [](Target target) { return TargetGetWarpSize(target); });
});

} // namespace tl
} // namespace tvm
