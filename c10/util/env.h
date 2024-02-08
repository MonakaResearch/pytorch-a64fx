#pragma once

#include <c10/macros/Export.h>
#include <optional>
#include <string>

namespace c10::utils {

// Checks an environment variable is set.
C10_API bool has_env(const char* name);

// Reads an environment variable and returns the content if it is set.
C10_API std::optional<std::string> get_env(const char* name);

// Reads an environment variable and returns
// - optional<true>,              if set equal to "1"
// - optional<false>,             if set equal to "0"
// - nullopt,   otherwise
//
// NB:
// Issues a warning if the value of the environment variable is not 0 or 1.
C10_API std::optional<bool> check_env(const char* name);
} // namespace c10::utils
