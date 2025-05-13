# CMakeList.txt : projet CMake pour verifCam, incluez la source et définissez
# la logique spécifique au projet ici.
#
cmake_minimum_required (VERSION 3.8)

# Activez Rechargement à chaud pour les compilateurs MSVC si cela est pris en charge.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("verifCam")
include(D:/vcpkg/scripts/buildsystems/vcpkg.cmake)
# Ajoutez une source à l'exécutable de ce projet.
add_executable (verifCam "verifCam.cpp" )

find_package(nlohmann_json CONFIG REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(stb REQUIRED)
find_package(OpenCV REQUIRED)
find_path(TINYGLTF_INCLUDE_DIRS "tiny_gltf.h")
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries (verifCam OpenGL::GL GLEW::GLEW glfw glm::glm ${Stb_INCLUDE_DIR} ${OpenCV_LIBS} ${TINYGLTF_INCLUDE_DIRS} nlohmann_json::nlohmann_json )
if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET verifCam PROPERTY CXX_STANDARD 20)
endif()

# TODO: Ajoutez des tests et installez des cibles si nécessaire.
