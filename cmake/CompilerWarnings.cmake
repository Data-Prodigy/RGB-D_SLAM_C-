add_library(project_warnings INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
target_compile_options(project_warnings INTERFACE
		-Wall				#catches unused variables, uninitialized values, missing returns
		-Wextra				#catches unused parameters and comparison bugs
		-Wpedantic			#Warns when you use compiler extensions that aren’t standard C++
		-Wshadow			#catches error of a local variable that hides another one with the same name,
		-Wconversion		#Implicit conversions that can lose information, like double to float or int64_t to int.
		-Wsign-conversion  #Implicit mixing of signed and unsigned, like int i against v.size()
)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
target_compile_options(project_warnings INTERFACE /W4)
endif()