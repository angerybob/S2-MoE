add_executable(llama-speculative-next speculative-next.cpp)
target_link_libraries(llama-speculative-next PRIVATE llama)
target_include_directories(llama-speculative-next PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_features(llama-speculative-next PRIVATE cxx_std_17)
install(TARGETS llama-speculative-next RUNTIME)

# The ordinary S2-MoE entry point dispatches hybrid GGUFs to the same driver.
add_library(llama-hybrid-driver STATIC speculative-next.cpp)
target_link_libraries(llama-hybrid-driver PRIVATE llama)
target_include_directories(llama-hybrid-driver PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_features(llama-hybrid-driver PRIVATE cxx_std_17)
target_compile_definitions(llama-hybrid-driver PRIVATE LLAMA_HYBRID_DRIVER=1)
