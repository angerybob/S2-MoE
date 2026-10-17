add_executable(llama-speculative-next speculative-next.cpp)
target_link_libraries(llama-speculative-next PRIVATE llama)
target_compile_features(llama-speculative-next PRIVATE cxx_std_17)
install(TARGETS llama-speculative-next RUNTIME)
