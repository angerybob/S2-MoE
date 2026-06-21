# EAGLE3 comprehensive end-to-end test debug
set(TARGET_COMPREHENSIVE llama-speculative-eagle3-comprehensive)
add_executable(${TARGET_COMPREHENSIVE} speculative-eagle3-comprehensive.cpp)
target_link_libraries(${TARGET_COMPREHENSIVE} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_COMPREHENSIVE} PRIVATE cxx_std_17)
install(TARGETS ${TARGET_COMPREHENSIVE} RUNTIME)

# EAGLE3 comprehensive end-to-end test debug
set(TARGET_COMPREHENSIVE llama-speculative-eagle3-comprehensive-fix)
add_executable(${TARGET_COMPREHENSIVE} speculative-eagle3-comprehensive-fix.cpp)
target_link_libraries(${TARGET_COMPREHENSIVE} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_COMPREHENSIVE} PRIVATE cxx_std_17)
install(TARGETS ${TARGET_COMPREHENSIVE} RUNTIME)

# Simple debug test
set(TARGET_DEBUG_SIMPLE llama-speculative-eagle3-simple-predict)
add_executable(${TARGET_DEBUG_SIMPLE} speculative-eagle3-simple-predict.cpp)
target_link_libraries(${TARGET_DEBUG_SIMPLE} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_DEBUG_SIMPLE} PRIVATE cxx_std_17)
install(TARGETS ${TARGET_DEBUG_SIMPLE} RUNTIME)

# EAGLE3 simple logic test (speculative decoding flow)
set(TARGET_SIMPLE_LOGIC llama-speculative-eagle3-simple-logic)
add_executable(${TARGET_SIMPLE_LOGIC} speculative-eagle3-simple-logic.cpp)
target_link_libraries(${TARGET_SIMPLE_LOGIC} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_SIMPLE_LOGIC} PRIVATE cxx_std_17)
install(TARGETS ${TARGET_SIMPLE_LOGIC} RUNTIME)

# EAGLE3 embedding layer sharing API test
set(TARGET_EMBD_SHARING llama-speculative-eagle3-embd-sharing-test)
add_executable(${TARGET_EMBD_SHARING} speculative-eagle3-embd-sharing-test.cpp)
target_link_libraries(${TARGET_EMBD_SHARING} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_EMBD_SHARING} PRIVATE cxx_std_17)
install(TARGETS ${TARGET_EMBD_SHARING} RUNTIME)

# EAGLE3 tree-based speculative decoder
set(TARGET_TREE_DECODER llama-speculative-eagle3-conti-predict)
add_executable(${TARGET_TREE_DECODER} speculative-eagle3-conti-predict.cpp)
target_link_libraries(${TARGET_TREE_DECODER} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_TREE_DECODER} PRIVATE cxx_std_17)
install(TARGETS ${TARGET_TREE_DECODER} RUNTIME)

# EAGLE3 nuclear diagnosis test
set(TARGET_NUCLEAR llama-speculative-eagle3-simple-input-analysis)
add_executable(${TARGET_NUCLEAR} speculative-eagle3-simple-input-analysis.cpp)
target_link_libraries(${TARGET_NUCLEAR} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_NUCLEAR} PRIVATE cxx_std_17)
install(TARGETS ${TARGET_NUCLEAR} RUNTIME)

# EAGLE3 batch test
set(TARGET_BATCH llama-speculative-eagle3-batch-test)
add_executable(${TARGET_BATCH} speculative-eagle3-batch-test.cpp)
target_link_libraries(${TARGET_BATCH} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_BATCH} PRIVATE cxx_std_17)
install(TARGETS ${TARGET_BATCH} RUNTIME)

# EAGLE3 batch test
set(TARGET_BATCH llama-speculative-eagle3-speclike)
add_executable(${TARGET_BATCH} speculative-eagle3-speclike.cpp)
target_link_libraries(${TARGET_BATCH} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_BATCH} PRIVATE cxx_std_17)
install(TARGETS ${TARGET_BATCH} RUNTIME)

# EAGLE3 comprehensive hs extraction
set(TARGET_COMPREHENSIVE llama-speculative-eagle3-tgt-hs-extraction)
add_executable(${TARGET_COMPREHENSIVE} speculative-eagle3-tgt-hs-extraction.cpp)
target_link_libraries(${TARGET_COMPREHENSIVE} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_COMPREHENSIVE} PRIVATE cxx_std_17)
install(TARGETS ${TARGET_COMPREHENSIVE} RUNTIME)

# EAGLE3 performance benchmark tool
set(TARGET_BENCH llama-speculative-eagle3-bench)
add_executable(${TARGET_BENCH} eagle3-bench.cpp)
target_link_libraries(${TARGET_BENCH} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_BENCH} PRIVATE cxx_std_17)
install(TARGETS ${TARGET_BENCH} RUNTIME)

# EAGLE3 tree-based performance benchmark tool
set(TARGET_TREE_BENCH llama-speculative-eagle3-tree-bench)
add_executable(${TARGET_TREE_BENCH} eagle3-tree-bench.cpp)
target_link_libraries(${TARGET_TREE_BENCH} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_TREE_BENCH} PRIVATE cxx_std_17)
install(TARGETS ${TARGET_TREE_BENCH} RUNTIME)
