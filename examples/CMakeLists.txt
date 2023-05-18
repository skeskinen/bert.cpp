link_libraries("-lrt")

add_executable(server server.cpp)
target_link_libraries(server PRIVATE bert ggml)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE bert ggml)

add_executable(test_tokenizer test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE bert ggml)
