include_directories(${CMAKE_SOURCE_DIR}/)

if (MSVC)
    if (BERT_AVX2)
        add_compile_options(/arch:AVX2)
    elseif (BERT_AVX)
        add_compile_options(/arch:AVX)
    endif()
endif()

if(WIN32)
add_definitions(-DNOMINMAX=1)
endif()


add_executable(server server.cpp)
if(NOT WIN32)
	target_link_libraries(server PRIVATE bert ggml)
else()
	target_link_libraries(server PRIVATE bert ggml ws2_32)
endif()
add_executable(main main.cpp)
target_link_libraries(main PRIVATE bert ggml)

add_executable(test_tokenizer test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE bert ggml)