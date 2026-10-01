cmake_minimum_required(VERSION 3.13)

# Board: "pico" (RP2040) or "pico2" (RP2350). Override with -DPICO_BOARD=pico2
if (NOT DEFINED PICO_BOARD)
    set(PICO_BOARD pico)
endif()

include(pico_sdk_import.cmake)
project(silicon_neuron C CXX ASM)
set(CMAKE_C_STANDARD 11)
pico_sdk_init()

add_executable(silicon_neuron
    main.c
    nn_core.c
    trace_io.c
    generated/model_data.c
)
target_include_directories(silicon_neuron PRIVATE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/generated)
target_compile_options(silicon_neuron PRIVATE -O2 -Wall)
target_link_libraries(silicon_neuron pico_stdlib hardware_clocks)

# Serial goes over UART0 (GP0/GP1) to the Raspberry Pi, NOT over USB:
# the Pico is powered through the INA226 during measurements and USB must be unplugged.
pico_enable_stdio_uart(silicon_neuron 1)
pico_enable_stdio_usb(silicon_neuron 0)

pico_add_extra_outputs(silicon_neuron)   # produces silicon_neuron.uf2
