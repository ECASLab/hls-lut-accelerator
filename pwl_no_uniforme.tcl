# Autor: Patrick Hugo Nepveu Nelson <patrick.cr1405@gmail.com>
# Año: 2026 ECASLab

catch {::common::set_param -quiet hls.xocc.mode csynth};

open_project pwl_no_uniforme_test
set_top pwl_no_uniforme

add_files "pwl_no_uniforme.cpp" -cflags "-I ./"
add_files -tb "pwl_no_uniforme_tb.cpp" -cflags "-I ./"

open_solution -flow_target vitis test_12_6
set_part xcu55c-fsvh2892-2L-e

create_clock -period 4 -name default

config_dataflow -strict_mode warning
config_rtl -deadlock_detection sim
config_interface -m_axi_conservative_mode=1
config_interface -m_axi_addr64
config_interface -m_axi_auto_max_ports=0
config_export -format xo -ipname pwl_no_uniforme


csim_design
csynth_design
#cosim_design

close_project
puts "Fin"
exit