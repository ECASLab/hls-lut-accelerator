# Author: Patrick Hugo Nepveu Nelson <patrick.nepveu.ecaslab@gmail.com>
# Year: 2026 
# ECASLab

catch {::common::set_param -quiet hls.xocc.mode csynth};

open_project 
set_top pwl_non_uniform

add_files "pwl_non_uniform.cpp" -cflags "-I ./"
add_files -tb "pwl_non_uniform_tb.cpp" -cflags "-I ./"

open_solution -flow_target vitis 
set_part xcu55c-fsvh2892-2L-e

create_clock -period 4 -name default

config_dataflow -strict_mode warning
config_rtl -deadlock_detection sim
config_interface -m_axi_conservative_mode=1
config_interface -m_axi_addr64
config_interface -m_axi_auto_max_ports=0
config_export -format xo -ipname pwl_non_uniform


#csim_design
#csynth_design
#cosim_design

close_project
puts "Finished"
exit