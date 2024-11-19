#!/bin/bash

# Build the project
cmake ..
make

# Counter for embedding sizes
embd_list=( 64)
total_embd=${#embd_list[@]}
counter_embd=1

for embd in "${embd_list[@]}"
do
    echo "Processing embedding size $embd ($counter_embd/$total_embd)..."
    /opt/intel/oneapi/advisor/latest/bin64/advixe-cl --collect=roofline --project-dir=./projects/advixe-project-mult-$embd -- ./matrix_multiplication 200 $embd
    /opt/intel/oneapi/advisor/latest/bin64/advixe-cl --report=roofline --format=text --report-output=./reports/mult-$embd.txt --project-dir=./projects/advixe-project-mult-$embd    
    ((counter_embd++))
done

# Counter for dimensions
dim_list=(768 1024)
compress_dim=(16)
total_dim=${#dim_list[@]}
counter_dim=1

for embd in "${embd_list[@]}"
do
    for dim in "${dim_list[@]}"
    do
        for comp in "${compress_dim[@]}"
            do
                echo "Processing embedding size $embd and dimension $dim ($counter_dim/$(($total_embd * $total_dim)))..."
                /opt/intel/oneapi/advisor/latest/bin64/advixe-cl --collect=roofline --project-dir=./projects/advixe-project-lars-$embd-$dim-$comp -- ./matrix_lars 200 $embd $dim $comp
               /opt/intel/oneapi/advisor/latest/bin64/advixe-cl --report=roofline --format=text --report-output=./reports/lars-$embd-$dim-$comp.txt --project-dir=./projects/advixe-project-lars-$embd-$dim-$comp
           ((counter_dim++))
           done
    done
done

