include("Library.jl")

# Performance Benchmark
B = [
    1.0 -1.0 4.0 0.0 2.0 9.0; 
    0.0 5.0 -2.0 7.0 8.0 4.0; 
    1.0 0.0 5.0 7.0 3.0 -2.0; 
    6.0 -1.0 2.0 3.0 0.0 8.0;
    -4.0 2.0 0.0 5.0 -5.0 3.0;
    0.0 7.0 -1.0 5.0 4.0 -2.0
]

b = [
    19.0;
    2.0;
    13.0;
    -7.0;
    -9.0;
    2.0
]

using Main.MatInv

@time Main.MatInv.gauss_jordan(B, b)
@time Main.MatInv.lu_decom(B, b)
@time Main.MatInv.jacobi(B, b)
@time Main.MatInv.gauss_siedel(B, b)