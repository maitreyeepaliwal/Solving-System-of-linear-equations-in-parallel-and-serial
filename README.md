# Linear-equations-solving-using-parallelism
###Implentation of Back Substitution, Conjugate Gradient and Gauss Seidel using OpenMP parallelization

Solving large systems of linear equations in serial is time consuming and slow process.
**Prallelizing** the algorithms using suitable parallel constructs can provide the correct solutions with a lesser time complexity. 
This project presents parallel implementation of 3 algorithms for solving system of linear equations: 
    1. Back Substitution, 
    2. Conjugate Gradient and
    3. Gauss Seidel 
    
OpenMP is used to parallelize the algos.
For all the 3 algos, serial and parallel implementations is provided. Execution time for each of the 6 functions(serial and parallel implementations of the three algos) is calculated.
Two code files: required and random are basically the same implementation, but random.c considered a random matrix input, while required.c considers specific input matrix considering the limitations of the algorithms.
