# Linear-equations-solving-using-parallelism
### Implentation of Back Substitution, Conjugate Gradient and Gauss Seidel using OpenMP parallelization

Solving large systems of linear equations in serial is time consuming and slow process.<br />
**Prallelizing** the algorithms using suitable parallel constructs can provide the correct solutions with a lesser time complexity. <br />
<br/>
This project presents parallel implementation of 3 algorithms for solving system of linear equations: <br />
1. Back Substitution, 
1. Conjugate Gradient and
1. Gauss Seidel 
    
OpenMP is used to parallelize the algos.<br />
For all the 3 algos, serial and parallel implementations is provided. Execution time for each of the 6 functions(serial and parallel implementations of the three algos) is calculated.<br />
Two code files: required and random are basically the same implementation, but random.c considered a random matrix input, while required.c considers specific input matrix considering the limitations of the algorithms.
