#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include<math.h>

//Functions
void backsubstitution_p(int *A, int *b, double *x, int n);
void backsubstitution_s(int *A, int *b, double *x, int n);
void conjugategradient_p(int *A, int *b, double *x, int n);
void conjugategradient_s(int *A, int *b, double *x, int n);
void seidel_p(int *A, int *b, double *x, int n);
void seidel_s(int *A, int *b, double *x, int n);


int main(int argc, char *argv[])
{
    long n;
    //input matrix dimensions
    printf("Enter matrix dimension for a square matrix: ");
    n = 3;

    //defining requirements
    int matrix[n*n];
    int b[n];
    double x[n];
    matrix[0] = 10;
    matrix[1] = -1;
    matrix[2] = 3;
    matrix[3] = 1;
    matrix[4] = 11;
    matrix[5] = -5;
    matrix[6] = 2;
    matrix[7] = -1;
    matrix[8] = 13;
    b[0] = 7;
    b[1] = 5;
    b[2] = 8;

    // matrices
    int matrix2[n*n],matrix3[n*n], matrix4[n*n], matrix5[n*n], matrix6[n*n];
    int b2[n], b3[n], b4[n], b5[n], b6[n];
    double x2[n], x3[n], x4[n], x5[n], x6[n];

    for(int i =0; i<n*n; i++)
    {
        matrix2[i] = matrix[i];
        matrix3[i] = matrix[i];
        matrix4[i] = matrix[i];
        matrix5[i] = matrix[i];
        matrix6[i] = matrix[i];
    }

    for(int i = 0; i<n ; i++)
    {
        b2[i] = b[i];
        b3[i] = b[i];
        b4[i] = b[i];
        b5[i] = b[i];
        b6[i] = b[i];
    }


    printf("\n\nA * x = b  \n");
    printf("To find : x \n\n");

    //A Matrix
    printf("Matrix A: \n");
    for(int i = 0; i<n*n; i++)
    {
        printf("%d   ",matrix[i]);
        if((i+1)%n == 0)
        {
            printf("\n");
        }
    }

    //B Matrix
    printf("\n\nMatrix b: \n");
    for(int i = 0; i<n; i++)
    {
        printf("%d\n", b[i]);
    }

    printf("\n\nFINDING SOLUTIONS: \n");

    //Back Substitution
    printf("\n 1. Back substitution: \n");
    printf("\t A. Serially: \n");
    double ta = omp_get_wtime();
    backsubstitution_s(matrix, b, x, n);
    ta = omp_get_wtime() - ta;

    printf("\t B. Parallely: \n");
    double ta1 = omp_get_wtime();
    backsubstitution_p(matrix2, b2, x2, n);
    ta1 = omp_get_wtime() - ta1;

    printf("\n\t Time for serial execution: %0.30f\n\n", ta);
    printf("\n\t Time for parallel execution: %0.30f\n\n", ta1);

    //Conjugate Gradient
    printf("\n 2. Conjugate Gradient: \n");
    printf("\t A. Serially: \n");
    double tb = omp_get_wtime();
    conjugategradient_s(matrix3, b3, x3, n);
    tb = omp_get_wtime() - tb;
    printf("\t B. Parallely: \n");
    double tb1 = omp_get_wtime();
    conjugategradient_p(matrix4,b4,x4,n);
    tb1 = omp_get_wtime() - tb1;

    printf("\n\t Time for serial execution: %0.30f\n\n", tb);
    printf("\n\t Time for parallel execution: %0.30f\n\n", tb1);

    //Gauss Seidel
    printf("\n 3. Gauss Seidel: \n");
    printf("\t A. Serially: \n");
    double tc = omp_get_wtime();
    seidel_s(matrix6, b6, x6, n);
    tc = omp_get_wtime() - tc;

    printf("\t B. Parallely: \n");
    double tc1 = omp_get_wtime();
    seidel_p(matrix5,b5,x5,n);
    tc1 = omp_get_wtime() - tc1;

    printf("\n\t Time for serial execution: %0.30f\n\n", tc);
    printf("\n\t Time for parallel execution: %0.30f\n\n", tc1);

    free(matrix);
    free(b);
    free(x);
    free(matrix2);
    free(b2);
    free(x2);
    free(matrix3);
    free(b3);
    free(x3);
    free(matrix4);
    free(b4);
    free(x4);
    free(matrix5);
    free(b5);
    free(x5);
    return 0;
}

// For Back substitution:
// Use Gaussian elimination to get a matrix into upper triangular form
//Solve a triangular system using the row oriented algorithm
void backsubstitution_p(int*A, int *b, double *x, int n)
{
    int t;
    printf("\nEnter number of threads: ");
    scanf("%d", &t);
    double temp;
    int i, j, k;
    for(i =0; i < n-1; i++)
    {
        #pragma omp parallel default(none) num_threads(t) shared(n,A,b,i) private(j,k,temp)
		#pragma omp for schedule(static)
        for(j = i+1; j < n; j++)
        {
                temp = (A[j*(n)+i]) / (A[i*(n)+i]);

                for(k = i; k < n; k++)
                {
                    A[j*(n)+k] -= temp * (A[i*(n)+k]);
                }
                b[j] -= temp * (b[i]);
        }
    }

    double tmp;
    #pragma omp parallel num_threads(t) default(none) private(i,j) shared(A, b, x, n, tmp)
    for(int i= n-1; i >=0; i--)
    {
        #pragma omp single
        tmp = b[i];

        #pragma omp for reduction(+: tmp)

        for(j = i+1; j< n; j++)
            tmp += -A[i*n+j]*x[j];

        #pragma omp single
        x[i] = tmp/A[i*n+i];
    }

    for(int i =0; i < n; i++)
    {
        printf("\t\t%lf\n",x[i]);
    }

    return;
}


void backsubstitution_s(int *A, int *b, double *x, int n)
{
    int i, j, k;
    for(i =0; i < n-1; i++)
    {
        for(j = i+1; j < n; j++)
        {

            if(j>i)
            {
                double temp = (A[j*(n)+i]) / (A[i*(n)+i]);

                for(k = i; k < n; k++)
                {
                    A[j*(n)+k] -= temp * (A[i*(n)+k]);
                }
                b[j] -= temp * (b[i]);
            }
        }
    }
    double tmp;
    for(int i= n-1; i >=0; i--)
    {
        tmp = b[i];
        for(j = i+1; j< n; j++)
            tmp += -A[i*n+j]*x[j];
        x[i] = tmp/A[i*n+i];
    }

    for(int i =0; i < n; i++)
    {
        printf("\t\t%f\n",x[i]);
    }

    return ;
}

void conjugategradient_p(int *A, int *b, double *x,  int n)
{
    int t;
    int max_iterations;
    printf("\nEnter number of iterations: ");
    scanf("%d", &max_iterations);

    printf("\nEnter number of threads: ");
    scanf("%d", &t);

    double r[n];
    double p[n];
    double px[n];

    #pragma omp parallel for num_threads(t)
    for( int i = 0 ; i<n ; i++)
    {
        x[i] = 0;
        p[i] = b[i];
        r[i] = b[i];
        px[i] = 0;
    }


    int q = max_iterations;

    double alpha = 0;

    while(q--)
    {
        double sum = 0;
        #pragma omp parallel  for num_threads(t)  reduction(+ : sum)
        for(int i = 0 ; i < n ; i++)
        {
            sum = r[i]*r[i] + sum;
        }

        double temp[n];
        #pragma omp parallel for num_threads(t)
        for( int i = 0; i<n ; i++ )
        {
            temp[i] = 0;
        }

        double num = 0;
        #pragma omp parallel for num_threads(t)
        for(int i = 0 ; i < n ; i++)
        {
            #pragma omp parallel for reduction(+ : temp[i])
            for(int j = 0 ; j < n ; j++ )
            {
                temp[i] = A[i*n+j]*p[j] + temp[i];
            }
        }
        #pragma omp parallel for num_threads(t) reduction(+ : num)
        for(int j = 0 ; j < n ; j++)
        {
            num = num + temp[j]*p[j];
        }

        alpha = sum / num;

        #pragma omp parallel for num_threads(t)
        for(int i = 0; i < n ; i++ )
        {
            px[i] = x[i];
            x[i] = x[i] + alpha*p[i];
            r[i] = r[i] - alpha*temp[i];
        }

        double beta = 0;
        #pragma omp parallel for num_threads(t) reduction(+ : beta)
        for(int i = 0 ; i < n ; i++)
        {
            beta = beta + r[i]*r[i];
        }

        beta = beta / sum;

        #pragma omp parallel for num_threads(t)
        for (int i = 0 ; i < n ; i++ )
        {
            p[i] = r[i] + beta*p[i];
        }

        int c=0;
        for(int i = 0 ; i<n ; i++ )
        {
            if(r[i]<0.000001)
                c++;
        }

        if(c==n)
            break;

        }

    for( int i = 0 ; i<n ; i++ )
        printf("\t\t%f\n", x[i]);

    return;
}


void conjugategradient_s(int *A, int *b, double *x,  int n)
{
    int max_iterations;
    printf("\nEnter number of iterations: ");
    scanf("%d", &max_iterations);
    double r[n];
    double p[n];
    double px[n];
    for( int i = 0 ; i<n ; i++)
    {
        x[i] = 0;
        p[i] = b[i];
        r[i] = b[i];
        px[i] = 0;
    }


    double alpha = 0;
    while(max_iterations--)
    {

        double sum = 0;
        for(int i = 0 ; i < n ; i++)
        {
            sum = r[i]*r[i] + sum;
        }

        double temp[n];
        for( int i = 0; i<n ; i++ )
        {
            temp[i] = 0;
        }

        double num = 0;
        for(int i = 0 ; i < n ; i++)
        {
            for(int j = 0 ; j < n ; j++ )
            {
                temp[i] = A[i*n+j]*p[j] + temp[i];
            }
        }
        for(int j = 0 ; j < n ; j++)
        {
            num = num + temp[j]*p[j];
        }

        alpha = sum / num;
        for(int i = 0; i < n ; i++ )
        {
            px[i] = x[i];
            x[i] = x[i] + alpha*p[i];
            r[i] = r[i] - alpha*temp[i];
        }
        double beta = 0;
        for(int i = 0 ; i < n ; i++)
        {
            beta = beta + r[i]*r[i];
        }
        beta = beta / sum;
        for (int i = 0 ; i < n ; i++ )
        {
            p[i] = r[i] + beta*p[i];
        }
        int c=0;
        for(int i = 0 ; i<n ; i++ )
        {
            if(r[i]<0.000001 )
                c++;
        }
        if(c==n)
            break;
    }
    for( int i = 0 ; i<n ; i++ )
        printf("\t\t%f\n", x[i]);

    return;
}


void seidel_p(int A[], int b[], double x[], int n)
{
    double *dx;
    dx = (double*) calloc(n,sizeof(double));
    int i,j,k;
    double dxi;
    int maxit;
    double epsilon = 1.0e-4;
    double m[n];
    int p;
    for(int i = 0; i<n; i++)
    {
        x[i] = 0;
        m[i] = 0;
    }

    printf("\nEnter number of iterations: ");
    scanf("%d", &maxit);

    printf("\nEnter number of threads: ");
    scanf("%d", &p);
    int z = n/p;

    int id;
    int start, stop;

    for(k=0; k<maxit; k++)
    {
            printf("\n%d th iteration => \n", k+1);
            #pragma omp parallel for num_threads(p) schedule(static, n) reduction(+:dxi)
            for(int i=0; i<n; i++)
            {
                dxi = b[i];

                    for(int j=0; j<n; j++)
                    {
                        if(j!=i)
                        {
                            dxi-=A[i*n + j] * x[j];
                        }

                        x[i] = dxi / A[i*n + i];
                    }



                printf("x %d  = %f \n", i+1,  x[i]);
            }
    }
}


void seidel_s(int A[], int b[], double x[], int n)
{
    double *dx;
    dx = (double*) calloc(n,sizeof(double));
    int i,j,k;
    double dxi;
    double epsilon = 1.0e-4;
    int maxit ;
    double m[n];

    for(int i = 0; i<n; i++)
    {
        x[i] = 0;
    }
    printf("\nEnter number of iterations: ");
    {
        scanf("%d", &maxit);
    }

    for(k=0; k<maxit; k++)
    {
        double sum = 0.0;
        printf("\n%d th iteration => \n", k+1);
        for(int i=0; i<n; i++)
        {
            dxi = b[i];
            for(int j=0; j<n; j++)
            {
                if(j!=i)
                {
                    dxi-=A[i*n + j] * x[j];
                }

                x[i] = dxi / A[i*n + i];
            }
            printf("x %d = %f \n", i+1, x[i]);
        }
    }
}

