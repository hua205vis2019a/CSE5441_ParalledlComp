/*
 NAME: Zewen Hua
 CLASS&SECTION: CSE5441 TuTh 12:45-14:05
 ASSIGNMENT NUMBER: Lab5_mpi
 COMPILER: mpicc
 COMPILATION INSTRUCTION: mpicc -fopenmp transform.o lab5.c -o lab5_mpi
 SUBMIT DATE: Dec.5, 2018
 DESCRIPTION: This is a solution to producer-consumer problem. It combines openmp and mpi. MPI has 5 rank, rank0 read the file and manage 4 thread using openmp to connect with rank1234 remote node to do produce transform and consume transform. Then use send&recv to communicate. Lastly, print out the result at rank0.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>

/* provided functions in transform.o */
extern uint16_t transformA(uint16_t input_val);
extern uint16_t transformB(uint16_t input_val);
extern uint16_t transformC(uint16_t input_val);
extern uint16_t transformD(uint16_t input_val);

/* struct for work entry input */
typedef struct work_entry{
    char cmd;
    uint16_t key;
    uint16_t alpha;
    uint16_t result;
}WORK;

static WORK ori_input;
static int rank, len, end = 1;
static char cmd;
static uint16_t key, alpha, beta, result;
static WORK temp;
static WORK ori[4];

static clock_t start_clock1 = 0;       // producer start clock for each
static clock_t end_clock1 = 0;         // producer end clock for each
static clock_t start_clock2 = 0;       // consumer start clock for each
static clock_t end_clock2 = 0;         // consumer end clock for each
static time_t start_time1 = 0;         // producer start time for each
static time_t end_time1 = 0;           // producer end time for each
static time_t start_time2 = 0;         // consumer start time for each
static time_t end_time2 = 0;           // consumer end time for each
static clock_t clock2_1;        // producer clock(2)
static clock_t clock2_2;        // consumer clock(2)
static double time2_1;          // producer time(2)
static double time2_2;          // consumer time(2)

/* producer for each line, validate values, return struct after producer transformation */
WORK produce_each(){
    scanf("%c%u", &ori_input.cmd, &ori_input.key);
    if(ori_input.cmd<'A'||(ori_input.cmd>'D'&&ori_input.cmd!='X')|| ori_input.key>1000)    /* invalid input */
        produce_each();
    return ori_input;
}

int main(int argc, char *argv[]) {
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    /* rank0 get the file input, send them to rank 1234, receive the transformed message, print them out*/
    if(rank == 0){
        while(end){
            len = 0;
            /* get source input */
            for(int i = 0;i < 4;i++){
                temp = produce_each();
                if(temp.cmd != 'X'){
                    ori[i] = temp;
                    len++;
                }
                else{
                    end = 0;
                    break;
                }
            }
            /* implement 4 threads using openmp */
            omp_set_num_threads(4);
            #pragma omp parallel
            {
                int id;
                id = omp_get_thread_num();
                /* test if it really has 4 threads */
                #pragma omp master
                {
                    int numthreads;
                    numthreads = omp_get_num_threads();
                    if(numthreads != 4){
                        printf("error: incorrect number of threads, %d\n", numthreads);
                        exit(1);
                    }
                }
                /* thread 0123 for rank 1234, send message from rank 0 to rank 1234 and recv them. */
                if(id == 0){
                    cmd = ori[0].cmd;
                    key = ori[0].key;
                    MPI_Send(&cmd, 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                    MPI_Send(&key, 1, MPI_UNSIGNED, 1, 0, MPI_COMM_WORLD);
                    MPI_Send(&end, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
                    MPI_Recv(&alpha, 1, MPI_UNSIGNED, 1, 0, MPI_COMM_WORLD, &status);
                    MPI_Recv(&result, 1, MPI_UNSIGNED, 1, 0, MPI_COMM_WORLD, &status);
                    ori[id].alpha = alpha;
                    ori[id].result = result;
                }
                if(id == 1){
                    cmd = ori[1].cmd;
                    key = ori[1].key;
                    MPI_Send(&cmd, 1, MPI_CHAR, 2, 0, MPI_COMM_WORLD);
                    MPI_Send(&key, 1, MPI_UNSIGNED, 2, 0, MPI_COMM_WORLD);
                    MPI_Send(&end, 1, MPI_INT, 2, 0, MPI_COMM_WORLD);
                    MPI_Recv(&alpha, 1, MPI_UNSIGNED, 2, 0, MPI_COMM_WORLD, &status);
                    MPI_Recv(&result, 1, MPI_UNSIGNED, 2, 0, MPI_COMM_WORLD, &status);
                    ori[id].alpha = alpha;
                    ori[id].result = result;
                }
                if(id == 2){
                    cmd = ori[2].cmd;
                    key = ori[2].key;
                    MPI_Send(&cmd, 1, MPI_CHAR, 3, 0, MPI_COMM_WORLD);
                    MPI_Send(&key, 1, MPI_UNSIGNED, 3, 0, MPI_COMM_WORLD);
                    MPI_Send(&end, 1, MPI_INT, 3, 0, MPI_COMM_WORLD);
                    MPI_Recv(&alpha, 1, MPI_UNSIGNED, 3, 0, MPI_COMM_WORLD, &status);
                    MPI_Recv(&result, 1, MPI_UNSIGNED, 3, 0, MPI_COMM_WORLD, &status);
                    ori[id].alpha = alpha;
                    ori[id].result = result;
                }
                if(id == 3){
                    cmd = ori[3].cmd;
                    key = ori[3].key;
                    MPI_Send(&cmd, 1, MPI_CHAR, 4, 0, MPI_COMM_WORLD);
                    MPI_Send(&key, 1, MPI_UNSIGNED, 4, 0, MPI_COMM_WORLD);
                    MPI_Send(&end, 1, MPI_INT, 4, 0, MPI_COMM_WORLD);
                    MPI_Recv(&alpha, 1, MPI_UNSIGNED, 4, 0, MPI_COMM_WORLD, &status);
                    MPI_Recv(&result, 1, MPI_UNSIGNED, 4, 0, MPI_COMM_WORLD, &status);
                    ori[id].alpha = alpha;
                    ori[id].result = result;
                }
            }
            /* print results out */
            for(int i = 0;i < len;i++)
                printf("%c\t%u\t%u\t%u\n",ori[i].cmd,ori[i].key,ori[i].alpha,ori[i].result);
            }
        }
    /* rank 1234, recv message from rank0 thread 0123, do 2 transform, and send it back */
        else{
            while(1){
                MPI_Recv(&cmd, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(&key, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(&end, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, &status);
                start_time1 = time(NULL);
                start_clock1 = clock();
                switch(cmd){
                    case 'A':
                        alpha = transformA(key);
                        break;
                    case 'B':
                        alpha = transformB(key);
                        break;
                    case 'C':
                        alpha = transformC(key);
                        break;
                    case 'D':
                        alpha = transformD(key);
                        break;
                }
                end_clock1 = clock();
                end_time1 = time(NULL);
                
                time2_1 += difftime(end_time1, start_time1);
                clock2_1 += end_clock1 - start_clock1;
                
                beta = (alpha + 1) % 1000;
                start_time2 = time(NULL);
                start_clock2 = clock();
                switch(cmd){
                    case 'A':
                        result = transformA(beta);
                        break;
                    case 'B':
                        result = transformB(beta);
                        break;
                    case 'C':
                        result = transformC(beta);
                        break;
                    case 'D':
                        result = transformD(beta);
                        break;
                }
                end_clock2 = clock();
                end_time2 = time(NULL);
                
                time2_2 += difftime(end_time2, start_time2);
                clock2_2 += end_clock2 - start_clock2;
                
                MPI_Send(&alpha, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD);
                MPI_Send(&result, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD);
                if(end == 0)
                    break;
            }
        }
    MPI_Finalize();
    printf ("###################running time###################\n");
    printf ("producer time(2): %fs, clock(2): %fs\n", time2_1, ((float)clock2_1)/CLOCKS_PER_SEC);
    printf ("consumer time(2): %fs, clock(2): %fs\n", time2_2, ((float)clock2_2)/CLOCKS_PER_SEC);

    return 0;
}
