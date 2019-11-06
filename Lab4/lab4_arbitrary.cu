#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define SIZE 64

__device__ uint16_t transformA(uint16_t input_val);
__device__ uint16_t transformB(uint16_t input_val);
__device__ uint16_t transformC(uint16_t input_val);
__device__ uint16_t transformD(uint16_t input_val);

typedef struct work_entry{
    char cmd;
    uint16_t key;
    uint16_t decoded_key;
}WORK;

typedef struct buffer{
    WORK b[SIZE];
    int len;
    int end;
}BUFFER;

__global__ void produce(BUFFER *pBuffer){
    int tid = threadIdx.x;
    if(tid < pBuffer->len){
        switch(pBuffer->b[tid].cmd){
            case 'A':
                pBuffer->b[tid].key = transformA(pBuffer->b[tid].key);
                break;
            case 'B':
                pBuffer->b[tid].key = transformB(pBuffer->b[tid].key);
                break;
            case 'C':
                pBuffer->b[tid].key = transformC(pBuffer->b[tid].key);
                break;
            case 'D':
                pBuffer->b[tid].key = transformD(pBuffer->b[tid].key);
                break;
            case 'X':
                pBuffer->b[tid].key = 0;
                break;
        }
    }
}

__global__ void consume(BUFFER *pBuffer){
    int tid = threadIdx.x;
    if(tid < pBuffer->len){
        switch(pBuffer->b[tid].cmd){
            case 'A':
                pBuffer->b[tid].decoded_key = transformA(pBuffer->b[tid].key);
                break;
            case 'B':
                pBuffer->b[tid].decoded_key = transformB(pBuffer->b[tid].key);
                break;
            case 'C':
                pBuffer->b[tid].decoded_key = transformC(pBuffer->b[tid].key);
                break;
            case 'D':
                pBuffer->b[tid].decoded_key = transformD(pBuffer->b[tid].key);
                break;
            case 'X':
                pBuffer->end = 1;
                break;
        }
    }
}

WORK produce_each(){
    static WORK ori_input;
    scanf("%c%u", &ori_input.cmd, &ori_input.key);
    if(ori_input.cmd<'A'||(ori_input.cmd>'D'&&ori_input.cmd!='X')|| ori_input.key>1000)    /* invalid input */
        produce_each();
    return ori_input;
}

int main(){
    static clock_t clock2_1;        // producer clock(2)
    static clock_t clock2_2;        // consumer clock(2)
    static double time2_1;          // producer time(2)
    static double time2_2;          // consumer time(2)

    BUFFER *pHostBuffer = (BUFFER*)malloc(sizeof(BUFFER));
    BUFFER *pBuffer;
    cudaMalloc((void**)&pBuffer, sizeof(BUFFER));
    
    dim3 dimGrid(1);
    dim3 dimBlock(64);

    while(!pHostBuffer->end){
        clock_t start_clock1 = 0;       // producer start clock for each
        clock_t end_clock1 = 0;         // producer end clock for each
        clock_t start_clock2 = 0;       // consumer start clock for each
        clock_t end_clock2 = 0;         // consumer end clock for each
        time_t start_time1 = 0;         // producer start time for each
        time_t end_time1 = 0;           // producer end time for each
        time_t start_time2 = 0;         // consumer start time for each
        time_t end_time2 = 0;           // consumer end time for each

        while(pHostBuffer->len < SIZE){
            pHostBuffer->b[pHostBuffer->len] = produce_each();
            pHostBuffer->len++;
            if(pHostBuffer->b[pHostBuffer->len-1].cmd == 'X')
                break;
        }

        cudaMemcpy(pBuffer, pHostBuffer, sizeof(BUFFER), cudaMemcpyHostToDevice);

        start_time1 = time(NULL);
        start_clock1 = clock();
        produce<<<dimGrid, dimBlock>>>(pBuffer);
        end_clock1 = clock();
        end_time1 = time(NULL);
        
        time2_1 += difftime(end_time1, start_time1);
        clock2_1 += end_clock1 - start_clock1;

        start_time2 = time(NULL);
        start_clock2 = clock();
        consume<<<dimGrid, dimBlock>>>(pBuffer);
        end_clock2 = clock();
        end_time2 = time(NULL);
        
        time2_2 += difftime(end_time2, start_time2);
        clock2_2 += end_clock2 - start_clock2;

        cudaMemcpy(pHostBuffer, pBuffer, sizeof(BUFFER), cudaMemcpyDeviceToHost);
        for(int i = 0; i < pHostBuffer->len; i++){
            if(pHostBuffer->b[i].cmd != 'X')
                printf("Q:%d\t%c\t%u\t%u\n", i, pHostBuffer->b[i].cmd, pHostBuffer->b[i].key, pHostBuffer->b[i].decoded_key);
        }
        pHostBuffer->len = 0;
    }
    
    printf ("###################running time###################\n");
    printf ("producer time(2): %fs, clock(2): %fs\n", time2_1, ((float)clock2_1)/CLOCKS_PER_SEC);
    printf ("consumer time(2): %fs, clock(2): %fs\n", time2_2, ((float)clock2_2)/CLOCKS_PER_SEC);
    
    return 0;
}


