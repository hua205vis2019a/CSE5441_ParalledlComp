#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define SIZE 128

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
    WORK a[SIZE];
    WORK b[SIZE];
    WORK c[SIZE];
    WORK d[SIZE];
    WORK x;
    int len_a;
    int len_b;
    int len_c;
    int len_d;
    int len_x;
    int end;
}BUFFER;

__global__ void produce_a(BUFFER *pBuffer){
    int tid = threadIdx.x;
    if(tid < pBuffer->len_a)
            pBuffer->a[tid].key = transformA(pBuffer->a[tid].key);
}

__global__ void produce_b(BUFFER *pBuffer){
    int tid = threadIdx.x;
    if(tid < pBuffer->len_b)
            pBuffer->b[tid].key = transformB(pBuffer->b[tid].key);
}

__global__ void produce_c(BUFFER *pBuffer){
    int tid = threadIdx.x;
    if(tid < pBuffer->len_c)
            pBuffer->c[tid].key = transformC(pBuffer->c[tid].key);
}

__global__ void produce_d(BUFFER *pBuffer){
    int tid = threadIdx.x;
    if(tid < pBuffer->len_d)
            pBuffer->d[tid].key = transformD(pBuffer->d[tid].key);
}

__global__ void produce_x(BUFFER *pBuffer){
    int tid = threadIdx.x;
    if(tid < pBuffer->len_x)
            pBuffer->x.key = 0;
}


__global__ void consume_a(BUFFER *pBuffer){
    int tid = threadIdx.x;
    if(tid < pBuffer->len_a)
            pBuffer->a[tid].decoded_key = transformA(pBuffer->a[tid].key);
}

__global__ void consume_b(BUFFER *pBuffer){
    int tid = threadIdx.x;
    if(tid < pBuffer->len_b)
            pBuffer->b[tid].decoded_key = transformB(pBuffer->b[tid].key);
}

__global__ void consume_c(BUFFER *pBuffer){
    int tid = threadIdx.x;
    if(tid < pBuffer->len_c)
            pBuffer->c[tid].decoded_key = transformC(pBuffer->c[tid].key);
}

__global__ void consume_d(BUFFER *pBuffer){
    int tid = threadIdx.x;
    if(tid < pBuffer->len_d)
            pBuffer->d[tid].decoded_key = transformD(pBuffer->d[tid].key);
}

__global__ void consume_x(BUFFER *pBuffer){
    int tid = threadIdx.x;
    if(tid < pBuffer->len_x)
            pBuffer->end = 1;
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
    dim3 dimBlock(128);

    WORK input;
    while(!pHostBuffer->end){
        clock_t start_clock1 = 0;       // producer start clock for each
        clock_t end_clock1 = 0;         // producer end clock for each
        clock_t start_clock2 = 0;       // consumer start clock for each
        clock_t end_clock2 = 0;         // consumer end clock for each
        time_t start_time1 = 0;         // producer start time for each
        time_t end_time1 = 0;           // producer end time for each
        time_t start_time2 = 0;         // consumer start time for each
        time_t end_time2 = 0;           // consumer end time for each

        while((pHostBuffer->len_a + pHostBuffer->len_b + pHostBuffer->len_c + pHostBuffer->len_d) < SIZE){
            input = produce_each();
            if(input.cmd == 'A'){
                pHostBuffer->a[pHostBuffer->len_a] = input;
                pHostBuffer->len_a++;
            }
            if(input.cmd == 'B'){
                pHostBuffer->b[pHostBuffer->len_b] = input;
                pHostBuffer->len_b++;
            }
            if(input.cmd == 'C'){
                pHostBuffer->c[pHostBuffer->len_c] = input;
                pHostBuffer->len_c++;
            }
            if(input.cmd == 'D'){
                pHostBuffer->d[pHostBuffer->len_d] = input;
                pHostBuffer->len_d++;
            }
            if(input.cmd == 'X'){
                pHostBuffer->x = input;
                pHostBuffer->len_x++;
                break;
            }    
        }

        cudaMemcpy(pBuffer, pHostBuffer, sizeof(BUFFER), cudaMemcpyHostToDevice);

        start_time1 = time(NULL);
        start_clock1 = clock();
        produce_a<<<dimGrid, dimBlock>>>(pBuffer);
        produce_b<<<dimGrid, dimBlock>>>(pBuffer);
        produce_c<<<dimGrid, dimBlock>>>(pBuffer);
        produce_d<<<dimGrid, dimBlock>>>(pBuffer);
        produce_x<<<dimGrid, dimBlock>>>(pBuffer);
        end_clock1 = clock();
        end_time1 = time(NULL);
        
        time2_1 += difftime(end_time1, start_time1);
        clock2_1 += end_clock1 - start_clock1;

        start_time2 = time(NULL);
        start_clock2 = clock();
        consume_a<<<dimGrid, dimBlock>>>(pBuffer);
        consume_b<<<dimGrid, dimBlock>>>(pBuffer);
        consume_c<<<dimGrid, dimBlock>>>(pBuffer);
        consume_d<<<dimGrid, dimBlock>>>(pBuffer);
        consume_x<<<dimGrid, dimBlock>>>(pBuffer);
        end_clock2 = clock();
        end_time2 = time(NULL);
        
        time2_2 += difftime(end_time2, start_time2);
        clock2_2 += end_clock2 - start_clock2;

        cudaMemcpy(pHostBuffer, pBuffer, sizeof(BUFFER), cudaMemcpyDeviceToHost);
        
        for(int i = 0; i < pHostBuffer->len_a; i++)
            printf("Q:%d\t%c\t%u\t%u\n", i, pHostBuffer->a[i].cmd, pHostBuffer->a[i].key, pHostBuffer->a[i].decoded_key);
        for(int i = 0; i < pHostBuffer->len_b; i++){
            int p = i + pHostBuffer->len_a;
            printf("Q:%d\t%c\t%u\t%u\n", p, pHostBuffer->b[i].cmd, pHostBuffer->b[i].key, pHostBuffer->b[i].decoded_key);
        }
        for(int i = 0; i < pHostBuffer->len_c; i++){
            int p = i + pHostBuffer->len_a + pHostBuffer->len_b;
            printf("Q:%d\t%c\t%u\t%u\n", p, pHostBuffer->c[i].cmd, pHostBuffer->c[i].key, pHostBuffer->c[i].decoded_key);
        }
        for(int i = 0; i < pHostBuffer->len_d; i++){
            int p = i + pHostBuffer->len_a + pHostBuffer->len_b + pHostBuffer->len_c;
            printf("Q:%d\t%c\t%u\t%u\n", p, pHostBuffer->d[i].cmd, pHostBuffer->d[i].key, pHostBuffer->d[i].decoded_key);    
        }
        
        pHostBuffer->len_a = 0;
        pHostBuffer->len_b = 0;
        pHostBuffer->len_c = 0;
        pHostBuffer->len_d = 0;
        pHostBuffer->len_x = 0;
    }
    
    printf ("###################running time###################\n");
    printf ("producer time(2): %fs, clock(2): %fs\n", time2_1, ((float)clock2_1)/CLOCKS_PER_SEC);
    printf ("consumer time(2): %fs, clock(2): %fs\n", time2_2, ((float)clock2_2)/CLOCKS_PER_SEC);
    
    return 0;
}