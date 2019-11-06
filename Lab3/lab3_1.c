/*
 NAME: Zewen Hua
 CLASS&SECTION: CSE5441 TuTh 12:45-14:05
 ASSIGNMENT NUMBER: Lab3
 COMPILER: Intel
 COMPILATION INSTRUCTION: icc transform.o lab3.c -o lab3_omp.out -fopenmp
 SUBMIT DATE: Nov.13, 2018
 DESCRIPTION: This is a solution to producer-consumer problem using openMP with 2, 8, 16, 32 threads. The size of work queue is 40. It has two parts: producer and consumer. Producer fills the queue, and consumer gets it empty. Resume the producer when read the X.
 */

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <time.h>
#include <omp.h>

/* buffer size */
#define BUFF_SIZE 40

/* provided functions in transform.o */
extern uint16_t transformA(uint16_t input_val);
extern uint16_t transformB(uint16_t input_val);
extern uint16_t transformC(uint16_t input_val);
extern uint16_t transformD(uint16_t input_val);

/* struct for source input */
typedef struct work_entry{
    char cmd;
    uint16_t encoded_key;
}WORK;

/* struct for buffer */
typedef struct buffer{
    WORK val_input[BUFF_SIZE];
    int len;
    int num1;
    int num2;
}BUFFER;

/* When the buffer is not full, then get a data line from the source file. If it is valid, transform it and store it into the buffer for consumer to get.*/
void producer(BUFFER *buffer, int *end){
    #pragma omp parallel
    {
        WORK each;
        char line[50];
        int flag = 1;           //data line is empty or not
        while(*end != 1 && flag){
            if(buffer->len == BUFF_SIZE)
                continue;
        /* get a data line from the source file */
        #pragma omp critical
        {
            flag = fgets(line, 50, stdin) != NULL;
        }
            if(flag){
                /* if the data is invalid, skip it */
                each.cmd = line[0];
                if(each.cmd < 'A' || (each.cmd > 'D' && each.cmd != 'X'))
                    continue;
                each.encoded_key = atoi(line + 1);
                if(each.encoded_key < 0 || each.encoded_key > 1000)    /* invalid input */
                    continue;
                /* if the data is valid, transform the key */
                switch(each.cmd){
                    case 'A':
                        each.encoded_key = transformA(each.encoded_key);
                        break;
                    case 'B':
                        each.encoded_key = transformB(each.encoded_key);
                        break;
                    case 'C':
                        each.encoded_key = transformC(each.encoded_key);
                        break;
                    case 'D':
                        each.encoded_key = transformD(each.encoded_key);
                        break;
                    case 'X':
                        each.encoded_key = 0;
                        break;
                }
                /* update the buffer with new data */
                #pragma omp critical
                {
                    buffer->val_input[buffer->num1] = each;
                    ++buffer->len;
                    if(++buffer->num1 == BUFF_SIZE)
                        buffer->num1 = 0;
                }
            }
        }
    }
}

/* When the buffer is not empty, then get a piece of data from the buffer and print it out.*/
void consumer(BUFFER *buffer, int *end){
    #pragma omp parallel
    {
        WORK input;
        int position;
        uint16_t decoded_key;
        while(*end == 0 || buffer->len > 0){
            int flag = 0;
            #pragma omp critical
            {
                if(buffer->len > 0){
                    position = buffer->num2;
                    input = buffer->val_input[buffer->num2];
                    if(++buffer->num2 >= BUFF_SIZE)
                        buffer->num2 = 0;
                    --buffer->len;
                    flag = 1;
                }
            }
            /* The buffer is empty, nothing left to consume */
            if(!flag)
                continue;
            /* Decode the encoded key */
            switch(input.cmd){
                case 'A':
                    decoded_key = transformA(input.encoded_key);
                    break;
                case 'B':
                    decoded_key = transformB(input.encoded_key);
                    break;
                case 'C':
                    decoded_key = transformC(input.encoded_key);
                    break;
                case 'D':
                    decoded_key = transformD(input.encoded_key);
                    break;
                case 'X':
                    *end = 1;
                    break;
            }
            /* Print the result out */
            if(input.cmd != 'X')
                printf("Q:%d\t%c\t%u\t%u\n", position, input.cmd, input.encoded_key, decoded_key);
        }
    }
}


int main(int argc, char** argv){
    int end = 0;
    int nt = atoi(argv[1]);
    BUFFER buffer = {
        .num1 = 0,
        .num2 = 0,
        .len = 0
    };
    
    static clock_t clock2_1;        // producer clock(2)
    static clock_t clock2_2;        // consumer clock(2)
    static double time2_1;          // producer time(2)
    static double time2_2;
    
    omp_set_num_threads(nt);
    #pragma omp parallel
    {
        int id;
        id = omp_get_thread_num();
        /* Verify the actual number of threads created */
        #pragma omp master
        {
            int numthreads;
            numthreads = omp_get_num_threads();
            if(numthreads!=nt){
                printf("error: incorrect number of threads, %d \n", numthreads);
                exit(1);
            }
        }
        clock_t start_clock1 = 0;       // producer start clock for each
        clock_t end_clock1 = 0;         // producer end clock for each
        clock_t start_clock2 = 0;       // consumer start clock for each
        clock_t end_clock2 = 0;         // consumer end clock for each
        time_t start_time1 = 0;         // producer start time for each
        time_t end_time1 = 0;           // producer end time for each
        time_t start_time2 = 0;         // consumer start time for each
        time_t end_time2 = 0;           // consumer end time for each
        
        if(id < nt/2){
            start_time1 = time(NULL);
            start_clock1 = clock();
            producer(&buffer, &end);
            end_clock1 = clock();
            end_time1 = time(NULL);
        }
        #pragma omp critical
        {
            time2_1 += difftime(end_time1, start_time1);
            clock2_1 += end_clock1 - start_clock1;
        }
        if(id >= nt/2){
            start_time2 = time(NULL);
            start_clock2 = clock();
            consumer(&buffer, &end);
            end_clock2 = clock();
            end_time2 = time(NULL);
        }
        #pragma omp critical
        {
            time2_2 += difftime(end_time2, start_time2);
            clock2_2 += end_clock2 - start_clock2;
        }
    }
    
    printf ("###################running time###################\n");
    printf ("producer time(2): %fs, clock(2): %fs\n", time2_1, ((float)clock2_1)/CLOCKS_PER_SEC);
    printf ("consumer time(2): %fs, clock(2): %fs\n", time2_2, ((float)clock2_2)/CLOCKS_PER_SEC);
    
    return 0;
}
