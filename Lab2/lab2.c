/*
 NAME: Zewen Hua
 CLASS&SECTION: CSE5441 TuTh 12:45-14:05
 ASSIGNMENT NUMBER: Lab2
 COMPILER: Intel
 COMPILATION INSTRUCTION: icc transform.o lab2.c -o lab2.out -lpthread
 SUBMIT DATE: Nov.5, 2018
 DESCRIPTION: This is a solution to producer-consumer problem using pthread. The size of work buffer is 5. It has two parts: producer and consumer.
*/

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <time.h>
#include <pthread.h>

#define BUFF_SIZE 5

/* provided functions in transform.o*/
extern uint16_t transformA(uint16_t input_val);
extern uint16_t transformB(uint16_t input_val);
extern uint16_t transformC(uint16_t input_val);
extern uint16_t transformD(uint16_t input_val);

/* struct for source input*/
typedef struct work_entry{
    char cmd;
    uint16_t key;
}WORK;

/* struct after producer*/
typedef struct producer_output{
    int position;
    char cmd;
    uint16_t encoded_key;
}POUTPUT;

/* final struct for output after consumer*/
typedef struct consumer_output{
    int position;
    char cmd;
    uint16_t encoded_key;
    uint16_t decoded_key;
}COUTPUT;

/* struct for pthread API */
typedef struct{
    POUTPUT val_input[BUFF_SIZE]; // the buffer
    COUTPUT final_output[BUFF_SIZE]; // final output of the buffer
    POUTPUT each;
    int len; // number of items in the buffer
    int num1;
    int num2;
    int end1;
    int end2;
    pthread_mutex_t mutex; //needed to add/remove data from the buffer
    pthread_cond_t can_produce; // signaled when items are removed
    pthread_cond_t can_consume; // signaled when items are added
}buffer_t;

/* producer for each line, validate values, return struct after producer transformation */
POUTPUT producer_each(){
    static POUTPUT each_input;
    static WORK ori_input;
    scanf("%s %" PRIu16, &ori_input.cmd, &ori_input.key);
    if(ori_input.cmd<'A'||(ori_input.cmd>'D'&&ori_input.cmd!='X')||ori_input.key<0||ori_input.key>1000){    /* invalid input */
        producer_each();
    }
    else{
        each_input.cmd = ori_input.cmd;
        switch(ori_input.cmd){
            case 'A':
                each_input.encoded_key = transformA(ori_input.key);
                break;
            case 'B':
                each_input.encoded_key = transformB(ori_input.key);
                break;
            case 'C':
                each_input.encoded_key = transformC(ori_input.key);
                break;
            case 'D':
                each_input.encoded_key = transformD(ori_input.key);
                break;
            case 'X':
                each_input.encoded_key = 0;
                break;
        }
    }
    return each_input;
}

void *producer(void *arg){
    buffer_t *buffer = (buffer_t*)arg;
    while(!buffer->end1){
        
        if(buffer->len == BUFF_SIZE){
            pthread_cond_wait(&buffer->can_produce, &buffer->mutex);
        }
        buffer->each = producer_each();
        buffer->val_input[buffer->num1].position = buffer->num1;
        buffer->val_input[buffer->num1].cmd = buffer->each.cmd;
        buffer->val_input[buffer->num1].encoded_key = buffer->each.encoded_key;
        pthread_mutex_lock(&buffer->mutex);
        ++buffer->len;
        if(buffer->each.cmd == 'X'){
            buffer->end1 = 1;
        }
        if(buffer->num1 < 4)
            ++buffer->num1;
        else
            buffer->num1 = 0;
        pthread_cond_signal(&buffer->can_consume);
        pthread_mutex_unlock(&buffer->mutex);
    }
    return NULL;
}

void *consumer(void *arg){
    buffer_t *buffer = (buffer_t*)arg;
    while(!buffer->end2){
        pthread_mutex_lock(&buffer->mutex);
        if(buffer->len == 0){
            pthread_cond_wait(&buffer->can_consume, &buffer->mutex);
        }
        --buffer->len;
        int j = buffer->num2;
        if(buffer->val_input[j].cmd == 'X'){
            buffer->end2 = 1;
        }
        
        if(buffer->end2 == 0){
        buffer->final_output[j].cmd = buffer->val_input[j].cmd;
        buffer->final_output[j].position = buffer->val_input[j].position;
        buffer->final_output[j].encoded_key = buffer->val_input[j].encoded_key;
        switch(buffer->val_input[j].cmd){
            case 'A':
                buffer->final_output[j].decoded_key = transformA(buffer->val_input[j].encoded_key);
                break;
            case 'B':
                buffer->final_output[j].decoded_key = transformB(buffer->val_input[j].encoded_key);
                break;
            case 'C':
                buffer->final_output[j].decoded_key = transformC(buffer->val_input[j].encoded_key);
                break;
            case 'D':
                buffer->final_output[j].decoded_key = transformD(buffer->val_input[j].encoded_key);
                break;
        }
        
        printf("Q:%d\t", buffer->final_output[j].position);
        printf("%c\t",buffer->final_output[j].cmd);
        printf("%" PRIu16, buffer->final_output[j].encoded_key);
        printf("\t");
        printf("%" PRIu16, buffer->final_output[j].decoded_key);
        printf("\n");
        if(buffer->num2 < 4)
            ++buffer->num2;
        else
            buffer->num2 = 0;
        }
        pthread_cond_signal(&buffer->can_produce);
        pthread_mutex_unlock(&buffer->mutex);
    }
    return NULL;
}
int main(int argc, char *argv[]){
    /* initialize the variables of the buffer */
    buffer_t buffer = {
        .len = 0,
        .num1 = 0,
        .num2 = 0,
        .end1 = 0,
        .end2 = 0
    };
    clock_t start_clock = 0;
    clock_t end_clock = 0;
    time_t start_time = 0;
    time_t end_time = 0;
    
    static clock_t clock2;
    static double time2;
    
    pthread_t prod, cons;
    pthread_attr_t attr;
    
    pthread_mutex_init(&buffer.mutex, NULL);
    pthread_cond_init(&buffer.can_produce, NULL);
    pthread_cond_init(&buffer.can_consume, NULL);
    
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    
    /* time and clock */
    start_time = time(NULL);
    start_clock = clock();
    pthread_create(&prod, &attr, producer, (void*)&buffer);
    pthread_create(&cons, &attr, consumer, (void*)&buffer);
    
    pthread_join(prod, NULL);
    pthread_join(cons, NULL);
    end_clock = clock();
    end_time = time(NULL);
    
    time2 += difftime(end_time, start_time);
    clock2 += end_clock - start_clock;
    
    pthread_mutex_destroy(&buffer.mutex);
    pthread_cond_destroy(&buffer.can_produce);
    pthread_cond_destroy(&buffer.can_consume);
    pthread_attr_destroy(&attr);
    
    printf ("###################running time###################\n");
    printf ("time(2): %fs, clock(2): %fs\n", time2, ((float)clock2)/CLOCKS_PER_SEC);
    
    pthread_exit(NULL);
    return 0;
}
