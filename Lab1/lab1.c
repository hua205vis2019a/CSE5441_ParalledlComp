/*
 NAME: Zewen Hua
 CLASS&SECTION: CSE5441 TuTh 12:45-14:05
 ASSIGNMENT NUMBER: Lab1
 COMPILER: Intel
 COMPILATION INSTRUCTION: icc transform.o lab1.c -o lab1.out
 SUBMIT DATE: Sept.26, 2018
 DESCRIPTION: This is a solution to producer-consumer problem. The size of work queue is 5. It has two parts: producer and consumer. After producer fills the queue, consumer gets it empty. Resume the producer when read the X.
*/
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <time.h>

#define QUEUESIZE 5

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
    int positon;
    char cmd;
    uint16_t encoded_key;
    uint16_t decoded_key;
}COUTPUT;

static POUTPUT each_input;          /* struct in producer for each line */
WORK ori_input;                     /* save the producer output for each line temporarily */
POUTPUT each;
POUTPUT val_input[QUEUESIZE];       /* work queue */
COUTPUT final_output[QUEUESIZE];    /* final output of queue */
POUTPUT *val_output;
int num = 0;                        /* number of output */

/* producer for each line, validate values, return struct after producer transformation */
POUTPUT producer_each(){
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

/* producer for full the work queue */
POUTPUT *producer(){
    for(int i = 0;i < QUEUESIZE;i++){
        each = producer_each();
        val_input[i].position = i;
        val_input[i].cmd = each.cmd;
        val_input[i].encoded_key = each.encoded_key;
        if(each.cmd == 'X'){
            break;
        }
    }
    return val_input;
}

/* consumer for empty the queue, print final output struct after transformation */
void consumer(POUTPUT *pro_input){
    for(int j = 0;j < QUEUESIZE;j++){
        if(pro_input[j].cmd == 'X'){
            break;
        }
        final_output[j].cmd = pro_input[j].cmd;
        final_output[j].positon = pro_input[j].position;
        final_output[j].encoded_key = pro_input[j].encoded_key;
        switch(pro_input[j].cmd){
            case 'A':
                final_output[j].decoded_key = transformA(pro_input[j].encoded_key);
                break;
            case 'B':
                final_output[j].decoded_key = transformB(pro_input[j].encoded_key);
                break;
            case 'C':
                final_output[j].decoded_key = transformC(pro_input[j].encoded_key);
                break;
            case 'D':
                final_output[j].decoded_key = transformD(pro_input[j].encoded_key);
                break;
        }
        num++;
        printf("Q:%d\t", final_output[j].positon);
        printf("%c\t",final_output[j].cmd);
        printf("%" PRIu16, final_output[j].encoded_key);
        printf("\t");
        printf("%" PRIu16, final_output[j].decoded_key);
        printf("\n");
    }
}

int main()
{
    clock_t start_clock1 = 0;       // producer start clock for each
    clock_t end_clock1 = 0;         // producer end clock for each
    clock_t start_clock2 = 0;       // consumer start clock for each
    clock_t end_clock2 = 0;         // consumer end clock for each
    time_t start_time1 = 0;         // producer start time for each
    time_t end_time1 = 0;           // producer end time for each
    time_t start_time2 = 0;         // consumer start time for each
    time_t end_time2 = 0;           // consumer end time for each
    static clock_t clock2_1;        // producer clock(2)
    static clock_t clock2_2;        // consumer clock(2)
    static double time2_1;          // producer time(2)
    static double time2_2;          // consumer time(2)
    
    /* producer and consumer, time(2)&clock(2) */
    for(int i = 0;i < (int)(num/QUEUESIZE) + 1;i++){
        start_time1 = time(NULL);
        start_clock1 = clock();
        val_output = producer();
        end_clock1 = clock();
        end_time1 = time(NULL);
        
        time2_1 += difftime(end_time1, start_time1);
        clock2_1 += end_clock1 - start_clock1;
        
        start_time2 = time(NULL);
        start_clock2 = clock();
        consumer(val_output);
        end_clock2 = clock();
        end_time2 = time(NULL);
        
        time2_2 += difftime(end_time2, start_time2);
        clock2_2 += end_clock2 - start_clock2;
    }
    printf ("###################running time###################\n");
    printf ("producer time(2): %fs, clock(2): %fs\n", time2_1, ((float)clock2_1)/CLOCKS_PER_SEC);
    printf ("consumer time(2): %fs, clock(2): %fs\n", time2_2, ((float)clock2_2)/CLOCKS_PER_SEC);

    return 0;
}
