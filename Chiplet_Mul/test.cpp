#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <thread>

#include "m5ops.h"

/*
��ʾ��ʾ�� 2 �� chiplet �� 10*10 �ľ���˹�����������Ϊ C���� C �Ĵ�СΪ
10*10����һά���󴢴棬�� chiplet0 �������� C ������ 0 �� 500*249 �Ľ������
chiplet1 ����ʣ�µĽ����
*/
extern "C" 
{
    const int N = 10;
    // ��main����д�ɶ��̵߳���ʽ
    void  thread_main(int chipletNumber,long long *martrix,long long *martrix2,long long *martrix3) {
        // ʾ���� totalChipletNumber Ϊ 2���ʲ���ʽ��д������
        if (chipletNumber == 0) {
            for (int i = 5; i < 10; i++) {
                for (int j = 0; j < 10; j++) {
                    for (int k = 0; k < 10; k++)
                        martrix3[i * 10 + j] = martrix3[i * 10 + j] + martrix[i * 10 + k] * martrix2[k * 10 + j];
                }
            }
            // m5_gadia_call(chipletNumber, 1, -2, 0);
            int position = 0;
            std::cout << " coming 0" << "coming while" << std::endl; 
            while (true) {
                int result = (int)m5_gadia_receive(chipletNumber);
                // ��� chiplet1 �Ƿ�����˾���˵Ĺ���
                if(result == -1)
                    continue;
                else if (result == -2)  //����ȴ���Chiplet�Ѿ���ɶ�д 
                    break;
                else {
                    martrix3[position] = result;
                    position++;
                }
            }
            // m5_gadia_call(chipletNumber, chipletNumber, 0, 0); // ��¼���� cycle
            return;
        }// the following is responsible for collect
        else if (chipletNumber == 1) {
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 10; j++) {
                    for (int k = 0; k < 10; k++)
                        martrix3[i * 10 + j] = martrix3[i * 10 + j] + martrix[i * 10 + k] * martrix2[k * 10 + j];
                    // chiplet 1 �ѽ��д�뵽�������С�
                    m5_gadia_call(chipletNumber, 0, (int)martrix3[i * 10 + j], 0);
                }
            }
            std::cout << "coming 1" << std::endl;
            m5_gadia_call(chipletNumber, 0, -2, 0);
            //
        }
    }

    int main() {
        // �����ʼ����ʼ
        long long *martrix = new long long[N * N];
        long long *martrix2 = new long long[N * N];
        long long *martrix3 = new long long[N * N];
        for (int i = 0; i < N * N; i++) {
            martrix[i] = rand() % 100;
            martrix2[i] = rand() % 100;
            martrix3[i] = 0;
        }
        // �����߳�
        std::thread t1(thread_main, 0, martrix, martrix2, martrix3);
        std::thread t2(thread_main, 1, martrix, martrix2, martrix3);
        t1.join();
        t2.join();
        // �����ά����
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++)
                std::cout << martrix3[i * N + j] << " ";
            std::cout << std::endl;
        }
        // ������Ӧ����Ӵ����ͷ�martrix, martrix2, martrix3��ָ����ڴ�
        delete[] martrix;
        delete[] martrix2;
        delete[] martrix3;
        return 0;
    }

}
