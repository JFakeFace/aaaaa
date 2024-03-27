#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <thread>

#include "m5ops.h"

/*
本示例示范 2 个 chiplet 做 10*10 的矩阵乘工作，假设结果为 C，则 C 的大小为
10*10，用一维矩阵储存，则 chiplet0 计算矩阵乘 C 索引从 0 到 500*249 的结果，而
chiplet1 计算剩下的结果。
*/
extern "C" 
{
    const int N = 10;
    // 将main函数写成多线程的形式
    void  thread_main(int chipletNumber,long long *martrix,long long *martrix2,long long *martrix3) {
        // 示例的 totalChipletNumber 为 2，故不显式的写出来。
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
                // 检测 chiplet1 是否完成了矩阵乘的工作
                if(result == -1)
                    continue;
                else if (result == -2)  //代表等待的Chiplet已经完成读写 
                    break;
                else {
                    martrix3[position] = result;
                    position++;
                }
            }
            // m5_gadia_call(chipletNumber, chipletNumber, 0, 0); // 记录结束 cycle
            return;
        }// the following is responsible for collect
        else if (chipletNumber == 1) {
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 10; j++) {
                    for (int k = 0; k < 10; k++)
                        martrix3[i * 10 + j] = martrix3[i * 10 + j] + martrix[i * 10 + k] * martrix2[k * 10 + j];
                    // chiplet 1 把结果写入到共享储存中。
                    m5_gadia_call(chipletNumber, 0, (int)martrix3[i * 10 + j], 0);
                }
            }
            std::cout << "coming 1" << std::endl;
            m5_gadia_call(chipletNumber, 0, -2, 0);
            //
        }
    }

    int main() {
        // 程序初始化开始
        long long *martrix = new long long[N * N];
        long long *martrix2 = new long long[N * N];
        long long *martrix3 = new long long[N * N];
        for (int i = 0; i < N * N; i++) {
            martrix[i] = rand() % 100;
            martrix2[i] = rand() % 100;
            martrix3[i] = 0;
        }
        // 启动线程
        std::thread t1(thread_main, 0, martrix, martrix2, martrix3);
        std::thread t2(thread_main, 1, martrix, martrix2, martrix3);
        t1.join();
        t2.join();
        // 输出二维数组
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++)
                std::cout << martrix3[i * N + j] << " ";
            std::cout << std::endl;
        }
        // 在这里应该添加代码释放martrix, martrix2, martrix3所指向的内存
        delete[] martrix;
        delete[] martrix2;
        delete[] martrix3;
        return 0;
    }

}
