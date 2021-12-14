#include <chrono>

class TictocTimer {
private:
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
public:

    TictocTimer(){
        start=std::chrono::high_resolution_clock::now();
        end=std::chrono::high_resolution_clock::now();
    }

    void tic(){
        start = std::chrono::high_resolution_clock::now();
    }
    
    void toc(){
        end = std::chrono::high_resolution_clock::now();
    }

    unsigned long get_time_us() {
        return (std::chrono::duration_cast<std::chrono::microseconds>(end-start)).count();
    }

};