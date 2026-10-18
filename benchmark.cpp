#include "quant_core.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
using namespace quantcore;
volatile float checksum_sink=0;
template<class F> double median_ms(F operation) {
    operation(); // Warm-up, excluded from timing.
    std::vector<double> samples;
    for(int i=0;i<7;++i) {
        auto start=std::chrono::steady_clock::now();
        auto output=operation();
        auto end=std::chrono::steady_clock::now();
        checksum_sink=output.front(); // Keep observable work outside the timed interval.
        samples.push_back(std::chrono::duration<double,std::milli>(end-start).count());
    }
    std::sort(samples.begin(),samples.end());return samples[3];
}
int main() {
    constexpr std::size_t rows=4096,columns=4096;
    std::mt19937 rng(42);
    std::normal_distribution<float> weight_distribution(0,0.02f),input_distribution(0,1);
    std::vector<float> weights(rows*columns),input(columns);
    for(auto& w:weights) w=weight_distribution(rng);
    for(auto& x:input) x=input_distribution(rng);
    Int8Linear model(rows,columns,weights);
    // Evaluate multiple held-out synthetic vectors; not real model task accuracy.
    double worst_relative_l2=0,mean_mse=0,min_cosine=1;
    for(int sample=0;sample<8;++sample) {
        for(auto& x:input) x=input_distribution(rng);
        const auto error=compare(fp32_matvec(rows,columns,weights,input),model.forward(input));
        worst_relative_l2=std::max(worst_relative_l2,error.relative_l2);
        mean_mse+=error.mse/8;min_cosine=std::min(min_cosine,error.cosine);
    }
    const double reference_ms=median_ms([&]{return fp32_matvec(rows,columns,weights,input);});
    const double quantized_ms=median_ms([&]{return model.forward(input);});
    std::cout<<std::setprecision(10)
        <<"Synthetic projection: "<<rows<<" x "<<columns<<" = "<<weights.size()<<" parameters\n"
        <<"FP32 weight bytes: "<<weights.size()*sizeof(float)<<"\n"
        <<"INT8 weight + scale bytes: "<<model.storage_bytes()<<"\n"
        <<"Weight storage reduction: "<<100*(1-static_cast<double>(model.storage_bytes())/(weights.size()*sizeof(float)))<<"%\n"
        <<"Worst relative L2 error (8 vectors): "<<worst_relative_l2<<"\n"
        <<"Mean output MSE: "<<mean_mse<<"\nMinimum output cosine: "<<min_cosine<<"\n"
        <<"FP32 median ms: "<<reference_ms<<"\nINT8 weight-only median ms: "<<quantized_ms<<"\n";
    return worst_relative_l2<0.02 ? 0 : 1;
}
