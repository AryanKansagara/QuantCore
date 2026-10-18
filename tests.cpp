#include "quant_core.hpp"
#include <iostream>
#include <random>
using namespace quantcore;
static int assertions=0;
void check(bool pass,const char* name) {
    ++assertions;if(!pass) throw std::runtime_error(name);
}
template<class F> void rejects(F f) {
    bool rejected=false;
    try {f();} catch(const std::invalid_argument&) {rejected=true;}
    check(rejected,"invalid input was accepted");
}
int main() {
    Int8Linear zero(2,3,std::vector<float>(6,0));
    check(zero.forward({1,2,3})==std::vector<float>({0,0}),"zero rows");
    Int8Linear exact(2,2,{127,-127,1,-1});
    auto y=exact.forward({2,1});
    check(std::abs(y[0]-127)<0.0001f && std::abs(y[1]-1)<0.0001f,"per-channel scales");
    check(exact.storage_bytes()==12,"weight and scale accounting");
    rejects([]{Int8Linear bad(0,3,{});});
    rejects([]{Int8Linear bad(2,3,{1});});
    rejects([]{Int8Linear bad(1,1,{std::numeric_limits<float>::infinity()});});
    rejects([&]{exact.forward({1});});
    rejects([&]{exact.forward({1,std::numeric_limits<float>::quiet_NaN()});});
    rejects([]{compare({1},{1,2});});
    std::mt19937 rng(7);std::normal_distribution<float> dist(0,1);
    std::vector<float> weights(32*64),input(64);
    for(auto& w:weights) w=dist(rng);
    Int8Linear model(32,64,weights);
    for(int trial=0;trial<16;++trial) {
        for(auto& x:input)x=dist(rng);
        auto error=compare(fp32_matvec(32,64,weights,input),model.forward(input));
        check(error.relative_l2<0.025,"quantization relative error");
        check(error.cosine>0.999,"output cosine");
    }
    for(std::size_t row=0;row<32;++row) {
        float max_abs=0;
        for(std::size_t column=0;column<64;++column)max_abs=std::max(max_abs,std::abs(weights[row*64+column]));
        for(std::size_t column=0;column<64;++column)
            check(std::abs(weights[row*64+column]-model.reconstructed(row,column))<=max_abs/254+1e-6f,"rounding error bound");
    }
    check(compare({0,0},{0,0}).relative_l2==0,"zero-reference error");
    std::cout<<"PASS: "<<assertions<<" assertions\n";
}
