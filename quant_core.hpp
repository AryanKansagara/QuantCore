#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace quantcore {
static_assert(sizeof(float)==4, "Benchmark assumes FP32 storage");
inline void validate(std::size_t rows,std::size_t columns,const std::vector<float>& weights) {
    if(!rows || !columns || columns>std::numeric_limits<std::size_t>::max()/rows || weights.size()!=rows*columns)
        throw std::invalid_argument("Invalid matrix dimensions");
    for(float v:weights) if(!std::isfinite(v)) throw std::invalid_argument("Nonfinite weight");
}
inline void validate_input(std::size_t columns,const std::vector<float>& input) {
    if(input.size()!=columns) throw std::invalid_argument("Input dimension mismatch");
    for(float v:input) if(!std::isfinite(v)) throw std::invalid_argument("Nonfinite input");
}
inline std::vector<float> fp32_matvec(std::size_t rows,std::size_t columns,
                                     const std::vector<float>& weights,const std::vector<float>& input) {
    // Weight validation belongs to load/construction; the reference checks shape here.
    if(!rows || !columns || columns>std::numeric_limits<std::size_t>::max()/rows || weights.size()!=rows*columns)
        throw std::invalid_argument("Invalid reference dimensions");
    validate_input(columns,input);
    std::vector<float> output(rows,0);
    for(std::size_t row=0;row<rows;++row)
        for(std::size_t column=0;column<columns;++column)
            output[row]+=weights[row*columns+column]*input[column];
    return output;
}

class Int8Linear {
    std::size_t rows_,columns_;
    std::vector<std::int8_t> weights_;
    std::vector<float> scales_;
public:
    Int8Linear(std::size_t rows,std::size_t columns,const std::vector<float>& weights)
        : rows_(rows),columns_(columns) {
        validate(rows,columns,weights);
        weights_.resize(weights.size()); scales_.resize(rows);
        for(std::size_t row=0;row<rows;++row) {
            float max_abs=0;
            for(std::size_t column=0;column<columns;++column)
                max_abs=std::max(max_abs,std::abs(weights[row*columns+column]));
            scales_[row]=max_abs==0 ? 1 : std::max(max_abs/127,std::numeric_limits<float>::min());
            for(std::size_t column=0;column<columns;++column) {
                const auto index=row*columns+column;
                const auto rounded=std::lround(weights[index]/scales_[row]);
                weights_[index]=static_cast<std::int8_t>(std::clamp(rounded,-127L,127L));
            }
        }
    }
    std::vector<float> forward(const std::vector<float>& input) const {
        validate_input(columns_,input);
        std::vector<float> output(rows_,0);
        for(std::size_t row=0;row<rows_;++row) {
            float sum=0;
            for(std::size_t column=0;column<columns_;++column)
                sum+=static_cast<float>(weights_[row*columns_+column])*input[column];
            output[row]=sum*scales_[row];
        }
        return output;
    }
    std::size_t storage_bytes() const { return weights_.size()*sizeof(std::int8_t)+scales_.size()*sizeof(float); }
    float reconstructed(std::size_t row,std::size_t column) const {
        if(row>=rows_ || column>=columns_) throw std::out_of_range("Weight index");
        return static_cast<float>(weights_[row*columns_+column])*scales_[row];
    }
};

struct Error { double relative_l2,mse,cosine; };
inline Error compare(const std::vector<float>& reference,const std::vector<float>& actual) {
    if(reference.empty() || reference.size()!=actual.size()) throw std::invalid_argument("Comparison dimensions");
    double squared=0,ref_norm=0,out_norm=0,dot=0;
    for(std::size_t i=0;i<reference.size();++i) {
        const double a=reference[i],b=actual[i],difference=a-b;
        squared+=difference*difference;ref_norm+=a*a;out_norm+=b*b;dot+=a*b;
    }
    return {ref_norm>0?std::sqrt(squared/ref_norm):(squared==0?0:std::numeric_limits<double>::infinity()),
            squared/reference.size(), ref_norm>0 && out_norm>0 ? dot/std::sqrt(ref_norm*out_norm) : (squared==0?1:0)};
}
} // namespace quantcore
