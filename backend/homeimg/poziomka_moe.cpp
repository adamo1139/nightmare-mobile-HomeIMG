// Scalar CPU reference for one epoch49 token. BF16 arrays use native-endian bits.
// Dense matrices are [out,in], fused INT8 matrices are [expert,in,out].
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

namespace {
float unpack(uint16_t word) {
    uint32_t bits = uint32_t(word) << 16;
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}
uint16_t pack(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
}
float rounded(float value) { return unpack(pack(value)); }
float dot(const uint16_t* x, const uint16_t* weight, int d) {
    float sum = 0;
    for (int i = 0; i < d; ++i) sum += unpack(x[i]) * unpack(weight[i]);
    return sum;
}
float glu(float g, float u) {
    return rounded(rounded(g / (1.f + std::exp(-g))) * u);
}
void project(const std::vector<float>& x, const int8_t* weight,
             const uint16_t* scale, std::vector<float>& y) {
    std::fill(y.begin(), y.end(), 0.f);
    for (size_t i = 0; i < x.size(); ++i)
        for (size_t j = 0; j < y.size(); ++j)
            y[j] += x[i] * weight[i*y.size()+j];
    for (size_t j = 0; j < y.size(); ++j) y[j] *= unpack(scale[j]);
}
}

extern "C" void poziomka_moe_token(int d, int h, int shared_h, const uint16_t* x,
    const uint16_t* router, const float* bias, const int8_t* wgu, const uint16_t* sgu,
    const int8_t* wd, const uint16_t* sd, const uint16_t* shared_gate,
    const uint16_t* shared_up, const uint16_t* shared_down, int* selected, uint16_t* output) {
    std::array<float,128> scores, rank;
    std::array<float,8> group;
    for (int e = 0; e < 128; ++e) {
        scores[e] = 1.f / (1.f + std::exp(-dot(x, router + e*d, d)));
        rank[e] = scores[e] + bias[e];
    }
    for (int g = 0; g < 8; ++g) {
        std::array<float,16> members;
        std::copy_n(rank.begin()+16*g,16,members.begin());
        std::partial_sort(members.begin(),members.begin()+2,members.end(),std::greater<float>());
        group[g] = members[0]+members[1];
    }
    std::array<int,8> groups;
    std::iota(groups.begin(),groups.end(),0);
    std::stable_sort(groups.begin(),groups.end(),[&](int a,int b){return group[a]>group[b];});
    std::array<int,32> candidates;
    for (int i=0;i<32;++i) candidates[i]=16*groups[i/16]+i%16;
    std::stable_sort(candidates.begin(),candidates.end(),[&](int a,int b){return rank[a]>rank[b];});
    std::copy_n(candidates.begin(),8,selected);
    float total=1e-20f;
    for(int k=0;k<8;++k) total+=scores[selected[k]];
    std::array<int,8> order;
    std::copy_n(selected,8,order.begin());
    std::sort(order.begin(),order.end());
    std::vector<float> input(d), gu(2*h), hidden(h), y(d), sum(d), shared(shared_h);
    std::transform(x,x+d,input.begin(),unpack);
    for (int e : order) {
        project(input,wgu+size_t(e)*d*2*h,sgu+e*2*h,gu);
        for(int i=0;i<h;++i) hidden[i]=glu(rounded(gu[i]),rounded(gu[h+i]));
        project(hidden,wd+size_t(e)*h*d,sd+e*d,y);
        float weight=rounded(scores[e]/total*2.5f);
        for(int j=0;j<d;++j) sum[j]+=rounded(y[j]*weight);
    }
    for(int i=0;i<shared_h;++i)
        shared[i]=glu(rounded(dot(x,shared_gate+i*d,d)),rounded(dot(x,shared_up+i*d,d)));
    for(int j=0;j<d;++j) {
        float v=0;
        for(int i=0;i<shared_h;++i) v+=shared[i]*unpack(shared_down[j*shared_h+i]);
        output[j]=pack(rounded(sum[j])+rounded(v));
    }
}
