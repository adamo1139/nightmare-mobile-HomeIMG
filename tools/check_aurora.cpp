// Host parity driver. stdin contains named row-major FP32 matrices.
#include "../backend/homeimg/aurora.hpp"
#include <iostream>
int main() {
    homeimg::Weights weights;
    std::string key;
    int rows, cols;
    while (std::cin >> key >> rows >> cols) {
        std::cin.get();
        auto& m = weights[key];
        m.resize(rows, cols);
        if (!std::cin.read(reinterpret_cast<char*>(m.data()), m.size()*sizeof(float))) return 1;
    }
    // Batched CFG must agree with independent trajectories, including prompt-specific KV caches.
    const auto& z=weights.at("latent");
    const auto& text=weights.at("text");
    for(int length : {1,2}){
        homeimg::Matrix negative=-text.topRows(length);
        std::vector<homeimg::Matrix> posCache(2),negCache(2);
        for(float t : {1.f,.37f}){
            auto both=homeimg::aurora_velocity(weights,{2,2,64,256},z,3,t,text,1.f,&posCache,&negative,&negCache);
            auto pos=homeimg::aurora_velocity(weights,{2,2,64,256},z,3,t,text);
            auto neg=homeimg::aurora_velocity(weights,{2,2,64,256},z,3,t,negative);
            if(!pos.isApprox(both.topRows(z.rows()),2e-5f) || !neg.isApprox(both.bottomRows(z.rows()),2e-5f))
                throw std::runtime_error("batched CFG parity failure");
        }
    }
    for(float t : {0.f,.37f,1.f}) {
        auto v = homeimg::aurora_velocity(weights,{2,2,64,256},weights.at("latent"),3,t,weights.at("text"));
        std::cout.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(float));
    }
}
