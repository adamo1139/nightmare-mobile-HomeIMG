// Desktop worker: stdin/stdout are contiguous [side*side,128] FP32 latents.
#include "../backend/homeimg/sample.hpp"
#include <iostream>
#include "../backend/homeimg/transformer_weights.hpp"
#include "../backend/homeimg/vae.hpp"

int main(int argc,char** argv) {
    try {
        if(argc==4 && std::string(argv[1])=="decode"){
            int side=std::stoi(argv[3]);
            if(side<1 || side>64)throw std::invalid_argument("invalid latent side");
            homeimg::Matrix z(side*side,128);
            if(!std::cin.read(reinterpret_cast<char*>(z.data()),z.size()*4))
                throw std::runtime_error("incomplete VAE input");
            auto rgb=homeimg::VaeDecoder(argv[2]).decode(z,side);
            if(!rgb.allFinite())throw std::runtime_error("non-finite decoded image");
            std::cout.write(reinterpret_cast<const char*>(rgb.data()),rgb.size()*4);
            return std::cout?0:1;
        }
        if(argc!=7) throw std::invalid_argument("expected model, side, steps, guidance, prompt, negative");
        std::string root=argv[1];
        int side=std::stoi(argv[2]),steps=std::stoi(argv[3]);
        float cfg=std::stof(argv[4]);
        if(side<=0 || side>64 || steps<1 || !std::isfinite(cfg))throw std::invalid_argument("invalid sampler options");
        homeimg::Matrix z(side*side,128);
        if(!std::cin.read(reinterpret_cast<char*>(z.data()),z.size()*sizeof(float)))
            throw std::runtime_error("incomplete input latent");
        homeimg::sample(root,argv[5],argv[6],side,steps,cfg,z,[](int,const homeimg::Matrix& latent){
            std::cout.write(reinterpret_cast<const char*>(latent.data()),latent.size()*sizeof(float));
            return bool(std::cout);
        });
    } catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
    return std::cout?0:1;
}
