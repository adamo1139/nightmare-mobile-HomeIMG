#include "../backend/homeimg/transformer_weights.hpp"
#include <iostream>
// Check velocity at the exact saved inputs to isolate the denoiser from Euler drift.
static void check_velocity(const std::string& root,const std::string& fixture) {
    homeimg::MappedWeights f(fixture);
    const auto weights=homeimg::load_transformer(root);
    auto noise=f.at("noise");
    int side=noise.shape.at(2);
    const auto context=f.at("context");
    const auto mask=f.at("valid");
    const auto times=f.at("times");
    int length=context.shape.at(1);
    for(size_t i=0;i+1<times.shape.at(0);++i) {
        const auto input=i?f.at("euler_"+std::to_string(i-1)):noise;
        homeimg::Matrix z=Eigen::Map<const homeimg::Matrix>(
            reinterpret_cast<const float*>(input.data),128,side*side).transpose();
        auto target=f.at("velocity_pair_"+std::to_string(i));
        for(int branch=0;branch<2;++branch){
            int count=0;
            for(int j=0;j<length;++j)count+=mask.data[branch*length+j]!=0;
            homeimg::Matrix text=Eigen::Map<const homeimg::Matrix>(
                reinterpret_cast<const float*>(context.data)+branch*length*1280,count,1280);
            auto result=homeimg::aurora_velocity(weights,{},z,side,
                reinterpret_cast<const float*>(times.data)[i],text);
            homeimg::Matrix ref=Eigen::Map<const homeimg::Matrix>(
                reinterpret_cast<const float*>(target.data)+branch*z.size(),128,side*side).transpose();
            auto error=(result-ref).cwiseAbs().eval();
            std::cout<<"step "<<i<<" branch "<<branch<<" velocity max "<<error.maxCoeff()<<std::endl;
            if(!(error.array()<=1e-4f+1e-4f*ref.array().abs()).all())
                throw std::runtime_error("velocity outside reference tolerance");
        }
    }
}
int main(int argc,char** argv){
    if(argc!=3)return 2;
    try {check_velocity(argv[1],argv[2]);}
    catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
