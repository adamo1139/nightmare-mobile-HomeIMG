#pragma once
#include <fstream>
#include "text_encoder.hpp"
#include "transformer_weights.hpp"
namespace homeimg {
template<class StepCallback>
bool sample(const std::string& root,const std::string& prompt,const std::string& negative_prompt,
            int side,int steps,float cfg,Matrix& z,StepCallback after_step,const AuroraConfig& config={},float override_shift=0.f) {
        std::ifstream settings(root+"/transformer/config.json");
        auto transformer=nlohmann::json::parse(settings);
        double ref=transformer.value("time_shift_ref",0.0);
        double shift=override_shift;
        if(shift==0) shift=ref ? 16.0*side/ref : transformer.value("time_shift",1.0);
        if(!std::isfinite(shift) || shift<=0)throw std::invalid_argument("invalid transformer time shift");
        homeimg::Matrix positive,negative;
        {
            homeimg::Tokenizer tokenizer(root+"/tokenizer/tokenizer.json");
            homeimg::TextEncoder encoder(root,config.masked?config.project:nullptr,config.masked,config.backend,config.expert,config.rowop);
            positive=encoder.refine(encoder.trunk(tokenizer.encode(prompt)));
            if(cfg!=1.f)negative=encoder.refine(encoder.trunk(tokenizer.encode(negative_prompt)));
        }
        auto weights=homeimg::load_transformer(root);
        struct WeightScope {
            const AuroraConfig& config;
            ~WeightScope(){if(config.weightScope)config.weightScope(config.backend,nullptr,0);}
        } scope{config};
        if(config.weightScope && steps>1)
            config.weightScope(config.backend,weights.mapping().base(),weights.mapping().bytes());
        auto time=[&](int i){
            // torch.linspace anchors the second half at zero.
            if(i==0)return 1.f;
            if(i==steps)return 0.f;
            float step=1.f/steps;
            float u=i<(steps+1)/2?std::fma(-step,float(i),1.f):step*(steps-i);
            volatile float offset=float(shift-1.0)*u;
            return float(shift)*u/(1.f+offset);
        };
        // Prompt keys/values are invariant across Euler steps and differ between CFG branches.
        std::vector<Matrix> positiveCache(steps>1 ? config.depth : 0);
        std::vector<Matrix> negativeCache(steps>1 && cfg!=1.f ? config.depth : 0);
        for(int i=0;i<steps;++i){
            float t=time(i);
            auto* posCache=positiveCache.empty()?nullptr:&positiveCache;
            auto* negCache=negativeCache.empty()?nullptr:&negativeCache;
            auto update=[&](const auto& pos,const float* neg){
                float dt=time(i+1)-t;
                homeimg_rows op{HOMEIMG_EULER,int(z.rows()),int(z.cols()),z.data(),pos.data(),neg,nullptr,
                                nullptr,nullptr,dt,cfg,z.data()};
                if(config.rowop && config.rowop(config.backend,&op))return;
                if(!neg)z+=dt*pos;
                else{
                    Eigen::Map<const Matrix> uncond(neg,z.rows(),z.cols());
                    z+=dt*(uncond+cfg*(pos-uncond));
                }
            };
            if(side==64 && cfg!=1.f){
                Matrix both=aurora_velocity(weights,config,z,side,t,positive,1.f,posCache,&negative,negCache);
                auto uncond=both.bottomRows(z.rows());
                update(both.topRows(z.rows()),uncond.data());
            }else{
                Matrix v=aurora_velocity(weights,config,z,side,t,positive,1.f,posCache);
                if(cfg!=1.f){
                    Matrix uncond=aurora_velocity(weights,config,z,side,t,negative,1.f,negCache);
                    update(v,uncond.data());
                }else update(v,nullptr);
            }
            if(!z.allFinite())throw std::runtime_error("non-finite sampled latent");
            if(!after_step(i+1,z))return false;
        }
    return true;
}
}
