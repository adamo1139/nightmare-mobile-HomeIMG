// FP32 CPU implementation. Matrices are row-major, one token per row.
// Flatten checkpoint tensors to [first dimension, remaining dimensions].
#pragma once
#include <Eigen/Dense>
#include "homeimg.h"
#include <cmath>
#include <map>
#include <vector>
#include <optional>
#include <exception>
#include <stdexcept>
#include <string>

namespace homeimg {
using Matrix = Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
using Weights = std::map<std::string, Matrix>;
struct AuroraConfig { int heads=20, depth=12, rope_ref=64, micro_cond=256; homeimg_projection project=nullptr; void* backend=nullptr; homeimg_attention attention=nullptr; homeimg_mlp mlp=nullptr; homeimg_cross cross=nullptr; homeimg_masked_attention masked=nullptr; homeimg_spatial spatial=nullptr; homeimg_rowop rowop=nullptr; homeimg_expert expert=nullptr; homeimg_weight_scope weightScope=nullptr; };

inline Matrix silu(const Matrix& x) {
    return x.array() / (1.f + (-x.array()).exp());
}
inline Matrix silu(Matrix&& x) {
    x.array() /= (1.f + (-x.array()).exp());
    return std::move(x);
}
inline Matrix norm(const Matrix& x, const Matrix& weight) {
    Eigen::ArrayXf invRms=(x.array().square().rowwise().mean()+1e-6f).rsqrt();
    Matrix y = x.array().colwise() * invRms;
    return y.array().rowwise() * weight.col(0).transpose().array();
}
template<class WeightStore>
inline Matrix linear(const Matrix& x,const WeightStore& w,const std::string& name,bool bias=false) {
    Matrix y=x*w.at(name+".weight").transpose();
    if(bias) y.rowwise()+=w.at(name+".bias").col(0).transpose();
    return y;
}
inline Matrix sinusoid(float value,int denominator) {
    Matrix y(1,256);
    for(int i=0;i<128;++i) {
        float a=value*std::exp(-std::log(10000.f)*i/denominator);
        y(i)=std::sin(a); y(i+128)=std::cos(a);
    }
    return y;
}
struct SpatialRope {
    Matrix cosine, sine;
    SpatialRope(int side, int ref, int width):cosine(side*side,width/2),sine(side*side,width/2) {
        for(int t=0;t<side*side;++t) for(int pair=0;pair<width/2;++pair) {
            float pos=pair<width/4?t/side:t%side;
            if(ref) pos=(pos+.5f)*ref/side-.5f;
            float angle=pos/std::pow(10000.f,float(pair%(width/4))/(width/4));
            cosine(t,pair)=std::cos(angle);
            sine(t,pair)=std::sin(angle);
        }
    }
    void apply(Matrix& x) const {
        for(int t=0;t<x.rows();++t) for(int j=0;j<x.cols();j+=2) {
            float c=cosine(t,j/2),s=sine(t,j/2),a=x(t,j),b=x(t,j+1);
            x(t,j)=a*c-b*s; x(t,j+1)=a*s+b*c;
        }
    }
};
inline Matrix attention(const Matrix& q,const Matrix& k,const Matrix& v,float scale) {
    // Tile query rows; every softmax still spans all keys.
    Matrix out(q.rows(),v.cols());
    for(int first=0;first<q.rows();first+=256) {
        int count=std::min<int>(256,q.rows()-first);
        Matrix scores=(q.middleRows(first,count)*k.transpose())*scale;
        scores.colwise()-=scores.rowwise().maxCoeff().eval();
        scores=scores.array().exp();
        scores.array().colwise()/=scores.rowwise().sum().eval().array();
        out.middleRows(first,count).noalias()=scores*v;
    }
    return out;
}
// The output slices of distinct heads do not overlap. For larger grids let Eigen
// parallelize each GEMM instead, to bound simultaneous attention scratch buffers.
template<class Function>
inline void attention_heads(int heads,int tokens,bool parallel,Function run) {
#ifdef _OPENMP
    if(parallel && heads>=4 && tokens>=256 && tokens<=1024){
        std::exception_ptr failure;
        #pragma omp parallel for schedule(static)
        for(int head=0;head<heads;++head){
            try{run(head);}
            catch(...){
                #pragma omp critical(homeimg_attention_error)
                { if(!failure)failure=std::current_exception(); }
            }
        }
        if(failure)std::rethrow_exception(failure);
        return;
    }
#endif
    for(int head=0;head<heads;++head)run(head);
}
// text contains valid conditioning tokens only (compact the padding mask before calling).
template<class WeightStore>
inline Matrix aurora_velocity(const WeightStore& w,const AuroraConfig& cfg,const Matrix& latent,
                              int side,float time,const Matrix& text,float self_scale=1.f,
                              std::vector<Matrix>* textCache=nullptr,const Matrix* negativeText=nullptr,
                              std::vector<Matrix>* negativeCache=nullptr) {
    if(side<=0 || latent.rows()!=static_cast<long long>(side)*side || !text.rows())
        throw std::invalid_argument("invalid Aurora input");
    auto biasAdd=[&](auto&& x,const float* bias){
        homeimg_rows op{HOMEIMG_BIAS,int(x.rows()),int(x.cols()),x.data(),nullptr,nullptr,nullptr,
                        nullptr,bias,0,1,x.data()};
        if(!cfg.rowop || !cfg.rowop(cfg.backend,&op))x.rowwise()+=Eigen::Map<const Eigen::RowVectorXf>(bias,x.cols());
    };
    auto activate=[&](Matrix x){
        homeimg_rows op{HOMEIMG_SILU,int(x.rows()),int(x.cols()),x.data(),nullptr,nullptr,nullptr,
                        nullptr,nullptr,0,1,x.data()};
        if(cfg.rowop && cfg.rowop(cfg.backend,&op))return x;
        return silu(std::move(x));
    };
    auto project=[&](const Matrix& input,const std::string& key,bool bias=false)->Matrix {
        if(cfg.project && (input.rows()>=16 || cfg.mlp)) {
            const auto& weight=w.at(key+".weight");
            Matrix result(input.rows(),weight.rows());
            if(cfg.project(cfg.backend,input.data(),weight.data(),input.rows(),input.cols(),weight.rows(),result.data())) {
                if(bias)biasAdd(result,w.at(key+".bias").data());
                return result;
            }
        }
        return linear(input,w,key,bias);
    };
    auto normalize=[&](const Eigen::Ref<const Matrix>& input,const std::string& key,
                       const float* gain=nullptr,const float* bias=nullptr)->Matrix {
        const auto& weight=w.at(key);
        Matrix y(input.rows(),input.cols());
        homeimg_rows op{HOMEIMG_RMS,int(y.rows()),int(y.cols()),input.data(),nullptr,nullptr,
                        weight.data(),gain,bias,0,1,y.data()};
        if(cfg.rowop && cfg.rowop(cfg.backend,&op))return y;
        y=norm(input,weight);
        if(gain)y.array().rowwise()*=(1+Eigen::Map<const Eigen::RowVectorXf>(gain,y.cols()).array());
        if(bias)y.rowwise()+=Eigen::Map<const Eigen::RowVectorXf>(bias,y.cols());
        return y;
    };
    auto residual=[&](auto&& dst,const Matrix& delta,const float* gain){
        homeimg_rows op{HOMEIMG_RESIDUAL,int(dst.rows()),int(dst.cols()),dst.data(),delta.data(),
                        nullptr,nullptr,gain,nullptr,0,1,dst.data()};
        if(cfg.rowop && cfg.rowop(cfg.backend,&op))return;
        if(gain)dst.array()+=delta.array().rowwise()*Eigen::Map<const Eigen::RowVectorXf>(gain,dst.cols()).array();
        else dst+=delta;
    };
    Matrix x=project(latent,"patch_in");
    int d=x.cols();
    if(cfg.heads<=0 || d%cfg.heads || (d/cfg.heads)%4) throw std::invalid_argument("invalid heads");
    int hd=d/cfg.heads;
    std::optional<SpatialRope> rope;
    auto mlp=[&](const Matrix& input,const std::string& prefix){
        return project(activate(project(input,prefix+".0",true)),prefix+".2",true);
    };
    Matrix emb=mlp(sinusoid(time*1000,127),"time_mlp");
    if(cfg.micro_cond) {
        int res=16*side,o=std::max(res,cfg.micro_cond),values[]={o,o,0,0,o,o,res,res};
        Matrix cond(1,2048);
        for(int i=0;i<8;++i) cond.middleCols(256*i,256)=sinusoid(values[i],128);
        emb+=mlp(cond,"cond_mlp");
    }
    Matrix ada=project(activate(std::move(emb)),"ada",true);
    for(int i=0;i<cfg.depth;++i) {
        std::string p="blocks."+std::to_string(i)+".";
        Matrix mod=w.at(p+"mod_table")+Eigen::Map<const Matrix>(ada.data(),6,d);
        auto conditioned=[&](const std::string& name,int offset)->Matrix {
            return normalize(x,p+name+".weight",mod.row(offset+1).data(),mod.row(offset).data());
        };
        {
            Matrix joined(x.rows(),d);
            Matrix qkv=project(conditioned("norm1",0),p+"attn.qkv");
            // Every image has its own softmax and rotary positions even when the
            // surrounding dense layers process both CFG branches as one matrix.
            int tokens=latent.rows();
            for(int row=0;row<x.rows();row+=tokens){
                if(cfg.spatial && hd==128 &&
                   cfg.spatial(cfg.backend,qkv.row(row).data(),side,cfg.heads,cfg.rope_ref,self_scale/std::sqrt(float(hd)),
                               w.at(p+"attn.q_norm.weight").data(),w.at(p+"attn.k_norm.weight").data(),joined.row(row).data()))
                    continue;
                if(!rope)rope.emplace(side,cfg.rope_ref,hd);
                attention_heads(cfg.heads,tokens,!cfg.attention,[&](int head){
                    int j=head*hd;
                    Matrix q=norm(qkv.block(row,j,tokens,hd),w.at(p+"attn.q_norm.weight"));
                    Matrix k=norm(qkv.block(row,d+j,tokens,hd),w.at(p+"attn.k_norm.weight"));
                    rope->apply(q);rope->apply(k);
                    Matrix v=qkv.block(row,2*d+j,tokens,hd);
                    float scale=self_scale/std::sqrt(float(hd));
                    Matrix result;
                    // Large-grid heads execute sequentially here; the Vulkan queue is shared.
                    if(cfg.attention){
                        result.resize(tokens,hd);
                        if(!cfg.attention(cfg.backend,q.data(),k.data(),v.data(),tokens,hd,scale,result.data()))
                            result=attention(q,k,v,scale);
                    }else result=attention(q,k,v,scale);
                    joined.block(row,j,tokens,hd)=result;
                });
            }
            residual(x,project(joined,p+"attn.out"),mod.row(2).data());
        }
        // Everything before the first cross-attention is identical for +/- CFG.
        int tokens=latent.rows();
        if(negativeText && i==0){
            x.conservativeResize(2*tokens,d);
            x.bottomRows(tokens)=x.topRows(tokens);
        }
        for(int row=0;row<x.rows();row+=tokens){
            const Matrix& prompt=row==0 ? text : *negativeText;
            auto* cache=row==0 ? textCache : negativeCache;
            Matrix local;
            Matrix& context=cache ? cache->at(i) : local;
            if(context.size()==0){
                Matrix kv=project(prompt,p+"cross.kv");
                if(prompt.rows()==1)context=project(kv.rightCols(d),p+"cross.out");
                else context=std::move(kv);
            }
            auto branch=x.middleRows(row,tokens);
            if(prompt.rows()==1){
                biasAdd(branch,context.data());
            }else{
                Matrix q=project(normalize(branch,p+"norm2.weight"),p+"cross.q");
                Matrix joined(tokens,d);
                attention_heads(cfg.heads,tokens,!cfg.cross,[&](int head){
                    int j=head*hd;
                    Matrix query=q.middleCols(j,hd),key=context.middleCols(j,hd),value=context.middleCols(d+j,hd);
                    float scale=1.f/std::sqrt(float(hd));
                    Matrix attended(tokens,hd);
                    if(!cfg.cross || !cfg.cross(cfg.backend,query.data(),key.data(),value.data(),
                            tokens,context.rows(),hd,scale,attended.data()))
                        attended=attention(query,key,value,scale);
                    joined.middleCols(j,hd)=attended;
                });
                residual(branch,project(joined,p+"cross.out"),nullptr);
            }
        }
        if(cfg.mlp){
            const auto& gate=w.at(p+"mlp.in_proj.weight");
            const auto& down=w.at(p+"mlp.out_proj.weight");
            if(gate.cols()==d && gate.rows()==2*down.cols() && down.rows()==d &&
               cfg.mlp(cfg.backend,x.data(),x.rows(),d,down.cols(),gate.data(),down.data(),
                       w.at(p+"norm3.weight").data(),mod.row(4).data(),mod.row(3).data(),mod.row(5).data()))
                continue;
        }
        Matrix gu=project(conditioned("norm3",3),p+"mlp.in_proj");
        int h=gu.cols()/2;
        Matrix hidden=(gu.leftCols(h).array() / (1.f+(-gu.leftCols(h).array()).exp())) * gu.rightCols(h).array();
        gu.resize(0,0);
        x.array()+=project(hidden,p+"mlp.out_proj").array().rowwise()*mod.row(5).array();
    }
    return project(normalize(x,"norm.weight"),"patch_out");
}
}
