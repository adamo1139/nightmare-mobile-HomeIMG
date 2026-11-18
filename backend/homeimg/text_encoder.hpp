// Single-prompt FP32 CPU path for the validated epoch49 architecture.
#pragma once
#include "aurora.hpp"
#include <numeric>
#include <array>
#include "safetensors.hpp"
#include "tokenizer.hpp"
namespace homeimg {
// Optional intermediate values used to localize numerical regressions.
struct TextTrace {
    Matrix first_attention, first_dense, first_moe, first_experts;
};

class TextEncoder {
    MappedWeights weights;
    homeimg_projection projection;
    homeimg_masked_attention attentionBackend;
    void* backend;
    homeimg_expert expertBackend;
    homeimg_rowop rowop;
    static Matrix read(const TensorView& view,size_t offset,int rows,int cols) {
        size_t count=size_t(rows)*cols;
        if(view.dtype=="F32" && offset+count<=view.bytes/4)
            return Eigen::Map<const Matrix>(reinterpret_cast<const float*>(view.data)+offset,rows,cols);
        if(view.dtype=="BF16" && offset+count<=view.bytes/2) {
            using BF=Eigen::Matrix<Eigen::bfloat16,-1,-1,Eigen::RowMajor>;
            return Eigen::Map<const BF>(reinterpret_cast<const Eigen::bfloat16*>(view.data)+offset,rows,cols).cast<float>();
        }
        throw std::runtime_error("invalid floating weight");
    }
    Matrix weight(const std::string& key) const {
        auto& w=weights.at(key);return read(w,0,w.shape.at(0),w.shape.size()>1?w.shape.at(1):1);
    }
    Matrix project(const Matrix& x,const std::string& key) const {
        Matrix w=weight(key+".weight");
        if(projection){
            Matrix y(x.rows(),w.rows());
            if(projection(backend,x.data(),w.data(),x.rows(),x.cols(),w.rows(),y.data()))return y;
        }
        return x*w.transpose();
    }
    Matrix rms(const Matrix& x,const std::string& key) const {
        Matrix w=weight(key+".weight");
        if(rowop){
            Matrix y(x.rows(),x.cols());
            homeimg_rows op{HOMEIMG_RMS,int(x.rows()),int(x.cols()),x.data(),nullptr,nullptr,w.data(),
                            nullptr,nullptr,0,1,y.data()};
            if(rowop(backend,&op))return y;
        }
        return norm(x,w);
    }
    Matrix activate(const Matrix& g,const Matrix& u) const {
        if(rowop){
            Matrix y(g.rows(),g.cols());
            homeimg_rows op{HOMEIMG_SWIGLU,int(g.rows()),int(g.cols()),g.data(),u.data(),nullptr,nullptr,
                            nullptr,nullptr,0,1,y.data()};
            if(rowop(backend,&op))return y;
        }
        return silu(g).array()*u.array();
    }
    Matrix dense(const Matrix& x,const std::string& p) const {
        Matrix h=activate(project(x,p+"gate_proj"),project(x,p+"up_proj"));
        return project(h,p+"down_proj");
    }
    void rope(Matrix& x,bool lm) const {
        homeimg_rows op{HOMEIMG_TEXT_ROPE,int(x.rows()),int(x.cols()),x.data(),nullptr,nullptr,nullptr,
                        nullptr,nullptr,0,lm?2.f:1.f,x.data()};
        if(rowop && rowop(backend,&op))return;
        int width=lm?64:x.cols();
        for(int t=0;t<x.rows();++t)for(int i=0;i<width/2;++i){
            float angle=t*std::pow(lm?84000.f:10000.f,-2.f*i/width);
            int j=lm?i:2*i,k=lm?i+width/2:j+1;
            float a=x(t,j),b=x(t,k);
            x(t,j)=a*std::cos(angle)-b*std::sin(angle);
            x(t,k)=a*std::sin(angle)+b*std::cos(angle);
        }
    }
    Matrix attend(const Matrix& x,const std::string& prefix,bool lm) const {
        int heads=lm?16:20,kv=lm?4:5,d=lm?128:64;
        Matrix qkv=project(x,prefix+(lm?"query_key_value":"qkv")),out(x.rows(),heads*d);
        for(int group=0;group<kv;++group){
            Matrix k=qkv.middleCols((heads+group)*d,d);
            Matrix v=qkv.middleCols((heads+kv+group)*d,d);
            if(lm)k=rms(k,prefix+"key_layernorm");
            rope(k,lm);
            for(int head=4*group;head<4*group+4;++head){
                Matrix q=qkv.middleCols(head*d,d);
                if(lm)q=rms(q,prefix+"query_layernorm");
                rope(q,lm);
                if(attentionBackend){
                    Matrix y(x.rows(),d);
                    if(attentionBackend(backend,q.data(),k.data(),v.data(),x.rows(),x.rows(),d,
                                        1.f/std::sqrt(float(d)),y.data(),lm)){
                        out.middleCols(head*d,d)=y;
                        continue;
                    }
                }
                Matrix scores=(q*k.transpose())/std::sqrt(float(d));
                if(lm)scores.triangularView<Eigen::StrictlyUpper>().setConstant(-INFINITY);
                scores.colwise()-=scores.rowwise().maxCoeff().eval();
                scores=scores.array().exp();
                scores.array().colwise()/=scores.rowwise().sum().eval().array();
                out.middleCols(head*d,d)=scores*v;
            }
        }
        return project(out,prefix+(lm?"dense":"out"));
    }
    Matrix expert(const Matrix& x,const std::string& key,int e,int cols) const {
        const auto& raw=weights.at(key);
        using I8=Eigen::Matrix<int8_t,-1,-1,Eigen::RowMajor>;
        if(raw.dtype!="I8" || raw.shape!=std::vector<size_t>{128,size_t(x.cols()),size_t(cols)})
            throw std::runtime_error("invalid expert tensor: "+key);
        auto ptr=reinterpret_cast<const int8_t*>(raw.data)+size_t(e)*x.cols()*cols;
        auto packed=Eigen::Map<const I8>(ptr,x.cols(),cols);
        Matrix y(x.rows(),cols);
        Matrix scales=read(weights.at(key+"_s"),size_t(e)*cols,1,cols);
        if(expertBackend && expertBackend(backend,x.data(),ptr,scales.data(),x.rows(),x.cols(),cols,y.data()))return y;
        y.noalias()=x*packed.cast<float>();
        return y.array().rowwise()*scales.row(0).array();
    }
    Matrix moe(const Matrix& x,const std::string& p,Matrix* experts) const {
        if(experts) experts->resize(x.rows(),8);
        Matrix logits=project(x,p+"gate");
        Matrix bias=weight(p+"gate.expert_bias");
        std::vector<std::vector<std::pair<int,float>>> selected(128);
        Matrix routes(x.rows(),16);
        homeimg_rows op{HOMEIMG_ROUTES,int(x.rows()),128,logits.data(),nullptr,nullptr,nullptr,
                        nullptr,bias.data(),0,1,routes.data()};
        if(rowop && rowop(backend,&op)){
            for(int t=0;t<x.rows();++t)for(int j=0;j<8;++j){
                int expert=int(routes(t,j));
                if(expert<0 || expert>=128)throw std::runtime_error("invalid Vulkan expert index");
                selected[expert].emplace_back(t,routes(t,8+j));
                if(experts)(*experts)(t,j)=expert;
            }
        }else{
            Matrix scores=(1.f+(-logits).array().exp()).inverse();
            Matrix rank=scores.rowwise()+bias.col(0).transpose();
            for(int t=0;t<x.rows();++t){
                std::array<float,8> group;
                std::array<int,128> ids;
                std::iota(ids.begin(),ids.end(),0);
                auto better=[&](int a,int b){return rank(t,a)>rank(t,b);};
                for(int g=0;g<8;++g){
                    auto begin=ids.begin()+16*g;
                    std::stable_sort(begin,begin+16,better);
                    group[g]=rank(t,begin[0])+rank(t,begin[1]);
                }
                std::array<int,8> groups;std::iota(groups.begin(),groups.end(),0);
                std::stable_sort(groups.begin(),groups.end(),[&](int a,int b){return group[a]>group[b];});
                std::vector<int> pool;
                for(int g:{groups[0],groups[1]})pool.insert(pool.end(),ids.begin()+16*g,ids.begin()+16*g+16);
                std::stable_sort(pool.begin(),pool.end(),better);
                float sum=1e-20f;
                for(int i=0;i<8;++i){
                    sum+=scores(t,pool[i]);
                    if(experts) (*experts)(t,i)=pool[i];
                }
                for(int i=0;i<8;++i)selected[pool[i]].emplace_back(t,scores(t,pool[i])/sum*2.5f);
            }
        }
        Matrix out=Matrix::Zero(x.rows(),x.cols());
        for(int e=0;e<128;++e){
            const auto& rows=selected[e];if(rows.empty())continue;
            Matrix input(rows.size(),x.cols());
            for(size_t i=0;i<rows.size();++i)input.row(i)=x.row(rows[i].first);
            Matrix gu=expert(input,p+"_wgu",e,640);
            Matrix h=activate(gu.leftCols(320),gu.rightCols(320));
            Matrix y=expert(h,p+"_wd",e,2048);
            for(size_t i=0;i<rows.size();++i)out.row(rows[i].first)+=rows[i].second*y.row(i);
        }
        return out+dense(x,p+"shared_experts.");
    }
public:
    explicit TextEncoder(const std::string& root,homeimg_projection project=nullptr,homeimg_masked_attention attn=nullptr,
                         void* ctx=nullptr,homeimg_expert expertFn=nullptr,homeimg_rowop rowsFn=nullptr):weights(load_component(root,Component::TextEncoder)),
                         projection(project),attentionBackend(attn),backend(ctx),expertBackend(expertFn),rowop(rowsFn){}
    Matrix trunk(const Tokens& tokens, TextTrace* trace=nullptr) const {
        Matrix x(tokens.count,2048);
        auto& embedding=weights.at("lm.word_embeddings.weight");
        for(int t=0;t<tokens.count;++t)x.row(t)=read(embedding,size_t(tokens.ids.at(t))*2048,1,2048);
        for(int i=0;i<16;++i){
            auto p="lm.layers."+std::to_string(i)+".";
            Matrix attention_output=attend(rms(x,p+"input_layernorm"),p+"attention.",true);
            x+=attention_output;
            if(trace && i==0) trace->first_attention=attention_output;
            Matrix h=rms(x,p+"post_attention_layernorm");
            Matrix update=i?moe(h,p+"mlp.",trace && i==1 ? &trace->first_experts : nullptr):dense(h,p+"mlp.");
            x+=update;
            if(trace && i==0) trace->first_dense=update;
            if(trace && i==1) trace->first_moe=update;
        }
        return rms(x,"lm.norm");
    }
    Matrix refine(const Matrix& lm) const {
        Matrix x=project(rms(lm,"in_norm"),"in_proj");
        for(int i=0;i<4;++i){
            auto p="blocks."+std::to_string(i)+".";
            x+=attend(rms(x,p+"norm1"),p+"attn.",false);
            Matrix gu=project(rms(x,p+"norm2"),p+"mlp.in_proj");
            Matrix h=activate(gu.leftCols(3456),gu.rightCols(3456));
            x+=project(h,p+"mlp.out_proj");
        }
        return rms(x,"norm");
    }
};
}
