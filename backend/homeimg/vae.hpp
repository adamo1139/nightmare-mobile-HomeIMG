// FLUX.2 small decoder. Each row is a spatial position; channels are contiguous.
#pragma once
#include "aurora.hpp"
#include "safetensors.hpp"

namespace homeimg {
class VaeDecoder {
    MappedWeights weights;
    homeimg_projection projection;
    homeimg_conv convolution;
    homeimg_cross attentionBackend;
    void* backend;
    homeimg_group group;
    using View=Eigen::Map<const Matrix>;
    View tensor(const std::string& name) const {
        const auto& t=weights.at(name);
        if(t.dtype!="F32" || t.shape.empty())throw std::runtime_error("invalid VAE weight "+name);
        return View(reinterpret_cast<const float*>(t.data),t.shape.front(),t.bytes/4/t.shape.front());
    }
    Matrix affine(const Matrix& x,const std::string& name) const {
        auto weight=tensor(name+".weight");
        Matrix y(x.rows(),weight.rows());
        if(!projection || !projection(backend,x.data(),weight.data(),x.rows(),x.cols(),weight.rows(),y.data()))
            y.noalias()=x*weight.transpose();
        y.rowwise()+=tensor(name+".bias").col(0).transpose();
        return y;
    }
    Matrix conv(const Matrix& x,int side,const std::string& name,int scale=1) const {
        auto w=tensor(name+".weight");
        if(w.cols()==x.cols() && scale==1)return affine(x,name);
        // Checkpoint convolution filters are flattened in [channel,dy,dx] order.
        if(w.cols()!=9*x.cols())throw std::runtime_error("invalid convolution "+name);
        int outputSide=side*scale;
        Matrix out(outputSide*outputSide,w.rows());
        if(convolution && convolution(backend,x.data(),w.data(),side,x.cols(),w.rows(),scale,out.data())){
            out.rowwise()+=tensor(name+".bias").col(0).transpose();
            return out;
        }
        // Size im2col batches by bytes, since decoder channel counts change by stage.
        int batch=std::min<Eigen::Index>(out.rows(),(8u<<20)/(sizeof(float)*w.cols()));
        batch=std::max(1,batch);
        Matrix patch(batch,w.cols());
        Eigen::RowVectorXf padding=Eigen::RowVectorXf::Zero(x.cols());
        for(int start=0;start<out.rows();start+=batch){
            int rows=std::min<int>(batch,out.rows()-start);
            for(int r=0;r<rows;++r){
                int y=(start+r)/outputSide, col=(start+r)%outputSide;
                const float* neighborhood[9];
                for(int dy=0;dy<3;++dy)for(int dx=0;dx<3;++dx){
                    int iy=y+dy-1,ix=col+dx-1;
                    neighborhood[3*dy+dx]=iy<0||ix<0||iy>=outputSide||ix>=outputSide
                        ? padding.data() : x.row((iy/scale)*side+ix/scale).data();
                }
                float* dst=patch.row(r).data();
                for(int c=0;c<x.cols();++c)
                    for(int i=0;i<9;++i)*dst++=neighborhood[i][c];
            }
            out.middleRows(start,rows).noalias()=patch.topRows(rows)*w.transpose();
        }
        out.rowwise()+=tensor(name+".bias").col(0).transpose();
        return out;
    }
    Matrix group_norm(const Matrix& x,const std::string& p,bool activate=false) const {
        Matrix y(x.rows(),x.cols());
        if(group && group(backend,x.data(),tensor(p+".weight").data(),tensor(p+".bias").data(),
                          x.rows(),x.cols(),activate,y.data()))return y;
        int width=x.cols()/32;
        for(int c=0;c<x.cols();c+=width){
            auto group=x.middleCols(c,width);
            float mean=group.cast<double>().mean();
            float var=(group.cast<double>().array()-mean).square().mean();
            y.middleCols(c,width)=(group.array()-mean)/std::sqrt(var+1e-6f);
        }
        y.array().rowwise()*=tensor(p+".weight").col(0).transpose().array();
        y.rowwise()+=tensor(p+".bias").col(0).transpose();
        if(activate)return silu(std::move(y));
        return y;
    }
    Matrix residual(const Matrix& x,int side,const std::string& p) const {
        Matrix h=conv(group_norm(x,p+".norm1",true),side,p+".conv1");
        h=conv(group_norm(h,p+".norm2",true),side,p+".conv2");
        if(h.cols()!=x.cols())h+=affine(x,p+".conv_shortcut");
        else h+=x;
        return h;
    }
public:
    explicit VaeDecoder(const std::string& root,homeimg_projection project=nullptr,homeimg_conv conv=nullptr,
                        homeimg_cross attn=nullptr,void* ctx=nullptr,homeimg_group groupFn=nullptr):weights(load_component(root,Component::Vae)),
                        projection(project),convolution(conv),attentionBackend(attn),backend(ctx),group(groupFn){}
    Matrix decode(const Matrix& z,int side) const {
        Matrix unpack(4*z.rows(),32);
        auto mean=tensor("bn.running_mean"),var=tensor("bn.running_var");
        for(int t=0;t<z.rows();++t)for(int i=0;i<128;++i){
            int row=2*(t/side)+(i%4)/2,col=2*(t%side)+i%2;
            unpack(row*2*side+col,i/4)=z(t,i)*std::sqrt(var(i,0)+1e-4f)+mean(i,0);
        }
        side*=2;
        Matrix x=conv(affine(unpack,"post_quant_conv"),side,"decoder.conv_in");
        x=residual(x,side,"decoder.mid_block.resnets.0");
        {
            const std::string a="decoder.mid_block.attentions.0.";
            Matrix n=group_norm(x,a+"group_norm"),q=affine(n,a+"to_q"),k=affine(n,a+"to_k"),v=affine(n,a+"to_v");
            Matrix joined(q.rows(),q.cols());
            float scale=1.f/std::sqrt(float(q.cols()));
            if(!attentionBackend || !attentionBackend(backend,q.data(),k.data(),v.data(),q.rows(),k.rows(),q.cols(),scale,joined.data()))
                joined=attention(q,k,v,scale);
            x+=affine(joined,a+"to_out.0");
        }
        x=residual(x,side,"decoder.mid_block.resnets.1");
        for(int block=0;block<4;++block){
            auto p="decoder.up_blocks."+std::to_string(block)+".";
            for(int j=0;j<3;++j)x=residual(x,side,p+"resnets."+std::to_string(j));
            if(block<3){
                x=conv(x,side,p+"upsamplers.0.conv",2);
                side*=2;
            }
        }
        return conv(group_norm(x,"decoder.conv_norm_out",true),side,"decoder.conv_out").cwiseMax(-1).cwiseMin(1);
    }
};
}
