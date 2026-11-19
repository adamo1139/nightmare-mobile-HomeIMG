#include "homeimg.h"
#include "sample.hpp"
#include "vae.hpp"
#include <cstdio>
#include <cfenv>
#if defined(__ANDROID__) && defined(_OPENMP)
#include <omp.h>
#endif

uint32_t homeimg_api_version(){return 1;}
uint32_t homeimg_capabilities(){return 1;}
int homeimg_generate_attention_blocks(const char *root,const char *prompt,const char *negative,int side,int steps,float cfg,
                     const float *noise,size_t count,uint8_t *rgb,size_t bytes,homeimg_progress progress,
                     void *user,char *error,size_t error_bytes,homeimg_projection project,void* backend,float shift,const homeimg_device_ops* operations,homeimg_self_block selfBlock,homeimg_cross_heads crossHeads){
    if(error && error_bytes)error[0]=0;
    try{
        if(!root || !prompt || !negative || !noise || !rgb ||
           (side<16 || side>64) || steps<1 || steps>1000 || !std::isfinite(cfg) || !std::isfinite(shift) || shift<0 ||
           count!=size_t(side)*side*128 || bytes!=size_t(side)*side*16*16*3)
            throw std::invalid_argument("invalid HomeIMG request or buffer size");
#if defined(__ANDROID__) && defined(_OPENMP)
        // Bound Eigen's CPU GEMMs on mobile, including the attention left on CPU in Vulkan mode.
        omp_set_num_threads(std::min(4,omp_get_num_procs()));
#endif
        auto advance=[&](int n){return !progress || progress(user,n,steps)!=0;};
        if(!advance(0))return 2;
        homeimg::Matrix z=Eigen::Map<const homeimg::Matrix>(noise,side*side,128);
        if(!z.allFinite())throw std::invalid_argument("non-finite noise");
        const homeimg_device_ops ops=operations?*operations:homeimg_device_ops{};
        homeimg::AuroraConfig config;
        config.project=project;config.backend=backend;config.attention=ops.attention;
        config.selfBlock=selfBlock;config.crossHeads=crossHeads;config.mlp=ops.mlp;config.cross=ops.cross;config.masked=ops.masked;config.spatial=ops.spatial;
        config.rowop=ops.rowop;config.expert=ops.expert;config.weightScope=ops.weight_scope;
        if(!homeimg::sample(root,prompt,negative,side,steps,cfg,z,
                           [&](int n,const auto&){return advance(n);},config,shift))return 2;
        auto image=homeimg::VaeDecoder(root,ops.conv?project:nullptr,ops.conv,ops.cross,backend,ops.group).decode(z,side);
        if(!image.allFinite())throw std::runtime_error("non-finite image");
        for(Eigen::Index i=0;i<image.size();++i)
            rgb[i]=static_cast<uint8_t>(std::nearbyint((image.data()[i]+1.f)*127.5f));
        return 0;
    }catch(const std::exception& e){
        if(error && error_bytes)std::snprintf(error,error_bytes,"%s",e.what());
        return 1;
    }catch(...){
        if(error && error_bytes)std::snprintf(error,error_bytes,"unexpected HomeIMG failure");
        return 1;
    }
}

int homeimg_generate_blocks(const char *root,const char *prompt,const char *negative,int side,int steps,float cfg,
                     const float *noise,size_t count,uint8_t *rgb,size_t bytes,homeimg_progress progress,
                     void *user,char *error,size_t error_bytes,homeimg_projection project,void* backend,float shift,const homeimg_device_ops* operations){
    return homeimg_generate_attention_blocks(root,prompt,negative,side,steps,cfg,noise,count,rgb,bytes,
        progress,user,error,error_bytes,project,backend,shift,operations,nullptr,nullptr);
}

int homeimg_generate_accelerated(const char *root,const char *prompt,const char *negative,int side,int steps,float cfg,
                     const float *noise,size_t count,uint8_t *rgb,size_t bytes,homeimg_progress progress,
                     void *user,char *error,size_t error_bytes,homeimg_projection project,void* backend,float shift,homeimg_attention attention){
    homeimg_device_ops ops{};ops.attention=attention;
    return homeimg_generate_blocks(root,prompt,negative,side,steps,cfg,noise,count,rgb,bytes,
                                   progress,user,error,error_bytes,project,backend,shift,&ops);
}

int homeimg_generate_shifted(const char *root,const char *prompt,const char *negative,int side,int steps,float cfg,
                     const float *noise,size_t count,uint8_t *rgb,size_t bytes,homeimg_progress progress,
                     void *user,char *error,size_t error_bytes,homeimg_projection project,void* backend,float shift){
    return homeimg_generate_accelerated(root,prompt,negative,side,steps,cfg,noise,count,rgb,bytes,
                                        progress,user,error,error_bytes,project,backend,shift,nullptr);
}

int homeimg_generate_projected(const char *root,const char *prompt,const char *negative,int side,int steps,float cfg,
                     const float *noise,size_t count,uint8_t *rgb,size_t bytes,homeimg_progress progress,
                     void *user,char *error,size_t error_bytes,homeimg_projection project,void* backend){
    return homeimg_generate_shifted(root,prompt,negative,side,steps,cfg,noise,count,rgb,bytes,
                                    progress,user,error,error_bytes,project,backend,0.f);
}

int homeimg_generate(const char *root,const char *prompt,const char *negative,int side,int steps,float cfg,
                     const float *noise,size_t count,uint8_t *rgb,size_t bytes,homeimg_progress progress,
                     void *user,char *error,size_t error_bytes){
    return homeimg_generate_projected(root,prompt,negative,side,steps,cfg,noise,count,rgb,bytes,
                                      progress,user,error,error_bytes,nullptr,nullptr);
}
