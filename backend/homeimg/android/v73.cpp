// Isolated optional plugin. Only this library links the pinned Hexagon backend.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-hexagon.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>
#include <android/log.h>

extern "C" void* homeimg_v73_open(){
    auto reg=ggml_backend_hexagon_reg();
    if(!reg || !ggml_backend_reg_dev_count(reg))return nullptr;
    auto device=ggml_backend_reg_dev_get(reg,0);
    return device?ggml_backend_dev_init(device,nullptr):nullptr;
}
extern "C" void homeimg_v73_close(void* handle){
    if(handle)ggml_backend_free(static_cast<ggml_backend_t>(handle));
}
extern "C" int homeimg_v73_project(void* handle,const float* x,const float* w,
                                  int tokens,int in,int out,float* y) {
    if(!handle || in%32 || out%32 || tokens<=0)return 0;
    const size_t limit=256u<<20;
    const size_t xbytes=size_t(tokens)*in*sizeof(float);
    if(xbytes>=limit)return 0;
    // Split by output channels: each shard of a row-major [out,in] weight
    // is contiguous; each graph maps at most 256 MiB incl. inputs and outputs.
    int shard=std::min<size_t>(out,(limit-xbytes)/(sizeof(float)*(size_t(in)+tokens)));
    shard-=shard%32;
    if(!shard)return 0;
    auto backend=static_cast<ggml_backend_t>(handle);
    for(int first=0;first<out;first+=shard){
        int n=std::min(shard,out-first);
        ggml_init_params params{4*ggml_tensor_overhead()+ggml_graph_overhead_custom(8,false)+1024,nullptr,true};
        std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init(params),ggml_free);
        if(!ctx)return 0;
        auto weights=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,in,n);
        auto inputs=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,in,tokens);
        auto result=ggml_mul_mat(ctx.get(),weights,inputs);
        if(!ggml_backend_supports_op(backend,result))return 0;
        auto graph=ggml_new_graph_custom(ctx.get(),8,false);
        ggml_build_forward_expand(graph,result);
        auto buft=ggml_backend_get_default_buffer_type(backend);
        if(ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(),buft)>limit)return 0;
        auto buffer=ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(),buft);
        if(!buffer)return 0;
        std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> guard(buffer,ggml_backend_buffer_free);
        ggml_backend_tensor_set(weights,w+size_t(first)*in,0,ggml_nbytes(weights));
        ggml_backend_tensor_set(inputs,x,0,xbytes);
        if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)return 0;
        if(n==out)ggml_backend_tensor_get(result,y,0,ggml_nbytes(result));
        else{
            std::vector<float> tile(size_t(tokens)*n);
            ggml_backend_tensor_get(result,tile.data(),0,ggml_nbytes(result));
            for(int t=0;t<tokens;++t)std::copy_n(tile.data()+size_t(t)*n,n,y+size_t(t)*out+first);
        }
    }
    return std::all_of(y,y+size_t(tokens)*out,[](float value){return std::isfinite(value);});
}
