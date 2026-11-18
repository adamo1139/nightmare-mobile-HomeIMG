// No Android framework required. Run on Android or with qemu-user + Bionic.
#include "../backend/homeimg/homeimg.h"
#include <dlfcn.h>
#include <cstdio>
#include <vector>
#include <stdexcept>
#include <string>
#include <fstream>
#include <cstring>

template<typename T> T symbol(void* handle,const char* name){
    auto p=dlsym(handle,name);
    if(!p)throw std::runtime_error(dlerror());
    return reinterpret_cast<T>(p);
}
static void* load(const char* path){
    void* lib=dlopen(path,RTLD_NOW|RTLD_LOCAL);
    if(!lib)throw std::runtime_error(dlerror());
    return lib;
}
int main(int argc,char** argv){
    if(argc!=4 && argc!=7){std::fprintf(stderr,"engine.so homeimg.so v73.so [model noise-token-major.f32 output.rgb]\n");return 2;}
    try {
        auto engine=load(argv[1]);
        auto get=symbol<const void*(*)(int)>(engine,"dit_engine_get_api");
        if(!get(111)||get(110))throw std::runtime_error("DiT ABI negotiation failed");
        auto core=load(argv[2]);
        auto api=symbol<decltype(&homeimg_api_version)>(core,"homeimg_api_version");
        auto generate=symbol<decltype(&homeimg_generate)>(core,"homeimg_generate");
        char error[512]{};
        if(api()!=1 || generate(nullptr,nullptr,nullptr,0,0,0,nullptr,0,nullptr,0,nullptr,nullptr,error,sizeof(error))!=1)
            throw std::runtime_error("HomeIMG ABI validation failed");
        auto plugin=load(argv[3]);
        auto open=symbol<void*(*)()>(plugin,"homeimg_v73_open");
        auto close=symbol<void(*)(void*)>(plugin,"homeimg_v73_close");
        auto device=open();
        std::printf("DiT ABI 111; HomeIMG API 1; Hexagon device: %s\n",device?"present":"unavailable");
        close(device);
        if(argc==7){
            std::vector<float> noise(16*16*128);std::vector<uint8_t> rgb(256*256*3);
            if(!std::ifstream(argv[5],std::ios::binary).read(reinterpret_cast<char*>(noise.data()),noise.size()*sizeof(float)))
                throw std::runtime_error("bad noise");
            if(generate(argv[4],"Rudy lis w zaśnieżonym lesie.","",16,1,1.f,noise.data(),noise.size(),
                        rgb.data(),rgb.size(),nullptr,nullptr,error,sizeof(error)))throw std::runtime_error(error);
            if(!std::ofstream(argv[6],std::ios::binary).write(reinterpret_cast<char*>(rgb.data()),rgb.size()))
                throw std::runtime_error("output failed");
        }
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
