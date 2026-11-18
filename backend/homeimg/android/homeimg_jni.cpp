#include "../homeimg.h"
#include <jni.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <random>
#include <array>
#include <chrono>
#include <cmath>
#include <mutex>
#include <cstdlib>
#include <dlfcn.h>
#include <sys/resource.h>
#include <android/log.h>

extern "C" void* homeimg_gpu_open();
extern "C" int homeimg_gpu_masked_attention(void*,const float*,const float*,const float*,int,int,int,float,float*,int);
extern "C" void homeimg_gpu_weights(void*,const void*,size_t);
extern "C" int homeimg_gpu_group(void*,const float*,const float*,const float*,int,int,int,float*);
extern "C" int homeimg_gpu_expert(void*,const float*,const int8_t*,const float*,int,int,int,float*);
extern "C" int homeimg_gpu_rows(void*,const homeimg_rows*);
extern "C" int homeimg_gpu_spatial(void*,const float*,int,int,int,float,const float*,const float*,float*);
extern "C" int homeimg_gpu_conv(void*,const float*,const float*,int,int,int,int,float*);
extern "C" int homeimg_gpu_mlp(void*,float*,int,int,int,const float*,const float*,const float*,const float*,const float*,const float*);
extern "C" int homeimg_gpu_cross(void*,const float*,const float*,const float*,int,int,int,float,float*);
extern "C" int homeimg_gpu_attention(void*,const float*,const float*,const float*,int,int,float,float*);
extern "C" const char* homeimg_gpu_diagnostic(void*);
extern "C" int64_t homeimg_gpu_dispatches(void*);
extern "C" void homeimg_gpu_close(void*);
extern "C" int homeimg_gpu_project(void*,const float*,const float*,int,int,int,float*);

namespace {
std::string utf8(JNIEnv* env,jbyteArray input){
    std::string value(env->GetArrayLength(input),'\0');
    env->GetByteArrayRegion(input,0,value.size(),reinterpret_cast<jbyte*>(value.data()));
    if(value.find('\0')!=std::string::npos)throw std::invalid_argument("NUL in text");
    return value;
}
std::mutex engine_mutex;
struct Accelerator {
    using Open=void*(*)();
    using Close=void(*)(void*);
    homeimg_projection project=nullptr;
    Close close=nullptr;
    void* backend=nullptr;
    const char* diagnostic="CPU selected";
    JNIEnv* env=nullptr;
    jobject activity=nullptr;
    jmethodID projectionStatus=nullptr;
    jmethodID backendNotice=nullptr;
    std::string lastNotice;
    int hits=0,misses=0,attentionHits=0,attentionMisses=0;
    long long attentionMicros=0;
    long long micros=0;
    int64_t gpuDispatches=0;
    void release(){
        if(backend){
            if(close==homeimg_gpu_close)gpuDispatches=homeimg_gpu_dispatches(backend);
            close(backend);backend=nullptr;
        }
    }
    ~Accelerator(){release();}
    void prepare(const std::string& skel,bool gpu=false){
        if(gpu){
            backend=homeimg_gpu_open();
            close=homeimg_gpu_close;
            project=homeimg_gpu_project;
            diagnostic="Vulkan compute unavailable";
            if(backend)test(true);
            return;
        }
        if(skel.empty())return;
        // This Activity has a dedicated process. Set the DSP path before loading ggml.
        std::string paths=skel+";/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp";
        setenv("ADSP_LIBRARY_PATH",paths.c_str(),1);
        setenv("DSP_LIBRARY_PATH",paths.c_str(),1);
        setenv("GGML_HEXAGON_ARCH","73",1);
        setenv("GGML_HEXAGON_OPFUSION","0",1);
        static void* library=dlopen("libhomeimg_v73.so",RTLD_NOW|RTLD_LOCAL);
        diagnostic="HTP plugin unavailable";
        if(!library)return;
        auto open=reinterpret_cast<Open>(dlsym(library,"homeimg_v73_open"));
        close=reinterpret_cast<Close>(dlsym(library,"homeimg_v73_close"));
        project=reinterpret_cast<homeimg_projection>(dlsym(library,"homeimg_v73_project"));
        diagnostic="Hexagon driver or v73 session unavailable";
        if(!open || !close || !project || !(backend=open()))return;
        test(false);
    }
    void test(bool gpu){
        // Non-symmetric rows detect wrong orientation and incorrect output strides.
        std::array<float,128*128> identity{};
        std::array<float,32*128> x{},y{};
        for(int j=0;j<128;++j)identity[128*j+j]=1;
        for(size_t j=0;j<x.size();++j)x[j]=float(int(j%31)-15)/16.f;
        bool ok=project(backend,x.data(),identity.data(),32,128,128,y.data());
        for(size_t j=0;j<x.size();++j)ok=ok && std::abs(x[j]-y[j])<1e-4f;
        diagnostic=gpu?(ok?"Vulkan GPU matrix test passed":"Vulkan GPU matrix test failed"):ok?"HTP v73 matrix test passed":"HTP v73 matrix test failed";
        if(!ok){close(backend);backend=nullptr;}
    }
    static void weights(void* ctx,const void* base,size_t bytes){
        homeimg_gpu_weights(static_cast<Accelerator*>(ctx)->backend,base,bytes);
    }
    static int group(void* ctx,const float* x,const float* w,const float* bias,int rows,int cols,int silu,float* y){
        return homeimg_gpu_group(static_cast<Accelerator*>(ctx)->backend,x,w,bias,rows,cols,silu,y);
    }
    static int expert(void* ctx,const float* x,const int8_t* w,const float* scales,int rows,int width,int cols,float* y){
        return homeimg_gpu_expert(static_cast<Accelerator*>(ctx)->backend,x,w,scales,rows,width,cols,y);
    }
    static int rows(void* ctx,const homeimg_rows* op){
        return homeimg_gpu_rows(static_cast<Accelerator*>(ctx)->backend,op);
    }
    static int spatial(void* ctx,const float* qkv,int side,int heads,int ref,float scale,const float* nq,const float* nk,float* y){
        auto& self=*static_cast<Accelerator*>(ctx);
        auto start=std::chrono::steady_clock::now();
        int ok=homeimg_gpu_spatial(self.backend,qkv,side,heads,ref,scale,nq,nk,y);
        self.attentionMicros+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count();
        if(ok)self.attentionHits+=heads;
        return ok;
    }
    static int textAttention(void* ctx,const float* q,const float* k,const float* v,int tokens,int keys,int width,
                             float scale,float* y,int causal){
        return homeimg_gpu_masked_attention(static_cast<Accelerator*>(ctx)->backend,q,k,v,tokens,keys,width,scale,y,causal);
    }
    static int convolution(void* ctx,const float* x,const float* w,int side,int in,int out,int scale,float* y){
        return homeimg_gpu_conv(static_cast<Accelerator*>(ctx)->backend,x,w,side,in,out,scale,y);
    }
    int mlpHits=0,mlpMisses=0;
    long long mlpMicros=0;
    static int feedForward(void* ctx,float* x,int rows,int width,int hidden,const float* gate,const float* down,
                           const float* norm,const float* scale,const float* bias,const float* strength){
        auto& self=*static_cast<Accelerator*>(ctx);
        auto start=std::chrono::steady_clock::now();
        int ok=homeimg_gpu_mlp(self.backend,x,rows,width,hidden,gate,down,norm,scale,bias,strength);
        self.mlpMicros+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count();
        if(ok)++self.mlpHits;else ++self.mlpMisses;
        return ok;
    }
    static int cross(void* ctx,const float* q,const float* k,const float* v,int tokens,int keys,int width,float scale,float* y){
        auto& self=*static_cast<Accelerator*>(ctx);
        auto start=std::chrono::steady_clock::now();
        int ok=homeimg_gpu_cross(self.backend,q,k,v,tokens,keys,width,scale,y);
        self.attentionMicros+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count();
        if(ok)++self.attentionHits;else ++self.attentionMisses;
        return ok;
    }
    static int attend(void* ctx,const float* q,const float* k,const float* v,int tokens,int width,float scale,float* y){
        auto& self=*static_cast<Accelerator*>(ctx);
        auto start=std::chrono::steady_clock::now();
        int ok=homeimg_gpu_attention(self.backend,q,k,v,tokens,width,scale,y);
        self.attentionMicros+=std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now()-start).count();
        if(ok)++self.attentionHits;else ++self.attentionMisses;
        return ok;
    }
    static int run(void* ctx,const float* x,const float* w,int t,int in,int out,float* y){
        auto& self=*static_cast<Accelerator*>(ctx);
        self.env->CallVoidMethod(self.activity,self.projectionStatus,self.hits,self.misses,JNI_TRUE);
        if(self.env->ExceptionCheck())throw std::runtime_error("Projection callback failed");
        auto start=std::chrono::steady_clock::now();
        int ok=self.project(self.backend,x,w,t,in,out,y);
        self.micros+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count();
        if(self.close==homeimg_gpu_close){
            const char* reason=homeimg_gpu_diagnostic(self.backend);
            if(self.lastNotice!=reason){
                self.lastNotice=reason;
                __android_log_print(ANDROID_LOG_WARN,"HomeIMG","%s",reason);
                auto message=self.env->NewStringUTF(reason);
                if(message){
                    self.env->CallVoidMethod(self.activity,self.backendNotice,message);
                    self.env->DeleteLocalRef(message);
                }
                if(self.env->ExceptionCheck())throw std::runtime_error("Backend notice failed");
            }
        }
        if(ok)++self.hits;else ++self.misses;
        self.env->CallVoidMethod(self.activity,self.projectionStatus,self.hits,self.misses,JNI_FALSE);
        if(self.env->ExceptionCheck())throw std::runtime_error("Projection callback failed");
        return ok;
    }
};
struct Callback {JNIEnv* env; jobject target; jmethodID method,profile; Accelerator& accelerator;
    std::chrono::steady_clock::time_point since{};
    long long projections=0,attention=0,mlp=0;};
int advance(void* user,int done,int total){
    auto& cb=*static_cast<Callback*>(user);
    auto now=std::chrono::steady_clock::now();
    if(done){
        auto duration=std::chrono::duration_cast<std::chrono::milliseconds>(now-cb.since).count();
        cb.env->CallVoidMethod(cb.target,cb.profile,done,jlong(duration),
            jlong((cb.accelerator.micros-cb.projections)/1000),
            jlong((cb.accelerator.attentionMicros-cb.attention)/1000),
            jlong((cb.accelerator.mlpMicros-cb.mlp)/1000));
        if(cb.env->ExceptionCheck())return 0;
    }
    cb.since=now;
    cb.projections=cb.accelerator.micros;
    cb.attention=cb.accelerator.attentionMicros;
    cb.mlp=cb.accelerator.mlpMicros;
    if(done==total && cb.accelerator.close!=homeimg_gpu_close)cb.accelerator.release();
    bool proceed=cb.env->CallBooleanMethod(cb.target,cb.method,done,total);
    return !cb.env->ExceptionCheck() && proceed;
}
}
extern "C" JNIEXPORT jintArray JNICALL
Java_com_abrah_nightmare_homeimg_HomeImgActivity_generate(JNIEnv* env,jobject activity,
        jbyteArray directory,jbyteArray positive,jbyteArray negative,jint resolution,jint steps,
        jfloat cfg,jfloat shift,jlong seed,jbyteArray skelDirectory,jboolean gpu,jboolean gpuAttention){
    try {
        std::unique_lock<std::mutex> lock(engine_mutex,std::try_to_lock);
        if(!lock)throw std::runtime_error("HomeIMG is already generating");
        if(resolution<256 || resolution>1024 || resolution%16!=0)throw std::invalid_argument("unsupported size");
        auto root=utf8(env,directory),prompt=utf8(env,positive),neg=utf8(env,negative);
        int side=resolution/16;
        std::vector<float> noise(side*side*128);
        std::mt19937_64 generator(static_cast<uint64_t>(seed));
        std::normal_distribution<float> normal;
        for(float& n:noise)n=normal(generator);
        std::vector<uint8_t> rgb(resolution*resolution*3);
        auto cls=env->GetObjectClass(activity);
        auto method=env->GetMethodID(cls,"progress","(II)Z");
        if(!method)return nullptr;

        char error[512]{};
        Accelerator accelerator;
        auto profile=env->GetMethodID(cls,"stepProfile","(IJJJJ)V");
        if(!profile)return nullptr;
        Callback cb{env,activity,method,profile,accelerator};
        accelerator.env=env;
        accelerator.activity=activity;
        accelerator.projectionStatus=env->GetMethodID(cls,"projectionStatus","(IIZ)V");
        if(!accelerator.projectionStatus)return nullptr;
        accelerator.backendNotice=env->GetMethodID(cls,"backendNotice","(Ljava/lang/String;)V");
        if(!accelerator.backendNotice)return nullptr;
        accelerator.prepare(utf8(env,skelDirectory),gpu);
        if(gpu && !accelerator.backend)throw std::runtime_error(accelerator.diagnostic);
        auto started=std::chrono::steady_clock::now();
        homeimg_device_ops ops{};
        if(gpu){
            ops.attention=gpuAttention?Accelerator::attend:nullptr;
            ops.spatial=gpuAttention?Accelerator::spatial:nullptr;
            ops.mlp=Accelerator::feedForward;ops.cross=Accelerator::cross;
            ops.conv=Accelerator::convolution;ops.masked=Accelerator::textAttention;
            ops.rowop=Accelerator::rows;ops.expert=Accelerator::expert;
            ops.group=Accelerator::group;ops.weight_scope=Accelerator::weights;
        }
        int status=homeimg_generate_blocks(root.c_str(),prompt.c_str(),neg.c_str(),side,steps,cfg,
            noise.data(),noise.size(),rgb.data(),rgb.size(),advance,&cb,error,sizeof(error),
            accelerator.backend?Accelerator::run:nullptr,&accelerator,shift,&ops);
        accelerator.release();
        struct rusage usage{};
        getrusage(RUSAGE_SELF,&usage);
        auto elapsed=std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now()-started).count();
        __android_log_print(ANDROID_LOG_INFO,"HomeIMG","%s; status=%d accelerated projections=%d CPU fallback=%d accelerator ms=%lld elapsed ms=%lld process peak RSS KiB=%ld",
                            accelerator.diagnostic,status,accelerator.hits,accelerator.misses,
                            accelerator.micros/1000,static_cast<long long>(elapsed),usage.ru_maxrss);
        if(!env->ExceptionCheck()){
            auto summary=env->GetMethodID(cls,"backendSummary","(IIJJJIIJIIJ)V");
            if(summary)env->CallVoidMethod(activity,summary,accelerator.hits,accelerator.misses,jlong(accelerator.micros/1000),jlong(elapsed),jlong(accelerator.gpuDispatches),accelerator.attentionHits,accelerator.attentionMisses,jlong(accelerator.attentionMicros/1000),accelerator.mlpHits,accelerator.mlpMisses,jlong(accelerator.mlpMicros/1000));
        }
        if(env->ExceptionCheck() || status==2)return nullptr;
        if(status)throw std::runtime_error(error);
        std::vector<jint> pixels(resolution*resolution);
        for(size_t i=0;i<pixels.size();++i)pixels[i]=static_cast<jint>(
            0xff000000u|(uint32_t(rgb[3*i])<<16)|(uint32_t(rgb[3*i+1])<<8)|rgb[3*i+2]);
        auto result=env->NewIntArray(pixels.size());
        if(result)env->SetIntArrayRegion(result,0,pixels.size(),pixels.data());
        return result;
    } catch(const std::exception& e){
        if(!env->ExceptionCheck())env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),e.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_abrah_nightmare_homeimg_HomeImgActivity_probe(JNIEnv* env,jobject,jbyteArray directory,jboolean gpu) {
    try{
        std::unique_lock<std::mutex> lock(engine_mutex,std::try_to_lock);
        if(!lock)return env->NewStringUTF("HomeIMG is busy");
        Accelerator accelerator;
        accelerator.prepare(utf8(env,directory),gpu);
        return env->NewStringUTF(accelerator.diagnostic);
    } catch(const std::exception& e) {
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),e.what());
        return nullptr;
    }
}
