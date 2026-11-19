#include <vulkan/vulkan.h>
#include "../homeimg.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <map>
#include <chrono>
#include <string>
#include <stdexcept>
#include <vector>

namespace {
const uint32_t shader[] = {
#include "projection.inc"
};
const uint32_t alignedShader[] = {
#include "projection_aligned.inc"
};
const uint32_t softmaxShader[] = {
#include "attention_softmax.inc"
};
const uint32_t normCode[]={
#include "mlp_norm.inc"
};
const uint32_t gateCode[]={
#include "mlp_gate.inc"
};
const uint32_t residualCode[]={
#include "mlp_residual.inc"
};
const uint32_t patchCode[]={
#include "vae_patch.inc"
};
const uint32_t prepareCode[]={
#include "spatial_prepare.inc"
};
const uint32_t joinCode[]={
#include "spatial_join.inc"
};
const uint32_t rowCode[]={
#include "row_update.inc"
};
const uint32_t expertCode[]={
#include "moe_expert.inc"
};
const uint32_t groupCode[]={
#include "vae_group.inc"
};
const uint32_t routeCode[]={
#include "moe_route.inc"
};
const uint32_t crossScoreCode[]={
#include "cross_scores.inc"
};
const uint32_t crossValueCode[]={
#include "cross_values.inc"
};
struct VulkanError : std::runtime_error {
    VkResult code;
    VulkanError(VkResult result,const char* operation):std::runtime_error(
        std::string(operation)+" VkResult "+std::to_string(result)),code(result){}
};
void check(VkResult status,const char* operation="Vulkan") {
    if(status!=VK_SUCCESS) throw VulkanError(status,operation);
}
struct Device {
    VkInstance instance{};
    VkPhysicalDevice physical{};
    VkDevice device{};
    VkQueue queue{};
    VkDescriptorSetLayout bindings{};
    VkPipelineLayout layout{};
    VkPipeline pipeline{};
    VkPipeline alignedPipeline{};
    VkPipeline softmax{},norm{},gate{},residual{},patch{},prepare{},join{},rowUpdate{},expert{},groupNorm{},route{},crossScores{},crossValues{};
    VkDescriptorSet mlpSets[6]{};
    VkDescriptorSet attentionSets[3]{};
    VkFence fence{};
    VkDescriptorPool descriptors{};
    VkDescriptorSet set{};
    VkCommandPool commands{};
    VkCommandBuffer cmd{};
    VkPhysicalDeviceMemoryProperties memory{};
    VkPhysicalDeviceProperties properties{};
    void init() {
        VkInstanceCreateInfo ic{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        check(vkCreateInstance(&ic,nullptr,&instance));
        uint32_t n=0;
        check(vkEnumeratePhysicalDevices(instance,&n,nullptr));
        if(!n)throw std::runtime_error("No Vulkan device");
        std::vector<VkPhysicalDevice> devices(n);
        check(vkEnumeratePhysicalDevices(instance,&n,devices.data()));
        uint32_t family=0;
        for(auto candidate:devices) {
            vkGetPhysicalDeviceQueueFamilyProperties(candidate,&n,nullptr);
            std::vector<VkQueueFamilyProperties> families(n);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate,&n,families.data());
            for(family=0;family<n;++family)if(families[family].queueFlags&VK_QUEUE_COMPUTE_BIT)break;
            if(family<n){physical=candidate;break;}
        }
        if(!physical)throw std::runtime_error("No Vulkan compute queue");
        vkGetPhysicalDeviceProperties(physical,&properties);
        vkGetPhysicalDeviceMemoryProperties(physical,&memory);
        float priority=1;
        VkDeviceQueueCreateInfo qc{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qc.queueFamilyIndex=family;qc.queueCount=1;qc.pQueuePriorities=&priority;
        VkDeviceCreateInfo dc{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dc.queueCreateInfoCount=1;dc.pQueueCreateInfos=&qc;
        check(vkCreateDevice(physical,&dc,nullptr,&device));
        vkGetDeviceQueue(device,family,0,&queue);
        VkDescriptorSetLayoutBinding entries[3]{};
        for(uint32_t i=0;i<3;++i)entries[i]={i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo bc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        bc.bindingCount=3;bc.pBindings=entries;
        check(vkCreateDescriptorSetLayout(device,&bc,nullptr,&bindings));
        VkPushConstantRange shape{VK_SHADER_STAGE_COMPUTE_BIT,0,16};
        VkPipelineLayoutCreateInfo lc{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        lc.setLayoutCount=1;lc.pSetLayouts=&bindings;lc.pushConstantRangeCount=1;lc.pPushConstantRanges=&shape;
        check(vkCreatePipelineLayout(device,&lc,nullptr,&layout));
        pipeline=createPipeline(shader,sizeof(shader));
        alignedPipeline=createPipeline(alignedShader,sizeof(alignedShader));
        softmax=createPipeline(softmaxShader,sizeof(softmaxShader));
        norm=createPipeline(normCode,sizeof(normCode));
        gate=createPipeline(gateCode,sizeof(gateCode));
        residual=createPipeline(residualCode,sizeof(residualCode));
        patch=createPipeline(patchCode,sizeof(patchCode));
        prepare=createPipeline(prepareCode,sizeof(prepareCode));
        join=createPipeline(joinCode,sizeof(joinCode));
        rowUpdate=createPipeline(rowCode,sizeof(rowCode));
        expert=createPipeline(expertCode,sizeof(expertCode));
        groupNorm=createPipeline(groupCode,sizeof(groupCode));
        route=createPipeline(routeCode,sizeof(routeCode));
        crossScores=createPipeline(crossScoreCode,sizeof(crossScoreCode));
        crossValues=createPipeline(crossValueCode,sizeof(crossValueCode));
        VkFenceCreateInfo fc{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        check(vkCreateFence(device,&fc,nullptr,&fence));
        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,30};
        VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dp.maxSets=10;dp.poolSizeCount=1;dp.pPoolSizes=&size;
        check(vkCreateDescriptorPool(device,&dp,nullptr,&descriptors));
        VkDescriptorSetAllocateInfo sa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        sa.descriptorPool=descriptors;sa.descriptorSetCount=1;sa.pSetLayouts=&bindings;
        check(vkAllocateDescriptorSets(device,&sa,&set));
        for(auto& s:attentionSets)check(vkAllocateDescriptorSets(device,&sa,&s));
        for(auto& s:mlpSets)check(vkAllocateDescriptorSets(device,&sa,&s));
        VkCommandPoolCreateInfo cp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cp.queueFamilyIndex=family;cp.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        check(vkCreateCommandPool(device,&cp,nullptr,&commands));
        VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ca.commandPool=commands;ca.commandBufferCount=1;ca.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        check(vkAllocateCommandBuffers(device,&ca,&cmd));
    }
    VkPipeline createPipeline(const uint32_t* code,size_t bytes){
        VkShaderModuleCreateInfo sc{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        sc.codeSize=bytes;sc.pCode=code;
        VkShaderModule module{};
        check(vkCreateShaderModule(device,&sc,nullptr,&module));
        VkComputePipelineCreateInfo pc{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pc.layout=layout;pc.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pc.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;pc.stage.module=module;pc.stage.pName="main";
        VkPipeline result{};
        auto status=vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&pc,nullptr,&result);
        vkDestroyShaderModule(device,module,nullptr);
        check(status);
        return result;
    }
    ~Device(){
        if(device){
            vkDestroyCommandPool(device,commands,nullptr);vkDestroyDescriptorPool(device,descriptors,nullptr);
            vkDestroyPipeline(device,pipeline,nullptr);vkDestroyPipeline(device,alignedPipeline,nullptr);
            for(auto p:{softmax,norm,gate,residual,patch,prepare,join,rowUpdate,expert,groupNorm,route,crossScores,crossValues})vkDestroyPipeline(device,p,nullptr);
            vkDestroyFence(device,fence,nullptr);
            vkDestroyPipelineLayout(device,layout,nullptr);vkDestroyDescriptorSetLayout(device,bindings,nullptr);
            vkDestroyDevice(device,nullptr);
        }
        if(instance)vkDestroyInstance(instance,nullptr);
    }
    void beginCommands(){
        check(vkResetCommandBuffer(cmd,0));
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(cmd,&begin));
    }
    void multiply(VkDescriptorSet descriptors,uint32_t rows,uint32_t width,uint32_t channels){
        bool aligned=rows%32==0 && channels%32==0 && width%16==0;
        uint32_t tile=aligned?32:16;
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,aligned?alignedPipeline:pipeline);
        vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,layout,0,1,&descriptors,0,nullptr);
        uint32_t dims[]={rows,width,channels};
        vkCmdPushConstants(cmd,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(dims),dims);
        vkCmdDispatch(cmd,(channels+tile-1)/tile,(rows+tile-1)/tile,1);
    }
    void finishCommands(){
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&barrier,0,nullptr,0,nullptr);
        check(vkEndCommandBuffer(cmd));
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};submit.commandBufferCount=1;submit.pCommandBuffers=&cmd;
        check(vkResetFences(device,1,&fence));
        check(vkQueueSubmit(queue,1,&submit,fence),"vkQueueSubmit");
        check(vkWaitForFences(device,1,&fence,VK_TRUE,UINT64_MAX),"vkWaitForFences");
    }
    void dispatch(uint32_t rows,uint32_t width,uint32_t channels,VkDescriptorSet descriptors){
        beginCommands();
        multiply(descriptors,rows,width,channels);
        finishCommands();
    }
    void computeBarrier(){
        VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        b.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
        b.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0,1,&b,0,nullptr,0,nullptr);
    }

};
struct StoragePressure {};
void checkStorage(VkResult status){
    if(status==VK_ERROR_OUT_OF_DEVICE_MEMORY || status==VK_ERROR_OUT_OF_HOST_MEMORY ||
       status==VK_ERROR_MEMORY_MAP_FAILED)throw StoragePressure{};
    check(status);
}
struct Buffer {
    Device& d;
    VkBuffer buffer{};
    VkDeviceMemory allocation{};
    void* data{};
    size_t capacity{};
    explicit Buffer(Device& dev):d(dev){}
    void reserve(size_t bytes,uint32_t binding=UINT32_MAX) {
        if(bytes<=capacity)return;
        release();

        VkBufferCreateInfo bc{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bc.size=bytes;bc.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        checkStorage(vkCreateBuffer(d.device,&bc,nullptr,&buffer));
        VkMemoryRequirements req;vkGetBufferMemoryRequirements(d.device,buffer,&req);
        constexpr auto flags=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        const auto preferred=binding==2 ? VK_MEMORY_PROPERTY_HOST_CACHED_BIT : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        uint32_t type=d.memory.memoryTypeCount;
        for(uint32_t i=0;i<d.memory.memoryTypeCount;++i){
            auto properties=d.memory.memoryTypes[i].propertyFlags;
            if(!(req.memoryTypeBits&(1u<<i)) || (properties&flags)!=flags)continue;
            if(type==d.memory.memoryTypeCount)type=i;
            if(properties&preferred){type=i;break;}
        }
        if(type==d.memory.memoryTypeCount)throw std::runtime_error("No coherent Vulkan storage");
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize=req.size;ai.memoryTypeIndex=type;
        checkStorage(vkAllocateMemory(d.device,&ai,nullptr,&allocation));
        checkStorage(vkBindBufferMemory(d.device,buffer,allocation,0));
        checkStorage(vkMapMemory(d.device,allocation,0,bytes,0,&data));
        VkDescriptorBufferInfo info{buffer,0,bytes};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet=d.set;write.dstBinding=binding;write.descriptorCount=1;
        write.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;write.pBufferInfo=&info;
        if(binding!=UINT32_MAX)vkUpdateDescriptorSets(d.device,1,&write,0,nullptr);
        capacity=bytes;
    }
    ~Buffer(){release();}
    void release(){
        if(!d.device)return;
        if(data)vkUnmapMemory(d.device,allocation);
        vkDestroyBuffer(d.device,buffer,nullptr);vkFreeMemory(d.device,allocation,nullptr);
        data=nullptr;buffer=VK_NULL_HANDLE;allocation=VK_NULL_HANDLE;capacity=0;
    }
};
// Destroy mapped scratch buffers before the device they belong to.
static VkDescriptorBufferInfo range(Buffer& buffer,size_t offset=0,size_t bytes=0){
    return {buffer.buffer,offset,bytes?bytes:buffer.capacity-offset};
}
static void bindRanges(Device& d,int index,VkDescriptorBufferInfo a,VkDescriptorBufferInfo b,VkDescriptorBufferInfo c){
    VkDescriptorBufferInfo info[]={a,b,c};
    VkWriteDescriptorSet writes[3]{};
    for(uint32_t i=0;i<3;++i){
        writes[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet=d.mlpSets[index];writes[i].dstBinding=i;
        writes[i].descriptorCount=1;writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo=info+i;
    }
    vkUpdateDescriptorSets(d.device,3,writes,0,nullptr);
}
void bind(Device& d,int index,Buffer& a,Buffer& b,Buffer& c){
    bindRanges(d,index,range(a),range(b),range(c));
}
struct Session {
    Device device;
    size_t storageLimit=0;
    // Bound compute per submission as well as bytes. The first small projection
    // measures the local driver before a 1024px MLP can occupy it for seconds.
    uint64_t macLimit=1ull<<30;
    int64_t dispatches=0;
    Buffer input{device},weight{device},result{device};
    uintptr_t mappedBase=0;
    size_t mappedBytes=0,cachedBytes=0,cacheBudget=0;
    std::map<std::pair<uintptr_t,size_t>,std::unique_ptr<Buffer>> cache;

    Buffer* cached(const float* data,size_t bytes){
        uintptr_t address=reinterpret_cast<uintptr_t>(data);
        if(address<mappedBase || address-mappedBase>mappedBytes ||
           bytes>mappedBytes-(address-mappedBase) || bytes>storageLimit)return nullptr;
        auto key=std::make_pair(address,bytes);
        auto it=cache.find(key);
        if(it!=cache.end())return it->second.get();
        if(bytes>cacheBudget-cachedBytes)return nullptr;
        try{
            auto entry=std::make_unique<Buffer>(device);
            entry->reserve(bytes);
            std::memcpy(entry->data,data,bytes);
            auto result=cache.emplace(key,std::move(entry)).first;
            cachedBytes+=bytes;
            return result->second.get();
        }catch(const StoragePressure&){return nullptr;}
    }
};
std::unique_ptr<Session> openSession(){
    auto session=std::make_unique<Session>();
    session->device.init();
    session->storageLimit=session->device.properties.limits.maxStorageBufferRange;
    VkDeviceSize heap=0;
    const auto& memory=session->device.memory;
    for(uint32_t i=0;i<memory.memoryTypeCount;++i)
        if(memory.memoryTypes[i].propertyFlags&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            heap=std::max(heap,memory.memoryHeaps[memory.memoryTypes[i].heapIndex].size);
    session->cacheBudget=std::min<VkDeviceSize>(512u<<20,heap/16);
    return session;
}
struct Gpu {
    std::unique_ptr<Session> session=openSession();
    int64_t previousDispatches=0;
    bool recovered=false;
    std::string diagnostic;
};
void project(Session& session,const float* x,const float* w,int tokens,int width,int out,float* y){
        auto& d=session.device;
        auto& input=session.input;auto& weight=session.weight;auto& result=session.result;
        int rows=0,channels=0;
        // Prefer the whole projection. If the descriptor range requires tiling,
        // minimize GPU submissions, then minimize repeated uploads of the input.
        for(;;){
            size_t elements=session.storageLimit/sizeof(float);
            int maxRows=std::min<size_t>(tokens,elements/width);
            size_t bestCalls=SIZE_MAX,bestUploads=SIZE_MAX;
            for(int r=maxRows;r>0;r=(r-1)/32*32){
                size_t perChannel=size_t(width)*r;
                int c=std::min<uint64_t>({static_cast<uint64_t>(out),elements/std::max(width,r),
                                          session.macLimit/perChannel});
                if(c<out && c>=32)c-=c%32;
                if(!c)continue;
                size_t shards=(out+c-1)/c;
                size_t calls=shards*((tokens+r-1)/r);
                size_t uploads=r==tokens?1:shards;
                if(calls<bestCalls || (calls==bestCalls && uploads<bestUploads)){
                    rows=r;channels=c;bestCalls=calls;bestUploads=uploads;
                }
                if(calls==1)break;
            }
            if(bestCalls==SIZE_MAX)throw std::runtime_error("no GPU tile fits the memory/work budget");
            size_t inputBytes=size_t(rows)*width*sizeof(float);
            size_t weightBytes=size_t(channels)*width*sizeof(float);
            size_t resultBytes=size_t(rows)*channels*sizeof(float);
            try{
                input.reserve(inputBytes,0);
                weight.reserve(weightBytes,1);
                result.reserve(resultBytes,2);
                break;
            }catch(const StoragePressure&){
                input.release();weight.release();result.release();
                session.storageLimit=std::max({inputBytes,weightBytes,resultBytes})/2;
            }
        }
        if(rows==tokens)std::memcpy(input.data,x,size_t(tokens)*width*sizeof(float));
        for(int c=0;c<out;c+=channels){
            int n=std::min(channels,out-c);
            const float* shard=w+size_t(c)*width;
            size_t bytes=size_t(n)*width*4;
            Buffer* cached=session.cached(shard,bytes);
            if(cached)bind(d,0,input,*cached,result);
            else std::memcpy(weight.data,shard,bytes);
            for(int r=0;r<tokens;r+=rows){
                int t=std::min(rows,tokens-r);
                if(rows!=tokens)std::memcpy(input.data,x+size_t(r)*width,size_t(t)*width*4);
                auto start=std::chrono::steady_clock::now();
                d.dispatch(t,width,n,cached?d.mlpSets[0]:d.set);
                uint64_t macs=uint64_t(t)*width*n;
                if(macs>=(1u<<24)){
                    auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                    // Target about 200 ms and grow gradually; ignore tiny probe kernels.
                    uint64_t estimate=static_cast<uint64_t>(macs*200.0/std::max(1.,ms));
                    session.macLimit=std::clamp<uint64_t>(estimate,1u<<24,std::min<uint64_t>(1ull<<38,session.macLimit*2));
                }
                ++session.dispatches;
                if(n==out){
                    std::memcpy(y+size_t(r)*out,result.data,size_t(t)*out*sizeof(float));
                    continue;
                }
                for(int i=0;i<t;++i)
                    std::memcpy(y+size_t(r+i)*out+c,static_cast<float*>(result.data)+size_t(i)*n,n*4);
            }
        }
        if(!std::all_of(y,y+size_t(tokens)*out,[](float f){return std::isfinite(f);}))
            throw std::runtime_error("non-finite GPU output");
}
}
extern "C" void* homeimg_gpu_open(){
    try{return new Gpu();}catch(...){return nullptr;}
}
extern "C" int64_t homeimg_gpu_dispatches(void* p){
    auto& gpu=*static_cast<Gpu*>(p);
    return gpu.previousDispatches+(gpu.session?gpu.session->dispatches:0);
}
extern "C" const char* homeimg_gpu_diagnostic(void* p){return static_cast<Gpu*>(p)->diagnostic.c_str();}
extern "C" void homeimg_gpu_close(void* p){delete static_cast<Gpu*>(p);}
extern "C" int homeimg_gpu_project(void* p,const float* x,const float* w,int tokens,int width,int out,float* y){
    if(!p || tokens<=0 || width<=0 || out<=0)return 0;
    auto& gpu=*static_cast<Gpu*>(p);
    if(!gpu.session)return 0;
    try{
        try{
            project(*gpu.session,x,w,tokens,width,out,y);
        }catch(const VulkanError& error){
            if(error.code!=VK_ERROR_DEVICE_LOST)throw;
            gpu.previousDispatches+=gpu.session->dispatches;
            gpu.session.reset();
            if(gpu.recovered)throw;
            gpu.recovered=true;
            gpu.diagnostic="VK_ERROR_DEVICE_LOST: retrying with smaller tiles";
            gpu.session=openSession();
            gpu.session->macLimit=1u<<26;
            project(*gpu.session,x,w,tokens,width,out,y);
            gpu.diagnostic="VK_ERROR_DEVICE_LOST: GPU session restored";
        }
        return 1;
    }catch(const std::exception& e){
        gpu.diagnostic=std::string(e.what())+" ["+std::to_string(tokens)+" x "+std::to_string(width)+
            " x "+std::to_string(out)+"]";
        return 0;
    }
}

// A single submission computes QK^T, stable row softmax and probabilities times V.
// The scratch matrix is tiled by query row; the key axis always spans the entire head.
static int gpuAttention(void* handle,const float* q,const float* k,const float* v,
                                    int tokens,int keys,int width,float scale,float* y,int causal,Buffer* joined=nullptr,int head=0,int dim=0,int side=0,int reference=0){
    if(!handle || keys<=0 || tokens<=0 || width<=0)return 0;
    auto& gpu=*static_cast<Gpu*>(handle);
    if(!gpu.session)return 0;
    try{
        auto& s=*gpu.session;
        auto& d=s.device;
        size_t bytes=size_t(tokens)*width*sizeof(float);
        size_t align=d.properties.limits.minStorageBufferOffsetAlignment;
        size_t keyBytes=size_t(keys)*width*sizeof(float);
        size_t keyOffset=(bytes+align-1)/align*align;
        size_t valueOffset=(keyOffset+keyBytes+align-1)/align*align;
        if((32*width*sizeof(float))%align || valueOffset+keyBytes>s.storageLimit)return 0;
        int tile=std::min<uint64_t>(tokens,std::max<uint64_t>(32,s.macLimit/(2ull*keys*width)));
        tile=std::min(tile,512);
        if(tile>=32)tile-=tile%32;
        s.input.reserve(valueOffset+keyBytes,0);
        s.weight.reserve(size_t(tile)*keys*4,1);
        s.result.reserve(size_t(tile)*width*4,2);
        if(!joined){
            auto* packed=static_cast<float*>(s.input.data);
            std::memcpy(packed,q,bytes);
            std::memcpy(packed+keyOffset/4,k,keyBytes);
            auto* vt=packed+valueOffset/4;
            for(int r=0;r<keys;r+=32)
                for(int c=0;c<width;++c)
                    for(int i=0;i<std::min(32,keys-r);++i)vt[size_t(c)*keys+r+i]=v[size_t(r+i)*width+c];
        }
        auto bind=[&](int set,int binding,VkBuffer buffer,size_t offset,size_t range){
            VkDescriptorBufferInfo info{buffer,offset,range};
            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet=d.attentionSets[set];write.dstBinding=binding;write.descriptorCount=1;
            write.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;write.pBufferInfo=&info;
            vkUpdateDescriptorSets(d.device,1,&write,0,nullptr);
        };
        bind(0,1,s.input.buffer,keyOffset,keyBytes);
        bind(0,2,s.weight.buffer,0,size_t(tile)*keys*4);
        bind(1,0,s.weight.buffer,0,size_t(tile)*keys*4);
        bind(2,0,s.weight.buffer,0,size_t(tile)*keys*4);
        bind(2,1,s.input.buffer,valueOffset,keyBytes);
        bind(2,2,s.result.buffer,0,size_t(tile)*width*4);
        if(joined)::bind(d,1,s.result,s.weight,*joined);
        for(int r=0;r<tokens;r+=tile){
            int n=std::min(tile,tokens-r);
            bind(0,0,s.input.buffer,size_t(r)*width*4,size_t(n)*width*4);
            d.beginCommands();
            if(joined && r==0){
                int shape[]={side,dim,head,reference};
                vkCmdBindPipeline(d.cmd,VK_PIPELINE_BIND_POINT_COMPUTE,d.prepare);
                vkCmdBindDescriptorSets(d.cmd,VK_PIPELINE_BIND_POINT_COMPUTE,d.layout,0,1,&d.mlpSets[0],0,nullptr);
                vkCmdPushConstants(d.cmd,d.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(shape),shape);
                vkCmdDispatch(d.cmd,tokens,1,1);
                d.computeBarrier();
                ++s.dispatches;
            }
            d.multiply(d.attentionSets[0],n,width,keys);
            d.computeBarrier();
            vkCmdBindPipeline(d.cmd,VK_PIPELINE_BIND_POINT_COMPUTE,d.softmax);
            vkCmdBindDescriptorSets(d.cmd,VK_PIPELINE_BIND_POINT_COMPUTE,d.layout,0,1,&d.attentionSets[1],0,nullptr);
            struct {uint32_t keys;float scale;int queryOffset;} params{uint32_t(keys),scale,causal?r:-1};
            vkCmdPushConstants(d.cmd,d.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(params),&params);
            vkCmdDispatch(d.cmd,n,1,1);
            d.computeBarrier();
            d.multiply(d.attentionSets[2],n,keys,width);
            if(joined){
                d.computeBarrier();
                vkCmdBindPipeline(d.cmd,VK_PIPELINE_BIND_POINT_COMPUTE,d.join);
                vkCmdBindDescriptorSets(d.cmd,VK_PIPELINE_BIND_POINT_COMPUTE,d.layout,0,1,&d.mlpSets[1],0,nullptr);
                int shape[]={dim,head,r};
                vkCmdPushConstants(d.cmd,d.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(shape),shape);
                vkCmdDispatch(d.cmd,n,1,1);
                d.computeBarrier();
            }
            d.finishCommands();
            s.dispatches+=joined?4:3;
            if(!joined)std::memcpy(y+size_t(r)*width,s.result.data,size_t(n)*width*4);
        }
        return joined || std::all_of(y,y+bytes/4,[](float value){return std::isfinite(value);});
    }catch(const std::exception& error){gpu.diagnostic=error.what();return 0;}
    catch(const StoragePressure&){gpu.diagnostic="Vulkan attention allocation failed";return 0;}
}

extern "C" int homeimg_gpu_masked_attention(void* handle,const float* q,const float* k,const float* v,
                                    int tokens,int keys,int width,float scale,float* y,int causal){
    return gpuAttention(handle,q,k,v,tokens,keys,width,scale,y,causal);
}
// Raw QKV and the concatenated head output stay in their device buffers.
static bool spatialHeads(Gpu& gpu,Buffer& raw,Buffer& weights,Buffer& joined,
                         int side,int heads,int ref,float scale){
    auto& s=*gpu.session;
    int tokens=side*side;
    size_t headBytes=size_t(tokens)*128*4;
    if(headBytes%s.device.properties.limits.minStorageBufferOffsetAlignment)return false;
    s.input.reserve(headBytes*3,0);
    bind(s.device,0,raw,weights,s.input);
    for(int head=0;head<heads;++head){
        if(!gpuAttention(&gpu,nullptr,nullptr,nullptr,tokens,tokens,128,scale,nullptr,0,
                         &joined,head,heads*128,side,ref))return false;
    }
    return true;
}
extern "C" int homeimg_gpu_spatial(void* handle,const float* qkv,int side,int heads,int ref,float scale,
                                   const float* nq,const float* nk,float* output){
    auto& gpu=*static_cast<Gpu*>(handle);
    if(!gpu.session || side<1 || heads<1)return 0;
    auto& s=*gpu.session;
    auto& d=s.device;
    int tokens=side*side,dim=heads*128;
    size_t bytes=size_t(tokens)*dim*4;
    size_t headBytes=size_t(tokens)*128*4;
    if(bytes*3>s.storageLimit || headBytes%d.properties.limits.minStorageBufferOffsetAlignment)return 0;
    try{
        Buffer raw(d),weights(d),joined(d);
        raw.reserve(bytes*3);weights.reserve(256*4);joined.reserve(bytes);
        std::memcpy(raw.data,qkv,bytes*3);std::memcpy(weights.data,nq,128*4);
        std::memcpy(static_cast<float*>(weights.data)+128,nk,128*4);
        if(!spatialHeads(gpu,raw,weights,joined,side,heads,ref,scale))return 0;
        std::memcpy(output,joined.data,bytes);
        return std::all_of(output,output+bytes/4,[](float v){return std::isfinite(v);});
    }catch(...){return 0;}
}
extern "C" int homeimg_gpu_self_block(void* handle,const homeimg_self_inputs* job){
    if(!handle || !job || job->side<1 || job->side>64 || job->heads<1 || job->heads>64 ||
       job->branches<1 || job->branches>2)return 0;
    auto& gpu=*static_cast<Gpu*>(handle);
    if(!gpu.session)return 0;
    auto& s=*gpu.session;
    auto& d=s.device;
    const int tokens=job->side*job->side,width=job->heads*128;
    const size_t rowBytes=size_t(width)*4,activation=tokens*rowBytes,weightBytes=width*rowBytes;
    if(3*std::max(activation,weightBytes)>s.storageLimit ||
       (size_t(tokens)*128*4)%d.properties.limits.minStorageBufferOffsetAlignment)return 0;
    try{
        Buffer localQkv(d),localOut(d),input(d),raw(d),joined(d),tileBuffer(d),params(d),qkNorm(d);
        auto stage=[&](Buffer& local,const float* src,size_t bytes)->Buffer&{
            if(auto* cached=s.cached(src,bytes))return *cached;
            local.reserve(bytes);
            std::memcpy(local.data,src,bytes);
            return local;
        };
        Buffer& qkvWeight=stage(localQkv,job->qkv_weight,3*weightBytes);
        Buffer& outWeight=stage(localOut,job->out_weight,weightBytes);
        input.reserve(activation);raw.reserve(3*activation);joined.reserve(activation);
        params.reserve(4*rowBytes);qkNorm.reserve(256*4);
        size_t slot=0;
        for(auto src:{job->norm,job->gain,job->bias,job->gate})
            std::memcpy(static_cast<float*>(params.data)+width*slot++,src,rowBytes);
        std::memcpy(qkNorm.data,job->q_norm,128*4);
        std::memcpy(static_cast<float*>(qkNorm.data)+128,job->k_norm,128*4);
        auto rowsPerTile=[&](int outputs){
            int rows=std::min<uint64_t>(tokens,std::max<uint64_t>(32,s.macLimit/(uint64_t(width)*outputs)));
            return rows>=32?rows/32*32:rows;
        };
        int qkvRows=rowsPerTile(3*width),outRows=rowsPerTile(width);
        tileBuffer.reserve(rowBytes*std::max(qkvRows,outRows));
        if(size_t(32)*rowBytes%d.properties.limits.minStorageBufferOffsetAlignment)return 0;
        auto elementwise=[&](int set,VkPipeline pipeline,int groupsX,int groupsY){
            vkCmdBindPipeline(d.cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);
            vkCmdBindDescriptorSets(d.cmd,VK_PIPELINE_BIND_POINT_COMPUTE,d.layout,0,1,&d.mlpSets[set],0,nullptr);
            vkCmdPushConstants(d.cmd,d.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(width),&width);
            vkCmdDispatch(d.cmd,groupsX,groupsY,1);
        };
        for(int branch=0;branch<job->branches;++branch){
            size_t base=size_t(branch)*tokens*width;
            std::memcpy(input.data,job->input+base,activation);
            for(int r=0;r<tokens;r+=qkvRows){
                int n=std::min(qkvRows,tokens-r);
                size_t offset=r*rowBytes,bytes=n*rowBytes;
                bindRanges(d,2,range(input,offset,bytes),range(params),range(tileBuffer,0,bytes));
                bindRanges(d,3,range(tileBuffer,0,bytes),range(qkvWeight),range(raw,3*offset,3*bytes));
                d.beginCommands();
                elementwise(2,d.norm,n,1);
                d.computeBarrier();
                d.multiply(d.mlpSets[3],n,width,3*width);
                d.computeBarrier();
                d.finishCommands();s.dispatches+=2;
            }
            if(!spatialHeads(gpu,raw,qkNorm,joined,job->side,job->heads,job->reference,
                             job->attention_scale))return 0;
            for(int r=0;r<tokens;r+=outRows){
                int n=std::min(outRows,tokens-r);
                size_t offset=r*rowBytes,bytes=n*rowBytes;
                bindRanges(d,2,range(joined,offset,bytes),range(outWeight),range(tileBuffer,0,bytes));
                bindRanges(d,3,range(input,offset,bytes),range(params),range(tileBuffer,0,bytes));
                d.beginCommands();
                d.multiply(d.mlpSets[2],n,width,width);
                d.computeBarrier();
                elementwise(3,d.residual,(width+63)/64,n);
                d.finishCommands();s.dispatches+=2;
                std::memcpy(job->output+base+size_t(r)*width,tileBuffer.data,bytes);
            }
        }
        return std::all_of(job->output,job->output+size_t(job->branches)*tokens*width,
                           [](float v){return std::isfinite(v);});
    }catch(const std::exception& error){gpu.diagnostic=error.what();return 0;}
    catch(const StoragePressure&){gpu.diagnostic="Vulkan self-attention block allocation failed";return 0;}
}
// Cross-attention over packed heads. Each tile reads the same uploaded text K/V.
extern "C" int homeimg_gpu_cross_heads(void* handle,const float* q,const float* kv,int rows,
                                       int keys,int heads,float scale,float* y){
    if(!handle || rows<=0 || keys<=0 || keys>256 || heads<=0 || heads>64)return 0;
    auto& gpu=*static_cast<Gpu*>(handle);
    if(!gpu.session)return 0;
    try{
        auto& s=*gpu.session;
        auto& d=s.device;
        size_t rowBytes=size_t(heads)*128*4,scoreRow=size_t(heads)*keys*4;
        size_t contextBytes=2*keys*rowBytes;
        if(contextBytes>s.storageLimit)return 0;
        int tile=std::min<uint64_t>({uint64_t(rows),65535,s.storageLimit/std::max(rowBytes,scoreRow),
                               std::max<uint64_t>(1,s.macLimit/(2ull*128*keys*heads))});
        if(!tile)return 0;
        Buffer scores(d);
        scores.reserve(tile*scoreRow);
        s.weight.reserve(contextBytes,1);s.input.reserve(tile*rowBytes,0);s.result.reserve(tile*rowBytes,2);
        std::memcpy(s.weight.data,kv,contextBytes);
        bind(d,0,s.input,s.weight,scores);bind(d,1,scores,s.weight,s.result);
        for(int r=0;r<rows;r+=tile){
            int n=std::min(tile,rows-r);
            size_t offset=size_t(r)*heads*128,bytes=n*rowBytes;
            std::memcpy(s.input.data,q+offset,bytes);
            struct {int rows,keys,heads;float scale;} shape{n,keys,heads,scale};
            d.beginCommands();
            vkCmdPushConstants(d.cmd,d.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(shape),&shape);
            vkCmdBindPipeline(d.cmd,VK_PIPELINE_BIND_POINT_COMPUTE,d.crossScores);
            vkCmdBindDescriptorSets(d.cmd,VK_PIPELINE_BIND_POINT_COMPUTE,d.layout,0,1,&d.mlpSets[0],0,nullptr);
            vkCmdDispatch(d.cmd,(keys+7)/8,(n+7)/8,heads);
            d.computeBarrier();
            vkCmdBindPipeline(d.cmd,VK_PIPELINE_BIND_POINT_COMPUTE,d.crossValues);
            vkCmdBindDescriptorSets(d.cmd,VK_PIPELINE_BIND_POINT_COMPUTE,d.layout,0,1,&d.mlpSets[1],0,nullptr);
            vkCmdDispatch(d.cmd,n,heads,1);
            d.finishCommands();s.dispatches+=2;
            std::memcpy(y+offset,s.result.data,bytes);
        }
        return std::all_of(y,y+size_t(rows)*heads*128,[](float v){return std::isfinite(v);});
    }catch(const std::exception& e){gpu.diagnostic=e.what();return 0;}
    catch(const StoragePressure&){gpu.diagnostic="Vulkan cross-attention allocation failed";return 0;}
}
extern "C" int homeimg_gpu_cross(void* handle,const float* q,const float* k,const float* v,
                                 int tokens,int keys,int width,float scale,float* y){
    return homeimg_gpu_masked_attention(handle,q,k,v,tokens,keys,width,scale,y,0);
}
extern "C" int homeimg_gpu_attention(void* handle,const float* q,const float* k,const float* v,
                                    int tokens,int width,float scale,float* y){
    return homeimg_gpu_cross(handle,q,k,v,tokens,tokens,width,scale,y);
}

// Fused Aurora feed-forward block. Stage each weight once for all query-row tiles;
// keep RMSNorm, both SwiGLU inputs, the hidden activation and residual on Vulkan.
extern "C" int homeimg_gpu_mlp(void* handle,float* x,int rows,int width,int hidden,const float* wgu,
        const float* wd,const float* norm,const float* scale,const float* bias,const float* strength){
    if(!handle || rows<=0 || width<=0 || hidden<=0)return 0;
    auto& gpu=*static_cast<Gpu*>(handle);
    if(!gpu.session)return 0;
    try{
        auto& s=*gpu.session;
        auto& d=s.device;
        size_t wbytes=size_t(width)*hidden*sizeof(float);
        if(wbytes>s.storageLimit)return 0;
        int tile=std::min<uint64_t>(rows,std::max<uint64_t>(32,s.macLimit/(3ull*width*hidden)));
        if(tile<rows)tile-=tile%32;
        size_t inner=size_t(tile)*hidden*4,outer=size_t(tile)*width*4;
        if(std::max(inner,outer)>s.storageLimit)return 0;
        Buffer localGate(d),localUp(d),localDown(d),params(d),up(d);
        auto stage=[&](Buffer& local,const float* source)->Buffer&{
            if(auto* cached=s.cached(source,wbytes))return *cached;
            local.reserve(wbytes);
            std::memcpy(local.data,source,wbytes);
            return local;
        };
        Buffer& wg=stage(localGate,wgu);
        Buffer& wu=stage(localUp,wgu+wbytes/4);
        Buffer& down=stage(localDown,wd);
        up.reserve(inner);params.reserve(4ull*width*4);
        s.input.reserve(outer,0);s.weight.reserve(inner,1);s.result.reserve(outer,2);
        int slot=0;
        for(auto ptr:{norm,scale,bias,strength})
            std::memcpy(static_cast<float*>(params.data)+size_t(slot++)*width,ptr,width*4);
        bind(d,0,s.input,params,s.result);
        bind(d,1,s.result,wg,s.weight);
        bind(d,2,s.result,wu,up);
        bind(d,3,s.weight,up,s.input); // binding 2 unused by the gate shader
        bind(d,4,s.weight,down,s.result);
        bind(d,5,s.input,params,s.result);
        auto elementwise=[&](int set,VkPipeline pipeline,int cols,uint32_t groupsX,uint32_t groupsY){
            vkCmdBindPipeline(d.cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);
            vkCmdBindDescriptorSets(d.cmd,VK_PIPELINE_BIND_POINT_COMPUTE,d.layout,0,1,&d.mlpSets[set],0,nullptr);
            vkCmdPushConstants(d.cmd,d.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(cols),&cols);
            vkCmdDispatch(d.cmd,groupsX,groupsY,1);
        };
        // Commit only after every tile succeeded, so CPU retry sees the original x.
        std::vector<float> output(size_t(rows)*width);
        for(int first=0;first<rows;first+=tile){
            int n=std::min(tile,rows-first);
            size_t bytes=size_t(n)*width*4;
            std::memcpy(s.input.data,x+size_t(first)*width,bytes);
            d.beginCommands();
            elementwise(0,d.norm,width,n,1);
            d.computeBarrier();
            d.multiply(d.mlpSets[1],n,width,hidden);
            d.multiply(d.mlpSets[2],n,width,hidden);
            d.computeBarrier();
            elementwise(3,d.gate,hidden,(hidden+63)/64,n);
            d.computeBarrier();
            d.multiply(d.mlpSets[4],n,hidden,width);
            d.computeBarrier();
            elementwise(5,d.residual,width,(width+63)/64,n);
            d.finishCommands();
            s.dispatches+=6;
            std::memcpy(output.data()+size_t(first)*width,s.result.data,bytes);
        }
        if(!std::all_of(output.begin(),output.end(),[](float f){return std::isfinite(f);}))return 0;
        std::memcpy(x,output.data(),output.size()*sizeof(float));
        return 1;
    }catch(const std::exception& error){gpu.diagnostic=error.what();return 0;}
    catch(const StoragePressure&){gpu.diagnostic="Vulkan MLP allocation failed";return 0;}
}

extern "C" int homeimg_gpu_conv(void* handle,const float* x,const float* w,int side,int channels,
                                 int filters,int scale,float* y){
    auto& gpu=*static_cast<Gpu*>(handle);
    if(!gpu.session)return 0;
    try{
        auto& s=*gpu.session;
        auto& d=s.device;
        const int outSide=side*scale,columns=channels*9;
        size_t weightBytes=size_t(columns)*filters*4;
        if(weightBytes>s.storageLimit)return 0;
        size_t maxRows=std::min<size_t>({size_t(outSide)*outSide,8192,s.storageLimit/(4*std::max(columns,filters)),
                                         std::max<uint64_t>(32,s.macLimit/(uint64_t(columns)*filters))});
        if(maxRows>=32)maxRows-=maxRows%32;
        if(!maxRows)return 0;
        Buffer patches(d);
        patches.reserve(maxRows*columns*4);
        s.weight.reserve(weightBytes,1);
        s.result.reserve(maxRows*filters*4,2);
        std::memcpy(s.weight.data,w,weightBytes);
        for(int first=0;first<outSide*outSide;first+=maxRows){
            int rows=std::min<int>(maxRows,outSide*outSide-first);
            int start=std::max(0,first/outSide-1)/scale;
            int end=std::min(outSide-1,(first+rows-1)/outSide+1)/scale;
            size_t bytes=size_t(end-start+1)*side*channels*4;
            if(bytes>s.storageLimit)return 0;
            s.input.reserve(bytes,0);
            std::memcpy(s.input.data,x+size_t(start)*side*channels,bytes);
            bind(d,0,s.input,s.weight,patches);bind(d,1,patches,s.weight,s.result);
            int shape[]={side,scale,channels,first};
            d.beginCommands();
            vkCmdBindPipeline(d.cmd,VK_PIPELINE_BIND_POINT_COMPUTE,d.patch);
            vkCmdBindDescriptorSets(d.cmd,VK_PIPELINE_BIND_POINT_COMPUTE,d.layout,0,1,&d.mlpSets[0],0,nullptr);
            vkCmdPushConstants(d.cmd,d.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(shape),shape);
            vkCmdDispatch(d.cmd,(channels+63)/64,rows,1);
            d.computeBarrier();
            d.multiply(d.mlpSets[1],rows,columns,filters);
            d.finishCommands();
            s.dispatches+=2;
            std::memcpy(y+size_t(first)*filters,s.result.data,size_t(rows)*filters*4);
        }
        return std::all_of(y,y+size_t(outSide)*outSide*filters,[](float f){return std::isfinite(f);});
    }catch(...){return 0;}
}

extern "C" int homeimg_gpu_rows(void* handle,const homeimg_rows* op){
    auto& gpu=*static_cast<Gpu*>(handle);
    if(!gpu.session)return 0;
    auto& s=*gpu.session;
    auto& d=s.device;
    try{
        int width=op->cols;
        if(op->kind==HOMEIMG_ROUTES){
            if(width!=128)return 0;
            size_t count=size_t(op->rows)*128;
            s.input.reserve(count*4,0);s.weight.reserve(128*4,1);s.result.reserve(op->rows*16*4,2);
            std::memcpy(s.input.data,op->x,count*4);std::memcpy(s.weight.data,op->bias,128*4);
            d.beginCommands();vkCmdBindPipeline(d.cmd,VK_PIPELINE_BIND_POINT_COMPUTE,d.route);
            vkCmdBindDescriptorSets(d.cmd,VK_PIPELINE_BIND_POINT_COMPUTE,d.layout,0,1,&d.set,0,nullptr);
            vkCmdDispatch(d.cmd,op->rows,1,1);d.finishCommands();++s.dispatches;
            std::memcpy(op->out,s.result.data,op->rows*16*4);
            return true;
        }
        size_t rowBytes=size_t(width)*4;
        int tile=std::min<uint64_t>({uint64_t(op->rows),65535,s.storageLimit/(3*rowBytes)});
        if(!tile)return 0;
        s.input.reserve(tile*rowBytes,0);s.result.reserve(tile*rowBytes,2);
        size_t extraRows=op->kind==HOMEIMG_RMS?3:op->kind==HOMEIMG_RESIDUAL?tile+1:
                         op->cfg==1?tile:2*tile;
        if(op->kind==HOMEIMG_BIAS || op->kind==HOMEIMG_SILU || op->kind==HOMEIMG_TEXT_ROPE)extraRows=1;
        if(op->kind==HOMEIMG_SWIGLU)extraRows=tile;
        s.weight.reserve(extraRows*rowBytes,1);
        float* params=static_cast<float*>(s.weight.data);
        if(op->kind==HOMEIMG_BIAS)std::memcpy(params,op->bias,rowBytes);
        if(op->kind==HOMEIMG_RMS){
            std::memcpy(params,op->norm,rowBytes);
            if(op->gain)std::memcpy(params+width,op->gain,rowBytes);else std::fill_n(params+width,width,0.f);
            if(op->bias)std::memcpy(params+2*width,op->bias,rowBytes);else std::fill_n(params+2*width,width,0.f);
        }else if(op->kind==HOMEIMG_RESIDUAL){
            if(op->gain)std::memcpy(params,op->gain,rowBytes);else std::fill_n(params,width,1.f);
        }
        std::vector<float> retry;
        float* dst=op->out;
        if(dst==op->x){retry.resize(size_t(op->rows)*width);dst=retry.data();}
        for(int r=0;r<op->rows;r+=tile){
            int n=std::min(tile,op->rows-r);
            size_t offset=size_t(r)*width;
            std::memcpy(s.input.data,op->x+offset,n*rowBytes);
            if(op->kind==HOMEIMG_SWIGLU)std::memcpy(params,op->extra+offset,n*rowBytes);
            if(op->kind==HOMEIMG_RESIDUAL)std::memcpy(params+width,op->extra+offset,n*rowBytes);
            if(op->kind==HOMEIMG_EULER){
                std::memcpy(params,op->extra+offset,n*rowBytes);
                if(op->cfg!=1)std::memcpy(params+size_t(n)*width,op->negative+offset,n*rowBytes);
            }
            struct {uint32_t cols,kind;float dt,cfg;} pc{uint32_t(width),uint32_t(op->kind),op->kind==HOMEIMG_TEXT_ROPE?float(r):op->dt,op->cfg};
            d.beginCommands();
            vkCmdBindPipeline(d.cmd,VK_PIPELINE_BIND_POINT_COMPUTE,op->kind==HOMEIMG_RMS?d.norm:d.rowUpdate);
            vkCmdBindDescriptorSets(d.cmd,VK_PIPELINE_BIND_POINT_COMPUTE,d.layout,0,1,&d.set,0,nullptr);
            vkCmdPushConstants(d.cmd,d.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(pc),&pc);
            if(op->kind==HOMEIMG_RMS)vkCmdDispatch(d.cmd,n,1,1);
            else vkCmdDispatch(d.cmd,(width+63)/64,n,1);
            d.finishCommands();++s.dispatches;
            std::memcpy(dst+offset,s.result.data,n*rowBytes);
        }
        if(!retry.empty())std::memcpy(op->out,dst,retry.size()*4);
        return 1;
    }catch(...){return 0;}
}

extern "C" int homeimg_gpu_expert(void* handle,const float* x,const int8_t* weights,const float* scales,
                                  int rows,int width,int cols,float* y){
    auto& gpu=*static_cast<Gpu*>(handle);
    if(!gpu.session || cols%4 || rows<=0)return 0;
    try{
        auto& s=*gpu.session;
        auto& d=s.device;
        size_t packed=size_t(width)*cols;
        size_t inputBytes=size_t(rows)*width*4,outputBytes=size_t(rows)*cols*4;
        if(std::max({packed+cols*4,inputBytes,outputBytes})>s.storageLimit)return 0;
        s.input.reserve(inputBytes,0);
        s.weight.reserve(packed+cols*4,1);
        s.result.reserve(outputBytes,2);
        std::memcpy(s.input.data,x,inputBytes);
        std::memcpy(s.weight.data,weights,packed);
        std::memcpy(static_cast<uint8_t*>(s.weight.data)+packed,scales,cols*4);
        int shape[]={width,cols};
        d.beginCommands();
        vkCmdBindPipeline(d.cmd,VK_PIPELINE_BIND_POINT_COMPUTE,d.expert);
        vkCmdBindDescriptorSets(d.cmd,VK_PIPELINE_BIND_POINT_COMPUTE,d.layout,0,1,&d.set,0,nullptr);
        vkCmdPushConstants(d.cmd,d.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(shape),shape);
        vkCmdDispatch(d.cmd,(cols+63)/64,rows,1);
        d.finishCommands();++s.dispatches;
        std::memcpy(y,s.result.data,outputBytes);
        return std::all_of(y,y+outputBytes/4,[](float f){return std::isfinite(f);});
    }catch(...){return 0;}
}

extern "C" int homeimg_gpu_group(void* handle,const float* x,const float* w,const float* bias,
                                  int rows,int cols,int silu,float* y){
    auto& gpu=*static_cast<Gpu*>(handle);
    if(!gpu.session || rows<=0 || cols<=0 || cols%32)return 0;
    try{
        auto& s=*gpu.session;
        auto& d=s.device;
        size_t rowBytes=size_t(cols)*4;
        int tile=std::min<size_t>({size_t(rows),32768,s.storageLimit/rowBytes});
        if(tile<rows)tile-=tile%256;
        if(!tile)return 0;
        Buffer moments(d);
        size_t statsBytes=size_t((rows+255)/256)*32*2*4;
        if(statsBytes>s.storageLimit)return 0;
        moments.reserve(statsBytes);
        s.input.reserve(tile*rowBytes,0);s.result.reserve(tile*rowBytes,2);
        s.weight.reserve(64*4+rowBytes*2,1);
        float* params=static_cast<float*>(s.weight.data);
        std::memcpy(params+64,w,rowBytes);std::memcpy(params+64+cols,bias,rowBytes);
        bind(d,0,s.input,s.result,moments);
        bind(d,1,moments,s.result,s.weight);
        auto dispatch=[&](VkDescriptorSet set,int mode,int n,int first,int gx,int gy){
            int shape[]={mode,n,cols,first};
            d.beginCommands();
            vkCmdBindPipeline(d.cmd,VK_PIPELINE_BIND_POINT_COMPUTE,d.groupNorm);
            vkCmdBindDescriptorSets(d.cmd,VK_PIPELINE_BIND_POINT_COMPUTE,d.layout,0,1,&set,0,nullptr);
            vkCmdPushConstants(d.cmd,d.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(shape),shape);
            vkCmdDispatch(d.cmd,gx,gy,1);d.computeBarrier();d.finishCommands();++s.dispatches;
        };
        for(int r=0;r<rows;r+=tile){
            int n=std::min(tile,rows-r);
            std::memcpy(s.input.data,x+size_t(r)*cols,n*rowBytes);
            dispatch(d.mlpSets[0],0,n,r/256,32,(n+255)/256);
        }
        dispatch(d.mlpSets[1],1,rows,0,32,1);
        for(int r=0;r<rows;r+=tile){
            int n=std::min(tile,rows-r);
            if(tile<rows)std::memcpy(s.input.data,x+size_t(r)*cols,n*rowBytes);
            dispatch(d.set,silu?3:2,n,0,n,1);
            std::memcpy(y+size_t(r)*cols,s.result.data,n*rowBytes);
        }
        return 1;
    }catch(...){return 0;}
}

extern "C" void homeimg_gpu_weights(void* handle,const void* base,size_t bytes){
    auto& gpu=*static_cast<Gpu*>(handle);
    if(!gpu.session)return;
    auto& s=*gpu.session;
    s.cache.clear();s.cachedBytes=0;
    s.mappedBase=reinterpret_cast<uintptr_t>(base);s.mappedBytes=bytes;
    if(!base){
        // The decoder uses a different set of weights and smaller convolution tiles.
        s.input.release();s.weight.release();s.result.release();
    }
}
