// Epoch49 component loader for a package accepted by the v1 manifest validator.
// Tensor views borrow the mapping and remain valid for the lifetime of the file.
#pragma once
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <nlohmann/json.hpp>

namespace homeimg {
struct TensorView {
    std::string dtype;
    std::vector<size_t> shape;
    const uint8_t* data;
    size_t bytes;
};

class MappedWeights {
    void* mapping = MAP_FAILED;
    size_t length = 0;
    std::map<std::string, TensorView> tensors;

    void index() {
        const auto* base = static_cast<const uint8_t*>(mapping);
        uint64_t header = 0;
        for (int i=7; i>=0; --i) header = (header << 8) | base[i];
        if (header > length-8 || header > (16u<<20)) throw std::runtime_error("invalid tensor header");
        const uint8_t* payload = base+8+header;
        const size_t capacity = length-8-header;
        const auto entries = nlohmann::json::parse(reinterpret_cast<const char*>(base+8),reinterpret_cast<const char*>(payload));
        static const std::map<std::string,size_t> widths{{"I8",1},{"BF16",2},{"F32",4},{"I64",8},{"BOOL",1}};
        std::map<size_t,size_t> spans;
        for (const auto& entry : entries.items()) {
            if (entry.key()=="__metadata__") continue;
            const auto& desc = entry.value();
            const auto dtype = desc.at("dtype").get<std::string>();
            const auto shape = desc.at("shape").get<std::vector<size_t>>();
            const auto range = desc.at("data_offsets").get<std::vector<size_t>>();
            size_t bytes = widths.at(dtype);
            for (size_t axis : shape) {
                if (!axis || bytes > capacity/axis) throw std::runtime_error("invalid tensor shape");
                bytes *= axis;
            }
            if (range.size()!=2 || range[0]>capacity || bytes>capacity-range[0] ||
                range[1]!=range[0]+bytes || !spans.emplace(range[0],range[1]).second)
                throw std::runtime_error("invalid tensor offsets");
            tensors.emplace(entry.key(),TensorView{dtype,shape,payload+range[0],bytes});
        }
        size_t cursor=0;
        for (auto span : spans) {
            if (span.first!=cursor) throw std::runtime_error("non-contiguous tensor offsets");
            cursor=span.second;
        }
        if (cursor!=capacity) throw std::runtime_error("unindexed tensor data");
    }
public:
    const void* base() const {return mapping;}
    size_t bytes() const {return length;}
    explicit MappedWeights(const std::string& path) {
        int fd=open(path.c_str(),O_RDONLY|O_CLOEXEC);
        if(fd<0) throw std::runtime_error("cannot open "+path);
        struct stat info{};
        if(fstat(fd,&info)==0 && S_ISREG(info.st_mode) && info.st_size>=8) {
            length=info.st_size;
            mapping=mmap(nullptr,length,PROT_READ,MAP_PRIVATE,fd,0);
        }
        close(fd);
        if(mapping==MAP_FAILED) throw std::runtime_error("cannot map "+path);
        try { index(); } catch (...) { munmap(mapping,length); throw; }
    }
    MappedWeights(const MappedWeights&)=delete;
    MappedWeights& operator=(const MappedWeights&)=delete;
    ~MappedWeights(){munmap(mapping,length);}
    const TensorView& at(const std::string& name) const { return tensors.at(name); }
    const auto& entries() const { return tensors; }
};
}

namespace homeimg {
enum class Component { TextEncoder, Transformer, Vae };

// Open components independently so callers can release the LM before denoising.
inline MappedWeights load_component(const std::string& verified_root, Component component) {
    const char* directory;
    switch (component) {
        case Component::TextEncoder: directory = "text_encoder"; break;
        case Component::Transformer: directory = "transformer"; break;
        case Component::Vae: directory = "vae"; break;
        default: throw std::invalid_argument("unknown HomeIMG component");
    }
    return MappedWeights(verified_root + "/" + directory + "/diffusion_pytorch_model.safetensors");
}
}
