#pragma once
#include "aurora.hpp"
#include "safetensors.hpp"

namespace homeimg {
// Keep a read-only file-backed mapping; Eigen only borrows a weight for an operation.
class TransformerWeights {
    MappedWeights file;
public:
    const MappedWeights& mapping() const {return file;}
    explicit TransformerWeights(const std::string& root):file(load_component(root,Component::Transformer)){}
    Eigen::Map<const Matrix> at(const std::string& name) const {
        const auto& t=file.at(name);
        if(t.dtype!="F32" || t.shape.empty() || t.shape.size()>2)
            throw std::runtime_error("invalid transformer weight "+name);
        return {reinterpret_cast<const float*>(t.data),static_cast<Eigen::Index>(t.shape[0]),
                static_cast<Eigen::Index>(t.shape.size()==1?1:t.shape[1])};
    }
};
inline TransformerWeights load_transformer(const std::string& root){return TransformerWeights(root);}
}
