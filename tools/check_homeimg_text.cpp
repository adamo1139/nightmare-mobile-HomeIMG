#include "../backend/homeimg/text_encoder.hpp"
#include <iostream>
int main(int argc,char** argv){
    if(argc!=3 && argc!=4)return 2;
    std::string root=argv[1];
    homeimg::Tokenizer tokenizer(root+"/tokenizer/tokenizer.json");
    homeimg::TextEncoder encoder(root);
    auto fixture=homeimg::MappedWeights(argv[2]);
    std::string prompt;std::getline(std::cin,prompt,'\0');
    auto tokens=tokenizer.encode(prompt);
    auto ids=reinterpret_cast<const int64_t*>(fixture.at("ids").data);
    if(!std::equal(tokens.ids.begin(),tokens.ids.end(),ids) ||
       !std::equal(tokens.valid.begin(),tokens.valid.end(),fixture.at("valid").data))return 1;
    homeimg::TextTrace trace;
    auto lm=encoder.trunk(tokens,argc==4?&trace:nullptr);
    auto context=encoder.refine(lm);
    bool ok=true;
    auto compare=[&](const homeimg::MappedWeights& data,const std::string& name,const auto& actual){
        const auto& view=data.at(name);
        if(view.dtype!="F32" || view.bytes!=size_t(actual.size())*sizeof(float))
            throw std::runtime_error("unexpected fixture dimensions for "+name);
        Eigen::Map<const homeimg::Matrix> ref(reinterpret_cast<const float*>(view.data),actual.rows(),actual.cols());
        auto err=(actual-ref).cwiseAbs().eval();
        std::cout<<name<<" max "<<err.maxCoeff()<<" rms "<<std::sqrt(err.array().square().mean())<<std::endl;
        ok &= (err.array() <= 1e-4f+1e-4f*ref.array().abs()).all();
    };
    compare(fixture,"lm",lm);
    compare(fixture,"context",context);
    if(argc==4){
        homeimg::MappedWeights layers(argv[3]);
        compare(layers,"attention",trace.first_attention);
        compare(layers,"dense",trace.first_dense);
        compare(layers,"experts",trace.first_experts);
        compare(layers,"moe",trace.first_moe);
    }
    return ok?0:1;
}
