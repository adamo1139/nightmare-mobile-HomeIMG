// Print header fields and a digest of each mapped view for comparison to safetensors.
#include "../backend/homeimg/safetensors.hpp"
#include <openssl/sha.h>
#include <iomanip>
#include <iostream>

int main(int argc,char** argv) {
    if(argc!=3) return 2;
    try {
        const std::map<std::string,homeimg::Component> components{
            {"text_encoder", homeimg::Component::TextEncoder},
            {"transformer", homeimg::Component::Transformer}, {"vae", homeimg::Component::Vae}};
        auto weights = homeimg::load_component(argv[1], components.at(argv[2]));
        for(const auto& entry:weights.entries()) {
            const auto& view=entry.second;
            unsigned char digest[SHA256_DIGEST_LENGTH];
            SHA256(view.data,view.bytes,digest);
            std::cout<<entry.first<<" "<<view.dtype<<" "<<nlohmann::json(view.shape).dump()<<" ";
            for(auto byte:digest) std::cout<<std::hex<<std::setw(2)<<std::setfill('0')<<int(byte);
            std::cout<<"\n";
        }
    } catch(const std::exception& e) { std::cerr<<e.what()<<"\n"; return 1; }
}
