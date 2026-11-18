// Tokenizer for the validated epoch49 tokenizer.json, with Python str.split whitespace.
#pragma once
#include <nlohmann/json.hpp>
#include <codecvt>
#include <locale>
#include <fstream>
#include <queue>
#include <map>
#include <tuple>

namespace homeimg {
struct Tokens {
    std::vector<int> ids = std::vector<int>(256,65535);
    std::vector<uint8_t> valid = std::vector<uint8_t>(256,0);
    int count=0;
};
class Tokenizer {
    std::map<std::string,int> vocab, special;
    std::map<std::pair<int,int>,std::pair<int,int>> merges;
    static bool space(char32_t c) {
        return (c>=9 && c<=13)||(c>=28 && c<=32)||c==133||c==160||c==5760||
            (c>=8192 && c<=8202)||c==8232||c==8233||c==8239||c==8287||c==12288;
    }
    void segment(const std::string& value, Tokens& out) const {
        if(value.empty() || out.count==256)return;
        std::wstring_convert<std::codecvt_utf8<char32_t>,char32_t> utf;
        std::vector<int> ids;
        auto chars=U"▁"+utf.from_bytes(value);
        const char hex[]="0123456789ABCDEF";
        for(auto c:chars) {
            std::string token=utf.to_bytes(c==32?U'▁':c);
            auto it=vocab.find(token);
            if(it!=vocab.end()) ids.push_back(it->second);
            else for(unsigned char b:token) {
                std::string key="<0x00>"; key[3]=hex[b>>4]; key[4]=hex[b&15];
                ids.push_back(vocab.at(key));
            }
        }
        using Candidate=std::tuple<int,int,int,int>; // rank, left, right, merged id
        std::priority_queue<Candidate,std::vector<Candidate>,std::greater<Candidate>> queue;
        std::vector<int> prev(ids.size()),next(ids.size());
        auto offer=[&](int i){
            if(i<0 || next[i]>=int(ids.size()))return;
            auto it=merges.find({ids[i],ids[next[i]]});
            if(it!=merges.end()) queue.emplace(it->second.first,i,next[i],it->second.second);
        };
        for(int i=0;i<int(ids.size());++i){ prev[i]=i-1; next[i]=i+1; offer(i); }
        while(!queue.empty()){
            auto [rank,i,j,result]=queue.top();queue.pop();
            if(ids[i]<0 || next[i]!=j || ids[j]<0)continue;
            auto it=merges.find({ids[i],ids[j]});
            if(it==merges.end() || it->second.first!=rank)continue;
            ids[i]=result;ids[j]=-1;next[i]=next[j];
            if(next[i]<int(ids.size()))prev[next[i]]=i;
            offer(prev[i]);offer(i);
        }
        for(int id:ids)if(id>=0 && out.count<256)out.ids[out.count++]=id;
    }
public:
    explicit Tokenizer(const std::string& path){
        std::ifstream input(path);auto json=nlohmann::json::parse(input);
        auto& model=json.at("model");vocab=model.at("vocab").get<decltype(vocab)>();
        int rank=0;
        for(auto& pair:model.at("merges")){
            std::string a=pair.at(0),b=pair.at(1);
            merges[{vocab.at(a),vocab.at(b)}]={rank++,vocab.at(a+b)};
        }
        for(auto& t:json.at("added_tokens"))special[t.at("content").get<std::string>()]=t.at("id");
    }
    Tokens encode(const std::string& prompt) const {
        std::wstring_convert<std::codecvt_utf8<char32_t>,char32_t> utf;
        std::u32string cleaned;
        bool pending=false;
        for(auto c:utf.from_bytes(prompt)){
            if(space(c)){pending=!cleaned.empty();continue;}
            if(pending)cleaned+=U' ';
            cleaned+=c;pending=false;
        }
        std::string text=utf.to_bytes(cleaned);
        Tokens out;out.ids[out.count++]=1;
        size_t start=0;
        while(start<text.size() && out.count<256){
            size_t end=text.size();std::string match;
            for(auto& s:special){
                auto pos=text.find(s.first,start);
                if(pos<end){end=pos;match=s.first;}
            }
            segment(text.substr(start,end-start),out);
            if(!match.empty() && out.count<256)out.ids[out.count++]=special.at(match);
            start=end+match.size();
        }
        std::fill_n(out.valid.begin(),out.count,1);
        return out;
    }
};
}
