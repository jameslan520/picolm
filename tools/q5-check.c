#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
static float f16(uint16_t h){
    uint32_t s=(h>>15)&1, e=(h>>10)&0x1f, f=h&0x3ff;
    if(e==0) return (s?-1:1)*ldexpf(f,-24);
    if(e==31) return f? NAN : (s?-INFINITY:INFINITY);
    return (s?-1:1)*ldexpf(1024+f, e-25);
}
typedef struct { char name[64]; uint32_t qt; uint64_t d0,d1,off; } tgt;
int main(int argc,char**argv){
    const char *path = argc>1 ? argv[1] : "/root/.picolm/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    FILE*fp=fopen(path,"rb");
    fseek(fp,0,SEEK_END); long sz=ftell(fp); fseek(fp,0,SEEK_SET);
    uint8_t*base=malloc(sz); fread(base,1,sz,fp); fclose(fp);
    uint8_t*p=base;
    uint32_t magic,ver; memcpy(&magic,p,4);p+=4; memcpy(&ver,p,4);p+=4;
    uint64_t nt,nm; memcpy(&nt,p,8);p+=8; memcpy(&nm,p,8);p+=8;
    for(uint64_t i=0;i<nm;i++){
        uint64_t l; memcpy(&l,p,8);p+=8+l;
        uint32_t t; memcpy(&t,p,4);p+=4;
        int done=0;
        while(!done){
            if(t<=3&&t!=0&&t!=1&&t!=2){ /*2,3*/ p+=2;done=1; }
            else if(t<=1){p+=1;done=1;}
            else if(t==7){p+=1;done=1;}
            else if(t<=6){p+=4;done=1;}          /* 4,5,6 */
            else if(t==8){uint64_t sl;memcpy(&sl,p,8);p+=8+sl;done=1;}
            else if(t==9){uint32_t at;memcpy(&at,p,4);p+=4;uint64_t n;memcpy(&n,p,8);p+=8;
                for(uint64_t j=0;j<n;j++){
                    if(at==8){uint64_t sl;memcpy(&sl,p,8);p+=8+sl;}
                    else if(at<=1||at==7)p+=1;
                    else if(at<=3)p+=2;
                    else if(at<=6)p+=4;
                    else if(at>=10)p+=8;
                } done=1;}
            else if(t<=12){p+=8;done=1;}
            else {fprintf(stderr,"bad t=%u\n",t);return 1;}
        }
    }
    tgt found[4]; int nf=0;
    for(uint64_t i=0;i<nt;i++){
        uint64_t l; memcpy(&l,p,8);p+=8;
        char nm2[256]; memcpy(nm2,p,l); nm2[l]=0; p+=l;
        uint32_t nd; memcpy(&nd,p,4);p+=4;
        uint64_t d[4]={0};
        for(uint32_t j=0;j<nd;j++){memcpy(&d[j],p,8);p+=8;}
        uint32_t qt; memcpy(&qt,p,4);p+=4;
        uint64_t off; memcpy(&off,p,8);p+=8;
        if((strcmp(nm2,"token_embd.weight")==0||strcmp(nm2,"output.weight")==0)&&nf<4){
            strcpy(found[nf].name,nm2); found[nf].qt=qt;
            found[nf].d0=d[0]; found[nf].d1=d[1]; found[nf].off=off; nf++;
        }
    }
    uint64_t data_base = ((uint64_t)(p-base) + 31) & ~31ULL;
    fprintf(stderr,"tensor-info ends=%ld data_base=%llu\n",(long)(p-base),(unsigned long long)data_base);
    for(int k=0;k<nf;k++){
        tgt*t=&found[k];
        fprintf(stderr,"%s type=%u dims=%llu x %llu off=%llu\n",t->name,t->qt,
            (unsigned long long)t->d0,(unsigned long long)t->d1,(unsigned long long)t->off);
        uint8_t*tw=base+data_base+t->off;
        int nb=(int)(t->d0/32);
        if(t->qt==6){
            double sum=0,sum2=0; float mn=1e30,mx=-1e30; int cnt=0;
            for(int b=0;b<nb;b++){
                uint16_t dd; memcpy(&dd,tw+b*22,2);
                float sc=f16(dd);
                uint32_t qh; memcpy(&qh,tw+b*22+2,4);
                uint8_t*qs=tw+b*22+6;
                for(int j=0;j<32;j++){
                    uint8_t low=(j<16)?(qs[j]&0xF):(qs[j-16]>>4);
                    int q=(int)low|(int)(((qh>>j)&1u)<<4);
                    float v=sc*(q-16);
                    sum+=v;sum2+=(double)v*v; if(v<mn)mn=v; if(v>mx)mx=v; cnt++;
                }
            }
            printf("%s Q5_0 row0: n=%d min=%.5f max=%.5f mean=%.5f std=%.5f\n",
                t->name,cnt,mn,mx,sum/cnt,sqrt(sum2/cnt-(sum/cnt)*(sum/cnt)));
        } else if(t->qt==8){
            double sum=0,sum2=0; float mn=1e30,mx=-1e30; int cnt=0;
            for(int b=0;b<nb;b++){
                uint16_t dd; memcpy(&dd,tw+b*34,2);
                float sc=f16(dd);
                int8_t*qs=(int8_t*)(tw+b*34+2);
                for(int j=0;j<32;j++){float v=sc*qs[j];
                    sum+=v;sum2+=(double)v*v; if(v<mn)mn=v; if(v>mx)mx=v; cnt++;}
            }
            printf("%s Q8_0 row0: n=%d min=%.5f max=%.5f mean=%.5f std=%.5f\n",
                t->name,cnt,mn,mx,sum/cnt,sqrt(sum2/cnt-(sum/cnt)*(sum/cnt)));
        }
    }
    return 0;
}
