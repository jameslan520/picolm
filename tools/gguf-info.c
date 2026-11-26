#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
static uint8_t *base, *p, *end;
static void chk(size_t n){ if(p+n>end){ fprintf(stderr,"OOB off=%zu need=%zu\n",(size_t)(p-base),n); exit(1);} }
static uint64_t rd64(){ chk(8); uint64_t v; memcpy(&v,p,8); p+=8; return v; }
static uint32_t rd32(){ chk(4); uint32_t v; memcpy(&v,p,4); p+=4; return v; }
static void rds(char *b){ uint64_t l=rd64(); chk(l); if(l>500){fprintf(stderr,"huge str len=%llu off=%zu\n",(unsigned long long)l,(size_t)(p-base));exit(1);} memcpy(b,p,l); b[l]=0; p+=l; }
static void skipval(int t){
  switch(t){
    case 0: case 1: chk(1); p+=1;break;
    case 2: case 3: chk(2); p+=2;break;
    case 4: case 5: case 6: chk(4); p+=4;break;
    case 7: chk(1); p+=1;break;
    case 8: {uint64_t l=rd64(); chk(l); p+=l;}break;
    case 9: {int at=rd32(); uint64_t n=rd64();
      if(n>100000000ULL){fprintf(stderr,"huge arr n=%llu at=%d off=%zu\n",(unsigned long long)n,at,(size_t)(p-base));exit(1);}
      for(uint64_t i=0;i<n;i++) skipval(at);}break;
    case 10: case 11: case 12: chk(8); p+=8;break;
    default: fprintf(stderr,"bad type %d off=%zu\n",t,(size_t)(p-base)); exit(1);
  }
}
int main(int argc,char**argv){
  if(argc<2){fprintf(stderr,"usage: gguf-info <file.gguf>\n");return 1;}
  FILE*f=fopen(argv[1],"rb");
  fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
  base=malloc(sz); fread(base,1,sz,f); fclose(f);
  end=base+sz; p=base;
  rd32();rd32(); uint64_t nt=rd64(), nm=rd64();
  fprintf(stderr,"nt=%llu nm=%llu\n",(unsigned long long)nt,(unsigned long long)nm);
  char b[512];
  for(uint64_t i=0;i<nm;i++){
    rds(b); uint32_t t=rd32();
    fprintf(stderr,"meta[%llu] %s t=%u off=%zu\n",(unsigned long long)i,b,t,(size_t)(p-base));
    skipval(t);
  }
  size_t off=(size_t)(p-base), al=(off+31)&~(size_t)31;
  fprintf(stderr,"after-meta off=%zu aligned=%zu\n",off,al);
  p=base+off; /* tensor info follows metadata directly */
  for(uint64_t i=0;i<nt;i++){
    rds(b); uint32_t nd=rd32();
    uint64_t d[4]={0,0,0,0};
    for(uint32_t j=0;j<nd;j++){ uint64_t v=rd64(); if(j<4)d[j]=v; }
    uint32_t qt=rd32(); rd64(); /* offset */
    printf("%-42s t=%-2u",b,qt);
    for(uint32_t j=0;j<nd&&j<4;j++) printf(" %llu",(unsigned long long)d[j]);
    printf("\n");
  }
  return 0;
}
