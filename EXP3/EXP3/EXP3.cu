#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <utility>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <ctime>
#include <cassert>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#define WIN32_LEAN_AND_MEAN
#define pb push_back 
#define all(c) (c).begin(),(c).end()
#include <Windows.h>
#include <MMSystem.h>
#pragma comment(lib, "winmm.lib")
using namespace std;
//groeten aan mijn nederlandse bezoekers!
#define _DTH cudaMemcpyDeviceToHost
#define _HTD cudaMemcpyHostToDevice
#define _DTD cudaMemcpyDeviceToDevice
//salutations à mes visiteurs français!
#define THREADS 256
#define INF (1<<30)
#define LARGE_NUM_CITIES 100
#define LARGE_K 60
#define LARGE_NUM_FANS 100
#define DO_GPU 1
//groeten aan mijn duitse bezoekers!
bool InitMMTimer(UINT wTimerRes);
void DestroyMMTimer(UINT wTimerRes, bool init);
const double eps=1e-8;
//saluti ai miei visitatori italiani!
inline bool _eq(double a,double b){return a+eps>=b && a-eps<=b;}

//for random data set generation (large)
int _gen_data_set(int *minJ,int *maxJ,int *minB,int *maxB,const int sz,const int fans_range0,const int fans_range1);

inline int _3d_flat(int i, int j, int k, int D1,int D0){return i*D1*D0+j*D0+k;}

double CPU_version(const int *minJ,const int *maxJ, const int *minB, const int *maxB, const int K,double *DP0, double *DP1,const int sz){//memset DP before call
	int totB=0,totJ=0;
	double ret=0.;
	for(int i=0;i<sz;i++){
		totB+=maxB[i];
		totJ+=maxJ[i];
	}
	const int bound=max(totB,totJ);
	const int D0=bound+1,D1=sz+1;
	DP0[0]=DP1[0]=1.;
	for(int i=0;i<sz;i++)for(int done=0;done<=K;done++){
		double p=double(K-done)/double(sz-i);
		for(int fans=0;fans<=bound;fans++){//Python????????????
			if(DP0[_3d_flat(i,done,fans,D1,D0)]>0.){
				DP0[_3d_flat(i+1,done,fans,D1,D0)]+=(1.-p)*DP0[_3d_flat(i,done,fans,D1,D0)];
				double r=double(maxJ[i]-minJ[i]+1);
				for(int f=minJ[i];f<=maxJ[i];f++){
					if(done<K && (fans+f)<=bound){
						DP0[_3d_flat(i+1,done+1,fans+f,D1,D0)]+=(p*DP0[_3d_flat(i,done,fans,D1,D0)])/r;
					}
				}
			}//???? ???? ?? ?? ???? ???? ????? ?????? ???????? ???? ???? ?????!
			if(DP1[_3d_flat(i,done,fans,D1,D0)]>0.){
				DP1[_3d_flat(i+1,done,fans,D1,D0)]+=(1.-p)*DP1[_3d_flat(i,done,fans,D1,D0)];
				double r=double(maxB[i]-minB[i]+1);
				for(int f=minB[i];f<=maxB[i];f++){
					if(done<K && (fans+f)<=bound){
						DP1[_3d_flat(i+1,done+1,fans+f,D1,D0)]+=(p*DP1[_3d_flat(i,done,fans,D1,D0)])/r;
					}
				}
			}
		}
	}
	for(int i=0;i<=bound;i++){
		ret+=DP0[_3d_flat(sz,K,i,D1,D0)]*DP1[_3d_flat(sz,K,i,D1,D0)];//????????????
	}
	return ret;
}

__device__ int D_3d_flat(int i, int j, int k, int D1,int D0){return D0*(i*D1+j)+k;}
__device__ double atomicAdd(double* address, double val){
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do{
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    }while (assumed != old);
    return __longlong_as_double(old);//Python est lent et ne devrait être utilisée par les gens paresseux!
}

//Ich hasse Python und Ruby!
__global__ void GPU_version(double *DP0, double *DP1, const int ii, const int bound, 
						const int sz,const int K,const int D1, const int D0,const int Jmin,
						const int Jmax,const int Bmin,const int Bmax,const double r0,const double r1){

	const int fans=threadIdx.x+blockIdx.x*blockDim.x;
	if(fans>bound)return;
	const int done=blockIdx.y;
	const int idx0=D_3d_flat(ii,done,fans,D1,D0);
	const double p=double(K-done)/double(sz-ii);

	int f;
	if(DP0[idx0]>0.){
		atomicAdd(&DP0[D_3d_flat(ii+1,done,fans,D1,D0)],((1.-p)*DP0[idx0]));
		if(done<K){
			for(f=Jmin;f<=Jmax;f++){
				if(fans+f<=bound){//Je suis simplement stupide graisse américain!
					atomicAdd(&DP0[D_3d_flat(ii+1,done+1,fans+f,D1,D0)],((r0*p*DP0[idx0])));
				}
			}
		}
	}
	if(DP1[idx0]>0.){//????? ? ??. ???? ??? ???? ??? ?? ??? ??? ?????? ?? ???!
		atomicAdd(&DP1[D_3d_flat(ii+1,done,fans,D1,D0)],((1.-p)*DP1[idx0]));
		if(done<K){
			for(f=Bmin;f<=Bmax;f++){
				if(fans+f<=bound){
					atomicAdd(&DP1[D_3d_flat(ii+1,done+1,fans+f,D1,D0)],((r1*p*DP1[idx0])));
				}
			}
		}
	}	
}

//Nicht alle Amerikaner sind Idioten..(nur mich!)
__global__ void GPU_last_step(const double *DP0,const double *DP1, const int bound, double *ret){
	const int offset=threadIdx.x+blockIdx.x*blockDim.x;
	__shared__ volatile double tot[THREADS];

	tot[threadIdx.x]= (offset<=bound) ? (DP0[offset]*DP1[offset]):double(0);
	__syncthreads();

	if(threadIdx.x<128){
		tot[threadIdx.x]+=tot[threadIdx.x+128];
	}
	__syncthreads();
	if(threadIdx.x<64){
		tot[threadIdx.x]+=tot[threadIdx.x+64];//???????
	}
	__syncthreads();
	if(threadIdx.x<32){
		tot[threadIdx.x]+=tot[threadIdx.x+32];
		tot[threadIdx.x]+=tot[threadIdx.x+16];
		tot[threadIdx.x]+=tot[threadIdx.x+8];
		tot[threadIdx.x]+=tot[threadIdx.x+4];
		tot[threadIdx.x]+=tot[threadIdx.x+2];
		tot[threadIdx.x]+=tot[threadIdx.x+1];
	}
	__syncthreads();
	if(threadIdx.x==0){//Deutsch Mädchen haben schöne Beine!
		atomicAdd(&ret[0],tot[0]);
	}
}

int main(){
	char ch;
	srand(time(NULL));
	const int K= LARGE_K;
	const int DPsize= (LARGE_NUM_CITIES+1)*(LARGE_NUM_CITIES+1)*((LARGE_NUM_CITIES+1)*(LARGE_NUM_FANS+1));//does not need to be this large, but just in case
	const int num_bytes=DPsize*sizeof(double);
	const int sz=LARGE_NUM_CITIES;
	double *DP0=(double *)malloc(num_bytes);
	double *DP1=(double *)malloc(num_bytes);
	//double *T0=(double *)malloc(num_bytes);
	int *minJ=(int *)malloc(LARGE_NUM_CITIES*sizeof(int));
	int *maxJ=(int *)malloc(LARGE_NUM_CITIES*sizeof(int));
	int *minB=(int *)malloc(LARGE_NUM_CITIES*sizeof(int));
	int *maxB=(int *)malloc(LARGE_NUM_CITIES*sizeof(int));

	const int d_bound_large=1+_gen_data_set(minJ,maxJ,minB,maxB,sz,LARGE_NUM_CITIES,LARGE_NUM_FANS);
	double CPU_ans=0.,GPU_ans=0.;
	//CPU
	cout<<"\nRunning CPU implementation..\n";
	UINT wTimerRes = 0;
	DWORD CPU_time=0,GPU_time=0;
	bool init = InitMMTimer(wTimerRes);
	DWORD startTime=timeGetTime();

	memset(DP0,0,num_bytes);
	memset(DP1,0,num_bytes);

	CPU_ans=CPU_version(minJ,maxJ,minB,maxB,K,DP0,DP1,LARGE_NUM_CITIES);
	
	DWORD endTime = timeGetTime();
	CPU_time=endTime-startTime;
	cout<<"CPU solution timing: "<<CPU_time<< " , answer= "<<CPU_ans<<'\n';
	DestroyMMTimer(wTimerRes, init);

	//GPU
	int compute_capability=0;
	cudaDeviceProp deviceProp;
	cudaError_t err=cudaGetDeviceProperties(&deviceProp, compute_capability);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	string ss= (deviceProp.major>=3 && deviceProp.minor>=5) ? "Capable!\n":"Not Sufficient compute capability!\n";
	cout<<ss;

	err=cudaDeviceReset();
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	cout<<"\nRunning GPU implementation..\n";
	if(DO_GPU && (deviceProp.major>=3 && deviceProp.minor>=5)){
		const int D_bound=d_bound_large;
		const int D0=D_bound,D1=sz+1;
		const int loc=_3d_flat(sz,K,0,D1,D0);
		const double ds=1.0;
		int ii=0,R0,R1;
		double r0,r1;
		double *D_DP0,*D_DP1,*D_ret;
		dim3 GridDim((D_bound+THREADS-1)/THREADS,K+1);//??????????????
		err=cudaMalloc((void**)&D_DP0,num_bytes);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMalloc((void**)&D_DP1,num_bytes);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMalloc((void**)&D_ret,sizeof(double));
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		wTimerRes = 0;
		init = InitMMTimer(wTimerRes);
		startTime = timeGetTime();
		
		err=cudaMemset(D_DP0,0,num_bytes);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMemset(D_DP1,0,num_bytes);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMemset(D_ret,0,sizeof(double));
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		
		err=cudaMemcpy(D_DP0,&ds,sizeof(double),_HTD);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMemcpy(D_DP1,&ds,sizeof(double),_HTD);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		
	
		for(;ii<sz;ii++){
			R0=(maxJ[ii]-minJ[ii]+1);R1=(maxB[ii]-minB[ii]+1);//Nederlandse meisjes zijn heel erg goed uit!
			r0=1./double(R0);r1=1./double(R1);
			GPU_version<<<GridDim,THREADS>>>(D_DP0,D_DP1,ii,D_bound,sz,K,D1,D0,minJ[ii],maxJ[ii],minB[ii],maxB[ii],r0,r1);
			err = cudaThreadSynchronize();
			if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		}

		

		GPU_last_step<<<((D_bound+THREADS-1)/THREADS),THREADS>>>(D_DP0+loc,D_DP1+loc,D_bound,D_ret);
		err = cudaThreadSynchronize();
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		err=cudaMemcpy(&GPU_ans,D_ret,sizeof(double),_DTH);//
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		endTime = timeGetTime();//Filles françaises sont nombreuses et très intelligent!
		GPU_time=endTime-startTime;
		cout<<"CUDA timing: "<<GPU_time<<" , answer= "<<GPU_ans<<'\n';
		DestroyMMTimer(wTimerRes, init);

		err=cudaFree(D_DP0);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaFree(D_DP1);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaFree(D_ret);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	}
	if(_eq(GPU_ans,CPU_ans)){
		cout<<"\nSuccess! The CPU answer and the GPU answer are equal! The CUDA implementation was "<<double(CPU_time)/double(GPU_time)<<" faster than the CPU serial implementation.\n";
	}else{
		cout<<"Error in calculation!\n";
	}
	
	free(DP0);
	free(DP1);
	//free(T0);
	free(minJ);
	free(maxJ);
	free(minB);
	free(maxB);
	
	std::cin>>ch;
	return 0;
}

bool InitMMTimer(UINT wTimerRes){
	TIMECAPS tc;
	if (timeGetDevCaps(&tc, sizeof(TIMECAPS)) != TIMERR_NOERROR) {return false;}
	wTimerRes = min(max(tc.wPeriodMin, 1), tc.wPeriodMax);
	timeBeginPeriod(wTimerRes); 
	return true;
}
void DestroyMMTimer(UINT wTimerRes, bool init){
	if(init)
		timeEndPeriod(wTimerRes);
}
int _gen_data_set(int *minJ,int *maxJ,int *minB,int *maxB,const int sz,const int fans_range0,const int fans_range1){
	int ret=0;
	assert(fans_range0>2 && fans_range1>2 && fans_range0<=sz && fans_range1<=sz);
	int mnfj,mxfj,mnfb,mxfb;
	for(int i=0;i<sz;i++){

		mnfj=(rand()%fans_range0)+1;
		mxfj=(rand()%fans_range0)+1;

		if(mnfj>mxfj)swap(mnfj,mxfj);

		mnfb=(rand()%fans_range1)+1;
		mxfb=(rand()%fans_range1)+1;

		if(mnfb>mxfb)swap(mnfb,mxfb);

		if(mxfb>mxfj){
			mxfj=mxfb-(rand()%3);
		}else if(mxfj>mxfb){
			mxfb=mxfj-(rand()%3);
		}
		minJ[i]=mnfj;maxJ[i]=mxfj;
		minB[i]=mnfb;maxB[i]=mxfb;
		
		ret+=max(mxfj,mxfb);
	}
	return ret;
}



