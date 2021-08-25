#define _USE_MATH_DEFINES
#include <cuda_runtime.h>
#include <cstdio>
#include <math.h>
#include <chrono>
#include <iostream>
#define PI M_PI

cudaArray_t bxarr, byarr, bzarr;
cudaTextureObject_t fieldX, fieldY, fieldZ;
__device__ cudaTextureObject_t cufieldX, cufieldY, cufieldZ;
__global__ void loadTex(cudaTextureObject_t bx,cudaTextureObject_t by,cudaTextureObject_t bz){
    cufieldX = bx;
    cufieldY = by;
    cufieldZ = bz;
    printf("%lld, %lld, %lld\n",cufieldX,cufieldY,cufieldZ);
}
__global__ void Permute302(double* src, double* dstx, double* dsty, double* dstz){
    unsigned long long id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < 302*302*83){
        dstx[id] = src[id*3+0];
        dsty[id] = src[id*3+1];
        dstz[id] = src[id*3+2];
    }
}

int PrepareTexture(double* field){

    auto ChDesc = cudaCreateChannelDesc<uint2>();
    auto ext = make_cudaExtent(302,302,83);
    auto err = cudaMalloc3DArray(&bxarr, &ChDesc, ext, cudaArrayDefault);
    if(err != 0){
        printf("Malloc3D for Bx err = %d\n", err);
        return err;
    }
    err = cudaMalloc3DArray(&byarr, &ChDesc, ext, cudaArrayDefault);
    if(err != 0){
        printf("Malloc3D for By err = %d\n", err);
        return err;
    }
    err = cudaMalloc3DArray(&bzarr, &ChDesc, ext, cudaArrayDefault);
    if(err != 0){
        printf("Malloc3D for Bz err = %d\n", err);
        return err;
    }

    double* cufield;
    err = cudaMalloc(&cufield, 302*302*83*3*sizeof(double));
    err = cudaMemcpy(cufield, field, 302*302*83*3*sizeof(double), cudaMemcpyHostToDevice);
    double* cuFx;
    double* cuFy;
    double* cuFz;
    cudaMalloc(&cuFx, 302*302*83*sizeof(double));
    cudaMalloc(&cuFy, 302*302*83*sizeof(double));
    cudaMalloc(&cuFz, 302*302*83*sizeof(double));
    Permute302<<<(302*302*83+63)/64, 64>>>(cufield, cuFx, cuFy, cuFz);
    cudaFree(cufield);


    cudaMemcpy3DParms cpy3dp={0};
    cpy3dp.srcPtr.ptr = cuFx;
    cpy3dp.srcPtr.pitch = 302*sizeof(double);
    cpy3dp.srcPtr.xsize = 302; // z size
    cpy3dp.srcPtr.ysize = 302; // x size
    cpy3dp.dstArray = bxarr;
    cpy3dp.extent = ext;
    cpy3dp.kind = cudaMemcpyDeviceToDevice;
    err = cudaMemcpy3D(&cpy3dp);
    if(err != 0){
        printf("Memcpy for Bx err = %d\n", err);
        return err;
    }

    cpy3dp.srcPtr.ptr = cuFy;
    cpy3dp.dstArray = byarr;
    err = cudaMemcpy3D(&cpy3dp);
    if(err != 0){
        printf("Memcpy for By err = %d\n", err);
        return err;
    }

    cpy3dp.srcPtr.ptr = cuFz;
    cpy3dp.dstArray = bzarr;
    err = cudaMemcpy3D(&cpy3dp);
    if(err != 0){
        printf("Memcpy for Bz err = %d\n", err);
        return err;
    }
    cudaFree(cuFx);
    cudaFree(cuFy);
    cudaFree(cuFz);



    struct cudaResourceDesc resDesc;
    memset(&resDesc,0,sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = bxarr;
    struct cudaTextureDesc texDesc;
    memset(&texDesc,0,sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeBorder;
    texDesc.addressMode[1]   = cudaAddressModeBorder;
    texDesc.addressMode[2]   = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode   = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    err = cudaCreateTextureObject(&fieldX, &resDesc, &texDesc, NULL);
    if(err != 0){
        printf("Create Texture for Bx err = %d\n", err);
        return err;
    }

    resDesc.res.array.array = byarr;
    cudaCreateTextureObject(&fieldY, &resDesc, &texDesc, NULL);
    if(err != 0){
        printf("Create Texture for By err = %d\n", err);
        return err;
    }

    resDesc.res.array.array = bzarr;
    cudaCreateTextureObject(&fieldZ, &resDesc, &texDesc, NULL);
    if(err != 0){
        printf("Create Texture for Bz err = %d\n", err);
        return err;
    }
    loadTex<<<1,1>>>(fieldX,fieldY,fieldZ);
    return 0;
}

int DestroyTexture(){
    cudaDestroyTextureObject(fieldX);
    cudaFreeArray(bxarr);
    cudaDestroyTextureObject(fieldY);
    cudaFreeArray(byarr);
    cudaDestroyTextureObject(fieldZ);
    cudaFreeArray(bzarr);
    return 0;
}


__device__ double4 fetch0(double x, double y, double z){
    float x0 = __double2float_rd(x);
    float y0 = __double2float_rd(y);
    float z0 = __double2float_rd(z);
    uint2 bxu2 = tex3D<uint2>(cufieldX,z0,x0,y0);
    uint2 byu2 = tex3D<uint2>(cufieldY,z0,x0,y0);
    uint2 bzu2 = tex3D<uint2>(cufieldZ,z0,x0,y0);
    double rx = __hiloint2double(bxu2.y, bxu2.x);
    double ry = __hiloint2double(byu2.y, byu2.x);
    double rz = __hiloint2double(bzu2.y, bzu2.x);
    return double4{rx,ry,rz,0.0};
}

//not rz; not rd

__device__ double4 fma414(double4 a, double b, double4 c){
    return double4{fma(a.x,b,c.x),fma(a.y,b,c.y),fma(a.z,b,c.z),0.0};
}
//a*b-c
__device__ double4 fms4(double4 a, double4 b, double4 c){
    return double4{fma(a.x,b.x,-c.x),fma(a.y,b.y,-c.y),fma(a.z,b.z,-c.z),0.0};
}
__device__ double4 operator+(const double4 &a, const double4 &b) {
    return make_double4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}
__device__ double4 operator*(const double &a, const double4 &b) {
    return make_double4(a*b.x, a*b.y, a*b.z, a*b.w);
}

__device__ double4 fetch(double x0, double y0, double z0){
    double x = fabs(x0*0.1);
    double y = fma(y0,0.1,41.0);
    double z = fabs(z0*0.1);
    double4 r0 = fetch0(x, y, z);
    double4 r1 = fetch0(x+1.0, y, z);
    double4 r2 = fetch0(x, y+1.0, z);
    double4 r3 = fetch0(x+1.0, y+1.0, z);
    double4 r4 = fetch0(x, y, z+1.0);
    double4 r5 = fetch0(x+1.0, y, z+1.0);
    double4 r6 = fetch0(x, y+1.0, z+1.0);
    double4 r7 = fetch0(x+1.0, y+1.0, z+1.0);
    double rx = x - floor(x);
    double ry = y - floor(y);
    double rz = z - floor(z);
    
    r0 = fma414(r0, -rz, r0);
    r0 = fma414(r4,  rz, r0);
    r1 = fma414(r1, -rz, r1);
    r1 = fma414(r5,  rz, r1);
    r2 = fma414(r2, -rz, r2);
    r2 = fma414(r6,  rz, r2);
    r3 = fma414(r3, -rz, r3);
    r3 = fma414(r7,  rz, r3);

    r0 = fma414(r0, -rx, r0);
    r0 = fma414(r1,  rx, r0);
    r2 = fma414(r2, -rx, r2);
    r2 = fma414(r3,  rx, r2);

    r0 = fma414(r0, -ry, r0);
    r0 = fma414(r2,  ry, r0);


    if(x0 < 0.0){
        r0.x = - r0.x;
    }
    if(z0 < 0.0){
        r0.z = - r0.z;
    }
    return r0;
}

__device__ double4 eqf(double4 pos, double4 v){
    double4 B = fetch(pos.x, pos.y, pos.z);
    double ax = v.y*B.z;
    ax = fma(v.z, B.y, -ax);
    double ay = v.z*B.x;
    ay = fma(v.x, B.z, -ay);
    double az = v.x*B.y;
    az = fma(v.y, B.x, -az);
    return double4{ax, ay, az, 0.0};
}

__device__ void next(double4 p0, double4 v0, double4 &pr, double4 &vr, double h){
    double4 a0 = eqf(p0, v0);
    //k1 : (v0, a0)
    
    double4 p1 = fma414(v0,0.5*h,p0);
    double4 v1 = fma414(a0,0.5*h,v0);
    double4 a1 = eqf(p1, v1);
    //k2 : (v1, a1)

    double4 p2 = fma414(v1,0.5*h,p0);
    double4 v2 = fma414(a1,0.5*h,v0);
    double4 a2 = eqf(p2, v2);
    //k3 : (v2, a2)

    double4 p3 = fma414(v2,0.5*h,p0);
    double4 v3 = fma414(a2,0.5*h,v0);
    double4 a3 = eqf(p3,v3);
    //k4 : (v3, a3)

    //pr = p0 + (h/6.0)*(v0+(2*v1)+(2*v2)+v3);
    //vr = v0 + (h/6.0)*(a0+(2*a1)+(2*a2)+a3);
    double4 tmp1 = fma414(v1, 2.0, v0);
    double4 tmp2 = fma414(v2, 2.0, v3);
    pr = fma414(tmp1+tmp2, h/6.0, p0);
    
    tmp1 = fma414(a1, 2.0, a0);
    tmp2 = fma414(a2, 2.0, a3);
    vr = fma414(tmp1+tmp2, h/6.0, v0);
}

__device__ void trace(double4 r, double4 v, double angle, double dist, double h, double4 &rstr, double4 &rstv){
    double c0 = cos(angle*PI/180.0);
    double s0 = sin(angle*PI/180.0);
    double4 rnow = r;
    double4 vnow = v;
    double d0 = rnow.z*c0 + rnow.x*s0;
    double tot = 0;
    for(int i = 0; i < 50000; i++){
        double4 rnxt, vnxt;
        next(rnow, vnow, rnxt, vnxt, h);
        double d = rnxt.z*c0 + rnxt.x*s0;
        if(d>dist){
            double k = (dist-d0)/(d-d0);
            rnxt.x = fma(k,(rnxt.x-rnow.x),rnow.x);
            rnxt.y = fma(k,(rnxt.y-rnow.y),rnow.y);
            rnxt.z = fma(k,(rnxt.z-rnow.z),rnow.z);
            vnxt.x = fma(k,(vnxt.x-vnow.x),vnow.x);
            vnxt.y = fma(k,(vnxt.y-vnow.y),vnow.y);
            vnxt.z = fma(k,(vnxt.z-vnow.z),vnow.z);
            d0 = d;
            tot += sqrt((rnxt.x-rnow.x)*(rnxt.x-rnow.x)+(rnxt.y-rnow.y)*(rnxt.y-rnow.y)+(rnxt.z-rnow.z)*(rnxt.z-rnow.z));
            rnow = rnxt;
            vnow = vnxt;
            break;
        }
        tot += sqrt((rnxt.x-rnow.x)*(rnxt.x-rnow.x)+(rnxt.y-rnow.y)*(rnxt.y-rnow.y)+(rnxt.z-rnow.z)*(rnxt.z-rnow.z));
        rnow = rnxt;
        vnow = vnxt;
        d0 = d;
    }
    rnow.w = tot;
    //return rnow;
    rstr.x = rnow.x * c0 - rnow.z * s0;
    rstr.z = rnow.x * s0 + rnow.z * c0;
    rstr.y = rnow.y;
    rstr.w = tot;
    
    rstv.x = vnow.x * c0 - vnow.z * s0;
    rstv.z = vnow.x * s0 + vnow.z * c0;
    rstv.y = vnow.y;
}


__global__ void Trace(double* src, double* dst, double angle, double dist, unsigned long long N){
    unsigned long long id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id<N){
        double4 r0;
        double4 v0;
        double4 r1;
        double4 v1;
        r0.x = src[id*6+0];
        r0.y = src[id*6+1];
        r0.z = src[id*6+2];
        v0.x = src[id*6+3];
        v0.y = src[id*6+4];
        v0.z = src[id*6+5];
        trace(r0, v0, angle, dist, 0.001, r1, v1);
        dst[id*7+0]=r1.x;
        dst[id*7+1]=r1.y;
        dst[id*7+2]=r1.z;
        
        dst[id*7+3]=v1.x;
        dst[id*7+4]=v1.y;
        dst[id*7+5]=v1.z;
        
        dst[id*7+6]=r1.w;
    }
}


int main(){
    //Prepare B field texture
    printf("hello!%d\n",1);
    FILE* bf = fopen("./bmap.bin", "rb");
    double* bfield = (double*)(malloc(302*83*302*3*sizeof(double)));
    fread(bfield,sizeof(double)*3, 302*83*302,bf);
    PrepareTexture(bfield);
    free(bfield);
    fclose(bf);

    printf("hello!%d\n",2);
    //Prepare Input Data
    FILE* fin = fopen("input.bin", "rb");
    double* input;
    cudaMallocHost(&input, 1024000*6*sizeof(double),cudaHostAllocDefault);
    fread(input, 6*sizeof(double), 1024000, fin);
    fclose(fin);

    double* result;
    cudaMallocHost(&result, 1024000*7*sizeof(double),cudaHostAllocDefault);

    double** cuinput = (double**)(malloc(4*sizeof(double*)));
    double** curesult = (double**)(malloc(4*sizeof(double*)));
    //double* cuinput;
    //double* curesult;
    //cudaMalloc(&cuinput,1024000*6*sizeof(double));
    //cudaMalloc(&curesult,1024000*7*sizeof(double));
    
    cudaStream_t streams[4];
    for(int i = 0; i < 4; i++){
        cudaStreamCreate(&streams[i]);
        cudaMalloc(&(cuinput[i]),10240*6*sizeof(double));
        cudaMalloc(&(curesult[i]),10240*7*sizeof(double));
    }

    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    auto start = std::chrono::system_clock::now();

    for(int i = 0; i < 1024000; i+=10240){
        int sid = (i/10240)%3;
        cudaMemcpyAsync(cuinput[sid], input+i*6, 10240*6*sizeof(double), cudaMemcpyHostToDevice, streams[sid]);
        Trace<<<320,32,0,streams[sid]>>>(cuinput[sid],curesult[sid],15,4000.0,10240);
        cudaError_t err = cudaGetLastError();
        if(err != 0){
            printf("kernel err %d (i = %d, stream = %d)\n", err, i, sid);
        }
        cudaMemcpyAsync(result+i*7, curesult[sid],10240*7*sizeof(double), cudaMemcpyDeviceToHost, streams[sid]);
    }
    
    cudaDeviceSynchronize();

    auto end   = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout<<"total time = "<<double(duration.count())/1000.0<<" ms."<<std::endl;
    for(int i = 0; i < 1024000; i += 102410){
        printf("%d %16.10f %16.10f %16.10f %16.10f\n",i, result[i*7+0],result[i*7+1],result[i*7+2],result[i*7+6]);

    }
    FILE* of = fopen("result.bin", "wb");
    fwrite(result,7*sizeof(double),1024000,of);
    fclose(of);

    cudaFreeHost(input);
    cudaFreeHost(result);
    for(int i = 0; i < 4; i++){
        cudaFree(cuinput[i]);
        cudaFree(curesult[i]);
        cudaStreamDestroy(streams[i]);
    }
    free(cuinput);
    free(curesult);
    DestroyTexture();
}
